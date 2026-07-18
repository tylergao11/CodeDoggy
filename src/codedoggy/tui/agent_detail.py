"""Full-fidelity Agent transcript model and prompt-toolkit renderer.

This module is intentionally independent from :mod:`codedoggy.tui.app` so the
detail surface can be built and tested while the main task cockpit evolves.
The eventual integration only needs to feed real messages/tool records into
``AgentDetailStore`` and place the returned formatted-text fragments in a
scrollable ``FormattedTextControl``.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Any, Iterable, Literal

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.utils import get_cwidth


DetailCategory = Literal["message", "tool", "file", "test"]
DetailBlockKind = Literal[
    "text",
    "metadata",
    "code",
    "diff",
    "command",
    "output",
]
DetailFilter = Literal["all", "message", "tool", "file", "test"]

DETAIL_FILTERS: tuple[DetailFilter, ...] = (
    "all",
    "message",
    "tool",
    "file",
    "test",
)
DETAIL_FILTER_LABELS: dict[DetailFilter, str] = {
    "all": "全部",
    "message": "消息",
    "tool": "工具",
    "file": "文件",
    "test": "测试",
}

# Kept separate so the final integration can merge these names into the
# cockpit Style without importing or mutating app.py from this module.
DETAIL_STYLE_RULES = {
    "detail.header": "bg:#0b0b0d #f5f5f7 bold",
    "detail.meta": "bg:#0b0b0d #8e8e93",
    "detail.active": "bg:#0b0b0d #64d2ff bold",
    "detail.separator": "bg:#0b0b0d #2c2c2e",
    "detail.text": "bg:#0b0b0d #f5f5f7",
    "detail.actor": "bg:#0b0b0d #64d2ff bold",
    "detail.tool": "bg:#0b0b0d #64d2ff",
    "detail.block": "bg:#141416 #d1d1d6",
    "detail.code": "bg:#141416 #d1d1d6",
    "detail.diff.add": "bg:#141416 #30d158",
    "detail.diff.remove": "bg:#141416 #ff453a",
    "detail.diff.hunk": "bg:#141416 #64d2ff",
    "detail.success": "bg:#141416 #30d158",
    "detail.error": "bg:#141416 #ff453a",
    "detail.warning": "bg:#141416 #ff9f0a",
}


@dataclass(frozen=True, slots=True)
class DetailBlock:
    """One complete, non-truncated block inside an Agent record."""

    kind: DetailBlockKind
    text: str
    status: str = "normal"
    label: str = ""


@dataclass(frozen=True, slots=True)
class DetailRecord:
    """One user-visible Agent message or tool execution record."""

    id: str
    sequence: int
    actor: str
    category: DetailCategory
    title: str
    blocks: tuple[DetailBlock, ...] = ()
    timestamp: str = ""
    status: str = "completed"


@dataclass(frozen=True, slots=True)
class AgentDetailSnapshot:
    """Immutable view consumed by the detail renderer."""

    task_id: str
    agent_id: str
    agent_label: str
    task_title: str
    status: str = "running"
    started_at: float = 0.0
    records: tuple[DetailRecord, ...] = ()


@dataclass(slots=True)
class _TranscriptState:
    task_id: str
    agent_id: str
    agent_label: str
    task_title: str
    status: str
    started_at: float
    records: dict[str, DetailRecord] = field(default_factory=dict)
    next_sequence: int = 1


class AgentDetailStore:
    """Thread-safe source for full Agent transcripts.

    ``upsert`` is deliberate: streamed Agent messages and in-flight tool calls
    can update one record without producing hundreds of near-duplicate rows.
    Existing records retain their original sequence and timestamp.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._states: dict[tuple[str, str], _TranscriptState] = {}

    def open(
        self,
        task_id: str,
        agent_id: str,
        *,
        agent_label: str,
        task_title: str,
        status: str = "running",
        started_at: float | None = None,
    ) -> AgentDetailSnapshot:
        key = (task_id, agent_id)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = _TranscriptState(
                    task_id=task_id,
                    agent_id=agent_id,
                    agent_label=_clean_label(agent_label),
                    task_title=_clean_text(task_title) or "未命名任务",
                    status=status,
                    started_at=time.monotonic() if started_at is None else started_at,
                )
                self._states[key] = state
            else:
                state.agent_label = _clean_label(agent_label)
                state.task_title = _clean_text(task_title) or state.task_title
                state.status = status
            return _snapshot(state)

    def upsert(
        self,
        task_id: str,
        agent_id: str,
        *,
        record_id: str,
        actor: str,
        category: DetailCategory,
        title: str,
        blocks: Iterable[DetailBlock] = (),
        timestamp: str | None = None,
        status: str = "completed",
    ) -> DetailRecord:
        key = (task_id, agent_id)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = _TranscriptState(
                    task_id=task_id,
                    agent_id=agent_id,
                    agent_label=_clean_label(actor),
                    task_title="未命名任务",
                    status="running",
                    started_at=time.monotonic(),
                )
                self._states[key] = state
            old = state.records.get(record_id)
            if old is None:
                sequence = state.next_sequence
                state.next_sequence += 1
                record_timestamp = timestamp or _clock_text()
            else:
                sequence = old.sequence
                record_timestamp = old.timestamp if timestamp is None else timestamp
            record = DetailRecord(
                id=record_id,
                sequence=sequence,
                actor=_clean_label(actor),
                category=_normalize_category(category),
                title=_clean_text(title),
                blocks=tuple(blocks),
                timestamp=record_timestamp,
                status=status,
            )
            state.records[record_id] = record
            return deepcopy(record)

    def set_status(self, task_id: str, agent_id: str, status: str) -> None:
        with self._lock:
            state = self._states.get((task_id, agent_id))
            if state is not None:
                state.status = status

    def snapshot(self, task_id: str, agent_id: str) -> AgentDetailSnapshot | None:
        with self._lock:
            state = self._states.get((task_id, agent_id))
            return None if state is None else _snapshot(state)


def snapshot_from_messages(
    messages: Iterable[Any],
    *,
    task_id: str,
    agent_id: str,
    agent_label: str,
    task_title: str,
    status: str = "running",
) -> AgentDetailSnapshot:
    """Build a full detail snapshot from existing OpenAI-style messages.

    This adapter is intentionally duck-typed so it can consume both CodeDoggy
    ``Message`` instances and restored session records. System/user prompts are
    not duplicated in the Agent detail page. Assistant prose, tool arguments,
    tool outputs, code, diffs and command results remain visible.
    """

    records: list[DetailRecord] = []
    tool_positions: dict[str, int] = {}
    sequence = 1
    for message in messages:
        role = getattr(message, "role", "")
        role_value = str(getattr(role, "value", role)).lower()
        if role_value == "assistant":
            content = _clean_text(getattr(message, "content", ""))
            if content:
                records.append(
                    DetailRecord(
                        id=f"message-{sequence}",
                        sequence=sequence,
                        actor=_clean_label(agent_label),
                        category="message",
                        title="进度",
                        blocks=(DetailBlock("text", content),),
                        timestamp=f"#{sequence:03d}",
                    )
                )
                sequence += 1
            for call in list(getattr(message, "tool_calls", None) or []):
                call_id = str(getattr(call, "id", "") or f"tool-{sequence}")
                name = str(getattr(call, "name", "tool") or "tool")
                arguments = getattr(call, "arguments", {})
                category = _tool_category(name, arguments)
                record = DetailRecord(
                    id=call_id,
                    sequence=sequence,
                    actor="TOOL",
                    category=category,
                    title=f"TOOL · {name}",
                    blocks=(_arguments_block(name, arguments),),
                    timestamp=f"#{sequence:03d}",
                    status="running",
                )
                tool_positions[call_id] = len(records)
                records.append(record)
                sequence += 1
        elif role_value == "tool":
            call_id = str(getattr(message, "tool_call_id", "") or "")
            name = str(getattr(message, "name", "") or "tool")
            output = str(getattr(message, "content", "") or "")
            result_block = _tool_result_block(name, output)
            if call_id in tool_positions:
                position = tool_positions[call_id]
                old = records[position]
                records[position] = replace(
                    old,
                    blocks=old.blocks + (result_block,),
                    status=_result_status(output),
                )
            else:
                records.append(
                    DetailRecord(
                        id=call_id or f"tool-result-{sequence}",
                        sequence=sequence,
                        actor="TOOL",
                        category=_tool_category(name, {}),
                        title=f"TOOL · {name}",
                        blocks=(result_block,),
                        timestamp=f"#{sequence:03d}",
                        status=_result_status(output),
                    )
                )
                sequence += 1
    return AgentDetailSnapshot(
        task_id=task_id,
        agent_id=agent_id,
        agent_label=_clean_label(agent_label),
        task_title=_clean_text(task_title) or "未命名任务",
        status=status,
        records=tuple(records),
    )


def filter_detail_records(
    records: Iterable[DetailRecord], active_filter: DetailFilter = "all"
) -> tuple[DetailRecord, ...]:
    """Return records for one UI tab without discarding stored detail."""

    active = active_filter if active_filter in DETAIL_FILTERS else "all"
    if active == "all":
        return tuple(records)
    if active == "tool":
        return tuple(item for item in records if item.category in {"tool", "file", "test"})
    return tuple(item for item in records if item.category == active)


def render_detail_header(
    snapshot: AgentDetailSnapshot,
    width: int,
    *,
    active_filter: DetailFilter = "all",
    elapsed_seconds: float | None = None,
) -> StyleAndTextTuples:
    """Render the compact fixed header and filter row."""

    width = max(36, width)
    elapsed = _elapsed_text(elapsed_seconds)
    right = f"{_status_text(snapshot.status)}"
    if elapsed:
        right += f" · {elapsed}"
    title = f"{snapshot.agent_label} · {snapshot.task_title}"
    title_budget = max(8, width - get_cwidth(right) - 5)
    title = _truncate_display(title, title_budget)
    gap = max(1, width - get_cwidth(title) - get_cwidth(right) - 2)
    fragments: StyleAndTextTuples = [
        ("class:detail.header", title),
        ("", " " * gap),
        ("class:detail.active", right),
        ("", "\n"),
        ("class:detail.separator", "─" * width + "\n"),
    ]
    for item in DETAIL_FILTERS:
        label = f"[{DETAIL_FILTER_LABELS[item]}]"
        style = "class:detail.active" if item == active_filter else "class:detail.meta"
        fragments.extend([(style, label), ("", "  ")])
    fragments.extend(
        [
            ("", "\n"),
            ("class:detail.separator", "─" * width + "\n"),
        ]
    )
    return fragments


def render_detail_body(
    snapshot: AgentDetailSnapshot,
    width: int,
    *,
    active_filter: DetailFilter = "all",
) -> StyleAndTextTuples:
    """Render every selected record without summarizing or truncating bodies."""

    width = max(36, width)
    records = filter_detail_records(snapshot.records, active_filter)
    if not records:
        return [("class:detail.meta", "\n  当前分类没有记录。\n")]
    fragments: StyleAndTextTuples = []
    for index, record in enumerate(records):
        if index:
            fragments.append(("class:detail.separator", "─" * width + "\n"))
        fragments.extend(_render_record(record, width))
    return fragments


def render_agent_detail(
    snapshot: AgentDetailSnapshot,
    width: int,
    *,
    active_filter: DetailFilter = "all",
    elapsed_seconds: float | None = None,
) -> StyleAndTextTuples:
    """Convenience composition for a complete scrollable detail document."""

    return render_detail_header(
        snapshot,
        width,
        active_filter=active_filter,
        elapsed_seconds=elapsed_seconds,
    ) + render_detail_body(snapshot, width, active_filter=active_filter)


def _render_record(record: DetailRecord, width: int) -> StyleAndTextTuples:
    timestamp = record.timestamp or f"#{record.sequence:03d}"
    actor = record.actor or "AGENT"
    header = f"{timestamp:<9}  {actor:<10}  {record.title}".rstrip()
    fragments: StyleAndTextTuples = []
    for line in _wrap_display(header, width):
        fragments.extend(
            [
                ("class:detail.actor", line),
                ("", "\n"),
            ]
        )
    body_width = max(12, width - 4)
    for block in record.blocks:
        if block.label:
            fragments.extend(
                [
                    ("class:detail.meta", f"  {block.label}\n"),
                ]
            )
        if block.kind in {"code", "diff", "command", "output", "metadata"}:
            fragments.append(("class:detail.separator", "  ┌" + "─" * (width - 4) + "┐\n"))
            for raw_line in block.text.splitlines() or [""]:
                wrapped = _wrap_display(raw_line, body_width)
                for line in wrapped:
                    style = _block_line_style(block, line)
                    padding = max(0, body_width - get_cwidth(line))
                    fragments.extend(
                        [
                            ("class:detail.separator", "  │"),
                            (style, line + " " * padding),
                            ("class:detail.separator", "│\n"),
                        ]
                    )
            fragments.append(("class:detail.separator", "  └" + "─" * (width - 4) + "┘\n"))
        else:
            for paragraph_line in block.text.splitlines() or [""]:
                for line in _wrap_display(paragraph_line, body_width):
                    fragments.extend(
                        [
                            ("class:detail.text", "  " + line),
                            ("", "\n"),
                        ]
                    )
    return fragments


def _arguments_block(name: str, arguments: Any) -> DetailBlock:
    if isinstance(arguments, str):
        text = arguments
    elif isinstance(arguments, dict):
        command = arguments.get("command") or arguments.get("cmd")
        if command is not None and _tool_category(name, arguments) == "test":
            return DetailBlock("command", f"$ {command}", label="调用参数")
        lines: list[str] = []
        for key, value in arguments.items():
            rendered = value if isinstance(value, str) else json.dumps(
                value, ensure_ascii=False, sort_keys=True
            )
            lines.append(f"{key}: {rendered}")
        text = "\n".join(lines) if lines else "{}"
    else:
        text = json.dumps(arguments, ensure_ascii=False, default=str)
    kind: DetailBlockKind = "diff" if "patch" in name.lower() else "metadata"
    return DetailBlock(kind, text, label="调用参数")


def _tool_result_block(name: str, output: str) -> DetailBlock:
    lowered = name.lower()
    if "patch" in lowered or _looks_like_diff(output):
        kind: DetailBlockKind = "diff"
    elif lowered in {"read_file", "read", "view_file"}:
        kind = "code"
    elif lowered in {"shell", "run_terminal_cmd", "run_command", "bash"}:
        kind = "output"
    else:
        kind = "output"
    return DetailBlock(kind, output, status=_result_status(output), label="返回结果")


def _tool_category(name: str, arguments: Any) -> DetailCategory:
    lowered = name.lower()
    argument_text = json.dumps(arguments, ensure_ascii=False, default=str).lower()
    if lowered in {"shell", "run_terminal_cmd", "run_command", "bash"} and any(
        marker in argument_text
        for marker in ("pytest", "unittest", "npm test", "cargo test", "vitest")
    ):
        return "test"
    if any(marker in lowered for marker in ("file", "patch", "write", "replace")):
        return "file"
    return "tool"


def _result_status(output: str) -> str:
    lowered = output.lower()
    if any(marker in lowered for marker in ("error", "failed", "traceback", "exception")):
        return "error"
    if any(marker in lowered for marker in ("passed", "success", "exit code: 0", "exit 0")):
        return "success"
    return "normal"


def _block_line_style(block: DetailBlock, line: str) -> str:
    if block.kind == "diff":
        if line.startswith("+++") or line.startswith("---"):
            return "class:detail.diff.hunk"
        if line.startswith("+"):
            return "class:detail.diff.add"
        if line.startswith("-"):
            return "class:detail.diff.remove"
        if line.startswith("@@"):
            return "class:detail.diff.hunk"
    if block.status == "success":
        return "class:detail.success"
    if block.status == "error":
        return "class:detail.error"
    if block.status == "warning":
        return "class:detail.warning"
    return "class:detail.block" if block.kind != "code" else "class:detail.code"


def _wrap_display(text: str, width: int) -> list[str]:
    """Wrap by terminal cell width while preserving every source character."""

    width = max(1, width)
    if not text:
        return [""]
    lines: list[str] = []
    current: list[str] = []
    used = 0
    for char in text:
        char_width = max(0, get_cwidth(char))
        if current and used + char_width > width:
            lines.append("".join(current))
            current = []
            used = 0
        current.append(char)
        used += char_width
    lines.append("".join(current))
    return lines


def _truncate_display(text: str, width: int) -> str:
    if get_cwidth(text) <= width:
        return text
    if width <= 1:
        return "…"
    out: list[str] = []
    used = 0
    for char in text:
        char_width = max(0, get_cwidth(char))
        if used + char_width > width - 1:
            break
        out.append(char)
        used += char_width
    return "".join(out).rstrip() + "…"


def _looks_like_diff(text: str) -> bool:
    return "\n+++ " in text or "\n--- " in text or "\n@@ " in text


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_label(value: Any) -> str:
    return _clean_text(value).upper() or "AGENT"


def _normalize_category(value: str) -> DetailCategory:
    return value if value in {"message", "tool", "file", "test"} else "tool"  # type: ignore[return-value]


def _clock_text() -> str:
    return time.strftime("%H:%M:%S")


def _elapsed_text(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = max(0, seconds)
    minutes, remain = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes:02d}:{remain:02d}"
    return f"{seconds:.1f}s"


def _status_text(status: str) -> str:
    return {
        "waiting": "等待",
        "pending": "准备中",
        "running": "推进中",
        "completed": "已完成",
        "failed": "失败",
        "cancelled": "已取消",
        "max_turns": "需继续",
    }.get(status, status)


def _snapshot(state: _TranscriptState) -> AgentDetailSnapshot:
    records = tuple(sorted(state.records.values(), key=lambda item: item.sequence))
    return AgentDetailSnapshot(
        task_id=state.task_id,
        agent_id=state.agent_id,
        agent_label=state.agent_label,
        task_title=state.task_title,
        status=state.status,
        started_at=state.started_at,
        records=deepcopy(records),
    )


__all__ = [
    "DETAIL_FILTERS",
    "DETAIL_FILTER_LABELS",
    "DETAIL_STYLE_RULES",
    "AgentDetailSnapshot",
    "AgentDetailStore",
    "DetailBlock",
    "DetailRecord",
    "filter_detail_records",
    "render_agent_detail",
    "render_detail_body",
    "render_detail_header",
    "snapshot_from_messages",
]
