"""Full-fidelity Agent transcript model and prompt-toolkit renderer.

This module is intentionally independent from :mod:`codedoggy.tui.app`: it
adapts the runtime transcript into immutable view data and renders it.  The
runtime transcript remains the single source of truth.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from collections.abc import Callable
from typing import Any, Iterable, Literal

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.utils import get_cwidth

from codedoggy.tui.open_path import paths_from_detail_record


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
    "detail.header": "bg:#0b0b0d #ff2d9a bold",
    "detail.meta": "bg:#0b0b0d #6f8791",
    "detail.active": "bg:#0b0b0d #16dfe8 bold",
    "detail.separator": "bg:#0b0b0d #123b43",
    "detail.border.left": "bg:#0b0b0d #8f1b58",
    "detail.border.right": "bg:#0b0b0d #0b6670",
    "detail.text": "bg:#0b0b0d #e8f2f2",
    "detail.actor": "bg:#0b0b0d #16dfe8 bold",
    "detail.actor.user": "bg:#0b0b0d #ff2d9a bold",
    "detail.actor.assistant": "bg:#0b0b0d #16dfe8 bold",
    "detail.actor.tool": "bg:#0b0b0d #ffb13b bold",
    "detail.tool": "bg:#0b0b0d #16dfe8",
    "detail.block": "bg:#071318 #cbdada",
    "detail.code": "bg:#071318 #f5f5f7",
    "detail.diff.add": "bg:#071318 #16dfe8",
    "detail.diff.remove": "bg:#071318 #ff2d9a",
    "detail.diff.hunk": "bg:#071318 #ffb13b",
    "detail.success": "bg:#071318 #16dfe8",
    "detail.error": "bg:#071318 #ff2d9a",
    "detail.warning": "bg:#071318 #ffd43b",
    "detail.link": "bg:#0b0b0d #16dfe8 bold underline",
    "detail.link.hint": "bg:#0b0b0d #ff9a3c bold",
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
    ``Message`` instances and restored session records. System prompts stay
    hidden. User instructions, assistant prose, tool arguments, tool outputs,
    code, diffs and command results remain visible. The initial instruction is
    suppressed only when it exactly duplicates the task title.
    """

    records: list[DetailRecord] = []
    tool_positions: dict[str, int] = {}
    sequence = 1
    for message in messages:
        role = _read_field(message, "role", "")
        role_value = str(getattr(role, "value", role)).lower()
        if role_value == "user":
            content = _clean_text(_read_field(message, "content", ""))
            if content and not (not records and content == _clean_text(task_title)):
                records.append(
                    DetailRecord(
                        id=f"message-{sequence}",
                        sequence=sequence,
                        actor="USER",
                        category="message",
                        title="补充指令",
                        blocks=(DetailBlock("text", content),),
                        timestamp=f"#{sequence:03d}",
                    )
                )
                sequence += 1
        elif role_value == "assistant":
            content = _clean_text(_read_field(message, "content", ""))
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
            for call in list(_read_field(message, "tool_calls", None) or []):
                call_id = str(_read_field(call, "id", "") or f"tool-{sequence}")
                name = str(_read_field(call, "name", "tool") or "tool")
                arguments = _read_field(call, "arguments", {})
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
            call_id = str(_read_field(message, "tool_call_id", "") or "")
            name = str(_read_field(message, "name", "") or "tool")
            output = str(_read_field(message, "content", "") or "")
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

    width = max(12, width)
    elapsed = _elapsed_text(elapsed_seconds)
    right = f"{_status_text(snapshot.status)}"
    if elapsed:
        right += f" · {elapsed}"
    title = f"{snapshot.agent_label} · {snapshot.task_title}"
    if get_cwidth(title) + get_cwidth(right) + 2 <= width:
        gap = width - get_cwidth(title) - get_cwidth(right)
        fragments: StyleAndTextTuples = [
            ("class:detail.header", title),
            ("", " " * gap),
            ("class:detail.active", right),
            ("", "\n"),
        ]
    else:
        fragments = [
            ("class:detail.header", _truncate_display(title, width)),
            ("", "\n"),
            ("class:detail.active", _truncate_display(right, width)),
            ("", "\n"),
        ]
    fragments.append(("class:detail.separator", "─" * width + "\n"))
    used = 0
    for item in DETAIL_FILTERS:
        label = f"[{DETAIL_FILTER_LABELS[item]}]"
        item_width = get_cwidth(label) + (2 if used else 0)
        if used and used + item_width > width:
            fragments.append(("", "\n"))
            used = 0
            item_width = get_cwidth(label)
        if used:
            fragments.append(("", "  "))
        style = "class:detail.active" if item == active_filter else "class:detail.meta"
        fragments.append((style, label))
        used += item_width
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
    path_mouse: Callable[[str], Any] | None = None,
) -> StyleAndTextTuples:
    """Render every selected record without summarizing or truncating bodies.

    ``path_mouse(path)`` optional: returns a prompt_toolkit mouse handler so
    image_gen paths open in the OS viewer on click (Grok-style link affordance).
    """

    width = max(12, width)
    records = filter_detail_records(snapshot.records, active_filter)
    if not records:
        return [("class:detail.meta", "\n  当前分类没有记录。\n")]
    fragments: StyleAndTextTuples = []
    for index, record in enumerate(records):
        if index:
            fragments.extend(
                [
                    ("class:detail.border.left", "╾"),
                    ("class:detail.separator", "┈" * max(1, width - 2)),
                    ("class:detail.border.right", "╼\n"),
                ]
            )
        fragments.extend(_render_record(record, width))
        if path_mouse is not None:
            for image_path in paths_from_detail_record(record):
                short = image_path
                try:
                    from pathlib import Path as _P

                    short = _P(image_path).name or image_path
                except Exception:  # noqa: BLE001
                    pass
                label = f"  ╭ ∪ 点击打开 {short} ╮"
                label = _truncate_display(label, width)
                handler = path_mouse(image_path)
                if handler is not None:
                    fragments.append(("class:detail.link.hint", label, handler))
                    fragments.append(("", "\n"))
                    hint = _truncate_display(f"     {image_path}", width)
                    fragments.append(("class:detail.link", hint, handler))
                    fragments.append(("", "\n"))
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
    fragments: StyleAndTextTuples = []
    prefix = f"{timestamp:<9}  "
    actor_piece = f"{actor:<10}  "
    actor_style = _actor_style(actor)
    if get_cwidth(prefix + actor_piece) < width:
        fragments.extend(
            [
                ("class:detail.meta", prefix),
                (actor_style, actor_piece),
                (
                    "class:detail.text",
                    _truncate_display(
                        record.title,
                        max(1, width - get_cwidth(prefix + actor_piece)),
                    ),
                ),
                ("", "\n"),
            ]
        )
    else:
        fragments.extend(
            [
                ("class:detail.meta", _truncate_display(timestamp, width)),
                ("", "\n"),
                (actor_style, _truncate_display(actor, width)),
                ("", "\n"),
                ("class:detail.text", _truncate_display(record.title, width)),
                ("", "\n"),
            ]
        )
    body_width = max(1, width - 4)
    for block in record.blocks:
        if block.label:
            fragments.extend(
                [
                    ("class:detail.meta", f"  {block.label}\n"),
                ]
            )
        if block.kind in {"code", "diff", "command", "output", "metadata"}:
            fragments.extend(
                [
                    ("class:detail.border.left", "  ┌"),
                    ("class:detail.separator", "─" * (width - 4)),
                    ("class:detail.border.right", "┐\n"),
                ]
            )
            for raw_line in block.text.splitlines() or [""]:
                wrapped = _wrap_display(raw_line, body_width)
                for line in wrapped:
                    style = _block_line_style(block, line)
                    padding = max(0, body_width - get_cwidth(line))
                    fragments.extend(
                        [
                            ("class:detail.border.left", "  │"),
                            (style, line + " " * padding),
                            ("class:detail.border.right", "│\n"),
                        ]
                    )
            fragments.extend(
                [
                    ("class:detail.border.left", "  └"),
                    ("class:detail.separator", "─" * (width - 4)),
                    ("class:detail.border.right", "┘\n"),
                ]
            )
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


def _actor_style(actor: str) -> str:
    normalized = actor.strip().upper()
    if normalized == "USER":
        return "class:detail.actor.user"
    if normalized in {"TOOL", "SYSTEM"}:
        return "class:detail.actor.tool"
    if normalized in {"ASSISTANT", "AGENT", "MAIN"}:
        return "class:detail.actor.assistant"
    return "class:detail.actor"


def _arguments_block(name: str, arguments: Any) -> DetailBlock:
    if isinstance(arguments, str):
        text = arguments
    elif isinstance(arguments, dict):
        command = arguments.get("command") or arguments.get("cmd")
        if command is not None and name.lower() in {
            "shell",
            "run_terminal_cmd",
            "run_command",
            "bash",
        }:
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
    hard_error = any(marker in lowered for marker in ("traceback", "exception", "error:"))
    failed_count = re.search(r"\b([1-9]\d*)\s+failed\b", lowered)
    if hard_error or failed_count:
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


def _read_field(value: Any, name: str, default: Any = None) -> Any:
    """Read one field from either a runtime dataclass or serialized mapping."""

    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _clean_label(value: Any) -> str:
    return _clean_text(value).upper() or "AGENT"


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


__all__ = [
    "DETAIL_FILTERS",
    "DETAIL_FILTER_LABELS",
    "DETAIL_STYLE_RULES",
    "AgentDetailSnapshot",
    "DetailBlock",
    "DetailFilter",
    "DetailRecord",
    "filter_detail_records",
    "render_agent_detail",
    "render_detail_body",
    "render_detail_header",
    "snapshot_from_messages",
]
