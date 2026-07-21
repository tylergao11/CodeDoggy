"""Full-fidelity Agent transcript model and prompt-toolkit renderer.

This module is intentionally independent from :mod:`codedoggy.tui.app`: it
adapts the runtime transcript into immutable view data and renders it.  The
runtime transcript remains the single source of truth.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.utils import get_cwidth

from codedoggy.tui.open_path import (
    link_label_for_path,
    paths_from_detail_record,
    tool_paths_from_arguments,
)
from codedoggy.tui.theme import DETAIL_STYLE_RULES

# Grok injects plan/MCP/scheduler hints as <system-reminder>...</system-reminder>
# on the model-facing user turn. Never paint those in the TUI transcript.
_SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder>\s*.*?\s*</system-reminder>",
    re.IGNORECASE | re.DOTALL,
)


DetailCategory = Literal["message", "tool", "file", "test"]
DetailBlockKind = Literal[
    "text",
    "thinking",
    "metadata",
    "code",
    "diff",
    "command",
    "output",
]
# UI tabs: message / tool always; plan when task is in plan lifecycle.
DetailFilter = Literal["message", "tool", "plan"]

DETAIL_FILTERS: tuple[DetailFilter, ...] = (
    "message",
    "tool",
    "plan",
)
DETAIL_FILTER_LABELS: dict[DetailFilter, str] = {
    "message": "消息",
    "tool": "工具",
    "plan": "计划",
}

# GrokBuild [scrollback.blocks.thinking] truncated_lines default.
THINKING_PREVIEW_LINES = 3
# GrokBuild max_thoughts_width spirit (soft cap for expanded body).
MAX_THOUGHTS_WIDTH = 120
# Paint-time caps: plan-mode memory tools can dump multi-MB blobs; wrapping
# every character freezes the TUI into a "dead loop" when opening detail.
MAX_BLOCK_CHARS = 12_000
THINKING_LABEL = "思考过程"


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
    # Full filesystem paths for Ctrl+click open (Write/Read/edit/attachments).
    open_paths: tuple[str, ...] = ()


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


def block_collapse_key(record_id: str, block_index: int) -> str:
    """Stable key for fold state in the TUI."""

    return f"{record_id}:{block_index}"


def _cap_display_text(text: str, *, max_chars: int = MAX_BLOCK_CHARS) -> str:
    """Truncate oversized tool/memory dumps for detail paint only."""

    if not text or len(text) <= max_chars:
        return text
    kept = text[:max_chars]
    return f"{kept}\n\n…(显示截断，原文 {len(text)} 字符)\n"


def default_collapsed_keys(records: Iterable[DetailRecord]) -> frozenset[str]:
    """GrokBuild default: tool arg/result bodies start collapsed (one-line tools).

    Thinking stays expanded. Matches Grok ``DisplayMode::Collapsed`` for tools.
    """

    keys: set[str] = set()
    for record in records:
        for index, block in enumerate(record.blocks):
            if block.kind == "thinking" or block.label == THINKING_LABEL:
                continue
            if block.label in {"调用参数", "返回结果"}:
                keys.add(block_collapse_key(record.id, index))
    return frozenset(keys)


def thinking_collapse_keys(records: Iterable[DetailRecord]) -> frozenset[str]:
    """All fold keys belonging to thinking blocks."""

    keys: set[str] = set()
    for record in records:
        for index, block in enumerate(record.blocks):
            if block.kind == "thinking" or block.label == THINKING_LABEL:
                keys.add(block_collapse_key(record.id, index))
    return frozenset(keys)


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
            content = strip_system_reminders(
                _clean_text(_read_field(message, "content", ""))
            )
            # Pure plan/MCP system-reminder injects: hide entirely from the UI.
            if not content:
                continue
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
            # Grok order: thinking block first, then visible assistant prose.
            reasoning = _extract_reasoning(message)
            if reasoning:
                records.append(
                    DetailRecord(
                        id=f"thinking-{sequence}",
                        sequence=sequence,
                        actor="THINK",
                        category="message",
                        title="思考",
                        blocks=(
                            DetailBlock(
                                "thinking",
                                _cap_display_text(reasoning),
                                label=THINKING_LABEL,
                            ),
                        ),
                        timestamp=f"#{sequence:03d}",
                        status="completed",
                    )
                )
                sequence += 1
            content = strip_system_reminders(
                _clean_text(_read_field(message, "content", ""))
            )
            if content:
                records.append(
                    DetailRecord(
                        id=f"message-{sequence}",
                        sequence=sequence,
                        actor=_clean_label(agent_label),
                        category="message",
                        title="进度",
                        blocks=(DetailBlock("text", _cap_display_text(content)),),
                        timestamp=f"#{sequence:03d}",
                    )
                )
                sequence += 1
            for call in list(_read_field(message, "tool_calls", None) or []):
                call_id = str(_read_field(call, "id", "") or f"tool-{sequence}")
                name = str(_read_field(call, "name", "tool") or "tool")
                arguments = _read_field(call, "arguments", {})
                category = _tool_category(name, arguments)
                # Grok-style headline: "Read path", "Ran cmd" — not "TOOL · name".
                headline = _tool_headline(name, arguments)
                open_paths = tool_paths_from_arguments(arguments)
                record = DetailRecord(
                    id=call_id,
                    sequence=sequence,
                    actor="TOOL",
                    category=category,
                    title=headline,
                    blocks=(_arguments_block(name, arguments),),
                    timestamp=f"#{sequence:03d}",
                    status="running",
                    open_paths=open_paths,
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
                        title=_tool_headline(name, {}),
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
    records: Iterable[DetailRecord], active_filter: DetailFilter = "message"
) -> tuple[DetailRecord, ...]:
    """Return records for one UI tab without discarding stored detail.

    Message tab: once a real assistant answer (进度) exists, **drop thinking**
    from the paint list so the UI no longer shows long 思考过程 after the
    model has spoken. Transcript truth is unchanged — only the view filter.
    """

    active = active_filter if active_filter in DETAIL_FILTERS else "message"
    if active == "plan":
        # Plan body is injected by the TUI from the plan file, not transcript.
        return ()
    if active == "tool":
        return tuple(
            item for item in records if item.category in {"tool", "file", "test"}
        )
    messages = tuple(item for item in records if item.category == "message")
    has_assistant_prose = any(
        item.actor not in {"THINK", "USER"}
        and any(block.kind != "thinking" for block in item.blocks)
        for item in messages
    )
    if has_assistant_prose:
        return tuple(
            item
            for item in messages
            if item.actor != "THINK"
            and not (
                item.blocks
                and all(b.kind == "thinking" for b in item.blocks)
            )
        )
    return messages


def render_detail_header(
    snapshot: AgentDetailSnapshot,
    width: int,
    *,
    active_filter: DetailFilter = "message",
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
    fragments.append(("class:detail.separator", "─" * min(width, 28) + "\n"))
    used = 0
    sep = " · "
    sep_w = get_cwidth(sep)
    for item in DETAIL_FILTERS:
        label = DETAIL_FILTER_LABELS[item]
        label_w = get_cwidth(label)
        need = label_w + (sep_w if used else 0)
        if used and used + need > width:
            fragments.append(("", "\n"))
            used = 0
            need = label_w
        if used:
            fragments.append(("class:detail.meta", sep))
            used += sep_w
        style = "class:detail.active" if item == active_filter else "class:detail.meta"
        fragments.append((style, label))
        used += label_w
    fragments.extend(
        [
            ("", "\n"),
            ("", "\n"),
        ]
    )
    return fragments


def _section_break(width: int) -> StyleAndTextTuples:
    """Reading rhythm between records: air + short soft rule + air.

    Inspired by Linear/Notion/Medium: space carries hierarchy; the rule is a
    quiet hairline, not a full-width terminal box rail.
    """
    rule_w = min(20, max(4, width // 4))
    rule_w = min(rule_w, max(1, width))
    rule = "─" * rule_w
    pad = max(0, (width - get_cwidth(rule)) // 2)
    line = ((" " * pad) + rule)[:width]
    return [
        ("", "\n"),
        ("class:detail.separator", line + "\n"),
        ("", "\n"),
    ]


def render_detail_body(
    snapshot: AgentDetailSnapshot,
    width: int,
    *,
    active_filter: DetailFilter = "message",
    path_mouse: Callable[[str], Any] | None = None,
    collapsed_keys: Collection[str] | None = None,
    fold_mouse: Callable[[str], Any] | None = None,
) -> StyleAndTextTuples:
    """Render every selected record without summarizing or truncating bodies.

    ``path_mouse(path)`` optional: returns a prompt_toolkit mouse handler so
    image_gen paths open in the OS viewer on click (Grok-style link affordance).

    ``collapsed_keys`` / ``fold_mouse`` optional: fold tool arg/result blocks.
    """

    width = max(12, width)
    collapsed = set(collapsed_keys or ())
    records = filter_detail_records(snapshot.records, active_filter)
    if not records:
        return [("class:detail.meta", "\n  当前分类没有记录。\n")]
    fragments: StyleAndTextTuples = []
    for index, record in enumerate(records):
        if index:
            fragments.extend(_section_break(width))
        fragments.extend(
            _render_record(
                record,
                width,
                collapsed_keys=collapsed,
                fold_mouse=fold_mouse,
                path_mouse=path_mouse,
            )
        )
        if path_mouse is not None:
            for file_path in paths_from_detail_record(record):
                label = f"  Ctrl+点击 {link_label_for_path(file_path)}"
                label = _truncate_display(label, width)
                handler = path_mouse(file_path)
                if handler is not None:
                    fragments.append(("", "\n"))
                    fragments.append(("class:detail.link.hint", label, handler))
                    fragments.append(("", "\n"))
    return fragments


def render_agent_detail(
    snapshot: AgentDetailSnapshot,
    width: int,
    *,
    active_filter: DetailFilter = "message",
    elapsed_seconds: float | None = None,
    collapsed_keys: Collection[str] | None = None,
) -> StyleAndTextTuples:
    """Convenience composition for a complete scrollable detail document."""

    return render_detail_header(
        snapshot,
        width,
        active_filter=active_filter,
        elapsed_seconds=elapsed_seconds,
    ) + render_detail_body(
        snapshot,
        width,
        active_filter=active_filter,
        collapsed_keys=collapsed_keys,
    )


def _render_record(
    record: DetailRecord,
    width: int,
    *,
    collapsed_keys: set[str],
    fold_mouse: Callable[[str], Any] | None,
    path_mouse: Callable[[str], Any] | None = None,
) -> StyleAndTextTuples:
    """Render one transcript row.

    Tools (GrokBuild Collapsed default): single muted line
    ``· Read path`` / ``· Ran cmd`` — arg/result bodies stay folded unless
    the user expands. Thinking / prose keep their own layout.
    """
    actor = record.actor or "AGENT"
    title = (record.title or "").strip()
    fragments: StyleAndTextTuples = []

    # ── Tool: Grok one-line collapsed header ─────────────────────────
    if actor.strip().upper() == "TOOL":
        st = (record.status or "").lower()
        if st == "running":
            bullet, bstyle = "…", "class:detail.tool"
        elif st in {"error", "failed"}:
            bullet, bstyle = "×", "class:detail.error"
        else:
            bullet, bstyle = "·", "class:detail.meta"
        # Whole header is a hit target: toggle all tool bodies for this record.
        tool_keys = [
            block_collapse_key(record.id, i)
            for i, b in enumerate(record.blocks)
            if b.label in {"调用参数", "返回结果"}
        ]
        all_collapsed = bool(tool_keys) and all(k in collapsed_keys for k in tool_keys)
        prefer = None
        if fold_mouse is not None and tool_keys:
            prefer = next(
                (
                    block_collapse_key(record.id, i)
                    for i, b in enumerate(record.blocks)
                    if b.label == "返回结果"
                ),
                tool_keys[0],
            )
        open_path = (record.open_paths[0] if record.open_paths else "") or ""
        # Prefer Ctrl+click open on the filename; plain click still folds.
        path_handler = (
            path_mouse(open_path) if (path_mouse is not None and open_path) else None
        )
        fold_handler = fold_mouse(prefer) if (fold_mouse is not None and prefer) else None
        head = _truncate_display(f"  {bullet}  {title}", width)
        if path_handler is not None and " " in title:
            verb, _, name = title.partition(" ")
            prefix = _truncate_display(f"  {bullet}  {verb} ", width)
            rest_w = max(1, width - get_cwidth(prefix))
            name_txt = _truncate_display(name, rest_w)
            if fold_handler is not None:
                fragments.append((bstyle, prefix, fold_handler))
            else:
                fragments.append((bstyle, prefix))
            fragments.append(("class:detail.link", name_txt + "\n", path_handler))
        elif fold_handler is not None:
            fragments.append((bstyle, head + "\n", fold_handler))
        elif path_handler is not None:
            fragments.append(("class:detail.link", head + "\n", path_handler))
        else:
            fragments.append((bstyle, head + "\n"))
        # Bodies only when expanded (not all_collapsed).
        if all_collapsed:
            return fragments
        for block_index, block in enumerate(record.blocks):
            key = block_collapse_key(record.id, block_index)
            if key in collapsed_keys:
                continue
            if block.label:
                fragments.append(
                    ("class:detail.meta", _truncate_display(f"    {block.label}", width) + "\n")
                )
            if block.kind in {"code", "diff", "command", "output", "metadata"}:
                fragments.extend(_render_structured_block(block, width))
            else:
                fragments.extend(_render_text_block(block.text, width))
        return fragments

    # ── Non-tool (user / think / assistant) ──────────────────────────
    # Web reading: small muted byline, then body — not a terminal "USER · 进度" rail.
    actor_short = {
        "USER": "你",
        "THINK": "思考",
        "MAIN": "MAIN",
    }.get(actor.strip().upper(), actor.strip()[:8] or "AGENT")
    actor_style = _actor_style(actor)
    # Byline only — drop redundant "进度" titles that add noise without meaning.
    byline = actor_short
    if title and title not in {"进度", "回复", "输出"}:
        byline = f"{actor_short}  ·  {title}"
    fragments.append((actor_style, f"  {byline}"))
    fragments.append(("", "\n"))
    fragments.append(("", "\n"))
    for block_index, block in enumerate(record.blocks):
        key = block_collapse_key(record.id, block_index)
        is_thinking = block.kind == "thinking" or block.label == THINKING_LABEL
        foldable = is_thinking or (
            bool(block.label) and block.label in {"调用参数", "返回结果"}
        )
        is_collapsed = foldable and key in collapsed_keys
        if block.label or is_thinking:
            if foldable:
                label = block.label or THINKING_LABEL
                line_count = max(
                    1, block.text.count("\n") + (1 if block.text else 0)
                )
                if fold_mouse is None and not is_collapsed:
                    if is_thinking:
                        label_text = f"  ◆ {label}"
                        style = "class:detail.thinking.header"
                    else:
                        label_text = f"  {label}"
                        style = "class:detail.meta"
                    fragments.append((style, _truncate_display(label_text, width)))
                    fragments.append(("", "\n"))
                else:
                    marker = "▶" if is_collapsed else "▼"
                    if is_thinking:
                        label_text = f"  ◆ {marker} {label} · {line_count} 行"
                        style = "class:detail.thinking.header"
                    else:
                        label_text = f"  {marker} {label}"
                        if is_collapsed:
                            preview = _first_line_preview(
                                block.text, max(8, width - 28)
                            )
                            label_text = f"  {marker} {label} · {line_count} 行"
                            if preview:
                                label_text += f" · {preview}"
                        style = "class:detail.fold.active"
                    label_text = _truncate_display(label_text, width)
                    if fold_mouse is not None:
                        fragments.append((style, label_text, fold_mouse(key)))
                    else:
                        fragments.append((style, label_text))
                    fragments.append(("", "\n"))
            elif block.label:
                fragments.append(("class:detail.meta", f"  {block.label}\n"))
        if is_collapsed:
            if is_thinking:
                fragments.extend(
                    _render_thinking_block(
                        block,
                        width,
                        max_lines=THINKING_PREVIEW_LINES,
                        truncated=True,
                    )
                )
            continue
        if is_thinking:
            fragments.extend(_render_thinking_block(block, width))
        elif block.kind in {"code", "diff", "command", "output", "metadata"}:
            fragments.extend(_render_structured_block(block, width))
        else:
            fragments.extend(_render_text_block(block.text, width))
    return fragments


def _render_thinking_block(
    block: DetailBlock,
    width: int,
    *,
    max_lines: int | None = None,
    truncated: bool = False,
) -> StyleAndTextTuples:
    """Muted thinking body with left rail (Grok thinking block spirit)."""

    width = max(12, min(width, MAX_THOUGHTS_WIDTH + 4))
    body_width = max(1, width - 6)
    lines = block.text.splitlines() or [""]
    total = len(lines)
    show = lines if max_lines is None else lines[: max(0, max_lines)]
    fragments: StyleAndTextTuples = []
    for raw in show:
        for piece in _wrap_display(raw, body_width):
            fragments.append(("class:detail.thinking.rail", "  ┃ "))
            fragments.append(("class:detail.thinking.body", piece))
            fragments.append(("", "\n"))
    if truncated and total > len(show):
        more = total - len(show)
        fragments.append(("class:detail.thinking.rail", "  ┃ "))
        fragments.append(
            ("class:detail.thinking.meta", f"… 另有 {more} 行")
        )
        fragments.append(("", "\n"))
    return fragments


def _extract_reasoning(message: Any) -> str:
    """Pull thinking text from reasoning_content or provider content blocks."""

    direct = _read_field(message, "reasoning_content", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    # Anthropic-style blocks may live on provider_data or content list.
    for candidate in (
        _read_field(message, "provider_data", None),
        _read_field(message, "content", None),
    ):
        text = _reasoning_from_blocks(candidate)
        if text:
            return text
    return ""


def _reasoning_from_blocks(value: Any) -> str:
    if not isinstance(value, (list, dict)):
        return ""
    blocks: list[Any]
    if isinstance(value, dict):
        # Nested: provider_data.anthropic_content_blocks / content
        for key in (
            "anthropic_content_blocks",
            "content_blocks",
            "content",
            "thinking",
        ):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip() and key == "thinking":
                return inner.strip()
            if isinstance(inner, list):
                blocks = inner
                break
        else:
            return ""
    else:
        blocks = value

    parts: list[str] = []
    redacted = False
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type") or "")
        if btype == "thinking":
            th = block.get("thinking") or block.get("text") or ""
            if isinstance(th, str) and th.strip():
                parts.append(th.strip())
        elif btype == "redacted_thinking":
            redacted = True
    if parts:
        return "\n\n".join(parts)
    if redacted:
        return "（思考已隐藏）"
    return ""


def _actor_style(actor: str) -> str:
    normalized = actor.strip().upper()
    if normalized == "USER":
        return "class:detail.actor.user"
    if normalized in {"THINK", "THINKING", "REASONING"}:
        return "class:detail.actor.think"
    if normalized in {"TOOL", "SYSTEM"}:
        return "class:detail.actor.tool"
    if normalized in {"ASSISTANT", "AGENT", "MAIN"}:
        return "class:detail.actor.assistant"
    return "class:detail.actor"


# Grok-style: LINE_NUMBER→CONTENT (and common N| / N: prefixes).
_LINE_PREFIX_RE = re.compile(r"^(\d+)(?:→|[|│:]\s?)(.*)$")
_OL_RE = re.compile(r"^(\s*)(\d+)([.)])\s+(.*)$")
_UL_RE = re.compile(r"^(\s*)([-*•])\s+(.*)$")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
_QUOTE_RE = re.compile(r"^(\s*)>\s?(.*)$")
_FENCE_RE = re.compile(r"^(\s*)```([^\s`]*)?")
# Inline spans (Grok markdown Strong/Emphasis spirit). Order: code, bold, italic.
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_INLINE_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_INLINE_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)")
_INLINE_STRIKE_RE = re.compile(r"~~(.+?)~~")


def _render_structured_block(block: DetailBlock, width: int) -> StyleAndTextTuples:
    """Grok-like rail + gutter + soft-wrap body (no heavy box blob)."""

    width = max(12, width)
    raw_lines = block.text.splitlines() or [""]
    line_count = len(raw_lines)
    fragments: StyleAndTextTuples = []

    # Quiet meta strip instead of ┌──┐ iron box.
    kind_label = {
        "code": "code",
        "diff": "diff",
        "command": "cmd",
        "output": "out",
        "metadata": "args",
    }.get(block.kind, block.kind)
    meta = f"  · {kind_label} · {line_count} lines"
    fragments.append(("class:detail.code.meta", _truncate_display(meta, width) + "\n"))

    if block.kind == "command":
        fragments.extend(_render_command_block(block, width))
        return fragments
    if block.kind == "diff":
        fragments.extend(_render_diff_block(block, width))
        return fragments
    if block.kind == "metadata":
        fragments.extend(_render_plain_rail_block(block, width, highlight=False))
        return fragments
    # code + output: line numbers when present or synthetic for code.
    # Highlight both code *and* output (shell dumps often carry source-like text).
    show_numbers = block.kind == "code" or _block_has_line_prefixes(raw_lines)
    use_hl = block.kind in {"code", "output"} or show_numbers
    fragments.extend(
        _render_gutter_block(
            block,
            width,
            show_numbers=show_numbers,
            highlight=use_hl,
        )
    )
    return fragments


def _block_has_line_prefixes(lines: list[str]) -> bool:
    hits = 0
    for line in lines[:20]:
        if _LINE_PREFIX_RE.match(line):
            hits += 1
    return hits >= 1


def _parse_code_line(raw: str) -> tuple[int | None, str]:
    """Return (line_no, body). line_no None when no prefix."""

    match = _LINE_PREFIX_RE.match(raw)
    if match:
        return int(match.group(1)), match.group(2)
    return None, raw


def _gutter_width_for_lines(lines: list[str], *, show_numbers: bool) -> int:
    if not show_numbers:
        return 0
    max_no = 0
    has_prefix = False
    for raw in lines:
        no, _ = _parse_code_line(raw)
        if no is not None:
            has_prefix = True
            max_no = max(max_no, no)
    if not has_prefix:
        max_no = max(1, len(lines))
    if max_no <= 0:
        max_no = 1
    return max(2, len(str(max_no)))


def _render_gutter_block(
    block: DetailBlock,
    width: int,
    *,
    show_numbers: bool,
    highlight: bool,
) -> StyleAndTextTuples:
    raw_lines = block.text.splitlines() or [""]
    sparse = any(_parse_code_line(raw)[0] is not None for raw in raw_lines)
    gutter_w = _gutter_width_for_lines(raw_lines, show_numbers=show_numbers)
    # "  ┃ " + gutter + "│ " + body
    prefix_fixed = 4  # "  ┃ "
    sep_w = 2 if show_numbers else 1  # "│ " or " "
    body_width = max(1, width - prefix_fixed - (gutter_w if show_numbers else 0) - sep_w)
    fragments: StyleAndTextTuples = []
    synth = 0
    for raw in raw_lines:
        parsed_no, body = _parse_code_line(raw)
        line_no: int | None
        if not show_numbers:
            line_no = None
        elif sparse:
            # Grok sparse numbers: only paint when N→ present; blank otherwise.
            line_no = parsed_no
        else:
            synth += 1
            line_no = synth
        wrapped = _wrap_display(body, body_width)
        for wrap_i, piece in enumerate(wrapped):
            fragments.append(("class:detail.code.rail", "  ┃ "))
            if show_numbers and gutter_w:
                if wrap_i == 0 and line_no is not None:
                    num = str(line_no).rjust(gutter_w)
                    g_style = (
                        "class:detail.code.gutter.mark"
                        if line_no % 10 == 0
                        else "class:detail.code.gutter"
                    )
                    fragments.append((g_style, num))
                else:
                    fragments.append(("class:detail.code.gutter", " " * gutter_w))
                fragments.append(("class:detail.code.gutter.sep", "│ "))
            else:
                fragments.append(("class:detail.code.gutter.sep", " "))
            if highlight:
                fragments.extend(_highlight_code_line(piece))
            else:
                style = _block_line_style(block, body if wrap_i == 0 else piece)
                fragments.append((style, piece))
            pad = max(0, body_width - get_cwidth(piece))
            if pad:
                fragments.append(("class:detail.code.plain", " " * pad))
            fragments.append(("", "\n"))
    return fragments


def _render_command_block(block: DetailBlock, width: int) -> StyleAndTextTuples:
    body_width = max(1, width - 6)
    fragments: StyleAndTextTuples = []
    for raw in block.text.splitlines() or [""]:
        text = raw
        for wrap_i, piece in enumerate(_wrap_display(text, body_width)):
            fragments.append(("class:detail.code.rail", "  ┃ "))
            if wrap_i == 0 and piece.startswith("$"):
                fragments.append(("class:detail.code.num", "$"))
                rest = piece[1:]
                if rest:
                    fragments.append(("class:detail.code.plain", rest))
            elif wrap_i == 0:
                fragments.append(("class:detail.code.plain", piece))
            else:
                fragments.append(("class:detail.code.plain", piece))
            fragments.append(("", "\n"))
    return fragments


def _render_diff_block(block: DetailBlock, width: int) -> StyleAndTextTuples:
    body_width = max(1, width - 8)
    fragments: StyleAndTextTuples = []
    for raw in block.text.splitlines() or [""]:
        if raw.startswith("+++") or raw.startswith("---") or raw.startswith("@@"):
            mark, mark_style = " ", "class:detail.diff.hunk"
            body_style = "class:detail.diff.hunk"
            body = raw
        elif raw.startswith("+"):
            mark, mark_style = "+", "class:detail.diff.add"
            body_style = "class:detail.diff.add"
            body = raw[1:]
        elif raw.startswith("-"):
            mark, mark_style = "-", "class:detail.diff.remove"
            body_style = "class:detail.diff.remove"
            body = raw[1:]
        else:
            mark, mark_style = " ", "class:detail.diff.gutter"
            body_style = "class:detail.block"
            body = raw
        for wrap_i, piece in enumerate(_wrap_display(body, body_width)):
            fragments.append(("class:detail.code.rail", "  ┃ "))
            if wrap_i == 0:
                fragments.append((mark_style, mark))
            else:
                fragments.append(("class:detail.diff.gutter", " "))
            fragments.append(("class:detail.code.gutter.sep", "│ "))
            fragments.append((body_style, piece))
            fragments.append(("", "\n"))
    return fragments


def _render_plain_rail_block(
    block: DetailBlock, width: int, *, highlight: bool
) -> StyleAndTextTuples:
    body_width = max(1, width - 6)
    fragments: StyleAndTextTuples = []
    for raw in block.text.splitlines() or [""]:
        for piece in _wrap_display(raw, body_width):
            fragments.append(("class:detail.code.rail", "  ┃ "))
            if highlight:
                fragments.extend(_highlight_code_line(piece))
            else:
                fragments.append(("class:detail.block", piece))
            fragments.append(("", "\n"))
    return fragments


def _render_text_block(text: str, width: int) -> StyleAndTextTuples:
    """Markdown-lite: lists, headings, quotes, fenced code."""

    width = max(12, width)
    fragments: StyleAndTextTuples = []
    lines = text.splitlines() or [""]
    in_fence = False
    fence_buf: list[str] = []
    list_marker_w = 0

    def flush_fence() -> None:
        nonlocal fence_buf
        if not fence_buf:
            return
        code = DetailBlock("code", "\n".join(fence_buf))
        fragments.extend(
            _render_gutter_block(code, width, show_numbers=True, highlight=True)
        )
        fence_buf = []

    for raw in lines:
        fence_m = _FENCE_RE.match(raw)
        if fence_m:
            if in_fence:
                in_fence = False
                flush_fence()
            else:
                in_fence = True
                fence_buf = []
                lang = (fence_m.group(2) or "").strip()
                if lang:
                    fragments.append(
                        (
                            "class:detail.code.meta",
                            _truncate_display(f"  · {lang}", width) + "\n",
                        )
                    )
            continue
        if in_fence:
            fence_buf.append(raw)
            continue

        heading = _HEADING_RE.match(raw)
        if heading:
            level = len(heading.group(1))
            style = {1: "class:detail.md.h1", 2: "class:detail.md.h2"}.get(
                level, "class:detail.md.h3"
            )
            body = heading.group(2)
            for i, piece in enumerate(_wrap_display(body, max(1, width - 4))):
                prefix = "  " if i == 0 else "  "
                fragments.append((style, prefix + piece + "\n"))
            list_marker_w = 0
            continue

        quote = _QUOTE_RE.match(raw)
        if quote:
            body = quote.group(2)
            for piece in _wrap_display(body, max(1, width - 6)):
                fragments.append(("class:detail.code.rail", "  ┃ "))
                # Quote body stays italic muted; still honor inline code/bold.
                for style, txt, *rest in _render_inline_md(piece):
                    if style == "class:detail.text":
                        style = "class:detail.md.quote"
                    if rest:
                        fragments.append((style, txt, rest[0]))
                    else:
                        fragments.append((style, txt))
                fragments.append(("", "\n"))
            list_marker_w = 0
            continue

        ol = _OL_RE.match(raw)
        if ol:
            indent, num, punct, body = ol.group(1), ol.group(2), ol.group(3), ol.group(4)
            marker = f"{num}{punct} "
            list_marker_w = get_cwidth(marker)
            lead = "  " + indent
            body_w = max(1, width - get_cwidth(lead) - list_marker_w)
            for i, piece in enumerate(_wrap_display(body, body_w)):
                if i == 0:
                    fragments.append(("class:detail.meta", lead))
                    fragments.append(("class:detail.md.ol", marker))
                    fragments.extend(_render_inline_md(piece))
                    fragments.append(("", "\n"))
                else:
                    pad = " " * (get_cwidth(lead) + list_marker_w)
                    fragments.append(("class:detail.meta", pad))
                    fragments.extend(_render_inline_md(piece))
                    fragments.append(("", "\n"))
            continue

        ul = _UL_RE.match(raw)
        if ul:
            indent, bullet, body = ul.group(1), ul.group(2), ul.group(3)
            marker = f"{bullet} "
            list_marker_w = get_cwidth(marker)
            lead = "  " + indent
            body_w = max(1, width - get_cwidth(lead) - list_marker_w)
            for i, piece in enumerate(_wrap_display(body, body_w)):
                if i == 0:
                    fragments.append(("class:detail.meta", lead))
                    fragments.append(("class:detail.md.ul", marker))
                    fragments.extend(_render_inline_md(piece))
                    fragments.append(("", "\n"))
                else:
                    pad = " " * (get_cwidth(lead) + list_marker_w)
                    fragments.append(("class:detail.meta", pad))
                    fragments.extend(_render_inline_md(piece))
                    fragments.append(("", "\n"))
            continue

        # Continuation of list: hang under previous marker.
        if list_marker_w and raw.startswith(" ") and raw.strip():
            lead_w = 2 + list_marker_w
            body_w = max(1, width - lead_w)
            for piece in _wrap_display(raw.strip(), body_w):
                fragments.append(("class:detail.meta", " " * lead_w))
                fragments.extend(_render_inline_md(piece))
                fragments.append(("", "\n"))
            continue

        list_marker_w = 0
        if not raw.strip():
            # Paragraph air (Medium/Notion): blank line stays blank, plus a little more.
            fragments.append(("", "\n"))
            continue
        # Comfortable reading column: indent body like a content pane, not edge-flush.
        gutter = "   "
        for piece in _wrap_display(raw, max(1, width - get_cwidth(gutter))):
            fragments.append(("class:detail.text", gutter))
            fragments.extend(_render_inline_md(piece))
            fragments.append(("", "\n"))

    if in_fence:
        flush_fence()
    return fragments


def _render_inline_md(text: str) -> StyleAndTextTuples:
    """Grok-lite inline: `code`, **bold**, *italic*, ~~strike~~."""

    if not text:
        return [("class:detail.text", "")]
    if not any(ch in text for ch in ("`", "*", "_", "~")):
        return [("class:detail.text", text)]

    # First split out inline code so * inside backticks is not emphasized.
    chunks: list[tuple[str, str]] = []  # (kind, text) kind=code|text
    pos = 0
    for match in _INLINE_CODE_RE.finditer(text):
        if match.start() > pos:
            chunks.append(("text", text[pos : match.start()]))
        chunks.append(("code", match.group(1)))
        pos = match.end()
    if pos < len(text):
        chunks.append(("text", text[pos:]))
    if not chunks:
        chunks = [("text", text)]

    fragments: StyleAndTextTuples = []
    for kind, piece in chunks:
        if kind == "code":
            fragments.append(("class:detail.md.inline", piece))
            continue
        fragments.extend(_render_inline_emphasis(piece))
    return fragments or [("class:detail.text", text)]


def _render_inline_emphasis(text: str) -> StyleAndTextTuples:
    """Apply bold / italic / strike on a non-code span."""

    if not text:
        return []
    # Process bold first, then italic on remaining plain spans.
    out: StyleAndTextTuples = []
    pos = 0
    for match in _INLINE_BOLD_RE.finditer(text):
        if match.start() > pos:
            out.extend(_render_inline_italic_strike(text[pos : match.start()]))
        bold = match.group(1) if match.group(1) is not None else match.group(2)
        out.append(("class:detail.md.bold", bold or ""))
        pos = match.end()
    if pos < len(text):
        out.extend(_render_inline_italic_strike(text[pos:]))
    return out or [("class:detail.text", text)]


def _render_inline_italic_strike(text: str) -> StyleAndTextTuples:
    if not text:
        return []
    out: StyleAndTextTuples = []
    pos = 0
    # Strike then italic (simple sequential).
    for match in _INLINE_STRIKE_RE.finditer(text):
        if match.start() > pos:
            out.extend(_render_inline_italic_only(text[pos : match.start()]))
        out.append(("class:detail.md.strike", match.group(1)))
        pos = match.end()
    if pos < len(text):
        out.extend(_render_inline_italic_only(text[pos:]))
    return out or [("class:detail.text", text)]


def _render_inline_italic_only(text: str) -> StyleAndTextTuples:
    if not text:
        return []
    out: StyleAndTextTuples = []
    pos = 0
    for match in _INLINE_ITALIC_RE.finditer(text):
        if match.start() > pos:
            out.append(("class:detail.text", text[pos : match.start()]))
        ital = match.group(1) if match.group(1) is not None else match.group(2)
        out.append(("class:detail.md.italic", ital or ""))
        pos = match.end()
    if pos < len(text):
        out.append(("class:detail.text", text[pos:]))
    return out or [("class:detail.text", text)]


_CODE_KEYWORDS = frozenset(
    {
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "False",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "None",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "True",
        "try",
        "while",
        "with",
        "yield",
        "const",
        "let",
        "var",
        "function",
        "return",
        "typeof",
        "new",
        "this",
        "void",
        "null",
        "undefined",
        "export",
        "default",
        "interface",
        "type",
        "enum",
        "public",
        "private",
        "protected",
        "static",
        "struct",
        "impl",
        "fn",
        "mut",
        "use",
        "mod",
        "pub",
        "crate",
        "self",
        "Self",
        "match",
        "loop",
        "move",
        "package",
        "func",
        "go",
        "defer",
        "chan",
        "map",
        "range",
        "select",
        "case",
        "switch",
    }
)

_TOKEN_RE = re.compile(
    r"(?P<cmt>//.*?$|#.*?$|/\*.*?\*/)"
    r"|(?P<str>'''(?:\\.|[^'\\])*'''|\"\"\"(?:\\.|[^\"\\])*\"\"\""
    r"|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:\\.|[^`\\])*`)"
    r"|(?P<num>\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)"
    r"|(?P<id>\b[A-Za-z_][A-Za-z0-9_]*\b)"
    r"|(?P<sym>[^\sA-Za-z0-9_]+)"
    r"|(?P<ws>\s+)",
    re.MULTILINE,
)


def _highlight_code_line(line: str) -> StyleAndTextTuples:
    """Heuristic token coloring — muted, no glowing pure white."""

    if not line:
        return [("class:detail.code.plain", "")]
    # Full-line comments / shell prompts
    stripped = line.lstrip()
    if stripped.startswith("#") or stripped.startswith("//"):
        return [("class:detail.code.cmt", line)]
    return _highlight_mixed(line)


def _highlight_mixed(line: str) -> StyleAndTextTuples:
    fragments: StyleAndTextTuples = []
    pos = 0
    for match in _TOKEN_RE.finditer(line):
        if match.start() > pos:
            fragments.append(("class:detail.code.plain", line[pos : match.start()]))
        if match.group("cmt") is not None:
            fragments.append(("class:detail.code.cmt", match.group("cmt")))
        elif match.group("str") is not None:
            fragments.append(("class:detail.code.str", match.group("str")))
        elif match.group("num") is not None:
            fragments.append(("class:detail.code.num", match.group("num")))
        elif match.group("id") is not None:
            ident = match.group("id")
            style = (
                "class:detail.code.kw"
                if ident in _CODE_KEYWORDS
                else "class:detail.code.plain"
            )
            fragments.append((style, ident))
        elif match.group("sym") is not None:
            fragments.append(("class:detail.code.sym", match.group("sym")))
        elif match.group("ws") is not None:
            fragments.append(("class:detail.code.plain", match.group("ws")))
        pos = match.end()
    if pos < len(line):
        fragments.append(("class:detail.code.plain", line[pos:]))
    return fragments or [("class:detail.code.plain", line)]


def _first_line_preview(text: str, width: int) -> str:
    first = (text.splitlines() or [""])[0].strip()
    if not first:
        return ""
    return _truncate_display(first, max(1, width))


def _tool_headline(name: str, arguments: Any) -> str:
    """GrokBuild-style one-line tool header (Read / Ran / Edited / …)."""
    n = (name or "tool").strip()
    low = n.lower()
    args = arguments if isinstance(arguments, dict) else {}

    def _path() -> str:
        for key in (
            "path",
            "target_file",
            "file_path",
            "file",
            "filename",
        ):
            val = args.get(key)
            if val:
                s = str(val).replace("\\", "/")
                # Collapsed: basename-ish short form.
                if "/" in s:
                    return s.rsplit("/", 1)[-1] or s
                return s
        return ""

    def _cmd() -> str:
        c = args.get("command") or args.get("cmd") or ""
        return " ".join(str(c).split())

    if low in {"read_file", "read", "view_file"}:
        p = _path()
        return f"Read {p}" if p else "Read"
    if low in {"write", "create_file"}:
        p = _path()
        return f"Wrote {p}" if p else "Wrote"
    if low in {"search_replace", "str_replace", "edit", "apply_patch"}:
        p = _path()
        return f"Edited {p}" if p else "Edited"
    if low in {
        "run_terminal_command",
        "run_terminal_cmd",
        "bash",
        "shell",
        "execute",
    }:
        c = _cmd()
        if len(c) > 48:
            c = c[:47] + "…"
        return f"Ran {c}" if c else "Ran command"
    if low in {"grep", "search", "glob"}:
        pat = str(args.get("pattern") or args.get("query") or "").strip()
        if len(pat) > 40:
            pat = pat[:39] + "…"
        return f"Searched {pat}" if pat else "Searched"
    if low in {"list_dir", "ls"}:
        p = _path() or str(args.get("target_directory") or "").strip()
        return f"Listed {p}" if p else "Listed"
    if low in {"web_fetch", "fetch"}:
        url = str(args.get("url") or "").strip()
        if len(url) > 40:
            url = url[:39] + "…"
        return f"Fetched {url}" if url else "Fetched"
    if low in {"web_search"}:
        q = str(args.get("query") or "").strip()
        if len(q) > 40:
            q = q[:39] + "…"
        return f"Searched {q}" if q else "Web search"
    if low in {"todo_write", "update_todo"}:
        return "Updated todos"
    if low in {"ask_user_question"}:
        return "Asked user"
    # Fallback: bare tool id (no TOOL · prefix).
    return n


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
            "run_terminal_command",
            "execute",
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
    return DetailBlock(kind, _cap_display_text(text), label="调用参数")


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
    return DetailBlock(
        kind,
        _cap_display_text(output),
        status=_result_status(output),
        label="返回结果",
    )


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
    return "class:detail.block" if block.kind != "code" else "class:detail.code.plain"


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


def strip_system_reminders(text: str) -> str:
    """Remove ``<system-reminder>…</system-reminder>`` blocks from display text.

    Plan-mode begin_turn injects these for the model; users should not see them
    in the task/plan detail transcript.
    """
    if not text or "<system-reminder>" not in text.lower():
        return text
    cleaned = _SYSTEM_REMINDER_RE.sub("", text)
    # Collapse leftover blank runs after stripping injects.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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
    "MAX_THOUGHTS_WIDTH",
    "THINKING_LABEL",
    "THINKING_PREVIEW_LINES",
    "AgentDetailSnapshot",
    "DetailBlock",
    "DetailFilter",
    "DetailRecord",
    "block_collapse_key",
    "default_collapsed_keys",
    "filter_detail_records",
    "render_agent_detail",
    "render_detail_body",
    "render_detail_header",
    "snapshot_from_messages",
    "strip_system_reminders",
    "thinking_collapse_keys",
]
