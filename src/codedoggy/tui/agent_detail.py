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
from codedoggy.tui.syntax import highlight_code_line

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

# Expanded tool body paint caps. Prefer change hunks over full-file dumps —
# the TUI is a review surface, not an editor. Click header to collapse.
# Hard caps prevent expand/Ctrl-misclick from flooding the terminal paint.
EXPANDED_DIFF_PREVIEW_LINES = 40  # edit hunks (already change-only)
EXPANDED_CODE_PREVIEW_LINES = 12  # reads / shell dumps (not whole files)
# Soft wrap guide for thinking (no longer a hard clip — panel width wins).
MAX_THOUGHTS_WIDTH = 120
# Paint-time caps for tool dumps (not thinking — thinking stays full).
MAX_BLOCK_CHARS = 4_000
# Extreme safety only for multi-MB reasoning blobs; normal thoughts are full.
MAX_THINKING_CHARS = 200_000
# Read-result preview: head/tail only (full path is openable via link).
READ_PREVIEW_HEAD_LINES = 6
READ_PREVIEW_TAIL_LINES = 2
# Cap lines inside a single edit hunk shown as a change block.
CHANGE_HUNK_MAX_LINES = 40
THINKING_LABEL = "思考过程"
CHANGE_LABEL = "变更"


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


def record_collapse_key(record_id: str) -> str:
    """Stable record-owned fold key for a complete tool row."""

    return f"record:{record_id}"


def _cap_display_text(text: str, *, max_chars: int = MAX_BLOCK_CHARS) -> str:
    """Truncate oversized tool/memory dumps for detail paint only."""

    if not text or len(text) <= max_chars:
        return text
    kept = text[:max_chars]
    return f"{kept}\n\n…(显示截断，原文 {len(text)} 字符)\n"


def default_collapsed_keys(records: Iterable[DetailRecord]) -> frozenset[str]:
    """Default fold set: tool bodies start collapsed.

    A tool owns one fold state for its complete body; individual result blocks
    never create a second, conflicting source of truth.
    """

    keys: set[str] = set()
    for record in records:
        if record.actor.strip().upper() == "TOOL" and record.blocks:
            keys.add(record_collapse_key(record.id))
    return frozenset(keys)


def snapshot_from_messages(
    messages: Iterable[Any],
    *,
    task_id: str,
    agent_id: str,
    agent_label: str,
    task_title: str,
    initial_user_text: str | None = None,
    status: str = "running",
) -> AgentDetailSnapshot:
    """Build a full detail snapshot from existing OpenAI-style messages.

    This adapter is intentionally duck-typed so it can consume both CodeDoggy
    ``Message`` instances and restored session records. System prompts stay
    hidden. User instructions, assistant prose, tool arguments, tool outputs,
    code, diffs and command results remain visible. The initial instruction is
    suppressed when it duplicates the separately rendered task prompt.
    """

    records: list[DetailRecord] = []
    tool_positions: dict[str, int] = {}
    sequence = 1
    initial_instruction = _clean_text(
        task_title if initial_user_text is None else initial_user_text
    )
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
            # The homepage owns the initial user prompt. Image chips are
            # display-only, so compare against the model-facing text rather
            # than the structural title that still contains the chip.
            if content and not (
                not records and content == initial_instruction
            ):
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
                                # Full reasoning by default — only extreme
                                # multi-MB blobs hit MAX_THINKING_CHARS.
                                _cap_display_text(
                                    reasoning, max_chars=MAX_THINKING_CHARS
                                ),
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
            result_status = _result_status(output)
            if call_id in tool_positions:
                position = tool_positions[call_id]
                old = records[position]
                # GrokBuild: edit tools keep the change hunk; skip trivial "ok".
                new_blocks = (
                    old.blocks + (result_block,)
                    if result_block is not None
                    else old.blocks
                )
                records[position] = replace(
                    old,
                    blocks=new_blocks,
                    status=result_status,
                )
            elif result_block is not None:
                records.append(
                    DetailRecord(
                        id=call_id or f"tool-result-{sequence}",
                        sequence=sequence,
                        actor="TOOL",
                        category=_tool_category(name, {}),
                        title=_tool_headline(name, {}),
                        blocks=(result_block,),
                        timestamp=f"#{sequence:03d}",
                        status=result_status,
                    )
                )
                sequence += 1
            else:
                # Edit ack only — surface a completed one-liner if no prior call.
                records.append(
                    DetailRecord(
                        id=call_id or f"tool-result-{sequence}",
                        sequence=sequence,
                        actor="TOOL",
                        category=_tool_category(name, {}),
                        title=_tool_headline(name, {}),
                        blocks=(),
                        timestamp=f"#{sequence:03d}",
                        status=result_status,
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

    Message tab includes thinking records as direct readable prose. Tools stay
    on the 工具 tab.
    """

    active = active_filter if active_filter in DETAIL_FILTERS else "message"
    if active == "plan":
        # Plan body is injected by the TUI from the plan file, not transcript.
        return ()
    if active == "tool":
        return tuple(
            item for item in records if item.category in {"tool", "file", "test"}
        )
    return tuple(item for item in records if item.category == "message")


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
    """Compact reading rhythm between records.

    One breathing line and a short left-aligned hairline keep adjacent records
    distinct without turning a tool-heavy transcript into a field of gaps.
    """
    rule_w = min(12, max(4, width - 2))
    line = ("  " + ("─" * rule_w))[:width]
    return [
        ("", "\n"),
        ("class:detail.separator", line + "\n"),
    ]


def render_detail_body(
    snapshot: AgentDetailSnapshot,
    width: int,
    *,
    active_filter: DetailFilter = "message",
    path_mouse: Callable[[str], Any] | None = None,
    collapsed_keys: Collection[str] | None = None,
    fold_mouse: Callable[..., Any] | None = None,
) -> StyleAndTextTuples:
    """Render every selected record without summarizing or truncating bodies.

    ``path_mouse(path)`` optional: returns a prompt_toolkit mouse handler so
    tool/file paths open in the OS viewer on click.

    ``collapsed_keys`` / ``fold_mouse`` optional: fold complete tool records.
    Thinking records render directly as readable body text.
    ``fold_mouse(key, open_path=None)`` — plain click folds; Ctrl+click opens
    ``open_path`` when set (never expands on Ctrl).
    """

    width = max(12, width)
    collapsed = set(collapsed_keys or ())
    records = filter_detail_records(snapshot.records, active_filter)
    if not records:
        return [("class:detail.meta", "\n  当前分类没有记录。\n")]
    fragments: StyleAndTextTuples = []
    for index, record in enumerate(records):
        if index and active_filter != "tool":
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
        # Tool rows own their primary file action in the header. Repeating a
        # separate "打开" row below every collapsed tool created most of the
        # visual noise in file-heavy plan sessions.
        if path_mouse is not None and record.actor.strip().upper() != "TOOL":
            for file_path in paths_from_detail_record(record):
                short = link_label_for_path(file_path)
                label = _truncate_display(short, max(1, width - 4))
                handler = path_mouse(file_path)
                if handler is not None:
                    fragments.append(("", "\n"))
                    fragments.append(("class:detail.meta", "  ↗ "))
                    fragments.append(("class:detail.link", label, handler))
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


def _tool_status_bullet(status: str) -> tuple[str, str]:
    """GrokBuild tool prefix glyph + style (diamond by default).

    Matches pager ``scrollback.blocks.tool.bullet = "diamond"`` spirit:
    a small mark before every tool headline, tinted by terminal status.
    """
    st = (status or "").strip().lower()
    if st in {"pending", "running", "waiting"}:
        return "…", "class:detail.tool"
    if st in {"error", "failed"}:
        return "×", "class:detail.error"
    if st in {"cancelled", "canceled"}:
        return "–", "class:detail.meta"
    # Completed / ok — filled diamond (Grok default tool bullet).
    return "◆", "class:detail.tool"


def _render_record(
    record: DetailRecord,
    width: int,
    *,
    collapsed_keys: set[str],
    fold_mouse: Callable[..., Any] | None,
    path_mouse: Callable[[str], Any] | None = None,
) -> StyleAndTextTuples:
    """Render one transcript row.

    Tools (GrokBuild Collapsed default): single muted line
    ``◆ Read path`` / ``◆ Ran cmd`` — arg/result bodies stay folded unless
    the user expands. Thinking renders directly without a redundant header.

    Ctrl+click on a path-bearing tool header opens the file (OS) and must
    never expand the body (expand can flood the terminal).
    """
    actor = record.actor or "AGENT"
    title = (record.title or "").strip()
    fragments: StyleAndTextTuples = []
    open_paths = paths_from_detail_record(record)
    first_open = open_paths[0] if open_paths else None

    # ── Tool: one record, one fold state ─────────────────────────────
    if actor.strip().upper() == "TOOL":
        bullet, bstyle = _tool_status_bullet(record.status)
        has_body = bool(record.blocks)
        if not has_body:
            handler = (
                path_mouse(first_open)
                if first_open is not None and path_mouse is not None
                else None
            )
            if handler is not None:
                prefix = f"  {bullet} "
                title_width = max(1, width - get_cwidth(prefix))
                fragments.append((bstyle, prefix))
                fragments.append(
                    (
                        "class:detail.tool.link",
                        _truncate_display(title, title_width) + "\n",
                        handler,
                    )
                )
            else:
                head = _truncate_display(f"  {bullet} {title}", width)
                fragments.append((bstyle, head + "\n"))
            return fragments

        fold_key = record_collapse_key(record.id)
        is_collapsed = fold_key in collapsed_keys
        if fold_mouse is not None:
            try:
                fold_handler = fold_mouse(fold_key, first_open)
            except TypeError:
                fold_handler = fold_mouse(fold_key)
        else:
            fold_handler = None

        chevron = "▸" if is_collapsed else "▾"
        prefix = f"  {chevron} {bullet} "
        title_width = max(1, width - get_cwidth(prefix))
        shown_title = _truncate_display(title, title_width)
        header_style = (
            "class:detail.fold.collapsed"
            if is_collapsed
            else "class:detail.fold.expanded"
        )
        if record.status.strip().lower() in {"error", "failed"}:
            header_style = bstyle
        if fold_handler is not None:
            title_style = header_style
            if first_open:
                if record.status.strip().lower() in {"error", "failed"}:
                    title_style = "class:detail.error.link"
                elif is_collapsed:
                    title_style = "class:detail.fold.collapsed.link"
                else:
                    title_style = "class:detail.fold.expanded.link"
            fragments.append((header_style, prefix, fold_handler))
            fragments.append((title_style, shown_title + "\n", fold_handler))
        else:
            head = _truncate_display(f"{prefix}{title}", width)
            fragments.append((header_style, head + "\n"))
        if is_collapsed:
            return fragments

        for block in record.blocks:
            if block.label:
                line_count = max(
                    1, block.text.count("\n") + (1 if block.text else 0)
                )
                section_label = {
                    "调用参数": "输入",
                    "返回结果": "结果",
                    CHANGE_LABEL: CHANGE_LABEL,
                }.get(block.label, block.label)
                label_text = _truncate_display(
                    f"    {section_label} · {line_count} 行", width
                )
                fragments.append(("class:detail.tool.section", label_text + "\n"))
            if block.kind in {"code", "diff", "command", "output", "metadata"}:
                paint_cap = (
                    EXPANDED_DIFF_PREVIEW_LINES
                    if block.kind == "diff"
                    else EXPANDED_CODE_PREVIEW_LINES
                )
                fragments.extend(
                    _render_structured_block(
                        block,
                        width,
                        max_lines=paint_cap,
                        show_meta=not bool(block.label),
                    )
                )
            else:
                fragments.extend(_render_text_block(block.text, width))
        if fold_handler is not None:
            fragments.append(
                (
                    "class:detail.fold.footer",
                    _truncate_display("  ▴ 收起工具详情", width) + "\n",
                    fold_handler,
                )
            )
        return fragments

    # ── Non-tool (user / think / assistant) ──────────────────────────
    # No repeated "MAIN" stamps — modal title already names the agent.
    # Only stamp USER turns (and rare non-generic titles).
    actor_norm = actor.strip().upper()
    actor_style = _actor_style(actor)
    is_think_actor = actor_norm in {"THINK", "THINKING", "REASONING"}
    generic_titles = {"进度", "回复", "输出", "思考", ""}
    if is_think_actor:
        byline = None  # block header "思考" is enough
    elif actor_norm == "USER":
        byline = "你"
        if title and title not in generic_titles and title not in {
            "补充指令",
            "指令",
        }:
            byline = f"你  ·  {title}"
    elif title and title not in generic_titles:
        # Named / special section only — never spam MAIN on every prose chunk.
        short = {
            "USER": "你",
            "THINK": "思考",
            "MAIN": "",
            "ASSISTANT": "",
            "AGENT": "",
        }.get(actor_norm, (actor.strip()[:12] or "").strip())
        byline = f"{short}  ·  {title}".strip(" ·") if short else title
    else:
        byline = None
    if byline:
        fragments.append((actor_style, f"  {byline}"))
        fragments.append(("", "\n"))
        fragments.append(("", "\n"))
    for block in record.blocks:
        is_thinking = block.kind == "thinking" or block.label == THINKING_LABEL
        if is_thinking:
            fragments.extend(_render_thinking_block(block, width))
            continue
        if block.label:
            fragments.append(("class:detail.meta", f"  {block.label}\n"))
        if block.kind in {"code", "diff", "command", "output", "metadata"}:
            paint_cap = (
                EXPANDED_DIFF_PREVIEW_LINES
                if block.kind == "diff"
                else EXPANDED_CODE_PREVIEW_LINES
            )
            fragments.extend(
                _render_structured_block(block, width, max_lines=paint_cap)
            )
        else:
            fragments.extend(_render_text_block(block.text, width))
    return fragments


def _render_thinking_block(
    block: DetailBlock,
    width: int,
) -> StyleAndTextTuples:
    """Thinking body with a soft left rail, rendered directly and in full."""

    # Use the full panel — do not hard-clip to MAX_THOUGHTS_WIDTH.
    width = max(12, width)
    body_width = max(1, width - 6)
    lines = block.text.splitlines() or [""]
    fragments: StyleAndTextTuples = []
    for raw in lines:
        for piece in _wrap_display(raw, body_width):
            fragments.append(("class:detail.thinking.rail", "  │ "))
            fragments.append(("class:detail.thinking.body", piece))
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
# Inline spans. Markdown strong uses color only; weight is intentionally absent.
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_INLINE_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_INLINE_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)")
_INLINE_STRIKE_RE = re.compile(r"~~(.+?)~~")


def _render_structured_block(
    block: DetailBlock,
    width: int,
    *,
    max_lines: int | None = None,
    show_meta: bool = True,
) -> StyleAndTextTuples:
    """Grok-like rail + gutter + soft-wrap body (no heavy box blob).

    ``max_lines`` caps paint for TUI (tool expand is not a full editor).
    When truncated, a quiet footer reports the undisplayed line count.
    """

    width = max(12, width)
    raw_lines = block.text.splitlines() or [""]
    line_count = len(raw_lines)
    truncated = False
    paint_block = block
    if max_lines is not None and line_count > max_lines:
        truncated = True
        kept = "\n".join(raw_lines[: max(0, max_lines)])
        paint_block = DetailBlock(
            block.kind,
            kept,
            status=block.status,
            label=block.label,
        )
        raw_lines = paint_block.text.splitlines() or [""]

    fragments: StyleAndTextTuples = []

    # Quiet meta strip instead of ┌──┐ iron box.
    kind_label = {
        "code": "code",
        "diff": "diff",
        "command": "cmd",
        "output": "out",
        "metadata": "args",
    }.get(block.kind, block.kind)
    if show_meta:
        if truncated:
            meta = f"  · {kind_label} · 前 {max_lines}/{line_count} 行"
        else:
            meta = f"  · {kind_label} · {line_count} lines"
        fragments.append(
            ("class:detail.code.meta", _truncate_display(meta, width) + "\n")
        )

    if paint_block.kind == "command":
        fragments.extend(_render_command_block(paint_block, width))
    elif paint_block.kind == "diff":
        fragments.extend(_render_diff_block(paint_block, width))
    elif paint_block.kind == "metadata":
        fragments.extend(_render_plain_rail_block(paint_block, width, highlight=False))
    else:
        # code + output: line numbers when present or synthetic for code.
        # Highlight both code *and* output (shell dumps often carry source-like text).
        show_numbers = paint_block.kind == "code" or _block_has_line_prefixes(raw_lines)
        use_hl = paint_block.kind in {"code", "output"} or show_numbers
        fragments.extend(
            _render_gutter_block(
                paint_block,
                width,
                show_numbers=show_numbers,
                highlight=use_hl,
            )
        )

    if truncated:
        more = line_count - int(max_lines or 0)
        foot = _truncate_display(f"  … 另有 {more} 行未显示", width)
        fragments.append(("class:detail.meta", foot + "\n"))
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
                # Quote body stays italic muted; still honor inline code/strong.
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
    """Grok-lite inline: code, strong color, italic, and strike."""

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
    """Apply strong color / italic / strike on a non-code span."""

    if not text:
        return []
    # Process Markdown strong first, then italic on remaining plain spans.
    out: StyleAndTextTuples = []
    pos = 0
    for match in _INLINE_BOLD_RE.finditer(text):
        if match.start() > pos:
            out.extend(_render_inline_italic_strike(text[pos : match.start()]))
        strong = match.group(1) if match.group(1) is not None else match.group(2)
        out.append(("class:detail.md.strong", strong or ""))
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


def _highlight_code_line(line: str) -> StyleAndTextTuples:
    return highlight_code_line(line, style_prefix="detail.code")


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
    """Tool args for paint — edits become change hunks, not full-file dumps."""
    if isinstance(arguments, str):
        text = arguments
    elif isinstance(arguments, dict):
        low = (name or "").lower()
        command = arguments.get("command") or arguments.get("cmd")
        if command is not None and low in {
            "shell",
            "run_terminal_cmd",
            "run_command",
            "bash",
            "run_terminal_command",
            "execute",
        }:
            return DetailBlock("command", f"$ {command}", label="调用参数")
        # Prefer change-only view for file mutations.
        change = _change_hunk_from_arguments(low, arguments)
        if change is not None:
            return change
        lines: list[str] = []
        for key, value in arguments.items():
            # Skip bulky payloads already represented as change hunks elsewhere.
            if key in {
                "contents",
                "content",
                "old_string",
                "new_string",
                "old_str",
                "new_str",
                "patch",
                "diff",
            }:
                if isinstance(value, str) and value.count("\n") + 1 > 3:
                    lines.append(f"{key}: …({len(value)} 字符)")
                    continue
            rendered = value if isinstance(value, str) else json.dumps(
                value, ensure_ascii=False, sort_keys=True
            )
            lines.append(f"{key}: {rendered}")
        text = "\n".join(lines) if lines else "{}"
    else:
        text = json.dumps(arguments, ensure_ascii=False, default=str)
    kind: DetailBlockKind = "diff" if "patch" in (name or "").lower() else "metadata"
    return DetailBlock(kind, _cap_display_text(text), label="调用参数")


def _change_hunk_from_arguments(
    name: str, arguments: dict[str, Any]
) -> DetailBlock | None:
    """Build a compact unified-diff style block for edit tools only."""

    path = ""
    for key in (
        "path",
        "target_file",
        "file_path",
        "file",
        "filename",
    ):
        val = arguments.get(key)
        if isinstance(val, str) and val.strip():
            path = val.strip().replace("\\", "/")
            if "/" in path:
                path = path.rsplit("/", 1)[-1]
            break

    low = name.lower()
    if low in {"search_replace", "str_replace", "edit", "replace_all"}:
        old = arguments.get("old_string")
        if old is None:
            old = arguments.get("old_str")
        new = arguments.get("new_string")
        if new is None:
            new = arguments.get("new_str")
        if not isinstance(old, str) and not isinstance(new, str):
            return None
        old_s = old if isinstance(old, str) else ""
        new_s = new if isinstance(new, str) else ""
        header = f"@@ {path or 'edit'} @@"
        body = _unified_hunk_lines(old_s, new_s)
        text = "\n".join([header, *body])
        return DetailBlock(
            "diff",
            _cap_display_text(text),
            label=CHANGE_LABEL,
        )

    if low in {"write", "create_file"}:
        contents = arguments.get("contents")
        if contents is None:
            contents = arguments.get("content")
        if not isinstance(contents, str):
            return None
        lines = contents.splitlines() or [""]
        total = len(lines)
        show = lines[:CHANGE_HUNK_MAX_LINES]
        header = f"@@ {path or 'write'} · new · {total} lines @@"
        body = [f"+{line}" for line in show]
        if total > len(show):
            body.append(f"+… 另有 {total - len(show)} 行未展示")
        text = "\n".join([header, *body])
        return DetailBlock(
            "diff",
            _cap_display_text(text),
            label=CHANGE_LABEL,
        )

    if low in {"apply_patch", "apply_diff"} or "patch" in low:
        patch = arguments.get("patch")
        if patch is None:
            patch = arguments.get("diff")
        if not isinstance(patch, str) or not patch.strip():
            return None
        return DetailBlock(
            "diff",
            _cap_display_text(patch.strip()),
            label=CHANGE_LABEL,
        )
    return None


def _unified_hunk_lines(old: str, new: str) -> list[str]:
    """Simple -/+ lines for a replace (no full-file context)."""

    old_lines = (old or "").splitlines() or ([""] if old == "" else [])
    new_lines = (new or "").splitlines() or ([""] if new == "" else [])
    # Cap each side so a huge replace still reads as a change, not a novel.
    half = max(4, CHANGE_HUNK_MAX_LINES // 2)
    out: list[str] = []
    if len(old_lines) > half:
        for line in old_lines[:half]:
            out.append(f"-{line}")
        out.append(f"-… 另有 {len(old_lines) - half} 行")
    else:
        for line in old_lines:
            out.append(f"-{line}")
    if len(new_lines) > half:
        for line in new_lines[:half]:
            out.append(f"+{line}")
        out.append(f"+… 另有 {len(new_lines) - half} 行")
    else:
        for line in new_lines:
            out.append(f"+{line}")
    return out or ["-(empty)", "+(empty)"]


def _compact_read_preview(text: str) -> str:
    """GrokBuild-style Read expand: short head/tail, never the whole file.

    Collapsed row already shows ``Read filename``; expand is a peek only.
    Full content is for the editor via the open-file link.
    """

    raw = text or ""
    lines = raw.splitlines() or [""]
    total = len(lines)
    head_n = READ_PREVIEW_HEAD_LINES
    tail_n = READ_PREVIEW_TAIL_LINES
    if total <= head_n + tail_n + 1:
        return raw
    head = lines[:head_n]
    tail = lines[-tail_n:] if tail_n else []
    omitted = total - head_n - tail_n
    mid = f"… 省略 {omitted} 行 · 完整内容请点下方打开"
    return "\n".join([*head, mid, *tail])


def _tool_result_block(name: str, output: str) -> DetailBlock | None:
    """Build the tool-result body, or None when Grok-style one-line is enough.

    Edit tools: change lives in the 变更 args block; a bare ``ok`` is omitted
    (matches GrokBuild compact edit rows — no full-file dump on expand).
    """
    lowered = name.lower()
    if "patch" in lowered or _looks_like_diff(output):
        kind: DetailBlockKind = "diff"
        body = _cap_display_text(output)
        label = CHANGE_LABEL if _looks_like_diff(output) else "返回结果"
    elif lowered in {"read_file", "read", "view_file"}:
        kind = "code"
        # Read = context, not a full buffer; keep a short head/tail only.
        body = _cap_display_text(_compact_read_preview(output))
        label = "返回结果"
    elif lowered in {
        "write",
        "create_file",
        "search_replace",
        "str_replace",
        "edit",
        "apply_patch",
        "apply_diff",
        "replace_all",
    }:
        # GrokBuild: compact edit — skip trivial success acks (变更 already has hunk).
        one = " ".join((output or "").split())
        low = one.lower()
        if (
            not one
            or low in {"ok", "done", "success", "updated", "wrote", "created"}
            or low.startswith("ok ")
            or (len(one) <= 40 and "error" not in low and "fail" not in low)
        ):
            return None
        kind = "output"
        if len(one) > 120:
            one = one[:119] + "…"
        body = one
        label = "返回结果"
    elif lowered in {"shell", "run_terminal_cmd", "run_command", "bash"}:
        kind = "output"
        body = _cap_display_text(output)
        label = "返回结果"
    else:
        kind = "output"
        body = _cap_display_text(output)
        label = "返回结果"
    return DetailBlock(
        kind,
        body,
        status=_result_status(output),
        label=label,
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
    "MAX_THOUGHTS_WIDTH",
    "THINKING_LABEL",
    "AgentDetailSnapshot",
    "DetailBlock",
    "DetailFilter",
    "DetailRecord",
    "default_collapsed_keys",
    "filter_detail_records",
    "render_agent_detail",
    "render_detail_body",
    "render_detail_header",
    "record_collapse_key",
    "snapshot_from_messages",
    "strip_system_reminders",
]
