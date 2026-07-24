"""BgTaskBlock — port spirit of ``scrollback/blocks/bg_task.rs``.

Lifecycle header (always painted)::

    Task started: “description”
    Task completed in 1.2s: “description”
    Task failed in 0.5s: “description” exit 1
    Task killed: “description”

When expanded (``collapsed=False``), stdout body follows under a bg_dark
panel band (Grok "open block viewer" spirit in scrollback) — same first-5 /
last-3 truncate as execute when ``truncated=True``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codedoggy.tui_v2.blocks.subagent import quoted_desc
from codedoggy.tui_v2.blocks.tool.common import (
    ELLIPSIS,
    FIRST_LINES_EXECUTE,
    LAST_LINES_EXECUTE,
    S_MUTED as S_COMMON_MUTED,
    S_PANEL,
    S_PRIMARY,
    body_lines_with_ellipsis,
    result_lines,
)

if TYPE_CHECKING:
    from prompt_toolkit.formatted_text import StyleAndTextTuples
else:
    StyleAndTextTuples = list  # type: ignore[misc, assignment]

S_BOLD = "class:grok.muted bold"
S_BOLD_SELECTED = "class:grok.primary bold"
S_MUTED = "class:grok.muted"

# Panel body style (bg_dark + primary), matching execute expanded stdout.
S_BODY = f"{S_PANEL} {S_PRIMARY}"

# "Task " prefix length for desc width budget
_LABEL_LEN = 5

# Cap full (non-truncated) expanded body so huge task logs stay scrollable.
_MAX_FULL_BODY_LINES = 200


def _format_elapsed_ms(ms: int | None) -> str:
    if ms is None:
        return "0.0s"
    secs = max(0.0, ms / 1000.0)
    if secs < 60.0:
        return f"{secs:.1f}s"
    mins = int(secs // 60.0)
    rem = secs - mins * 60.0
    return f"{mins}m{rem:.0f}s"


def _normalize_desc(description: str) -> str:
    """Collapse newlines for single-line display (Grok bg_task.rs).

    Callers pass preferred label already: description if non-empty, else
    command-like text — a single ``description`` arg carries either.
    """
    return (description or "").replace("\n", " ")


def _paint_header(
    description: str,
    *,
    width: int,
    status: str,
    elapsed_ms: int | None,
    selected: bool,
    exit_code: int | None,
    signal: str | None,
) -> list[StyleAndTextTuples]:
    """One-line lifecycle header (content only)."""
    _ = signal  # API parity; not painted yet
    st = (status or "").lower()
    bold = S_BOLD_SELECTED if selected else S_BOLD
    muted = S_MUTED
    w = max(8, int(width))
    display = _normalize_desc(description)

    if st in {"pending", "running", "started"}:
        verb = "started: "
        desc = quoted_desc(display, w - _LABEL_LEN - len(verb))
        return [[(bold, "Task "), (muted, verb), (muted, desc)]]

    if st in {"completed", "done", "success"}:
        time_str = _format_elapsed_ms(elapsed_ms)
        verb = f"completed in {time_str}: "
        desc = quoted_desc(display, w - _LABEL_LEN - len(verb))
        return [[(bold, "Task "), (muted, verb), (muted, desc)]]

    if st in {"failed", "error"}:
        time_str = _format_elapsed_ms(elapsed_ms)
        verb = f"failed in {time_str}: "
        suffix = f" exit {exit_code}" if exit_code is not None else ""
        desc = quoted_desc(display, w - _LABEL_LEN - len(verb) - len(suffix))
        row: StyleAndTextTuples = [
            (bold, "Task "),
            (muted, verb),
            (muted, desc),
        ]
        if suffix:
            row.append((muted, suffix))
        return [row]

    if st in {"cancelled", "canceled", "killed"}:
        verb = "killed: "
        desc = quoted_desc(display, w - _LABEL_LEN - len(verb))
        return [[(bold, "Task "), (muted, verb), (muted, desc)]]

    # Fallback — treat unknown status like started
    verb = "started: "
    desc = quoted_desc(display, w - _LABEL_LEN - len(verb))
    return [[(bold, "Task "), (muted, verb), (muted, desc)]]


def _soft_wrap_lines(lines: list[str], content_w: int) -> list[str]:
    """Hard-break long lines to content width (execute spirit)."""
    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        if len(line) <= content_w:
            wrapped.append(line)
        else:
            for i in range(0, len(line), content_w):
                wrapped.append(line[i : i + content_w])
    return wrapped


def _paint_body(
    output: str,
    *,
    width: int,
    truncated: bool,
) -> list[StyleAndTextTuples]:
    """Expanded stdout body under optional blank separator."""
    lines = result_lines(output)
    if not lines:
        return [
            [("", "")],  # blank separator
            [(S_COMMON_MUTED, "  (no output)")],
        ]

    content_w = max(20, int(width) - 2)
    wrapped = _soft_wrap_lines(lines, content_w)
    total = len(wrapped)
    threshold = FIRST_LINES_EXECUTE + LAST_LINES_EXECUTE

    if truncated and total > threshold:
        body = body_lines_with_ellipsis(
            wrapped,
            FIRST_LINES_EXECUTE,
            LAST_LINES_EXECUTE,
            show_hidden_count=True,
        )
    elif not truncated and total > _MAX_FULL_BODY_LINES:
        kept = wrapped[:_MAX_FULL_BODY_LINES]
        more = total - _MAX_FULL_BODY_LINES
        body = kept + [f"{ELLIPSIS} ({more} more lines)"]
    else:
        body = wrapped

    rows: list[StyleAndTextTuples] = [[("", "")]]  # blank separator
    for text in body:
        rows.append([(S_BODY, text)])
    return rows


def paint_bg_task(
    description: str,
    *,
    width: int,
    status: str,
    elapsed_ms: int | None = None,
    selected: bool = False,
    exit_code: int | None = None,
    signal: str | None = None,
    collapsed: bool = True,
    truncated: bool = False,
    output: str = "",
) -> list[StyleAndTextTuples]:
    """Paint bg-task lifecycle header + optional expanded stdout body.

    Status mapping:
      pending / running → started
      completed / done / success → completed
      failed / error → failed
      cancelled / canceled / killed → killed

    Optional ``exit_code`` / ``signal`` are accepted for callers that have
    process detail; only failed status appends muted `` exit N`` when
    ``exit_code`` is not None. ``signal`` is reserved (no display yet).

    When ``collapsed`` (default), returns header only. When expanded and
    ``output`` is non-empty, appends a blank separator then body lines in a
    bg_dark panel (``class:grok.bg_dark class:grok.primary``). Truncated
    mode keeps first 5 + last 3 wrap rows with a mid ellipsis; full mode
    shows all lines capped at 200 with a trailing ``… (N more lines)``.
    Empty expanded output shows muted ``  (no output)``.
    """
    rows = _paint_header(
        description,
        width=width,
        status=status,
        elapsed_ms=elapsed_ms,
        selected=selected,
        exit_code=exit_code,
        signal=signal,
    )
    if collapsed:
        return rows
    rows.extend(
        _paint_body(output, width=width, truncated=truncated)
    )
    return rows


__all__ = ["paint_bg_task"]
