"""OtherToolCallBlock — unknown / generic tools.

Grok source: ``blocks/tool/other.rs``
Header: bold name (or ``Label: content``) + muted summary; optional body.
"""

from __future__ import annotations

from codedoggy.tui_v2.blocks.tool.common import (
    S_BOLD,
    S_MUTED,
    S_PRIMARY,
    Rows,
    arg_str,
    empty_row,
    is_running,
    result_lines,
    row,
    truncate_str,
    wrap_text,
)


def paint_other(
    name: str,
    arguments: dict,
    result: str,
    *,
    width: int,
    collapsed: bool,
    status: str,
    selected: bool = False,
) -> Rows:
    summary = arg_str(
        arguments,
        "summary",
        "description",
        "title",
        "message",
        default="",
    )
    # Prefer a human target from common keys when summary empty.
    if not summary:
        for key in ("path", "query", "url", "pattern", "command", "todos"):
            val = arguments.get(key)
            if isinstance(val, str) and val:
                summary = val
                break
            if isinstance(val, list):
                summary = f"{len(val)} items"
                break

    running = is_running(status)
    muted = collapsed and not running
    text_style = S_MUTED if muted else S_PRIMARY
    bold_style = S_MUTED + " bold" if muted else S_BOLD
    if selected:
        text_style = f"{text_style} reverse"
        bold_style = f"{bold_style} reverse"

    display_name = name or "tool"
    frags: list[tuple[str, str]]
    if ": " in display_name:
        label, content = display_name.split(": ", 1)
        frags = [(bold_style, f"{label} "), (text_style, content)]
    else:
        frags = [(bold_style, display_name)]

    if summary:
        s = f"  {summary}"
        if collapsed:
            used = sum(len(t) for _, t in frags)
            if used + len(s) <= width:
                frags.append((S_MUTED, s))
        else:
            frags.append((S_MUTED, s))

    if collapsed:
        # Hard truncate whole header line.
        total = "".join(t for _, t in frags)
        if len(total) > width > 0:
            # Rebuild single truncated span.
            frags = [(bold_style, truncate_str(total, width))]
        return [row(*frags)]

    rows: Rows = [row(*frags)]

    lines = result_lines(result)
    if not lines:
        return rows

    rows.append(empty_row())
    content_w = max(20, width - 2)
    for line in lines:
        for piece in wrap_text(line, content_w):
            rows.append(row((S_MUTED, piece)))
    return rows
