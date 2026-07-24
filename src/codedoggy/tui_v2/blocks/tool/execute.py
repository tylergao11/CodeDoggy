"""ExecuteToolCallBlock — shell command with stdout body.

Grok source: ``blocks/tool/execute.rs``
Header: dim ``$ `` + command; stdout body under bg_dark panel.
"""

from __future__ import annotations

from codedoggy.tui_v2.blocks.tool.common import (
    ELLIPSIS,
    FIRST_LINES_EXECUTE,
    HEADER_SHELL,
    LAST_LINES_EXECUTE,
    S_DIM,
    S_ERROR,
    S_MUTED,
    S_PANEL,
    S_PRIMARY,
    Rows,
    arg_str,
    empty_row,
    is_running,
    result_lines,
    row,
    truncate_str,
)


def paint_execute(
    arguments: dict,
    result: str,
    *,
    width: int,
    collapsed: bool,
    status: str,
    selected: bool = False,
    truncated: bool = False,
) -> Rows:
    command = arg_str(
        arguments,
        "command",
        "cmd",
        "shell_command",
        default="",
    )
    description = arg_str(arguments, "description", "desc", default="").strip()
    running = is_running(status)
    muted = collapsed and not running

    display_cmd = command.replace("\n", " ")
    if not display_cmd.strip():
        display_cmd = ELLIPSIS

    rows: Rows = []

    # Description-first when present and collapsed (Grok density).
    if description and collapsed:
        style = S_MUTED if muted else S_PRIMARY
        if selected:
            style = f"{style} reverse"
        title = description.replace("\n", " ")
        # Strip leading Run/Running when we might show Label style elsewhere.
        low = title.lower()
        for prefix in ("running ", "run "):
            if low.startswith(prefix):
                title = title[len(prefix) :]
                break
        budget = max(1, width)
        title = truncate_str(title, budget)
        rows.append(row((style, title)))
        return rows

    # Shell header: `$ command`
    dollar_style = S_DIM
    cmd_style = S_MUTED if muted else S_PRIMARY
    if selected:
        dollar_style = f"{dollar_style} reverse"
        cmd_style = f"{cmd_style} reverse"
    budget = max(1, width - len(HEADER_SHELL))
    shown = truncate_str(display_cmd, budget) if collapsed else display_cmd
    rows.append(row((dollar_style, HEADER_SHELL), (cmd_style, shown)))

    if collapsed:
        return rows

    # Expanded / truncated stdout
    lines = result_lines(result)
    if not lines and status.lower() in {"failed", "error"} and result:
        rows.append(empty_row())
        for err_line in result.splitlines():
            rows.append(row((S_ERROR, err_line)))
        return rows

    if not lines:
        return rows

    rows.append(empty_row())
    content_w = max(20, width - 2)
    # Soft-wrap each source line into display rows; optional first5+last3 truncate.
    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        if len(line) <= content_w:
            wrapped.append(line)
        else:
            # Hard-break long lines (execute approximates word_wrap).
            for i in range(0, len(line), content_w):
                wrapped.append(line[i : i + content_w])

    total = len(wrapped)
    threshold = FIRST_LINES_EXECUTE + LAST_LINES_EXECUTE
    # Expanded (default): full body. Truncated: first 5 + last 3 of wrap rows.
    if truncated and total > threshold:
        head = wrapped[:FIRST_LINES_EXECUTE]
        tail = wrapped[total - LAST_LINES_EXECUTE :]
        hidden = total - threshold
        body = head + [f"{ELLIPSIS} +{hidden} lines"] + tail
    else:
        body = wrapped

    for text in body:
        rows.append(row((f"{S_PANEL} {S_PRIMARY}", text)))

    return rows
