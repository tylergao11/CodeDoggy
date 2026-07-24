"""ReadToolCallBlock — file read with gutter line numbers.

Grok source: ``blocks/tool/read.rs``
Header: fixed ``Read `` + path (bold prefix, path color); optional range / (empty).
Expanded body: full content. Truncated (``truncated=True``): first 5 + last 3 lines.
"""

from __future__ import annotations

from codedoggy.tui_v2.blocks.tool.common import (
    ELLIPSIS,
    FIRST_LINES_READ,
    HEADER_READ,
    LAST_LINES_READ,
    S_DIM,
    S_ERROR,
    S_MUTED,
    S_PANEL,
    S_PRIMARY,
    Rows,
    arg_path,
    digit_count,
    display_path,
    empty_row,
    header_row,
    is_running,
    result_lines,
    row,
)


def _line_range(arguments: dict) -> tuple[int, int] | None:
    """1-based inclusive range from Doggy args (offset/limit or start_line/end_line)."""
    start = arguments.get("offset") or arguments.get("start_line") or arguments.get("start")
    limit = arguments.get("limit")
    end = arguments.get("end_line") or arguments.get("end")
    if start is None and end is None and limit is None:
        return None
    try:
        s = int(start) if start is not None else 1
        if end is not None:
            e = int(end)
        elif limit is not None:
            e = s + int(limit) - 1
        else:
            e = s
        if s < 1:
            s = 1
        if e < s:
            e = s
        return s, e
    except (TypeError, ValueError):
        return None


def _skill_name(path: str) -> str | None:
    """Skill name when path ends with …/skills/<name>/SKILL.md."""
    norm = path.replace("\\", "/")
    if not norm.endswith("/SKILL.md") and not norm.endswith("SKILL.md"):
        return None
    parts = [p for p in norm.split("/") if p]
    if len(parts) >= 2 and parts[-1] == "SKILL.md":
        # Prefer parent of SKILL.md when under a skills/ directory.
        parent = parts[-2]
        if "skills" in parts:
            return parent
        return parent
    return None


def paint_read(
    arguments: dict,
    result: str,
    *,
    width: int,
    collapsed: bool,
    status: str,
    selected: bool = False,
    truncated: bool = False,
) -> Rows:
    path = arg_path(arguments)
    running = is_running(status)
    muted = collapsed and not running
    skill = _skill_name(path)

    if skill:
        rows: Rows = [
            header_row(
                "Skill ",
                skill,
                muted=muted,
                selected=selected,
                target_style=S_MUTED if muted else "class:grok.path",
            )
        ]
        return rows

    # Fixed present verb (Grok individual tool block); tense is verb_group only.
    prefix = HEADER_READ
    shown = display_path(path or "?", collapsed=collapsed)
    lr = _line_range(arguments)
    lines = result_lines(result)
    total_lines = len(lines) if lines else None

    suffixes: list[tuple[str, str]] = []
    detail = S_DIM  # Grok dim_details default
    if lr is not None:
        start, end = lr
        span = end - start + 1
        # Grok LineRange Display is "start-end"
        if total_lines is not None and total_lines > span:
            suffixes.append((detail, f" ({start}-{end} of {total_lines})"))
        else:
            suffixes.append((detail, f" ({start}-{end})"))
    # Empty-file annotation when finished with empty content body.
    if (
        not lines
        and not running
        and result == ""
        and status.lower() not in {"failed", "error", "running", "pending"}
        and path
    ):
        suffixes.append((detail, " (empty)"))

    rows = [
        header_row(
            prefix,
            shown,
            *suffixes,
            muted=muted,
            selected=selected,
        )
    ]

    if collapsed:
        return rows

    # Expanded / truncated content
    if not lines and status.lower() in {"failed", "error"}:
        rows.append(empty_row())
        for err_line in (result or "error").splitlines() or ["error"]:
            rows.append(row((S_ERROR, err_line)))
        return rows

    if not lines:
        return rows

    rows.append(empty_row())
    base_line = lr[0] if lr else 1
    gutter_w = digit_count(base_line + len(lines) - 1)
    content_width = max(20, width - gutter_w - 2)

    # Expanded (default): full body. Truncated: first 5 + last 3 of source lines.
    # Grok wraps first then truncates wrap rows; we approximate on source lines.
    total = len(lines)
    threshold = FIRST_LINES_READ + LAST_LINES_READ
    if truncated and total > threshold:
        indices = list(range(FIRST_LINES_READ)) + list(
            range(total - LAST_LINES_READ, total)
        )
        show_ellipsis_at = FIRST_LINES_READ
    else:
        indices = list(range(total))
        show_ellipsis_at = None

    painted = 0
    try:
        from codedoggy.tui_v2.syntax_hl import highlight_code_line
    except Exception:  # noqa: BLE001
        highlight_code_line = None  # type: ignore[assignment]

    for idx in indices:
        if show_ellipsis_at is not None and painted == show_ellipsis_at:
            rows.append(row((f"{S_PANEL} {S_MUTED}", ELLIPSIS)))
        line_no = base_line + idx
        gutter = f"{line_no:>{gutter_w}}  "
        text = lines[idx]
        if len(text) > content_width:
            text = text[:content_width]
        body_frags: list[tuple[str, str]]
        if highlight_code_line is not None and text:
            try:
                body_frags = [
                    (f"{S_PANEL} {st}", tx)
                    for st, tx in highlight_code_line(text, path=path or None)
                ]
            except Exception:  # noqa: BLE001
                body_frags = [(f"{S_PANEL} {S_PRIMARY}", text)]
        else:
            body_frags = [(f"{S_PANEL} {S_PRIMARY}", text)]
        if not body_frags:
            body_frags = [(f"{S_PANEL} {S_PRIMARY}", text or "")]
        rows.append(row((f"{S_PANEL} {S_DIM}", gutter), *body_frags))
        painted += 1

    return rows
