"""ListDirToolCallBlock — directory listing.

Grok source: ``blocks/tool/list_dir.rs``
Header: fixed ``List `` + path + optional ``(N entries)``.
"""

from __future__ import annotations

from codedoggy.tui_v2.blocks.tool.common import (
    HEADER_LIST,
    S_MUTED,
    S_PANEL,
    S_PRIMARY,
    Rows,
    arg_path,
    display_path,
    empty_row,
    header_row,
    is_running,
    result_lines,
    row,
)


def paint_list_dir(
    arguments: dict,
    result: str,
    *,
    width: int,
    collapsed: bool,
    status: str,
    selected: bool = False,
) -> Rows:
    path = arg_path(arguments) or arg_path({"path": arguments.get("target_directory", "")})
    if not path:
        path = str(arguments.get("target_directory") or arguments.get("path") or ".")

    running = is_running(status)
    muted = collapsed and not running
    # Fixed present verb (Grok individual tool block); tense is verb_group only.
    prefix = HEADER_LIST

    shown = display_path(path, collapsed=collapsed)
    lines = [ln for ln in result_lines(result) if ln.strip()]
    entry_count = len(lines)

    suffixes: list[tuple[str, str]] = []
    if not running and entry_count > 0 and status.lower() not in {"failed", "error"}:
        if entry_count == 1:
            suffixes.append((S_MUTED, " (1 entry)"))
        else:
            suffixes.append((S_MUTED, f" ({entry_count} entries)"))

    rows: Rows = [
        header_row(prefix, shown, *suffixes, muted=muted, selected=selected)
    ]

    if collapsed or not lines:
        return rows

    rows.append(empty_row())
    for line in result_lines(result):
        # Indent output by 2 spaces (Grok list_dir).
        rows.append(row((f"{S_PANEL} {S_PRIMARY}", f"  {line}")))
    return rows
