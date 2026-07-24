"""SearchToolCallBlock — grep/glob pattern search.

Grok source: ``blocks/tool/search.rs``
Header: fixed ``Search `` + quoted pattern + summary.
"""

from __future__ import annotations

import re
from codedoggy.tui_v2.blocks.tool.common import (
    HEADER_SEARCH,
    S_BOLD,
    S_MUTED,
    S_PANEL,
    S_PATH,
    S_PRIMARY,
    S_SUCCESS,
    Rows,
    arg_str,
    empty_row,
    is_running,
    result_lines,
    row,
)


_RG_LINE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<text>.*)$")
_RG_COUNT = re.compile(r"^(?P<path>.+):(?P<count>\d+)$")


def _pattern(arguments: dict) -> str:
    return arg_str(arguments, "pattern", "query", "regex", default="")


def _glob(arguments: dict) -> str | None:
    g = arg_str(arguments, "glob", "include", default="")
    return g or None


def _search_path(arguments: dict) -> str | None:
    p = arg_str(arguments, "path", "directory", default="")
    return p or None


def _is_trivial(pattern: str) -> bool:
    return pattern == "" or pattern == "."


def _parse_file_matches(result: str) -> list[tuple[str, list[tuple[int, str]]]]:
    """Group content-mode lines ``path:line:text`` by file."""
    groups: dict[str, list[tuple[int, str]]] = {}
    order: list[str] = []
    for raw in result_lines(result):
        m = _RG_LINE.match(raw)
        if not m:
            continue
        path = m.group("path")
        ln = int(m.group("line"))
        text = m.group("text")
        if path not in groups:
            groups[path] = []
            order.append(path)
        groups[path].append((ln, text))
    return [(p, groups[p]) for p in order]


def _match_summary(
    match_count: int,
    file_count: int,
    *,
    mode: str = "content",
) -> str:
    if match_count == 0:
        return "(no files)" if mode == "files_with_matches" else "(no matches)"
    if mode == "files_with_matches":
        return "(1 file)" if match_count == 1 else f"({match_count} files)"
    if mode == "count":
        if file_count > 1:
            return f"({match_count} matches across {file_count} files)"
        if match_count == 1:
            return "(1 match)"
        return f"({match_count} matches)"
    # content
    if file_count > 1:
        return f"({match_count} matches in {file_count} files)"
    if match_count == 1:
        return "(1 match)"
    return f"({match_count} matches)"


def paint_search(
    arguments: dict,
    result: str,
    *,
    width: int,
    collapsed: bool,
    status: str,
    selected: bool = False,
) -> Rows:
    pattern = _pattern(arguments)
    glob = _glob(arguments)
    path = _search_path(arguments)
    mode = arg_str(arguments, "output_mode", default="content") or "content"
    running = is_running(status)
    muted = collapsed and not running

    # Fixed present verb (Grok individual tool block); tense is verb_group only.
    prefix = HEADER_SEARCH

    pattern_style = S_MUTED if muted else S_SUCCESS
    text_style = S_MUTED if muted else S_PRIMARY
    path_style = S_MUTED if muted else S_PATH
    detail_style = S_MUTED

    p_style = S_MUTED + " bold" if muted else S_BOLD
    if selected:
        p_style = f"{p_style} reverse"
        pattern_style = f"{pattern_style} reverse"

    frags: list[tuple[str, str]] = [(p_style, prefix)]

    if _is_trivial(pattern) and glob:
        frags.append((pattern_style, glob))
    else:
        # Python repr matches Rust {:?} for simple strings.
        frags.append((pattern_style, repr(pattern)))
        if glob:
            frags.append((text_style, " in "))
            frags.append((pattern_style, glob))

    if path:
        frags.append((text_style, " in "))
        frags.append((path_style, path))

    groups = _parse_file_matches(result)
    if groups:
        match_count = sum(len(m) for _, m in groups)
        file_count = len(groups)
    else:
        # files_with_matches / plain paths
        paths = [ln for ln in result_lines(result) if ln.strip()]
        match_count = len(paths)
        file_count = len(paths)

    summary = f" {_match_summary(match_count, file_count, mode=mode)}"
    frags.append((detail_style, summary))

    rows: Rows = [row(*frags)]

    if collapsed:
        return rows

    # Metadata line
    rows.append(empty_row())
    mode_label = {
        "content": "pattern",
        "files_with_matches": "files",
        "count": "count",
    }.get(mode, "pattern")
    meta_parts = [f"mode: {mode_label}"]
    ft = arg_str(arguments, "type", "file_type", default="")
    if ft:
        meta_parts.append(f"type: {ft}")
    if arguments.get("case_insensitive") or arguments.get("-i"):
        meta_parts.append("case-insensitive: true")
    if arguments.get("multiline"):
        meta_parts.append("multiline: true")
    rows.append(row((S_MUTED, "  " + ", ".join(meta_parts))))

    if not result.strip():
        rows.append(empty_row())
        rows.append(row((S_MUTED, "  (no results)")))
        return rows

    rows.append(empty_row())

    if groups:
        for i, (fpath, matches) in enumerate(groups):
            if i > 0:
                rows.append(empty_row())
            rows.append(row((f"{S_PANEL} {S_PATH}", f"  {fpath}")))
            for ln, text in matches:
                gutter = f"{ln:>4}"
                rows.append(
                    row(
                        (f"{S_PANEL} {S_PRIMARY}", "    "),
                        (f"{S_PANEL} {S_MUTED}", gutter),
                        (f"{S_PANEL} {S_PRIMARY}", "  "),
                        (f"{S_PANEL} {S_PRIMARY}", text.rstrip()),
                    )
                )
    else:
        for path_line in result_lines(result):
            if not path_line.strip():
                continue
            if mode == "count" and ":" in path_line:
                file_part, _, count_part = path_line.rpartition(":")
                rows.append(
                    row(
                        (f"{S_PANEL} {S_PATH}", f"  {file_part}"),
                        (f"{S_PANEL} {S_PRIMARY}", f":{count_part}"),
                    )
                )
            else:
                rows.append(row((f"{S_PANEL} {S_PATH}", f"  {path_line}")))

    return rows
