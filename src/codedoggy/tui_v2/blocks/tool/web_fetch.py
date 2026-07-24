"""WebFetchToolCallBlock — URL fetch with content preview.

Grok source: ``blocks/tool/web_fetch.rs``
Header: fixed ``Fetch `` + url.
"""

from __future__ import annotations

from codedoggy.tui_v2.blocks.tool.common import (
    HEADER_FETCH,
    MAX_INLINE_WEB,
    S_COMMAND,
    S_DIM,
    S_MUTED,
    S_PANEL,
    S_PRIMARY,
    TRUNCATED_INLINE_WEB,
    Rows,
    arg_str,
    empty_row,
    header_row,
    is_running,
    result_lines,
    row,
    truncate_str,
)


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def paint_web_fetch(
    arguments: dict,
    result: str,
    *,
    width: int,
    collapsed: bool,
    status: str,
    selected: bool = False,
) -> Rows:
    url = arg_str(arguments, "url", "uri", "href", default="")
    running = is_running(status)
    muted = collapsed and not running

    # Fixed present verb (Grok individual tool block); tense is verb_group only.
    prefix = HEADER_FETCH

    display_url = url
    if collapsed:
        display_url = truncate_str(url, max(1, width - len(prefix)))

    rows: Rows = [
        header_row(
            prefix,
            display_url,
            muted=muted,
            selected=selected,
            target_style=S_MUTED if muted else S_COMMAND,
        )
    ]

    if collapsed:
        return rows

    # Optional metadata from arguments / structured result prefixes
    status_code = arguments.get("status_code") or arguments.get("status")
    content_type = arguments.get("content_type")
    nbytes = arguments.get("bytes") or arguments.get("size")
    meta_bits: list[str] = []
    if status_code is not None:
        meta_bits.append(f"status: {status_code}")
    if content_type:
        meta_bits.append(f"content_type: {content_type}")
    if nbytes is not None:
        try:
            meta_bits.append(f"size: {_format_bytes(int(nbytes))}")
        except (TypeError, ValueError):
            meta_bits.append(f"size: {nbytes}")

    if meta_bits:
        rows.append(empty_row())
        rows.append(row((S_MUTED, "  " + ", ".join(meta_bits))))

    lines = result_lines(result)
    if not lines:
        if status.lower() not in {"failed", "error"}:
            rows.append(empty_row())
            rows.append(row((S_MUTED, "  (no content)")))
        return rows

    rows.append(empty_row())
    rows.append(row((f"{S_PANEL} {S_PRIMARY}", "")))  # top pad

    max_inline = TRUNCATED_INLINE_WEB if collapsed else MAX_INLINE_WEB
    # Expanded uses MAX; "truncated" projection uses tighter cap via width heuristic:
    # Doggy passes collapsed=False for both truncated and expanded; use full max.
    max_inline = MAX_INLINE_WEB
    total = len(lines)
    for i, line in enumerate(lines):
        if i >= max_inline:
            rows.append(
                row(
                    (
                        f"{S_PANEL} {S_DIM}",
                        f"  ... ({total - max_inline} more lines, press Enter to view)",
                    )
                )
            )
            break
        rows.append(row((f"{S_PANEL} {S_PRIMARY}", f"  {line}")))

    rows.append(row((f"{S_PANEL} {S_PRIMARY}", "")))  # bottom pad
    return rows
