"""WebSearchToolCallBlock — web search with citations preview.

Grok source: ``blocks/tool/web_search.rs``
Header: ``Web Search `` + query + optional ``(N sites)``.
"""

from __future__ import annotations

from urllib.parse import urlparse

from codedoggy.tui_v2.blocks.tool.common import (
    HEADER_WEB_SEARCH,
    MAX_INLINE_WEB,
    S_COMMAND,
    S_DIM,
    S_MUTED,
    S_PANEL,
    S_PRIMARY,
    Rows,
    arg_str,
    empty_row,
    header_row,
    is_running,
    result_lines,
    row,
    truncate_str,
)

MAX_INLINE_SOURCES = 3


def _unique_domains(citations: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for url in citations:
        try:
            host = urlparse(url).hostname
        except Exception:  # noqa: BLE001
            host = None
        if not host or host in seen:
            continue
        seen.add(host)
        out.append(host)
    return out


def paint_web_search(
    arguments: dict,
    result: str,
    *,
    width: int,
    collapsed: bool,
    status: str,
    selected: bool = False,
) -> Rows:
    query = arg_str(arguments, "query", "q", "search", default="")
    label = arg_str(arguments, "label", default="") or HEADER_WEB_SEARCH
    if not label.endswith(" "):
        label = label + " "
    citations = arguments.get("citations") or arguments.get("sources") or []
    if not isinstance(citations, list):
        citations = []
    citations = [str(c) for c in citations]

    running = is_running(status)
    muted = collapsed and not running
    # Grok block source always uses "Web Search " (label override optional).
    prefix = label

    domains = _unique_domains(citations)
    suffixes: list[tuple[str, str]] = []
    display_query = query

    if collapsed:
        site_count = len(domains)
        suffix = ""
        if site_count > 0:
            s = "" if site_count == 1 else "s"
            suffix = f" ({site_count} site{s})"
        if len(prefix) + len(suffix) < width:
            budget = max(1, width - len(prefix) - len(suffix))
            display_query = truncate_str(query, budget)
            if suffix:
                suffixes.append((S_DIM, suffix))
        else:
            display_query = truncate_str(query, max(1, width - len(prefix)))

    rows: Rows = [
        header_row(
            prefix,
            display_query,
            *suffixes,
            muted=muted,
            selected=selected,
            target_style=S_MUTED if muted else S_COMMAND,
        )
    ]

    if collapsed:
        return rows

    lines = result_lines(result)
    if lines:
        rows.append(empty_row())
        rows.append(row((f"{S_PANEL} {S_PRIMARY}", "")))
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
        rows.append(row((f"{S_PANEL} {S_PRIMARY}", "")))
    elif status.lower() in {"failed", "error"} and result:
        rows.append(empty_row())
        rows.append(row((S_MUTED, f"  {result}")))
    else:
        rows.append(empty_row())
        rows.append(row((S_MUTED, "  (no content)")))

    if domains:
        rows.append(empty_row())
        shown = domains[:MAX_INLINE_SOURCES]
        bits = ", ".join(shown)
        remaining = len(domains) - MAX_INLINE_SOURCES
        more = f" (+{remaining} more)" if remaining > 0 else ""
        rows.append(row((S_MUTED, "  Sources: "), (S_PRIMARY, bits + more)))

    return rows
