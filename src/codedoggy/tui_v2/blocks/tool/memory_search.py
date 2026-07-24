"""MemorySearchToolCallBlock — port of ``blocks/tool/memory_search.rs``.

Header: ``Memory Search `` + query + `` (N results)``.
Expanded: numbered path:range + score/source + snippet preview.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from codedoggy.tui_v2.blocks.tool.common import (
    S_BOLD,
    S_COMMAND,
    S_DIM,
    S_ERROR,
    S_MUTED,
    S_PANEL,
    S_PRIMARY,
    Rows,
    arg_str,
    empty_row,
    is_running,
    row,
    truncate_str,
)

HEADER = "Memory Search "
# Fixed present verb on tool blocks; tense variants live in verb_group only.
HEADER_RUNNING = "Memory Search "

# Grok memory_search body (tools/builtins/memory_search.py):
#   ### Result N (score: 0.90, source: memory)
#   **File:** path (lines 10-25)
#   ```
#   snippet
#   ```
# Also tolerate: (score: 0.9, source) and separate **Lines:** 10-25.
_RESULT_HEADER_RE = re.compile(
    r"^###\s+Result\s+\d+\s*\(\s*score:\s*([0-9]*\.?[0-9]+)\s*,\s*"
    r"(?:source:\s*)?([^)]+?)\s*\)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_FILE_RE = re.compile(
    r"^\*\*File:\*\*\s*(.+?)(?:\s*\(\s*lines?\s+(\d+)\s*[-–]\s*(\d+)\s*\))?\s*$",
    re.IGNORECASE,
)
_LINES_RE = re.compile(
    r"^\*\*Lines?:\*\*\s*(\d+)\s*[-–]\s*(\d+)\s*$",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"```(?:[^\n`]*)\n(.*?)```", re.DOTALL)


@dataclass
class MemoryResult:
    score: float
    source: str
    path: str
    start_line: int
    end_line: int
    snippet: str


def _parse_json_results(text: str) -> list[MemoryResult] | None:
    """Parse JSON list/object when text looks like JSON. None if not JSON data."""
    if not (text.startswith("[") or text.startswith("{")):
        return None
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(data, dict):
        data = data.get("results") or data.get("items") or data.get("matches") or []
    if not isinstance(data, list):
        return None
    out: list[MemoryResult] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        path = str(
            item.get("path")
            or item.get("file")
            or item.get("source_path")
            or "?"
        )
        snippet = str(
            item.get("snippet")
            or item.get("text")
            or item.get("content")
            or ""
        )
        try:
            score = float(item.get("score") or item.get("relevance") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        source = str(item.get("source") or item.get("scope") or "memory")
        try:
            start = int(item.get("start_line") or item.get("start") or 0)
        except (TypeError, ValueError):
            start = 0
        try:
            end = int(item.get("end_line") or item.get("end") or start)
        except (TypeError, ValueError):
            end = start
        out.append(
            MemoryResult(
                score=score,
                source=source,
                path=path,
                start_line=start,
                end_line=end,
                snippet=snippet,
            )
        )
    return out


def _parse_grok_markdown(text: str) -> list[MemoryResult] | None:
    """Parse Grok-style ``### Result N (score, source)`` markdown. None if none."""
    headers = list(_RESULT_HEADER_RE.finditer(text))
    if not headers:
        return None

    out: list[MemoryResult] = []
    for i, m in enumerate(headers):
        try:
            score = float(m.group(1))
        except (TypeError, ValueError):
            score = 0.0
        source = (m.group(2) or "").strip() or "memory"

        body_start = m.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[body_start:body_end]

        path = "?"
        start_line = 0
        end_line = 0
        snippet = ""

        fence = _FENCE_RE.search(body)
        if fence:
            snippet = fence.group(1).strip()
            meta = body[: fence.start()]
        else:
            meta = body

        for ln in meta.splitlines():
            ln_s = ln.strip()
            if not ln_s:
                continue
            fm = _FILE_RE.match(ln_s)
            if fm:
                path = (fm.group(1) or "?").strip() or "?"
                if fm.group(2) is not None:
                    start_line = int(fm.group(2))
                    end_line = int(fm.group(3))
                continue
            lm = _LINES_RE.match(ln_s)
            if lm:
                start_line = int(lm.group(1))
                end_line = int(lm.group(2))

        # No fence: take remaining non-meta body lines as a light snippet
        if not snippet and not fence:
            leftover = [
                ln.strip()
                for ln in meta.splitlines()
                if ln.strip()
                and not _FILE_RE.match(ln.strip())
                and not _LINES_RE.match(ln.strip())
                and not ln.strip().startswith("#")
            ]
            if leftover:
                snippet = "\n".join(leftover[:8])

        out.append(
            MemoryResult(
                score=score,
                source=source,
                path=path,
                start_line=start_line,
                end_line=end_line,
                snippet=snippet,
            )
        )
    return out


def parse_memory_results(result: str) -> list[MemoryResult]:
    """Best-effort parse of memory search tool output.

    Order: JSON list/object → Grok markdown (``### Result N``) → line fallback.
    """
    text = (result or "").strip()
    if not text:
        return []

    json_out = _parse_json_results(text)
    if json_out is not None:
        return json_out

    md_out = _parse_grok_markdown(text)
    if md_out is not None:
        return md_out

    # Fallback: one result per non-empty line as snippet
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return [
        MemoryResult(
            score=0.0,
            source="memory",
            path="?",
            start_line=0,
            end_line=0,
            snippet=ln[:200],
        )
        for ln in lines[:20]
    ]


def _shorten_path(path: str, budget: int = 40) -> str:
    if len(path) <= budget:
        return path
    return "…" + path[-(budget - 1) :]


def paint_memory_search(
    arguments: dict,
    result: str,
    *,
    width: int,
    collapsed: bool,
    status: str,
    selected: bool = False,
) -> Rows:
    query = arg_str(arguments, "query", "q", "search", default="") or "…"
    running = is_running(status)
    muted = collapsed and not running
    results = [] if running else parse_memory_results(result)
    failed = status.lower() in {"failed", "error"}

    prefix = HEADER  # fixed present; HEADER_RUNNING kept as alias
    p_style = S_MUTED + " bold" if muted else S_BOLD
    q_style = S_MUTED if muted else S_COMMAND
    if selected:
        p_style = f"{p_style} reverse"
        q_style = f"{q_style} reverse"

    count = len(results)
    if count > 0:
        s = "" if count == 1 else "s"
        suffix = f" ({count} result{s})"
    else:
        suffix = ""

    budget = max(1, width - len(prefix) - len(suffix))
    shown_q = truncate_str(query, budget)
    parts: list[tuple[str, str]] = [(p_style, prefix), (q_style, shown_q)]
    if suffix:
        parts.append((S_DIM if not muted else S_MUTED, suffix))
    rows: Rows = [row(*parts)]

    if collapsed:
        return rows

    if failed and result:
        rows.append(empty_row())
        for err_line in result.splitlines()[:8]:
            rows.append(row((S_ERROR, err_line)))
        return rows

    if not results:
        rows.append(empty_row())
        rows.append(row((S_MUTED, "  (no results)")))
        return rows

    for i, r in enumerate(results[:15]):
        rows.append(empty_row())
        path_disp = _shorten_path(r.path, max(12, width - 24))
        range_s = (
            f"{path_disp}:{r.start_line}-{r.end_line}"
            if r.start_line or r.end_line
            else path_disp
        )
        meta = f"  (score: {r.score:.2f}, {r.source})"
        rows.append(
            row(
                (S_MUTED, f"  {i + 1}. "),
                (S_BOLD, range_s),
                (S_DIM, meta),
            )
        )
        snips = [ln.strip() for ln in r.snippet.splitlines() if ln.strip()][:3]
        for sl in snips:
            display = truncate_str(sl, max(8, width - 4))
            rows.append(row((f"{S_PANEL} {S_PRIMARY}", f"  {display}")))

    return rows


__all__ = ["MemoryResult", "paint_memory_search", "parse_memory_results"]
