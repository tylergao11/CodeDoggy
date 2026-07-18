"""Host MemoryBackend adapter for ``memory_search`` over curated MemoryStore files.

Purpose
-------
Provide a host-injected backend for ``tools/builtins/memory_search.py`` so
product sessions can search ``MEMORY.md`` / ``USER.md`` without shipping a full
Grok MemoryBackend stack (embeddings, BM25 index, cross-session vector recall).

NOT a full Grok MemoryBackend port.
Fidelity: **C/A** — result fields match the memory_search contract
(``score``, ``source``, ``path``, ``start_line``, ``end_line``, ``snippet``);
ranking is honest simple multi-token substring overlap — **not** BM25, **not**
embeddings / vector similarity. Documented as Contract + Approximate.

Wire (main agent / bootstrap owns injection; this module only builds):

  from codedoggy.host.memory_backend import build_memory_backend
  extra['memory_backend'] = build_memory_backend(memory_store)
  # or RuntimeKernel.wire_host_adapters() → wire_default_host_extras()
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from codedoggy.memory.defaults import ENTRY_DELIMITER

# Defaults when memory_search omits max_results / min_score
DEFAULT_MAX_RESULTS: int = 8
DEFAULT_MIN_SCORE: float = 0.15

# Cap snippet length so tool output stays readable
_SNIPPET_MAX: int = 480

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")


@dataclass(frozen=True)
class MemorySearchHit:
    """One search hit in the shape memory_search.py expects."""

    score: float
    source: str
    path: str
    start_line: int
    end_line: int
    snippet: str
    staleness_note: str = ""

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "score": self.score,
            "source": self.source,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "snippet": self.snippet,
        }
        if self.staleness_note:
            d["staleness_note"] = self.staleness_note
        return d


class SimpleMemoryStoreBackend:
    """Honest simple search over a bound ``MemoryStore`` (curated files only).

    Scans live ``memory_entries`` / ``user_entries`` (post mid-session writes).
    Score = fraction of query tokens found as case-insensitive substrings in
    the entry, with a full-phrase bonus. No IDF, no stemming, no vectors.
    """

    def __init__(self, memory_store: Any) -> None:
        self._store = memory_store

    def search(
        self,
        query: str,
        max_results: int | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []

        limit = DEFAULT_MAX_RESULTS if max_results is None else max(0, int(max_results))
        floor = DEFAULT_MIN_SCORE if min_score is None else float(min_score)
        if limit == 0:
            return []

        hits: list[MemorySearchHit] = []
        for target, source, filename in (
            ("memory", "memory", "MEMORY.md"),
            ("user", "user", "USER.md"),
        ):
            entries = _entries_for(self._store, target)
            if not entries:
                continue
            path = str(_path_for(self._store, target, filename))
            for idx, entry in enumerate(entries):
                score = _term_overlap_score(q, entry)
                if score < floor:
                    continue
                start, end = _entry_line_span(entries, idx)
                hits.append(
                    MemorySearchHit(
                        score=score,
                        source=source,
                        path=path,
                        start_line=start,
                        end_line=end,
                        snippet=_clip_snippet(entry),
                    )
                )

        hits.sort(key=lambda h: (-h.score, h.source, h.start_line))
        return [h.as_dict() for h in hits[:limit]]


def build_memory_backend(memory_store: Any) -> SimpleMemoryStoreBackend | None:
    """Factory: wrap a MemoryStore as a memory_search backend, or None.

    Returns ``None`` when ``memory_store`` is missing so host wiring can skip
    injection and leave memory_search on Grok soft-disabled text.
    """
    if memory_store is None:
        return None
    return SimpleMemoryStoreBackend(memory_store)


# ── scoring / layout helpers ─────────────────────────────────────────────


def _term_overlap_score(query: str, text: str) -> float:
    """Fraction of query tokens found in text (0..1). Honest, not BM25.

    Full-query substring match → 1.0. Empty token set after filter → 0.0
    unless the whole stripped query is a substring.
    """
    hay = text.lower()
    q_low = query.strip().lower()
    if not q_low:
        return 0.0
    if q_low in hay:
        return 1.0

    tokens = [t.lower() for t in _TOKEN_RE.findall(query) if len(t) > 1]
    if not tokens:
        return 0.0
    # unique order-preserving
    seen: set[str] = set()
    uniq: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    hits = sum(1 for t in uniq if t in hay)
    if hits == 0:
        return 0.0
    return hits / float(len(uniq))


def _entry_line_span(entries: list[str], index: int) -> tuple[int, int]:
    """1-based line span of entry ``index`` in §-joined file layout."""
    if index <= 0:
        start = 1
    else:
        head = ENTRY_DELIMITER.join(entries[:index]) + ENTRY_DELIMITER
        start = head.count("\n") + 1
    body = entries[index]
    nlines = body.count("\n") + 1 if body else 1
    return start, start + nlines - 1


def _clip_snippet(text: str, limit: int = _SNIPPET_MAX) -> str:
    t = text.strip()
    if len(t) <= limit:
        return t
    return t[: max(0, limit - 1)] + "…"


def _entries_for(store: Any, target: str) -> list[str]:
    if target == "user":
        raw = getattr(store, "user_entries", None)
    else:
        raw = getattr(store, "memory_entries", None)
    if not isinstance(raw, list):
        return []
    return [str(e) for e in raw if e]


def _path_for(store: Any, target: str, filename: str) -> Any:
    path_for = getattr(store, "_path_for", None)
    if callable(path_for):
        try:
            return path_for(target)
        except Exception:  # noqa: BLE001
            pass
    mem_dir = getattr(store, "memory_dir", None)
    if mem_dir is not None:
        from pathlib import Path

        return Path(mem_dir) / filename
    return filename
