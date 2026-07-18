"""In-memory cache for web_fetch text results.

Ported from:
  grok-build/.../implementations/grok_build/web_fetch/cache.rs
    FetchCache::new, get, insert_text
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class _CachedPage:
    output: Any
    inserted: float  # time.monotonic()


class FetchCache:
    """TTL cache; only non-truncated inline text is stored."""

    def __init__(self, ttl_secs: float, max_entries: int) -> None:
        self._entries: dict[str, _CachedPage] = {}
        self._ttl = float(ttl_secs)
        self._max_entries = max_entries

    def get(self, url: str) -> Any | None:
        entry = self._entries.get(url)
        if entry is None:
            return None
        if time.monotonic() - entry.inserted < self._ttl:
            return entry.output
        return None

    def insert_text(self, url: str, output: Any, was_truncated: bool) -> None:
        if was_truncated:
            return
        if len(self._entries) >= self._max_entries and url not in self._entries:
            # Evict oldest (max elapsed).
            oldest_key = max(
                self._entries.items(),
                key=lambda kv: time.monotonic() - kv[1].inserted,
            )[0]
            del self._entries[oldest_key]
        self._entries[url] = _CachedPage(output=output, inserted=time.monotonic())
