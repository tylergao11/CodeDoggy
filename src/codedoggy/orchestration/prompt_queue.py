"""Prompt queue + interjection buffer.

Source:
  - ``xai-interjection-core`` buffer + format
  - shell ``pending_interjections`` / ``drain_pending_interjections`` (safe points)

Deleted inventions: mid-stream interrupt flags, urgent stream coupling.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from codedoggy.orchestration.interjection import drain_formatted, format_interjection
from codedoggy.orchestration.types import Interjection


@dataclass
class PromptQueueItem:
    text: str
    prompt_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    urgent: bool = False


class PromptQueue:
    """FIFO queue for user prompts waiting on a busy session."""

    def __init__(self) -> None:
        self._q: deque[PromptQueueItem] = deque()
        self._lock = threading.Lock()

    def push(self, item: PromptQueueItem) -> None:
        with self._lock:
            if item.urgent:
                self._q.appendleft(item)
            else:
                self._q.append(item)

    def pop(self) -> PromptQueueItem | None:
        with self._lock:
            if not self._q:
                return None
            return self._q.popleft()

    def peek(self) -> PromptQueueItem | None:
        with self._lock:
            return self._q[0] if self._q else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._q)

    def clear(self) -> None:
        with self._lock:
            self._q.clear()


class InterjectionBuffer:
    """Grok ``pending_interjections`` — FIFO raw entries; frame on drain.

    Drain at safe points only (loop head / post-tool), via ``drain_formatted``.
    """

    def __init__(self) -> None:
        self._items: deque[Interjection] = deque()
        self._lock = threading.Lock()

    def push(self, text: str, *, prompt_id: str | None = None) -> None:
        with self._lock:
            self._items.append(Interjection(text=text, prompt_id=prompt_id))

    def is_empty(self) -> bool:
        with self._lock:
            return not self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def drain(self) -> list[Interjection]:
        """Raw drain (Grok EventQueue drain_all). Prefer ``drain_formatted``."""
        with self._lock:
            out = list(self._items)
            self._items.clear()
            return out

    def drain_formatted(
        self,
        sanitize_text: Callable[[str], str] | None = None,
    ) -> list[str]:
        """Source: ``buffer.rs::drain_formatted``."""
        raw = self.drain()
        return drain_formatted(raw, sanitize_text=sanitize_text)

    def peek_text(self) -> str | None:
        with self._lock:
            if not self._items:
                return None
            return self._items[0].text


__all__ = [
    "InterjectionBuffer",
    "PromptQueue",
    "PromptQueueItem",
    "format_interjection",
]
