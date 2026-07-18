"""Memory selection for resident audit — interface for Hermes integration."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from codedoggy.audit.types import MemorySelectRequest, MemorySelectResult


@runtime_checkable
class MemorySelector(Protocol):
    """Pick memory slices relevant to one mutation review.

    Next step (Hermes): implementors combine curated MEMORY/USER, session_search
    FTS hits, and optional external providers. Keep this interface stable.
    """

    def select(self, request: MemorySelectRequest) -> MemorySelectResult:
        ...


class NoopMemorySelector:
    """No memory injected into the auditor (tests / bare loop)."""

    def select(self, request: MemorySelectRequest) -> MemorySelectResult:
        return MemorySelectResult()


class CuratedMemorySelector:
    """Select from a bound MemoryStore (frozen snapshot or live entries).

    ``prefer_frozen=True`` (default): system-prompt snapshot.
    ``prefer_frozen=False``: live MEMORY/USER entries (mid-session writes).
    """

    def __init__(self, store: Any | None = None, *, prefer_frozen: bool = True) -> None:
        self.store = store
        self.prefer_frozen = prefer_frozen

    def bind_store(self, store: Any | None) -> None:
        self.store = store

    def select(self, request: MemorySelectRequest) -> MemorySelectResult:
        if self.store is None:
            return MemorySelectResult()

        blocks: list[str] = []
        if self.prefer_frozen:
            fn = getattr(self.store, "system_prompt_blocks", None)
            if callable(fn):
                text = fn()
                if text and text.strip():
                    blocks.append(text.strip())
        else:
            live_fn = getattr(self.store, "live_system_prompt_blocks", None)
            if callable(live_fn):
                text = live_fn()
                if text and text.strip():
                    blocks.append(text.strip())
            else:
                # Last resort: re-read frozen if live API missing.
                fn = getattr(self.store, "system_prompt_blocks", None)
                if callable(fn):
                    text = fn()
                    if text and text.strip():
                        blocks.append(text.strip())

        # Budget
        budget = request.max_curated_chars
        out: list[str] = []
        used = 0
        for b in blocks:
            if used >= budget:
                break
            room = budget - used
            if len(b) > room:
                out.append(b[: room - 20] + "\n… (truncated)")
                used = budget
            else:
                out.append(b)
                used += len(b)

        return MemorySelectResult(
            curated_blocks=out,
            raw={
                "source": "curated",
                "prefer_frozen": self.prefer_frozen,
                "goal": request.goal,
                "path": request.mutation.path,
            },
        )
