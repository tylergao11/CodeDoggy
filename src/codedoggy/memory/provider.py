"""Memory provider protocol — hermes-agent/agent/memory_provider.py surface.

Source lifecycle: initialize / system_prompt_block / prefetch / sync_turn /
get_tool_schemas / handle_tool_call / shutdown.

CodeDoggy builtins: curated MEMORY.md (system freeze) + SessionStore FTS
(prefetch only). One external provider max (MemoryManager).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MemoryProvider(Protocol):
    """Pluggable memory backend."""

    @property
    def name(self) -> str:
        ...

    def system_prompt_block(self) -> str:
        """Text injected into system prompt (may be empty)."""
        ...

    def prefetch(
        self, query: str, *, session_id: str = "", cwd: str = ""
    ) -> str:
        """On-demand recall for this user turn (may be empty)."""
        ...

    def queue_prefetch(
        self, query: str, *, session_id: str = "", cwd: str = ""
    ) -> None:
        """Optional: warm next-turn prefetch (default no-op)."""
        ...

    def sync_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        session_id: str = "",
        cwd: str = "",
    ) -> None:
        """Optional: post-turn persistence hook."""
        ...

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Optional extra tools from this provider."""
        ...


class BaseMemoryProvider:
    """Convenience base — hermes-agent MemoryProvider optional hooks as no-ops."""

    name: str = "base"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str = "", **kwargs: Any) -> None:
        return None

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(
        self, query: str, *, session_id: str = "", cwd: str = ""
    ) -> str:
        return ""

    def queue_prefetch(
        self, query: str, *, session_id: str = "", cwd: str = ""
    ) -> None:
        return None

    def sync_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        session_id: str = "",
        cwd: str = "",
        messages: list[Any] | None = None,
    ) -> None:
        return None

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return []

    def handle_tool_call(
        self, tool_name: str, args: dict[str, Any], **kwargs: Any
    ) -> str:
        raise NotImplementedError(
            f"Provider {self.name} does not handle tool {tool_name}"
        )

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        return None

    def on_session_end(self, messages: list[Any] | None = None) -> None:
        return None

    def on_pre_compress(self, messages: list[Any] | None = None) -> str:
        """Before context compression — optional extract (Hermes on_pre_compress)."""
        return ""

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        """Hermes: session_id rotated without provider teardown.

        ``rewound=True``: same id, truncated transcript — invalidate turn caches.
        """
        self._session_id = new_session_id  # type: ignore[attr-defined]
        return None

    def on_memory_write(
        self, action: str, target: str, content: str, metadata: Any = None
    ) -> None:
        return None

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Hermes: parent observes subagent task+result (subagent has no provider)."""
        return None

    def shutdown(self) -> None:
        return None


class CuratedMemoryProvider(BaseMemoryProvider):
    """Builtin: frozen MEMORY.md / USER.md snapshot."""

    name = "builtin_curated"

    def __init__(self, store: Any | None = None) -> None:
        self.store = store

    def system_prompt_block(self) -> str:
        if self.store is None:
            return ""
        fn = getattr(self.store, "system_prompt_blocks", None)
        if callable(fn):
            return (fn() or "").strip()
        return ""


class SessionFtsProvider(BaseMemoryProvider):
    """Builtin: SessionStore FTS hits for prefetch + warm cache for next turn."""

    name = "builtin_session_fts"
    # Default: conversational roles only — never surface raw tool dumps in prefetch.
    DEFAULT_ROLES = ("user", "assistant")

    def __init__(self, store: Any | None = None, *, max_hits: int = 6) -> None:
        self.store = store
        self.max_hits = max_hits
        self._warm: str = ""
        self._warm_query: str = ""
        self._warm_cwd: str = ""
        self._session_id: str = ""

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        self._session_id = new_session_id
        # Hermes rewound: invalidate per-turn warm cache for truncated transcript
        if reset or rewound or kwargs.get("rewound"):
            self._warm = ""
            self._warm_query = ""
            self._warm_cwd = ""

    def prefetch(
        self,
        query: str,
        *,
        session_id: str = "",
        cwd: str = "",
    ) -> str:
        q = (query or "").strip()
        # Prefer warm cache when query shares tokens with last warm key
        if self._warm and q and self._warm_query:
            if _query_overlap(q, self._warm_query) >= 0.3:
                if not cwd or cwd == self._warm_cwd:
                    return self._warm
        return self._search(q, session_id=session_id, cwd=cwd)

    def queue_prefetch(
        self,
        query: str,
        *,
        session_id: str = "",
        cwd: str = "",
    ) -> None:
        """Warm FTS result for the next turn (Hermes queue_prefetch spirit)."""
        q = (query or "").strip()
        self._warm_query = q
        self._warm_cwd = cwd or ""
        self._warm = self._search(q, session_id=session_id, cwd=cwd)

    def sync_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        session_id: str = "",
        cwd: str = "",
    ) -> None:
        # Archive is create-time via runner; warm next prefetch from this turn.
        # Blend user + short assistant keywords for better next-turn recall.
        blend = (user_text or "").strip()
        if assistant_text:
            # First ~120 chars of assistant often name files/decisions
            blend = f"{blend} {(assistant_text or '')[:120]}".strip()
        if blend:
            self.queue_prefetch(blend, session_id=session_id, cwd=cwd)

    def _search(
        self,
        query: str,
        *,
        session_id: str = "",
        cwd: str = "",
    ) -> str:
        if self.store is None or not (query or "").strip():
            return ""
        try:
            hits = self.store.search(
                query.strip()[:240],
                limit=self.max_hits,
                # Include prior turns of the current session (no exclude).
                # Cross-session hits remain available unless cwd scopes them.
                exclude_session_id=None,
                session_id=None,
                cwd=cwd or None,
                roles=list(self.DEFAULT_ROLES),
                # Failed/cancelled/aborted partial transcripts remain visible
                # to explicit session_search, but are not learned as memory.
                completed_only=True,
            )
        except Exception:
            return ""
        if not hits:
            return ""
        lines = [
            "### Session FTS recall",
            "Prior turns matching this prompt (reference only):",
        ]
        for h in hits[: self.max_hits]:
            title = getattr(h, "title", None) or getattr(h, "goal", None) or h.session_id[:8]
            snip = (h.snippet or h.content or "").replace("\n", " ")
            if len(snip) > 280:
                snip = snip[:277] + "…"
            lines.append(
                f"- [{h.role}] session={h.session_id[:8]}… title={title!r}: {snip}"
            )
        return "\n".join(lines)


def _query_overlap(a: str, b: str) -> float:
    """Jaccard-ish token overlap for warm-cache hit (0..1)."""
    ta = {t.lower() for t in a.split() if len(t) > 2}
    tb = {t.lower() for t in b.split() if len(t) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))
