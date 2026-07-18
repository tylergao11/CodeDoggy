"""Hermes-style multi-source memory selection for turn prefetch."""

from __future__ import annotations

from typing import Any

from codedoggy.memory.select_types import MemorySelectRequest, MemorySelectResult
from codedoggy.memory.session_store import SessionStore


class HermesMemorySelector:
    """Combine curated MEMORY/USER blocks + session FTS hits.

    Used by main-agent prefetch. Provider hits stay empty until plugins fill them.

    Parameters
    ----------
    prefer_frozen:
        True (default) → system_prompt snapshot (stable; refreshed after flush).
        False → live MEMORY/USER entries (includes mid-session tool writes
        that have not refreshed the freeze).
    include_current_session:
        True (default) → FTS may hit prior turns of the same session_id
        (current turn is only in SessionStore after the turn ends).
        False → exclude request.session_id (cross-session only).
    """

    def __init__(
        self,
        *,
        curated_store: Any | None = None,
        session_store: SessionStore | None = None,
        prefer_frozen: bool = True,
        include_current_session: bool = True,
    ) -> None:
        self.curated_store = curated_store
        self.session_store = session_store
        self.prefer_frozen = prefer_frozen
        self.include_current_session = include_current_session

    def bind_curated(self, store: Any | None) -> None:
        self.curated_store = store

    def bind_session_store(self, store: SessionStore | None) -> None:
        self.session_store = store

    def select(self, request: MemorySelectRequest) -> MemorySelectResult:
        curated = self._select_curated(request)
        sessions = self._select_sessions(request)
        return MemorySelectResult(
            curated_blocks=curated,
            session_hits=sessions,
            provider_hits=[],
            raw={
                "source": "hermes",
                "goal": request.goal,
                "path": request.path,
                "prefer_frozen": self.prefer_frozen,
                "include_current_session": self.include_current_session,
                "curated_n": len(curated),
                "session_n": len(sessions),
            },
        )

    def _select_curated(self, request: MemorySelectRequest) -> list[str]:
        if self.curated_store is None:
            return []
        blocks: list[str] = []
        if self.prefer_frozen:
            fn = getattr(self.curated_store, "system_prompt_blocks", None)
            if callable(fn):
                text = fn()
                if text and text.strip():
                    blocks.append(text.strip())
        else:
            live_fn = getattr(self.curated_store, "live_system_prompt_blocks", None)
            if callable(live_fn):
                text = live_fn()
                if text and text.strip():
                    blocks.append(text.strip())
            else:
                for key in ("user", "memory"):
                    fmt = getattr(self.curated_store, "format_live_block", None)
                    if callable(fmt):
                        block = fmt(key)
                        if block and str(block).strip():
                            blocks.append(str(block).strip())
        return _budget_blocks(blocks, request.max_curated_chars)

    def _select_sessions(self, request: MemorySelectRequest) -> list[str]:
        if self.session_store is None:
            return []
        query = (
            (request.query_hint or "").strip()
            or (request.goal or "").strip()
            or (request.path or "").strip()
        )
        if not query or query == "(prefetch)":
            # Still allow query_hint / goal only
            query = (
                (request.query_hint or "").strip()
                or (request.goal or "").strip()
            )
        if not query:
            return []
        extra = request.extra or {}
        roles = extra.get("roles")
        if roles is None:
            roles = ["user", "assistant"]
        elif not roles:
            roles = None
        else:
            roles = [str(r) for r in roles]

        cwd = extra.get("cwd")
        cwd_s = str(cwd) if cwd else None

        session_filter = extra.get("session_id")
        if isinstance(session_filter, str) and session_filter.strip():
            session_filter = session_filter.strip()
        else:
            session_filter = None

        exclude: str | None = None
        if not self.include_current_session and request.session_id:
            exclude = request.session_id
            if session_filter == exclude:
                session_filter = None

        try:
            hits = self.session_store.search(
                query,
                limit=request.max_session_hits,
                exclude_session_id=exclude,
                session_id=session_filter,
                cwd=cwd_s,
                roles=roles,
            )
        except Exception:  # noqa: BLE001
            return []
        lines: list[str] = []
        for h in hits:
            title = h.title or h.goal or h.session_id[:8]
            snippet = (h.snippet or h.content or "").replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:277] + "…"
            lines.append(
                f"- [{h.role}] session={h.session_id[:8]}… title={title!r} "
                f"msg#{h.message_id}: {snippet}"
            )
        return lines


def _budget_blocks(blocks: list[str], budget: int) -> list[str]:
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
    return out
