"""Hermes-aligned pre-turn recall (agent/memory_manager + conversation_loop).

Source:
  - hermes-agent/agent/memory_manager.py — build_memory_context_block / sanitize
  - hermes-agent/agent/conversation_loop.py — inject into *current user* at
    API-call time only; original messages list never mutated; not SYSTEM.

CodeDoggy maps that to:
  - raw prefetch text from MemoryManager.prefetch_all / HermesMemorySelector
  - wrap with build_memory_context_block
  - pass as ephemeral sample overlay (loop applies messages_with_ephemeral_memory)
  - archive / live transcript keep the clean user text only
"""

from __future__ import annotations

import logging
from typing import Any

from codedoggy.memory.context_fence import build_memory_context_block

logger = logging.getLogger(__name__)


def prefetch_for_turn(
    *,
    selector: Any,
    session: Any,
    session_id: str | None,
    user_text: str,
    max_session_hits: int = 6,
) -> str | None:
    """Return Hermes-fenced recall block for ephemeral user injection, or None."""
    if selector is None:
        return None
    try:
        from codedoggy.audit.types import MemorySelectRequest, MutationEvent

        goal = getattr(session, "goal", None) if session is not None else None
        cwd = None
        if session is not None:
            c = getattr(session, "cwd", None)
            cwd = str(c) if c is not None else None
        req = MemorySelectRequest(
            goal=goal if isinstance(goal, str) else None,
            mutation=MutationEvent(
                path="(prefetch)",
                tool_name="prefetch",
                call_id="prefetch",
                after="",
            ),
            trajectory_summary="(main-agent turn prefetch)",
            session_id=session_id,
            query_hint=(user_text or "")[:240],
            max_session_hits=max_session_hits,
            max_curated_chars=0,
            extra={"cwd": cwd, "roles": ["user", "assistant"]},
        )
        result = selector.select(req)
        hits = list(getattr(result, "session_hits", None) or [])
        if not hits:
            return None
        body = "\n".join(hits[:max_session_hits])
        return build_memory_context_block(body) or None
    except Exception:  # noqa: BLE001
        logger.warning("memory prefetch_for_turn failed", exc_info=True)
        return None


def fence_prefetch_raw(raw: str | None) -> str | None:
    """Wrap MemoryManager.prefetch_all raw merge with Hermes fence."""
    if not raw or not str(raw).strip():
        return None
    return build_memory_context_block(str(raw)) or None
