"""Grok ↔ Hermes memory seam — single owner of memory lifecycle against Grok runtime.

**Ownership (CodeDoggy architecture law):**
  - Grok owns: turn loop, sample messages, compaction, transcript pairs, tool exec
  - Hermes owns (this module + MemoryManager): curated freeze, FTS archive recall,
    external provider slot, prefetch fence, session-bound provider state

**Hermes sources (do not invent):**
  - agent/memory_manager.py — prefetch/sync/boundary/shutdown
  - agent/conversation_loop.py — inject fence into current user at API time only
  - agent/conversation_compression.py — on_pre_compress before discard
  - tools/memory_tool.py — MEMORY.md/USER.md freeze + drift

Call sites must go through these helpers so runner/kernel/compactor stay thin.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def bind_session(
    memory_manager: Any | None,
    *,
    session_id: str,
    cwd: str = "",
    platform: str = "cli",
    agent_context: str = "primary",
    **kwargs: Any,
) -> None:
    """Hermes initialize_all at session start / resume."""
    if memory_manager is None:
        return
    try:
        memory_manager.initialize_all(
            session_id=session_id,
            hermes_home=kwargs.pop("hermes_home", ""),
            platform=platform,
            agent_context=agent_context,
            cwd=cwd,
            **kwargs,
        )
    except Exception:  # noqa: BLE001
        logger.warning("hermes bind_session/initialize_all failed", exc_info=True)


def build_system_memory_block(memory_manager: Any | None, curated: Any | None = None) -> str:
    """Static system memory: manager providers, else curated store alone."""
    if memory_manager is not None:
        try:
            block = memory_manager.build_system_prompt()
            if block and str(block).strip():
                return str(block).strip()
        except Exception:  # noqa: BLE001
            logger.warning("build_system_prompt failed", exc_info=True)
    if curated is not None:
        fn = getattr(curated, "system_prompt_blocks", None)
        if callable(fn):
            try:
                return (fn() or "").strip()
            except Exception:  # noqa: BLE001
                logger.warning("curated system_prompt_blocks failed", exc_info=True)
    return ""


def prefetch_fenced(
    memory_manager: Any | None,
    *,
    user_text: str,
    session_id: str = "",
    cwd: str = "",
    selector: Any | None = None,
    session: Any | None = None,
) -> str | None:
    """Prefetch + Hermes ``build_memory_context_block`` (or None).

    Prefer MemoryManager.prefetch_all; fall back to selector path.
    """
    from codedoggy.memory.prefetch import fence_prefetch_raw, prefetch_for_turn

    if memory_manager is not None:
        try:
            raw = memory_manager.prefetch_all(
                user_text or "", session_id=session_id, cwd=cwd
            )
            fenced = fence_prefetch_raw(raw)
            if fenced:
                return fenced
        except Exception:  # noqa: BLE001
            logger.warning("prefetch_all failed", exc_info=True)
    if selector is not None:
        try:
            return prefetch_for_turn(
                selector=selector,
                session=session,
                session_id=session_id,
                user_text=user_text or "",
            )
        except Exception:  # noqa: BLE001
            logger.warning("prefetch_for_turn failed", exc_info=True)
    return None


def on_turn_begin(
    memory_manager: Any | None,
    curated: Any | None,
    *,
    turn_number: int,
    user_text: str,
) -> None:
    """Hermes on_turn_start + reset curated consolidation breaker."""
    if curated is not None:
        reset = getattr(curated, "reset_consolidation_failures", None)
        if callable(reset):
            try:
                reset()
            except Exception:  # noqa: BLE001
                logger.debug("reset_consolidation_failures failed", exc_info=True)
    if memory_manager is None:
        return
    try:
        memory_manager.on_turn_start(turn_number, user_text or "")
    except Exception:  # noqa: BLE001
        logger.debug("on_turn_start failed", exc_info=True)


def on_turn_end(
    memory_manager: Any | None,
    *,
    user_text: str,
    assistant_text: str,
    session_id: str = "",
    cwd: str = "",
    messages: list[Any] | None = None,
) -> None:
    """Hermes post-turn: sync_all then queue_prefetch_all (background)."""
    if memory_manager is None:
        return
    try:
        memory_manager.sync_all(
            user_text or "",
            assistant_text or "",
            session_id=session_id,
            cwd=cwd,
            messages=messages,
        )
    except Exception:  # noqa: BLE001
        logger.warning("sync_all failed", exc_info=True)
    try:
        # Warm next turn (Hermes run_agent post-turn order)
        memory_manager.queue_prefetch_all(
            user_text or "", session_id=session_id, cwd=cwd
        )
    except Exception:  # noqa: BLE001
        logger.debug("queue_prefetch_all failed", exc_info=True)


def on_pre_compress(
    memory_manager: Any | None,
    messages: list[Any] | None = None,
) -> str:
    """Hermes on_pre_compress before fold discards middle — may feed summarizer."""
    if memory_manager is None:
        return ""
    try:
        pre = getattr(memory_manager, "on_pre_compress", None)
        if callable(pre):
            return pre(list(messages or [])) or ""
    except Exception:  # noqa: BLE001
        logger.debug("on_pre_compress failed", exc_info=True)
    return ""


def on_transcript_rewound(
    memory_manager: Any | None,
    *,
    session_id: str,
) -> None:
    """Hermes on_session_switch(rewound=True) — same id, truncated transcript."""
    if memory_manager is None or not session_id:
        return
    try:
        memory_manager.on_session_switch(
            session_id, parent_session_id=session_id, reset=False, rewound=True
        )
    except Exception:  # noqa: BLE001
        logger.debug("on_transcript_rewound failed", exc_info=True)


def commit_session_boundary(
    memory_manager: Any | None,
    messages: list[Any] | None,
    *,
    new_session_id: str,
    parent_session_id: str = "",
    reason: str = "new_session",
) -> None:
    """Hermes new_session boundary: end → switch (async when manager supports it)."""
    if memory_manager is None or not new_session_id:
        return
    try:
        commit = getattr(memory_manager, "commit_session_boundary_async", None)
        if callable(commit):
            commit(
                list(messages or []),
                new_session_id=new_session_id,
                parent_session_id=parent_session_id,
                reason=reason,
            )
            return
        switch = getattr(memory_manager, "on_session_switch", None)
        if callable(switch):
            # Fallback: end then switch (sync) when async boundary missing
            end = getattr(memory_manager, "on_session_end", None)
            if callable(end):
                try:
                    end(list(messages or []))
                except Exception:  # noqa: BLE001
                    logger.warning("on_session_end (boundary fallback) failed", exc_info=True)
            switch(
                new_session_id,
                parent_session_id=parent_session_id,
                reset=True,
                reason=reason,
            )
    except Exception:  # noqa: BLE001
        logger.warning("commit_session_boundary failed", exc_info=True)


def on_session_close(
    memory_manager: Any | None,
    *,
    messages: list[Any] | None = None,
    timeout_s: float = 5.0,
) -> None:
    """Hermes session end: extract hooks, drain, shutdown providers."""
    if memory_manager is None:
        return
    try:
        memory_manager.on_session_end(list(messages or []))
    except Exception:  # noqa: BLE001
        logger.warning("on_session_end failed", exc_info=True)
    try:
        flush = getattr(memory_manager, "flush_pending", None)
        if callable(flush):
            flush(timeout=timeout_s)
    except Exception:  # noqa: BLE001
        logger.debug("flush_pending failed", exc_info=True)
    try:
        shut = getattr(memory_manager, "shutdown_all", None) or getattr(
            memory_manager, "shutdown", None
        )
        if callable(shut):
            try:
                shut(timeout_s=timeout_s)
            except TypeError:
                shut()
    except Exception:  # noqa: BLE001
        logger.warning("memory shutdown failed", exc_info=True)


def notify_curated_write(memory_manager: Any | None, target: str = "memory") -> None:
    """After MEMORY.md/USER.md mutation or flush — refresh freeze spine."""
    if memory_manager is None:
        return
    try:
        memory_manager.notify_memory_write(target)
    except Exception:  # noqa: BLE001
        logger.debug("notify_memory_write failed", exc_info=True)


def sample_messages_with_memory(
    messages: list[Any],
    prefetch_block: str | None,
) -> list[Any]:
    """Sample-time only: inject fenced prefetch into current user (never archive)."""
    from codedoggy.memory.context_fence import messages_with_ephemeral_memory

    return messages_with_ephemeral_memory(messages, prefetch_block)
