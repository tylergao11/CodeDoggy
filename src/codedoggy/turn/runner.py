"""TurnRunner that binds sampler + tools into Session.handle_prompt."""

from __future__ import annotations

import logging
from typing import Any

from codedoggy.session.types import TurnRequest, TurnResult, TurnStatus
from codedoggy.tools.registry import FinalizedToolset
from codedoggy.turn.hooks import LoopHooks
from codedoggy.turn.loop import run_agent_loop
from codedoggy.turn.sampler import Sampler
from codedoggy.turn.types import Message

logger = logging.getLogger(__name__)


class AgentTurnRunner:
    """Session-facing adapter over :func:`run_agent_loop`.

    Holds a **live transcript** across ``handle_prompt`` calls so Grok
    compaction and Hermes session lifetime share one continuous window.
    SessionStore receives **archive copies** at message-create time (full
    tool bodies) so FTS is not limited to post-prune live content.
    """

    def __init__(
        self,
        *,
        sampler: Sampler,
        tools: FinalizedToolset,
        hooks: LoopHooks | None = None,
        system_prompt: str | None = None,
        context_compactor: Any | None = None,
        resume_live: bool = True,
    ) -> None:
        self.sampler = sampler
        self.tools = tools
        self.hooks = hooks
        self.system_prompt = system_prompt
        self.context_compactor = context_compactor
        self.resume_live = resume_live
        # Last completed loop messages (incl. system); next turn strips system.
        self.live_messages: list[Message] = []

    def clear_live_history(self) -> None:
        """Drop in-process cross-prompt history (tests / new topic)."""
        self.live_messages = []

    def rewind_context(self, *, as_reference: bool = True) -> dict[str, Any]:
        """Inject last compaction checkpoint into the live window.

        Returns a small status dict (harness API — not a UI product).
        """
        compactor = self.context_compactor
        if compactor is None:
            return {"ok": False, "reason": "no context_compactor"}
        path = getattr(compactor, "last_checkpoint_path", None)
        if not path:
            return {"ok": False, "reason": "no checkpoint"}
        rew = getattr(compactor, "rewind_from_checkpoint", None)
        if not callable(rew):
            return {"ok": False, "reason": "rewind unsupported"}
        before = len(self.live_messages)
        self.live_messages = rew(self.live_messages, as_reference=as_reference)
        return {
            "ok": True,
            "checkpoint": str(path),
            "messages_before": before,
            "messages_after": len(self.live_messages),
            "as_reference": as_reference,
        }

    def run(self, request: TurnRequest, *, session: Any) -> TurnResult:
        tools = self.tools
        if getattr(session, "extensions", None) is not None:
            ext_tools = session.extensions.tools
            if ext_tools is not None:
                tools = ext_tools

        max_turns = getattr(session, "max_turns", None)
        cwd = getattr(session, "cwd")
        session_id = str(getattr(session, "id", "")) or None

        def _cancelled() -> bool:
            check = getattr(session, "is_cancel_requested", None)
            return bool(check()) if callable(check) else False

        system_prompt = self.system_prompt
        ext = getattr(session, "extensions", None)
        mem = getattr(ext, "memory", None) if ext is not None else None
        memory_manager = getattr(ext, "memory_manager", None) if ext is not None else None
        session_store = getattr(ext, "session_store", None) if ext is not None else None

        # Memory pillar: prefer MemoryManager system blocks, else curated store
        if memory_manager is not None:
            try:
                blocks = memory_manager.build_system_prompt()
                if blocks:
                    system_prompt = (
                        f"{system_prompt}\n\n{blocks}" if system_prompt else blocks
                    )
            except Exception:  # noqa: BLE001
                logger.warning("memory_manager.build_system_prompt failed", exc_info=True)
        elif mem is not None:
            blocks_fn = getattr(mem, "system_prompt_blocks", None)
            if callable(blocks_fn):
                blocks = blocks_fn()
                if blocks:
                    system_prompt = (
                        f"{system_prompt}\n\n{blocks}" if system_prompt else blocks
                    )

        # Lazy imports avoid audit ↔ turn package cycles at import time.
        from codedoggy.audit.hooks import resolve_audit_hooks
        from codedoggy.audit.memory_select import CuratedMemorySelector
        from codedoggy.memory.hermes_select import HermesMemorySelector
        from codedoggy.memory.prefetch import inject_prefetch_block, prefetch_for_turn

        audit = getattr(ext, "audit", None) if ext is not None else None
        selector = None
        if audit is not None:
            sel = getattr(audit, "memory_selector", None)
            selector = sel
            if isinstance(sel, CuratedMemorySelector) and sel.store is None and mem is not None:
                sel.bind_store(mem)
            if isinstance(sel, HermesMemorySelector):
                if sel.curated_store is None and mem is not None:
                    sel.bind_curated(mem)
                if sel.session_store is None and session_store is not None:
                    sel.bind_session_store(session_store)
        elif memory_manager is not None:
            selector = memory_manager.as_audit_selector()
        elif mem is not None or session_store is not None:
            selector = HermesMemorySelector(
                curated_store=mem,
                session_store=session_store,
            )

        # Prefetch: MemoryManager.prefetch_all when present, else FTS selector path
        if memory_manager is not None:
            try:
                pre = memory_manager.prefetch_all(
                    request.text or "", session_id=session_id or ""
                )
                if pre:
                    system_prompt = inject_prefetch_block(
                        system_prompt,
                        "## Prefetched memory (MemoryManager)\n"
                        "Reference only — latest user message wins:\n" + pre,
                    )
            except Exception:  # noqa: BLE001
                logger.warning("memory_manager.prefetch_all failed", exc_info=True)
        else:
            system_prompt = inject_prefetch_block(
                system_prompt,
                prefetch_for_turn(
                    selector=selector,
                    session=session,
                    session_id=session_id,
                    user_text=request.text,
                ),
            )

        hooks = resolve_audit_hooks(session, explicit_hooks=self.hooks)

        # Optional: use audit model as cheap summarizer for context fold.
        compactor = self.context_compactor
        if compactor is None:
            from codedoggy.context.compactor import ContextCompactor

            summary_client = None
            if audit is not None:
                auditor = getattr(audit, "auditor", None)
                summary_client = getattr(auditor, "client", None)
            compactor = ContextCompactor.from_env(
                summary_client=summary_client,
                memory_store=mem,
                session_store=session_store,
                memory_manager=memory_manager,
            )

        # Always clear per-turn suppress (bootstrap path shares one compactor).
        on_start = getattr(compactor, "on_turn_start", None)
        if callable(on_start):
            on_start()

        prior = self.live_messages if self.resume_live and self.live_messages else None

        def _archive(msg: Message) -> None:
            if session_store is None or not session_id:
                return
            try:
                role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                if role == "system":
                    return  # system rebuilt each turn; skip FTS noise
                tool_calls = None
                if msg.tool_calls:
                    tool_calls = [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in msg.tool_calls
                    ]
                session_store.append_message(
                    session_id,
                    role,
                    msg.content,
                    tool_name=msg.name,
                    tool_call_id=msg.tool_call_id,
                    tool_calls=tool_calls,
                )
            except Exception:  # noqa: BLE001
                logger.warning("session archive append failed", exc_info=True)

        if session_store is not None and session_id:
            try:
                session_store.ensure_session(
                    session_id,
                    cwd=str(cwd),
                    goal=getattr(session, "goal", None),
                )
            except Exception:  # noqa: BLE001
                pass

        loop = run_agent_loop(
            user_text=request.text,
            sampler=self.sampler,
            tools=tools,
            cwd=cwd,
            max_turns=max_turns,
            system_prompt=system_prompt,
            is_cancelled=_cancelled,
            hooks=hooks,
            session=session,
            session_id=session_id,
            prompt_id=request.prompt_id,
            context_compactor=compactor,
            prior_messages=prior,
            on_archive_message=_archive,
        )

        # Carry live window into the next prompt (may already be pruned/folded).
        if self.resume_live:
            self.live_messages = list(loop.messages)

        # Grok: clear UNTIL_SUCCESS suppress when a model sample completed.
        if not loop.error and compactor is not None:
            on_ok = getattr(compactor, "on_model_success", None)
            if callable(on_ok):
                on_ok()

        # Hermes MemoryManager post-turn: single spine via sync_all only.
        # Providers warm inside sync_turn (avoids queue clobbering richer blend).
        if memory_manager is not None:
            try:
                memory_manager.sync_all(
                    request.text or "",
                    loop.final_text or "",
                    session_id=session_id or "",
                )
            except Exception:  # noqa: BLE001
                logger.warning("memory_manager post-turn failed", exc_info=True)

        meta_extra = {
            "live_messages": len(loop.messages),
            "resumed_prior": bool(prior),
            "has_memory_manager": memory_manager is not None,
            "has_policy": getattr(ext, "policy", None) is not None if ext else False,
        }

        if loop.error and not loop.aborted and not loop.cancelled:
            return TurnResult(
                status=TurnStatus.ERROR,
                final_text=loop.final_text,
                tools_called=loop.tools_called,
                error=loop.error,
                metadata={
                    "rounds": loop.rounds,
                    **loop.metadata,
                    **meta_extra,
                },
            )
        if loop.cancelled:
            return TurnResult(
                status=TurnStatus.CANCELLED,
                final_text=loop.final_text,
                tools_called=loop.tools_called,
                metadata={"rounds": loop.rounds, **loop.metadata, **meta_extra},
            )
        if loop.max_turns_reached:
            return TurnResult(
                status=TurnStatus.MAX_TURNS_REACHED,
                final_text=loop.final_text,
                tools_called=loop.tools_called,
                metadata={
                    "rounds": loop.rounds,
                    "hint": loop.metadata.get("hint"),
                    **{k: v for k, v in loop.metadata.items() if k != "hint"},
                    **meta_extra,
                },
            )
        if loop.aborted:
            return TurnResult(
                status=TurnStatus.ERROR,
                final_text=loop.final_text,
                tools_called=loop.tools_called,
                error=loop.error or "aborted",
                metadata={
                    "rounds": loop.rounds,
                    "aborted": True,
                    **loop.metadata,
                    **meta_extra,
                },
            )
        return TurnResult(
            status=TurnStatus.COMPLETED,
            final_text=loop.final_text,
            tools_called=loop.tools_called,
            metadata={"rounds": loop.rounds, **loop.metadata, **meta_extra},
        )



