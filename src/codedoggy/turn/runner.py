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
        # Hermes: transcript truncated under same session_id → rewound
        # Caller may pass session via live; best-effort if bound on self later.
        mm = getattr(self, "_memory_manager", None)
        sid = getattr(self, "_session_id", None)
        if mm is not None and sid:
            from codedoggy.memory.hermes_seam import on_transcript_rewound

            on_transcript_rewound(mm, session_id=str(sid))
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

        # Prefer RuntimeKernel as single source for handles + system prompt base
        kernel = getattr(session, "_kernel", None)
        if kernel is None:
            ext0 = getattr(session, "extensions", None)
            kernel = getattr(ext0, "kernel", None) if ext0 is not None else None

        system_prompt = (
            getattr(kernel, "base_system_prompt", None) or self.system_prompt
        )
        ext = getattr(session, "extensions", None)
        mem = (
            getattr(kernel, "memory", None)
            if kernel is not None
            else (getattr(ext, "memory", None) if ext is not None else None)
        )
        memory_manager = (
            getattr(kernel, "memory_manager", None)
            if kernel is not None
            else (getattr(ext, "memory_manager", None) if ext is not None else None)
        )
        session_store = (
            getattr(kernel, "session_store", None)
            if kernel is not None
            else (getattr(ext, "session_store", None) if ext is not None else None)
        )

        # Hermes seam: system memory block (curated freeze + provider static)
        from codedoggy.memory.hermes_seam import (
            build_system_memory_block,
            on_turn_begin,
            on_turn_end,
            prefetch_fenced,
        )

        blocks = build_system_memory_block(memory_manager, mem)
        if blocks:
            system_prompt = (
                f"{system_prompt}\n\n{blocks}" if system_prompt else blocks
            )

        from codedoggy.memory.hermes_select import HermesMemorySelector

        selector = None
        if memory_manager is not None:
            selector = memory_manager.as_memory_selector()
        elif mem is not None or session_store is not None:
            selector = HermesMemorySelector(
                curated_store=mem,
                session_store=session_store,
            )

        cwd_s = str(cwd) if cwd is not None else ""
        # Hermes: fenced prefetch for sample-time user inject only
        prefetch_user_block = prefetch_fenced(
            memory_manager,
            user_text=request.text or "",
            session_id=session_id or "",
            cwd=cwd_s,
            selector=selector,
            session=session,
        )

        hooks = self.hooks

        compactor = self.context_compactor
        if compactor is None:
            from codedoggy.context.compactor import ContextCompactor

            compactor = ContextCompactor.from_env(
                summary_client=None,
                memory_store=mem,
                session_store=session_store,
                memory_manager=memory_manager,
            )

        # Grok residual: bind ModelConfig.context_window onto budget once per run.
        # ChatSampler wraps client; walk getattr carefully (client/config chain).
        _bind_compactor_model_window(compactor, self.sampler)
        # Hermes: let compactor know session_id for post-fold rewound notify
        if compactor is not None and session_id:
            try:
                compactor._session_id = session_id  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

        # Always clear per-turn suppress (bootstrap path shares one compactor).
        on_start = getattr(compactor, "on_turn_start", None)
        if callable(on_start):
            on_start()
        turn_n = int(getattr(session, "_turn_count", 0) or 0) + 1
        on_turn_begin(
            memory_manager,
            mem,
            turn_number=turn_n,
            user_text=request.text or "",
        )

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

        # Mid-turn tool path: provider tools + session_search read stores from
        # ctx.extra. Kernel.tool_extra is the single source; refresh every run
        # so late-bound handles (policy, graph, mm) are visible.
        tool_extra: dict[str, Any] = {}
        if kernel is not None:
            refresh = getattr(kernel, "refresh_tool_extra", None)
            if callable(refresh):
                refresh()
            tool_extra = dict(getattr(kernel, "tool_extra", None) or {})
        # Defensive fill when session was not fully kernel-wired (or keys missing)
        if "memory_manager" not in tool_extra and memory_manager is not None:
            tool_extra["memory_manager"] = memory_manager
        if "memory_store" not in tool_extra and mem is not None:
            tool_extra["memory_store"] = mem
        if "session_store" not in tool_extra and session_store is not None:
            tool_extra["session_store"] = session_store
        if ext is not None:
            if "policy" not in tool_extra:
                pol = getattr(ext, "policy", None)
                if pol is not None:
                    tool_extra["policy"] = pol
            if "graph" not in tool_extra:
                gr = getattr(ext, "graph", None)
                if gr is not None:
                    tool_extra["graph"] = gr
        if prefetch_user_block:
            tool_extra = dict(tool_extra)
            tool_extra["prefetch_user_block"] = prefetch_user_block

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
            tool_extra=tool_extra,
        )

        # Normalize tool pairs before carrying live history
        from codedoggy.context.select import sanitize_tool_pairs

        live = sanitize_tool_pairs(list(loop.messages))
        if self.resume_live:
            self.live_messages = live

        # Grok: clear UNTIL_SUCCESS suppress when a model sample completed.
        if not loop.error and compactor is not None:
            on_ok = getattr(compactor, "on_model_success", None)
            if callable(on_ok):
                on_ok()

        # Hermes post-turn: sync_all + queue_prefetch_all (seam)
        on_turn_end(
            memory_manager,
            user_text=request.text or "",
            assistant_text=loop.final_text or "",
            session_id=session_id or "",
            cwd=cwd_s,
            messages=list(loop.messages) if loop.messages else None,
        )

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


def _bind_compactor_model_window(compactor: Any, sampler: Any) -> None:
    """If sampler exposes a ModelConfig, bind its window into the compactor.

    Resolution order (Grok / ChatSampler):
      sampler.client.config → sampler.config → sampler.client (if config-like)
    """
    if compactor is None or sampler is None:
        return
    bind = getattr(compactor, "bind_model_window", None)
    if not callable(bind):
        return

    config = _resolve_model_config(sampler)
    if config is None:
        return

    cw = getattr(config, "context_window", None)
    mt = getattr(config, "max_tokens", None)
    if cw is None:
        extra = getattr(config, "extra", None)
        if isinstance(extra, dict):
            raw = extra.get("context_window")
            if raw is None:
                raw = extra.get("num_ctx")
            cw = raw
    if cw is None and mt is None:
        return
    try:
        bind(
            context_window=int(cw) if cw else None,
            max_completion_tokens=int(mt) if mt else None,
        )
    except Exception:  # noqa: BLE001
        logger.debug("bind_model_window from sampler failed", exc_info=True)


def _resolve_model_config(sampler: Any) -> Any | None:
    """Walk ChatSampler → client → config without assuming types."""
    # 1) sampler.client.config (ChatSampler + OpenAICompatClient)
    client = getattr(sampler, "client", None)
    if client is not None:
        cfg = getattr(client, "config", None)
        if callable(cfg):
            try:
                cfg = cfg()
            except TypeError:
                cfg = None
        if cfg is not None and (
            hasattr(cfg, "context_window") or hasattr(cfg, "max_tokens")
        ):
            return cfg
        # client itself might carry the knobs
        if hasattr(client, "context_window") or hasattr(client, "max_tokens"):
            return client
    # 2) sampler.config
    cfg = getattr(sampler, "config", None)
    if callable(cfg):
        try:
            cfg = cfg()
        except TypeError:
            cfg = None
    if cfg is not None and (
        hasattr(cfg, "context_window") or hasattr(cfg, "max_tokens")
    ):
        return cfg
    # 3) sampler itself looks like ModelConfig
    if hasattr(sampler, "context_window") or hasattr(sampler, "max_tokens"):
        return sampler
    return None



