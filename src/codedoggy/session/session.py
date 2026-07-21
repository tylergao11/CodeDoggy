"""Session: workspace-bound outer runtime."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from codedoggy.session.config import SessionConfig
from codedoggy.session.extensions import SessionExtensions, TurnRunner
from codedoggy.session.types import (
    SessionId,
    SessionPhase,
    TurnRequest,
    TurnResult,
    TurnStatus,
)

logger = logging.getLogger(__name__)


class SessionError(Exception):
    """Base error for session operations."""


class SessionClosedError(SessionError):
    """Session is closed."""


class SessionBusyError(SessionError):
    """A turn is already in progress."""


class StubTurnRunner:
    """Default runner until a real agentic loop is bound."""

    def run(self, request: TurnRequest, *, session: Session) -> TurnResult:
        return TurnResult(
            status=TurnStatus.NOT_IMPLEMENTED,
            final_text=None,
            metadata={
                "hint": "bind a TurnRunner via SessionExtensions",
                "cwd": str(session.cwd),
                "prompt_preview": request.text[:80],
            },
        )


class Session:
    """Holds cwd, phase, and extension handles; accepts user prompts."""

    def __init__(
        self,
        config: SessionConfig,
        extensions: SessionExtensions | None = None,
    ) -> None:
        self._id = SessionId(config.session_id or SessionId.new())
        self._cwd = config.cwd
        self._max_turns = config.max_turns
        self._goal = config.goal
        self._config = config
        self._ext = extensions or SessionExtensions()
        self._phase = SessionPhase.IDLE
        self._turn_count = 0
        self._closed = False
        self._closing = False
        self._close_finalizing = False
        self._close_finalizer_thread: threading.Thread | None = None
        self._cancel_requested = False
        self._cancel_event = threading.Event()
        self._kernel: Any | None = getattr(self._ext, "kernel", None)
        # Serialise phase / close / live-history transitions (Actor-lite)
        self._lock = threading.RLock()
        # A full-prompt driver owns the queue until it is empty.  This mirrors
        # Grok's pager/actor serialization and avoids recursive queue draining.
        self._prompt_drain_active = False
        self._prompt_ingress_stopped = False
        # Close is a barrier: resources stay live until the active turn exits.
        self._turn_done = threading.Event()
        self._turn_done.set()
        self._close_done = threading.Event()
        self._turn_thread_id: int | None = None
        # Host surfaces (TUI, RPC) observe every full turn, including prompts
        # injected by the scheduler.  Listeners are copied under the session
        # lock and invoked outside it; a start listener may enrich request
        # metadata with stream/live-message callbacks before the runner starts.
        self._turn_listeners: list[Any] = []

    @classmethod
    def create(
        cls,
        cwd: str | Path,
        *,
        max_turns: int | None = None,
        session_id: str | None = None,
        goal: str | None = None,
        extensions: SessionExtensions | None = None,
        **config_extra: Any,
    ) -> Session:
        cfg = SessionConfig(
            cwd=Path(cwd),
            max_turns=max_turns,
            session_id=session_id,
            goal=goal,
            extra=dict(config_extra) if config_extra else {},
        )
        return cls(cfg, extensions=extensions)

    @property
    def id(self) -> SessionId:
        return self._id

    @property
    def cwd(self) -> Path:
        return self._cwd

    @property
    def phase(self) -> SessionPhase:
        return self._phase

    @property
    def max_turns(self) -> int | None:
        return self._max_turns

    @property
    def goal(self) -> str | None:
        """Session-level intent anchor (goal mode + memory select)."""
        return self._goal

    def set_goal(self, goal: str | None) -> None:
        """Update session goal — single writer via RuntimeKernel when present."""
        self._ensure_open()
        g = goal.strip() if isinstance(goal, str) and goal.strip() else goal
        self._goal = g
        self._config.goal = g
        kernel = self._kernel or getattr(self._ext, "kernel", None)
        if kernel is not None:
            set_g = getattr(kernel, "set_goal", None)
            if callable(set_g):
                set_g(g)
                return
        # No kernel: still refresh runner system prompt goal line
        runner = self._ext.turn_runner
        sp = getattr(runner, "system_prompt", None)
        if isinstance(sp, str) and runner is not None:
            from codedoggy.session.kernel import rebuild_system_prompt

            runner.system_prompt = rebuild_system_prompt(sp, g)  # type: ignore[attr-defined]

    def interject(self, text: str, *, prompt_id: str | None = None) -> None:
        """Grok mid-turn interjection — drained before next sample."""
        self._ensure_open()
        kernel = self._kernel or getattr(self._ext, "kernel", None)
        if kernel is not None and hasattr(kernel, "interject"):
            kernel.interject(text, prompt_id=prompt_id)
            return
        raise SessionError("interjection buffer not available (no orchestration kernel)")

    def enqueue_prompt(
        self,
        text: str,
        *,
        prompt_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Park a full prompt for the next turn after the current one.

        Host API (Actor-lite):

        - Concurrent :meth:`handle_prompt` while a turn is running → **interjection**
          only (soft ``TurnStatus.QUEUED``; drained mid-turn at safe points).
        - :meth:`enqueue_prompt` → **next full turn** after the current one
          completes (lands in kernel ``prompt_queue``; never touches the
          interjection buffer).

        Enqueue while IDLE does **not** auto-start a turn — the host must call
        :meth:`handle_prompt`, or a post-turn drain will run queued items after
        a completed turn. Creates the kernel queue if missing.

        Returns the queue length after push. RLock serialises with phase transitions.
        """
        self._ensure_open()
        with self._lock:
            if self._prompt_ingress_stopped:
                raise SessionClosedError("full-prompt ingress is stopped")
            kernel = self._kernel or getattr(self._ext, "kernel", None)
            if kernel is None:
                raise SessionError(
                    "prompt queue not available (no orchestration kernel)"
                )
            enq = getattr(kernel, "enqueue_prompt", None)
            if callable(enq):
                return int(enq(text, prompt_id=prompt_id, metadata=metadata))
            # Kernel without helper — push directly, create queue if needed
            from codedoggy.orchestration.prompt_queue import PromptQueue, PromptQueueItem

            q = getattr(kernel, "prompt_queue", None)
            if q is None:
                q = PromptQueue()
                kernel.prompt_queue = q
            q.push(
                PromptQueueItem(
                    text=text,
                    prompt_id=prompt_id,
                    metadata=dict(metadata or {}),
                )
            )
            return len(q)

    def submit_prompt(
        self,
        text: str,
        *,
        prompt_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Submit a *full* host prompt and drive it immediately when idle.

        This is the Grok pager ingress used by synthetic prompts such as
        scheduler fires.  It is intentionally distinct from :meth:`interject`:
        a busy session parks the prompt for the next turn; an idle session
        consumes it immediately through the same serialized queue.

        Returns the queue length observed immediately after enqueue.
        """
        with self._lock:
            self._ensure_open()
            if self._prompt_ingress_stopped:
                raise SessionClosedError("full-prompt ingress is stopped")
            queued = self.enqueue_prompt(
                text,
                prompt_id=prompt_id,
                metadata=metadata,
            )
            should_drive = (
                self._phase is SessionPhase.IDLE and not self._prompt_drain_active
            )
        if should_drive:
            self._drain_prompt_queue()
        return queued

    def enter_plan_mode(self, plan_file: str | None = None) -> None:
        """Grok plan mode — only plan file edits allowed."""
        self._ensure_open()
        kernel = self._kernel or getattr(self._ext, "kernel", None)
        if kernel is not None and hasattr(kernel, "enter_plan_mode"):
            kernel.enter_plan_mode(plan_file)
            return
        raise SessionError("session mode not available (no orchestration kernel)")

    def exit_plan_mode(self, *, approved: bool = True) -> None:
        self._ensure_open()
        kernel = self._kernel or getattr(self._ext, "kernel", None)
        if kernel is not None and hasattr(kernel, "exit_plan_mode"):
            kernel.exit_plan_mode(approved=approved)
            return
        raise SessionError("session mode not available (no orchestration kernel)")

    def _resolve_parked_plan_then_remind(self, request: TurnRequest) -> TurnRequest:
        """Grok resume re-park then normal plan turn reminders."""
        kernel = self._kernel or getattr(self._ext, "kernel", None)
        injects: list[str] = []
        if kernel is not None and hasattr(kernel, "wait_or_resolve_parked_plan_approval"):
            try:
                parked = kernel.wait_or_resolve_parked_plan_approval()
            except Exception:  # noqa: BLE001
                logger.exception("parked plan approval failed")
                parked = None
            if parked and str(parked).strip():
                injects.append(str(parked).strip())
        request = self._inject_plan_mode_turn_reminder(request)
        if not injects:
            return request
        block = "\n\n".join(
            f"<system-reminder>\n{p}\n</system-reminder>" for p in injects
        )
        text = request.text or ""
        text = f"{block}\n\n{text}" if text.strip() else block
        return TurnRequest(
            text=text,
            prompt_id=request.prompt_id,
            metadata=dict(request.metadata or {}),
        )

    def _inject_plan_mode_turn_reminder(self, request: TurnRequest) -> TurnRequest:
        """Pending→Active + inject plan/exit system-reminders (Grok turn start)."""
        kernel = self._kernel or getattr(self._ext, "kernel", None)
        state = getattr(kernel, "session_mode_state", None) if kernel else None
        if state is None or not hasattr(state, "begin_turn"):
            return request
        try:
            body = state.begin_turn()
        except Exception:  # noqa: BLE001
            logger.exception("plan mode begin_turn failed")
            return request
        if kernel is not None:
            if hasattr(kernel, "refresh_tool_extra"):
                try:
                    kernel.refresh_tool_extra()
                except Exception:  # noqa: BLE001
                    pass
            if hasattr(kernel, "persist_plan_mode_state"):
                try:
                    kernel.persist_plan_mode_state()
                except Exception:  # noqa: BLE001
                    pass
        if not body or not str(body).strip():
            return request
        reminder = f"<system-reminder>\n{body.strip()}\n</system-reminder>"
        text = request.text or ""
        if text.strip():
            text = f"{reminder}\n\n{text}"
        else:
            text = reminder
        return TurnRequest(
            text=text,
            prompt_id=request.prompt_id,
            metadata=dict(request.metadata or {}),
        )

    def _complete_plan_mode_turn(self) -> None:
        """ExitPending → Inactive after turn (Grok complete_deferred_exit)."""
        kernel = self._kernel or getattr(self._ext, "kernel", None)
        state = getattr(kernel, "session_mode_state", None) if kernel else None
        if state is None or not hasattr(state, "end_turn"):
            return
        try:
            state.end_turn()
        except Exception:  # noqa: BLE001
            logger.exception("plan mode end_turn failed")
            return
        if kernel is not None:
            if hasattr(kernel, "refresh_tool_extra"):
                try:
                    kernel.refresh_tool_extra()
                except Exception:  # noqa: BLE001
                    pass
            if hasattr(kernel, "persist_plan_mode_state"):
                try:
                    kernel.persist_plan_mode_state()
                except Exception:  # noqa: BLE001
                    pass

    def enter_goal_mode(self) -> None:
        self._ensure_open()
        kernel = self._kernel or getattr(self._ext, "kernel", None)
        if kernel is not None and hasattr(kernel, "enter_goal_mode"):
            kernel.enter_goal_mode()
            return
        raise SessionError("session mode not available (no orchestration kernel)")

    def exit_goal_mode(self) -> None:
        self._ensure_open()
        kernel = self._kernel or getattr(self._ext, "kernel", None)
        if kernel is not None and hasattr(kernel, "exit_goal_mode"):
            kernel.exit_goal_mode()
            return
        raise SessionError("session mode not available (no orchestration kernel)")

    def new_session(
        self,
        *,
        title: str | None = None,
        clear_live: bool = True,
        reason: str = "new_session",
    ) -> str:
        """Hermes /new — rotate session id; memory providers rebind async."""
        with self._lock:
            self._ensure_open()
            if self._phase is SessionPhase.TURN_RUNNING:
                raise SessionBusyError("cannot rotate session during a turn")
            kernel = self._kernel or getattr(self._ext, "kernel", None)
            if kernel is None or not hasattr(kernel, "new_session"):
                raise SessionError("new_session requires RuntimeKernel")
            new_id = kernel.new_session(
                title=title, clear_live=clear_live, reason=reason
            )
            # Keep Session handle id in sync with kernel while the same lock
            # still excludes handle_prompt from reserving a turn.
            self._id = SessionId(new_id)
            self._config.session_id = new_id
            return new_id

    def rewind_context(self, *, as_reference: bool = True) -> dict:
        """Restore last context checkpoint into the live transcript window.

        Context-pillar thicken API. Call between prompts when fold lost detail.
        Notifies Hermes providers with rewound=True (same session_id).
        """
        with self._lock:
            self._ensure_open()
            if self._phase is SessionPhase.TURN_RUNNING:
                raise SessionBusyError("cannot rewind during a turn")
            return self._rewind_context_locked(as_reference=as_reference)

    def _rewind_context_locked(self, *, as_reference: bool) -> dict:
        """Rewind implementation; caller holds ``self._lock``."""
        runner = self._ext.turn_runner
        # Prefer runner method; fall back to context handle + live_messages
        rew = getattr(runner, "rewind_context", None)
        if callable(rew):
            # Stash memory handles so runner fires on_transcript_rewound once.
            # Do not notify again here — double rewound was a seam bug.
            if runner is not None:
                try:
                    k = self._kernel or getattr(self._ext, "kernel", None)
                    runner._memory_manager = (  # type: ignore[attr-defined]
                        getattr(k, "memory_manager", None)
                        or getattr(self._ext, "memory_manager", None)
                    )
                    runner._session_id = str(self._id)  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
            result = rew(as_reference=as_reference)
            if isinstance(result, dict) and result.get("ok"):
                self._persist_live_snapshot()
            return result
        compactor = self._ext.context
        live = getattr(runner, "live_messages", None)
        rew2 = getattr(compactor, "rewind_from_checkpoint", None)
        if callable(rew2) and isinstance(live, list):
            path = getattr(compactor, "last_checkpoint_path", None)
            if not path:
                return {"ok": False, "reason": "no checkpoint"}
            new = rew2(live, as_reference=as_reference)
            runner.live_messages = new  # type: ignore[attr-defined]
            # Fallback path (no runner.rewind_context) — seam notify once here
            from codedoggy.memory.hermes_seam import on_transcript_rewound

            mm = getattr(self._ext, "memory_manager", None)
            k = self._kernel or getattr(self._ext, "kernel", None)
            if mm is None and k is not None:
                mm = getattr(k, "memory_manager", None)
            on_transcript_rewound(mm, session_id=str(self._id))
            self._persist_live_snapshot()
            return {"ok": True, "checkpoint": str(path), "messages_after": len(new)}
        return {"ok": False, "reason": "rewind unavailable"}

    def _persist_live_snapshot(self) -> None:
        """Persist the canonical live window after an out-of-turn mutation."""
        kernel = self._kernel or getattr(self._ext, "kernel", None)
        store = getattr(kernel, "session_store", None) if kernel is not None else None
        runner = self._ext.turn_runner
        live = getattr(runner, "live_messages", None)
        save = getattr(store, "save_context_snapshot", None)
        if callable(save) and isinstance(live, list):
            try:
                save(str(self._id), live)
            except Exception:  # noqa: BLE001
                logger.warning("rewound context snapshot save failed", exc_info=True)

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def extensions(self) -> SessionExtensions:
        return self._ext

    @property
    def config(self) -> SessionConfig:
        return self._config

    @property
    def is_closed(self) -> bool:
        # Closing sessions reject new work even while the active turn drains.
        return self._closing or self._closed

    def bind_extensions(self, extensions: SessionExtensions) -> None:
        self._ensure_open()
        if self._phase is SessionPhase.TURN_RUNNING:
            raise SessionBusyError("cannot rebind extensions during a turn")
        self._ext = extensions
        self._kernel = getattr(extensions, "kernel", None)

    def bind_turn_runner(self, runner: TurnRunner) -> None:
        self.bind_extensions(self._ext.with_turn_runner(runner))

    def add_turn_listener(self, listener: Any) -> None:
        """Observe ``start``/``end`` for all full turns in this Session."""
        if not callable(listener):
            raise TypeError("turn listener must be callable")
        with self._lock:
            if listener not in self._turn_listeners:
                self._turn_listeners.append(listener)

    def remove_turn_listener(self, listener: Any) -> None:
        with self._lock:
            self._turn_listeners = [
                item for item in self._turn_listeners if item != listener
            ]

    def _notify_turn_listeners(
        self,
        event: str,
        request: TurnRequest,
        result: TurnResult | None,
    ) -> None:
        with self._lock:
            listeners = list(self._turn_listeners)
        for listener in listeners:
            try:
                listener(event, request, result)
            except Exception:  # host observability must never break execution
                logger.exception("session turn listener failed event=%s", event)

    def handle_prompt(
        self,
        text: str,
        *,
        prompt_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TurnResult:
        """Run one user prompt through the bound turn runner.

        Grok mid-turn: if a turn is already running, queue via :meth:`interject`
        and return ``TurnStatus.QUEUED`` (not COMPLETED — that lied about turn finish).
        Concurrent handle_prompt is **interjection only** — it does not also push
        :class:`~codedoggy.orchestration.prompt_queue.PromptQueue` (would double-run).

        For a deferred full turn while busy, use :meth:`enqueue_prompt` instead.

        After a turn ends, drain kernel ``prompt_queue`` as full subsequent prompts
        (one item per finally; nested handle_prompt continues the chain).
        """
        with self._lock:
            self._ensure_open()
            if self._prompt_ingress_stopped:
                raise SessionClosedError("prompt ingress is stopped")
            if self._phase is SessionPhase.TURN_RUNNING:
                # Grok: mid-turn → interjection only (drained before next sample).
                # Do NOT also push PromptQueue (would double-run after turn).
                try:
                    self.interject(text, prompt_id=prompt_id)
                except SessionError as e:
                    raise SessionBusyError("a turn is already running") from e
                logger.info(
                    "session.interject.queued id=%s prompt_id=%s",
                    self._id,
                    prompt_id,
                )
                return TurnResult(
                    status=TurnStatus.QUEUED,
                    final_text="(queued as interjection)",
                    metadata={
                        "interjected": True,
                        "queued_interjection": True,
                        "prompt_id": prompt_id,
                    },
                )

            request = TurnRequest(
                text=text,
                prompt_id=prompt_id,
                metadata=metadata or {},
            )
            self._begin_turn_locked()

        return self._run_started_turn(request)

    def _begin_turn_locked(self) -> None:
        """Reserve the single turn slot while ``self._lock`` is held."""
        self._cancel_requested = False
        # Per-request token identity matters: an abandoned transport owner may
        # outlive this turn.  Never clear/reuse its Event (ABA); the next turn
        # receives a fresh token while the old worker permanently sees set().
        self._cancel_event = threading.Event()
        self._phase = SessionPhase.TURN_RUNNING
        self._turn_thread_id = threading.get_ident()
        self._turn_done.clear()

    def _run_started_turn(self, request: TurnRequest) -> TurnResult:
        """Execute a request whose turn slot was already reserved."""
        prompt_id = request.prompt_id

        logger.info(
            "session.turn.start id=%s prompt_id=%s",
            self._id,
            prompt_id,
        )

        result: TurnResult | None = None
        self._notify_turn_listeners("start", request, None)
        try:
            if self._cancel_requested:
                result = TurnResult(status=TurnStatus.CANCELLED)
                return result

            request = self._resolve_parked_plan_then_remind(request)

            runner = self._ext.turn_runner or StubTurnRunner()
            result = runner.run(request, session=self)
            with self._lock:
                self._turn_count += 1
            return result
        except Exception as e:
            logger.exception("session.turn.error id=%s", self._id)
            with self._lock:
                self._turn_count += 1
            result = TurnResult(status=TurnStatus.ERROR, error=str(e))
            return result
        finally:
            self._notify_turn_listeners("end", request, result)
            self._complete_plan_mode_turn()
            with self._lock:
                closing = self._closing
                if not closing and not self._closed:
                    self._phase = SessionPhase.IDLE
                self._turn_thread_id = None
                self._turn_done.set()
            logger.info(
                "session.turn.end id=%s turns=%s",
                self._id,
                self._turn_count,
            )
            if closing:
                # Final teardown has one daemon owner.  The turn thread must
                # not inherit a blocking Graph/MCP/memory close operation.
                self._start_close_finalizer()
            elif not self._prompt_drain_active:
                with self._lock:
                    drain_allowed = not self._prompt_ingress_stopped
                # Grok: do not auto-wake queued full prompts after cancel
                # (CODEDOGGY_DRAIN_AFTER_CANCEL=1 restores old drain-on-cancel).
                cancelled_turn = (
                    result is not None
                    and getattr(result, "status", None) is not None
                    and str(getattr(result.status, "value", result.status)).lower()
                    == "cancelled"
                )
                if cancelled_turn:
                    try:
                        from codedoggy.orchestration.subagent_policy import (
                            drain_prompt_queue_after_cancel,
                        )

                        if not drain_prompt_queue_after_cancel():
                            drain_allowed = False
                    except Exception:  # noqa: BLE001
                        drain_allowed = False
                if drain_allowed:
                    self._drain_prompt_queue()

    def _push_prompt_queue(
        self,
        text: str,
        *,
        prompt_id: str | None,
        urgent: bool = False,
    ) -> None:
        """Internal push (prefer public :meth:`enqueue_prompt`)."""
        try:
            if urgent:
                from codedoggy.orchestration.prompt_queue import (
                    PromptQueue,
                    PromptQueueItem,
                )

                kernel = self._kernel or getattr(self._ext, "kernel", None)
                if kernel is None:
                    return
                q = getattr(kernel, "prompt_queue", None)
                if q is None:
                    q = PromptQueue()
                    kernel.prompt_queue = q
                q.push(PromptQueueItem(text=text, prompt_id=prompt_id, urgent=True))
            else:
                self.enqueue_prompt(text, prompt_id=prompt_id)
        except Exception:  # noqa: BLE001
            logger.debug("prompt_queue push failed", exc_info=True)

    def _drain_prompt_queue(self) -> None:
        """Drive parked full prompts serially while the session remains idle."""
        with self._lock:
            if (
                self._closed
                or self._closing
                or self._prompt_ingress_stopped
                or self._phase is not SessionPhase.IDLE
                or self._prompt_drain_active
            ):
                return
            self._prompt_drain_active = True

        try:
            while True:
                with self._lock:
                    if (
                        self._closed
                        or self._closing
                        or self._prompt_ingress_stopped
                        or self._phase is not SessionPhase.IDLE
                    ):
                        return
                    kernel = self._kernel or getattr(self._ext, "kernel", None)
                    q = (
                        getattr(kernel, "prompt_queue", None)
                        if kernel is not None
                        else None
                    )
                    pop = getattr(q, "pop", None) if q is not None else None
                    if not callable(pop):
                        return
                    item = pop()
                    while item is not None and not str(
                        getattr(item, "text", None) or item
                    ).strip():
                        item = pop()
                    if item is None:
                        return
                    request = TurnRequest(
                        text=str(getattr(item, "text", None) or item),
                        prompt_id=getattr(item, "prompt_id", None),
                        metadata=dict(getattr(item, "metadata", None) or {}),
                    )
                    self._begin_turn_locked()
                self._run_started_turn(request)
        except Exception:  # noqa: BLE001
            logger.exception("drained prompt_queue item failed")
        finally:
            with self._lock:
                self._prompt_drain_active = False

    def cancel(self) -> None:
        """Request cancellation of the active turn."""
        with self._lock:
            self._cancel_requested = True
            self._cancel_event.set()
        logger.info("session.cancel_requested id=%s", self._id)

    def stop_prompt_ingress(self, *, clear_queue: bool = True) -> int:
        """Freeze host/full-prompt ingress and optionally discard queued work.

        This is a host-lifecycle barrier, not a normal user cancellation.  It
        prevents a finishing turn from starting scheduler work after its TUI or
        RPC owner has already shut down.
        """
        cleared = 0
        with self._lock:
            self._prompt_ingress_stopped = True
            kernel = self._kernel or getattr(self._ext, "kernel", None)
            queue = getattr(kernel, "prompt_queue", None) if kernel is not None else None
            if clear_queue and queue is not None:
                try:
                    cleared = len(queue)
                except Exception:  # noqa: BLE001
                    cleared = 0
                clear = getattr(queue, "clear", None)
                if callable(clear):
                    clear()
        return int(cleared)

    def wait_for_turn(self, timeout_s: float | None = None) -> bool:
        """Wait for the active full-turn barrier without exposing internals."""
        timeout = None if timeout_s is None else max(0.0, float(timeout_s))
        return bool(self._turn_done.wait(timeout=timeout))

    def is_cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def cancel_event(self) -> threading.Event:
        """Per-turn cancellation token passed through the model transport."""
        return self._cancel_event

    def close(self, *, timeout_s: float | None = None) -> None:
        """Close the session and refuse further prompts.

        Prefers :meth:`RuntimeKernel.close` (memory shutdown, subagent pool,
        session store, context end hooks). Falls back to extension handles
        when no kernel is bound.
        """
        timeout = None if timeout_s is None else max(0.0, float(timeout_s))
        deadline = None if timeout is None else time.monotonic() + timeout
        caller = threading.get_ident()
        with self._lock:
            if self._closed:
                return
            if self._closing:
                same_turn = self._turn_thread_id == caller
                stop_ingress = False
            else:
                self._closing = True
                self._cancel_requested = True
                self._cancel_event.set()
                same_turn = self._turn_thread_id == caller
                stop_ingress = True
            turn_running = self._phase is SessionPhase.TURN_RUNNING

        if stop_ingress:
            # Quiesce synthetic host ingress before waiting for the active turn;
            # otherwise a scheduler tick could advance a one-shot task after the
            # Session has stopped accepting prompts.
            kernel = self._kernel or getattr(self._ext, "kernel", None)
            extra = getattr(kernel, "tool_extra", None) if kernel is not None else None
            handle = (extra or {}).get("scheduler_runtime")
            # Signal immediately without joining here.  The single close owner
            # performs the potentially blocking join inside kernel.close().
            stop_event = getattr(handle, "stop_event", None)
            set_stop = getattr(stop_event, "set", None)
            if callable(set_stop):
                try:
                    set_stop()
                except Exception:  # noqa: BLE001
                    logger.debug("scheduler ingress signal failed", exc_info=True)

        if turn_running:
            if same_turn:
                # The active turn's finally block owns final teardown.
                return
            remaining = _remaining_timeout(deadline)
            if not self._turn_done.wait(timeout=remaining):
                logger.warning(
                    "session close deferred: active turn did not stop within %.1fs id=%s",
                    float(timeout or 0.0),
                    self._id,
                )
                return
        self._start_close_finalizer()
        remaining = _remaining_timeout(deadline)
        if not self._close_done.wait(timeout=remaining):
            logger.warning(
                "session close deferred: teardown continues after %.1fs id=%s",
                float(timeout or 0.0),
                self._id,
            )

    def _start_close_finalizer(self) -> None:
        """Elect and start the one teardown owner without blocking its caller."""
        with self._lock:
            if self._closed:
                return
            if self._close_finalizing:
                return
            self._close_finalizing = True
            self._phase = SessionPhase.CLOSED
            owner = threading.Thread(
                target=self._run_close_finalizer,
                name=f"codedoggy-close-{self._id}",
                daemon=True,
            )
            self._close_finalizer_thread = owner
        owner.start()

    def _run_close_finalizer(self) -> None:
        """Close owned resources; always runs on the elected daemon owner."""
        kernel = self._kernel or getattr(self._ext, "kernel", None)
        try:
            if kernel is not None and callable(getattr(kernel, "close", None)):
                kernel.close()
            else:
                # Legacy path without kernel — best-effort mirror of kernel.close
                seen: set[int] = set()
                for obj in (
                    getattr(self._ext, "context", None),
                    getattr(
                        getattr(self._ext, "turn_runner", None),
                        "context_compactor",
                        None,
                    ),
                ):
                    if obj is None:
                        continue
                    oid = id(obj)
                    if oid in seen:
                        continue
                    seen.add(oid)
                    on_end = getattr(obj, "on_session_end", None)
                    if callable(on_end):
                        on_end()
                runner = getattr(self._ext, "turn_runner", None)
                # Snapshot before clear so providers can extract on end
                snap = list(getattr(runner, "live_messages", None) or [])
                clear = getattr(runner, "clear_live_history", None)
                if callable(clear):
                    clear()
                # Hermes seam: end → flush → shutdown (never raw mm.shutdown alone)
                from codedoggy.memory.hermes_seam import on_session_close

                mm = getattr(self._ext, "memory_manager", None)
                on_session_close(mm, messages=snap, timeout_s=5.0)
                mem = getattr(self._ext, "memory", None)
                close_mem = getattr(mem, "close", None)
                if callable(close_mem):
                    close_mem()
                ss = getattr(self._ext, "session_store", None)
                close_ss = getattr(ss, "close", None)
                if callable(close_ss):
                    close_ss()
        except Exception:  # noqa: BLE001
            logger.exception("session.end context cleanup failed")
        finally:
            with self._lock:
                self._closed = True
                self._closing = True
                self._close_finalizing = False
                self._close_finalizer_thread = None
                self._phase = SessionPhase.CLOSED
                self._close_done.set()
        logger.info("session.closed id=%s turns=%s", self._id, self._turn_count)

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise SessionClosedError(f"session {self._id} is closed")

    def __repr__(self) -> str:
        return (
            f"Session(id={self._id!r}, cwd={str(self._cwd)!r}, "
            f"phase={self._phase.value}, turns={self._turn_count})"
        )

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _remaining_timeout(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())
