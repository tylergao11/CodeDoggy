"""Session: workspace-bound outer runtime."""

from __future__ import annotations

import logging
import threading
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
        self._cancel_requested = False
        self._kernel: Any | None = getattr(self._ext, "kernel", None)
        # Serialise phase / close / live-history transitions (Actor-lite)
        self._lock = threading.RLock()
        # Nested drain depth (one item per finally; cap against re-enqueue loops)
        self._prompt_drain_depth = 0

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
        """Session-level intent anchor (resident audit + future memory select)."""
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

    def enqueue_prompt(self, text: str, *, prompt_id: str | None = None) -> int:
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
            kernel = self._kernel or getattr(self._ext, "kernel", None)
            if kernel is None:
                raise SessionError(
                    "prompt queue not available (no orchestration kernel)"
                )
            enq = getattr(kernel, "enqueue_prompt", None)
            if callable(enq):
                return int(enq(text, prompt_id=prompt_id))
            # Kernel without helper — push directly, create queue if needed
            from codedoggy.orchestration.prompt_queue import PromptQueue, PromptQueueItem

            q = getattr(kernel, "prompt_queue", None)
            if q is None:
                q = PromptQueue()
                kernel.prompt_queue = q
            q.push(PromptQueueItem(text=text, prompt_id=prompt_id))
            return len(q)

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
        self._ensure_open()
        if self._phase is SessionPhase.TURN_RUNNING:
            raise SessionBusyError("cannot rotate session during a turn")
        kernel = self._kernel or getattr(self._ext, "kernel", None)
        if kernel is None or not hasattr(kernel, "new_session"):
            raise SessionError("new_session requires RuntimeKernel")
        new_id = kernel.new_session(
            title=title, clear_live=clear_live, reason=reason
        )
        # Keep Session handle id in sync with kernel
        self._id = SessionId(new_id)
        self._config.session_id = new_id
        return new_id

    def rewind_context(self, *, as_reference: bool = True) -> dict:
        """Restore last context checkpoint into the live transcript window.

        Context-pillar thicken API. Call between prompts when fold lost detail.
        Notifies Hermes providers with rewound=True (same session_id).
        """
        self._ensure_open()
        if self._phase is SessionPhase.TURN_RUNNING:
            raise SessionBusyError("cannot rewind during a turn")
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
            return rew(as_reference=as_reference)
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
            return {"ok": True, "checkpoint": str(path), "messages_after": len(new)}
        return {"ok": False, "reason": "rewind unavailable"}

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
        return self._closed

    def bind_extensions(self, extensions: SessionExtensions) -> None:
        self._ensure_open()
        if self._phase is SessionPhase.TURN_RUNNING:
            raise SessionBusyError("cannot rebind extensions during a turn")
        self._ext = extensions
        self._kernel = getattr(extensions, "kernel", None)

    def bind_turn_runner(self, runner: TurnRunner) -> None:
        self.bind_extensions(self._ext.with_turn_runner(runner))

    # Safety cap: nested finally-chain drain (one item per level).
    _PROMPT_DRAIN_DEPTH_CAP = 32

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
            self._cancel_requested = False
            self._phase = SessionPhase.TURN_RUNNING

        logger.info(
            "session.turn.start id=%s prompt_id=%s",
            self._id,
            prompt_id,
        )

        try:
            if self._cancel_requested:
                return TurnResult(status=TurnStatus.CANCELLED)

            runner = self._ext.turn_runner or StubTurnRunner()
            result = runner.run(request, session=self)
            with self._lock:
                self._turn_count += 1
            return result
        except Exception as e:
            logger.exception("session.turn.error id=%s", self._id)
            with self._lock:
                self._turn_count += 1
            return TurnResult(status=TurnStatus.ERROR, error=str(e))
        finally:
            with self._lock:
                if not self._closed:
                    self._phase = SessionPhase.IDLE
            logger.info(
                "session.turn.end id=%s turns=%s",
                self._id,
                self._turn_count,
            )
            # Consume PromptQueue (full prompts parked while busy) — one per finally
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
        """Run one parked full prompt after IDLE (chain via nested finally).

        Processes **one** item per call so drain cannot nest forever in a
        single while-loop; each drained ``handle_prompt`` finishes → IDLE →
        finally drains the next. Depth is capped against pathological re-enqueue.
        """
        with self._lock:
            if self._closed or self._phase is not SessionPhase.IDLE:
                return
            if self._prompt_drain_depth >= self._PROMPT_DRAIN_DEPTH_CAP:
                logger.warning(
                    "session.prompt_queue drain depth cap=%s id=%s",
                    self._PROMPT_DRAIN_DEPTH_CAP,
                    self._id,
                )
                return
            kernel = self._kernel or getattr(self._ext, "kernel", None)
            q = getattr(kernel, "prompt_queue", None) if kernel is not None else None
            if q is None or len(q) == 0:
                return
            pop = getattr(q, "pop", None)
            if not callable(pop):
                return
            item = pop()
            # Skip empty items under lock; still at most one non-empty start
            while item is not None and not str(
                getattr(item, "text", None) or item
            ).strip():
                item = pop()
            if item is None:
                return
            self._prompt_drain_depth += 1

        text = getattr(item, "text", None) or str(item)
        pid = getattr(item, "prompt_id", None)
        try:
            self.handle_prompt(str(text), prompt_id=pid)
        except Exception:  # noqa: BLE001
            logger.exception("drained prompt_queue item failed")
        finally:
            with self._lock:
                self._prompt_drain_depth = max(0, self._prompt_drain_depth - 1)

    def cancel(self) -> None:
        """Request cancellation of the active turn."""
        self._cancel_requested = True
        logger.info("session.cancel_requested id=%s", self._id)

    def is_cancel_requested(self) -> bool:
        return self._cancel_requested

    def close(self) -> None:
        """Close the session and refuse further prompts.

        Prefers :meth:`RuntimeKernel.close` (memory shutdown, subagent pool,
        session store, context end hooks). Falls back to extension handles
        when no kernel is bound.
        """
        if self._closed:
            return
        self._closed = True
        self._phase = SessionPhase.CLOSED
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
        logger.info("session.closed id=%s turns=%s", self._id, self._turn_count)

    def _ensure_open(self) -> None:
        if self._closed:
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
