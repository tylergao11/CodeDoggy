"""Session: workspace-bound outer runtime."""

from __future__ import annotations

import logging
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
        """Update the session goal without starting a turn."""
        self._ensure_open()
        self._goal = goal.strip() if isinstance(goal, str) and goal.strip() else goal

    def rewind_context(self, *, as_reference: bool = True) -> dict:
        """Restore last context checkpoint into the live transcript window.

        Context-pillar thicken API. Call between prompts when fold lost detail.
        """
        self._ensure_open()
        if self._phase is SessionPhase.TURN_RUNNING:
            raise SessionBusyError("cannot rewind during a turn")
        runner = self._ext.turn_runner
        # Prefer runner method; fall back to context handle + live_messages
        rew = getattr(runner, "rewind_context", None)
        if callable(rew):
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

    def bind_turn_runner(self, runner: TurnRunner) -> None:
        self.bind_extensions(self._ext.with_turn_runner(runner))

    def handle_prompt(
        self,
        text: str,
        *,
        prompt_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TurnResult:
        """Run one user prompt through the bound turn runner."""
        self._ensure_open()
        if self._phase is SessionPhase.TURN_RUNNING:
            raise SessionBusyError("a turn is already running")

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
            self._turn_count += 1
            return result
        except Exception as e:
            logger.exception("session.turn.error id=%s", self._id)
            self._turn_count += 1
            return TurnResult(status=TurnStatus.ERROR, error=str(e))
        finally:
            if not self._closed:
                self._phase = SessionPhase.IDLE
            logger.info(
                "session.turn.end id=%s turns=%s",
                self._id,
                self._turn_count,
            )

    def cancel(self) -> None:
        """Request cancellation of the active turn."""
        self._cancel_requested = True
        logger.info("session.cancel_requested id=%s", self._id)

    def is_cancel_requested(self) -> bool:
        return self._cancel_requested

    def close(self) -> None:
        """Close the session and refuse further prompts."""
        if self._closed:
            return
        self._closed = True
        self._phase = SessionPhase.CLOSED
        # Hermes ContextEngine.on_session_end — clear per-session compact state
        # (both extensions.context and runner-owned compactor if distinct).
        try:
            seen: set[int] = set()
            for obj in (
                getattr(self._ext, "context", None),
                getattr(getattr(self._ext, "turn_runner", None), "context_compactor", None),
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
            clear = getattr(runner, "clear_live_history", None)
            if callable(clear):
                clear()
            mm = getattr(self._ext, "memory_manager", None)
            shutdown = getattr(mm, "shutdown", None)
            if callable(shutdown):
                shutdown()
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
