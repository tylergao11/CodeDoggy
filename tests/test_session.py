"""Session lifecycle tests."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from codedoggy.orchestration.prompt_queue import InterjectionBuffer
from codedoggy.session import (
    Session,
    SessionConfig,
    SessionExtensions,
    SessionPhase,
    TurnRequest,
    TurnResult,
    TurnStatus,
)
from codedoggy.session.kernel import RuntimeKernel
from codedoggy.session.session import SessionBusyError, SessionClosedError


class FakeRunner:
    def run(self, request: TurnRequest, *, session: Session) -> TurnResult:
        return TurnResult(
            status=TurnStatus.COMPLETED,
            final_text=f"echo:{request.text}",
            tools_called=[],
        )


def test_create_and_default_runner(tmp_path: Path) -> None:
    s = Session.create(tmp_path)
    assert s.phase is SessionPhase.IDLE
    assert s.cwd == tmp_path.resolve()
    r = s.handle_prompt("hi")
    assert r.status is TurnStatus.NOT_IMPLEMENTED
    assert s.phase is SessionPhase.IDLE
    assert s.turn_count == 1
    s.close()
    assert s.phase is SessionPhase.CLOSED


def test_bind_turn_runner(tmp_path: Path) -> None:
    s = Session.create(tmp_path)
    s.bind_turn_runner(FakeRunner())
    r = s.handle_prompt("world")
    assert r.status is TurnStatus.COMPLETED
    assert r.final_text == "echo:world"
    s.close()


def test_closed_rejects_prompt(tmp_path: Path) -> None:
    s = Session.create(tmp_path)
    s.close()
    with pytest.raises(SessionClosedError):
        s.handle_prompt("nope")


def test_context_manager(tmp_path: Path) -> None:
    with Session.create(tmp_path) as s:
        s.handle_prompt("x")
    assert s.is_closed


def test_config_max_turns(tmp_path: Path) -> None:
    cfg = SessionConfig(cwd=tmp_path, max_turns=3)
    s = Session(cfg)
    assert s.max_turns == 3
    s.close()


def test_handle_prompt_while_busy_interjects(tmp_path: Path) -> None:
    """Grok: concurrent handle_prompt mid-turn queues interjection (no hard busy)."""
    entered = threading.Event()
    release = threading.Event()
    main_results: list[TurnResult] = []

    class BlockingRunner:
        def run(self, request: TurnRequest, *, session: Session) -> TurnResult:
            entered.set()
            assert release.wait(timeout=5.0), "release timed out"
            return TurnResult(
                status=TurnStatus.COMPLETED,
                final_text=f"done:{request.text}",
                tools_called=[],
            )

    buf = InterjectionBuffer()
    runner = BlockingRunner()
    kernel = RuntimeKernel(
        cwd=tmp_path,
        session_id="interject-busy",
        turn_runner=runner,
        interjection_buffer=buf,
    )
    s = Session.create(
        tmp_path,
        extensions=SessionExtensions(turn_runner=runner, kernel=kernel),
    )

    def _main() -> None:
        main_results.append(s.handle_prompt("main work"))

    t = threading.Thread(target=_main, name="main-turn")
    t.start()
    assert entered.wait(timeout=5.0), "runner never entered"
    assert s.phase is SessionPhase.TURN_RUNNING

    soft = s.handle_prompt("change of plan", prompt_id="inj-1")
    # Must NOT look like a finished turn (audit: COMPLETED was a lie)
    assert soft.status is TurnStatus.QUEUED
    assert soft.final_text == "(queued as interjection)"
    assert soft.metadata.get("interjected") is True
    assert soft.metadata.get("queued_interjection") is True
    # Turn count must not bump for interjection-only path
    assert s.turn_count == 0
    assert len(buf) == 1
    drained = buf.drain()
    assert drained[0].text == "change of plan"
    assert drained[0].prompt_id == "inj-1"

    release.set()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert len(main_results) == 1
    assert main_results[0].status is TurnStatus.COMPLETED
    assert main_results[0].final_text == "done:main work"
    assert s.turn_count == 1
    assert s.phase is SessionPhase.IDLE
    s.close()


def test_handle_prompt_busy_without_kernel_raises(tmp_path: Path) -> None:
    """Without orchestration kernel, nested handle_prompt stays hard-busy."""

    class NestedRunner:
        def run(self, request: TurnRequest, *, session: Session) -> TurnResult:
            with pytest.raises(SessionBusyError):
                session.handle_prompt("nested")
            return TurnResult(status=TurnStatus.COMPLETED, final_text="ok")

    s = Session.create(tmp_path)
    s.bind_turn_runner(NestedRunner())
    r = s.handle_prompt("outer")
    assert r.status is TurnStatus.COMPLETED
    assert r.final_text == "ok"
    s.close()


def test_session_close_calls_kernel_close(tmp_path: Path) -> None:
    """Session.close must tear down kernel: memory + subagent + store."""

    class FakeMM:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self, *, timeout_s: float = 5.0) -> None:
            self.shutdown_calls += 1

    class FakeCoord:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self, wait: bool = False) -> None:
            self.shutdown_calls += 1

    class FakeStore:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    mm = FakeMM()
    coord = FakeCoord()
    store = FakeStore()
    kernel = RuntimeKernel(
        cwd=tmp_path,
        session_id="close-test",
        memory_manager=mm,
        subagent_coordinator=coord,
        session_store=store,
    )
    s = Session.create(
        tmp_path,
        extensions=SessionExtensions(kernel=kernel),
    )
    s.close()
    assert kernel.closed is True
    assert mm.shutdown_calls == 1
    assert coord.shutdown_calls == 1
    assert store.closed is True
