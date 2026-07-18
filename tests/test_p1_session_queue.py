"""Attack tests: Session Actor-lite prompt queue vs interjection.

Contract:
  - concurrent handle_prompt mid-turn → interjection only (QUEUED, no full re-run)
  - enqueue_prompt mid-turn → prompt_queue only (no interject buffer), drained
    as a separate full turn after the main turn completes
  - enqueue while IDLE does not auto-start a turn
"""

from __future__ import annotations

import threading
from pathlib import Path

from codedoggy.orchestration.prompt_queue import InterjectionBuffer, PromptQueue
from codedoggy.session import Session, SessionExtensions, SessionPhase, TurnStatus
from codedoggy.session.kernel import RuntimeKernel
from codedoggy.session.types import TurnRequest, TurnResult


def test_enqueue_while_busy_lands_in_queue_not_interjection(tmp_path: Path) -> None:
    """While busy, enqueue_prompt must not touch interjection_buffer; drains after turn."""
    entered = threading.Event()
    release = threading.Event()
    seen: list[str] = []

    class BlockingRunner:
        def run(self, request: TurnRequest, *, session: Session) -> TurnResult:
            seen.append(request.text)
            entered.set()
            assert release.wait(timeout=5.0), "release timed out"
            return TurnResult(
                status=TurnStatus.COMPLETED,
                final_text=f"done:{request.text}",
                tools_called=[],
            )

    buf = InterjectionBuffer()
    pq = PromptQueue()
    runner = BlockingRunner()
    kernel = RuntimeKernel(
        cwd=tmp_path,
        session_id="enq-busy",
        turn_runner=runner,
        interjection_buffer=buf,
        prompt_queue=pq,
    )
    s = Session.create(
        tmp_path,
        extensions=SessionExtensions(turn_runner=runner, kernel=kernel),
    )

    main_results: list[TurnResult] = []

    def _main() -> None:
        main_results.append(s.handle_prompt("main work", prompt_id="main"))

    t = threading.Thread(target=_main, name="main-turn")
    t.start()
    assert entered.wait(timeout=5.0)
    assert s.phase is SessionPhase.TURN_RUNNING

    n = s.enqueue_prompt("queued full turn", prompt_id="q1")
    assert n == 1
    assert len(pq) == 1
    assert len(buf) == 0  # must NOT interject
    peek = pq.peek()
    assert peek is not None
    assert peek.text == "queued full turn"
    assert peek.prompt_id == "q1"
    # Still only the main turn; queue item not started yet
    assert s.turn_count == 0
    assert seen == ["main work"]

    release.set()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert len(main_results) == 1
    assert main_results[0].status is TurnStatus.COMPLETED
    assert main_results[0].final_text == "done:main work"
    # Drain ran the parked prompt as a second full turn
    assert "queued full turn" in seen
    assert seen == ["main work", "queued full turn"]
    assert s.turn_count == 2
    assert s.phase is SessionPhase.IDLE
    assert len(pq) == 0
    assert len(buf) == 0
    s.close()


def test_mid_turn_handle_prompt_interject_only_no_double_full_run(
    tmp_path: Path,
) -> None:
    """Mid-turn handle_prompt → QUEUED + interjection; same text must not full-run twice."""
    entered = threading.Event()
    release = threading.Event()
    seen: list[str] = []

    class BlockingRunner:
        def run(self, request: TurnRequest, *, session: Session) -> TurnResult:
            seen.append(request.text)
            entered.set()
            assert release.wait(timeout=5.0), "release timed out"
            return TurnResult(
                status=TurnStatus.COMPLETED,
                final_text=f"done:{request.text}",
                tools_called=[],
            )

    buf = InterjectionBuffer()
    pq = PromptQueue()
    runner = BlockingRunner()
    kernel = RuntimeKernel(
        cwd=tmp_path,
        session_id="inj-only",
        turn_runner=runner,
        interjection_buffer=buf,
        prompt_queue=pq,
    )
    s = Session.create(
        tmp_path,
        extensions=SessionExtensions(turn_runner=runner, kernel=kernel),
    )

    main_results: list[TurnResult] = []

    def _main() -> None:
        main_results.append(s.handle_prompt("main work"))

    t = threading.Thread(target=_main, name="main-turn")
    t.start()
    assert entered.wait(timeout=5.0)

    soft = s.handle_prompt("side note", prompt_id="inj-1")
    assert soft.status is TurnStatus.QUEUED
    assert soft.status is not TurnStatus.COMPLETED
    assert soft.metadata.get("interjected") is True
    assert soft.metadata.get("queued_interjection") is True
    assert len(buf) == 1
    assert len(pq) == 0  # must NOT land in prompt_queue
    assert s.turn_count == 0

    release.set()
    t.join(timeout=5.0)
    assert not t.is_alive()
    # Only main full turn — interjected text must not re-run as full prompt
    assert seen == ["main work"]
    assert s.turn_count == 1
    assert s.phase is SessionPhase.IDLE
    assert len(pq) == 0
    # BlockingRunner does not drain interjections — entry remains until explicit drain
    drained = buf.drain()
    assert len(drained) == 1
    assert drained[0].text == "side note"
    assert drained[0].prompt_id == "inj-1"
    s.close()


def test_enqueue_while_idle_does_not_auto_start(tmp_path: Path) -> None:
    """enqueue_prompt while IDLE parks only; host must call handle_prompt."""
    seen: list[str] = []

    class CountingRunner:
        def run(self, request: TurnRequest, *, session: Session) -> TurnResult:
            seen.append(request.text)
            return TurnResult(
                status=TurnStatus.COMPLETED,
                final_text=f"echo:{request.text}",
            )

    runner = CountingRunner()
    # No prompt_queue pre-bound — enqueue must create it on kernel
    kernel = RuntimeKernel(
        cwd=tmp_path,
        session_id="enq-idle",
        turn_runner=runner,
        interjection_buffer=InterjectionBuffer(),
    )
    assert kernel.prompt_queue is None
    s = Session.create(
        tmp_path,
        extensions=SessionExtensions(turn_runner=runner, kernel=kernel),
    )

    n = s.enqueue_prompt("parked", prompt_id="p0")
    assert n == 1
    assert kernel.prompt_queue is not None
    assert len(kernel.prompt_queue) == 1
    assert seen == []
    assert s.turn_count == 0
    assert s.phase is SessionPhase.IDLE

    # Host starts a turn; after it ends, drain processes the parked item
    r = s.handle_prompt("host start")
    assert r.status is TurnStatus.COMPLETED
    assert seen == ["host start", "parked"]
    assert s.turn_count == 2
    assert len(kernel.prompt_queue) == 0
    s.close()


def test_enqueue_creates_queue_when_missing(tmp_path: Path) -> None:
    """enqueue_prompt creates kernel.prompt_queue if absent."""
    kernel = RuntimeKernel(cwd=tmp_path, session_id="create-q")
    assert kernel.prompt_queue is None
    s = Session.create(
        tmp_path,
        extensions=SessionExtensions(kernel=kernel),
    )
    n = s.enqueue_prompt("x")
    assert n == 1
    assert kernel.prompt_queue is not None
    assert len(kernel.prompt_queue) == 1
    s.close()
