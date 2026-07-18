"""Session lifecycle tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.session import (
    Session,
    SessionConfig,
    SessionPhase,
    TurnRequest,
    TurnResult,
    TurnStatus,
)
from codedoggy.session.session import SessionClosedError


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
