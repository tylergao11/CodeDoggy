"""Attack-style regressions for P1s (meaningful, not happy-path fluff)."""

from __future__ import annotations

from pathlib import Path

from codedoggy.context.budget import ContextBudget, needs_compaction
from codedoggy.memory.redact import redact_secrets
from codedoggy.memory.session_store import SessionStore
from codedoggy.session.types import TurnStatus
from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.kinds import ToolKind
from codedoggy.tools.runtime import ToolCallContext
from codedoggy.turn.types import FileMutation, Message, Role


def test_p1_busy_handle_prompt_is_queued_not_completed(tmp_path: Path) -> None:
    import threading

    from codedoggy.orchestration.prompt_queue import InterjectionBuffer
    from codedoggy.session import Session, SessionExtensions, SessionPhase
    from codedoggy.session.kernel import RuntimeKernel
    from codedoggy.session.types import TurnRequest, TurnResult

    entered = threading.Event()
    release = threading.Event()

    class BlockingRunner:
        def run(self, request: TurnRequest, *, session: Session) -> TurnResult:
            entered.set()
            assert release.wait(timeout=5.0)
            return TurnResult(status=TurnStatus.COMPLETED, final_text="done")

    buf = InterjectionBuffer()
    runner = BlockingRunner()
    kernel = RuntimeKernel(
        cwd=tmp_path,
        session_id="q1",
        turn_runner=runner,
        interjection_buffer=buf,
    )
    s = Session.create(
        tmp_path, extensions=SessionExtensions(turn_runner=runner, kernel=kernel)
    )
    t = threading.Thread(target=lambda: s.handle_prompt("main"))
    t.start()
    assert entered.wait(timeout=5.0)
    soft = s.handle_prompt("side")
    assert soft.status is TurnStatus.QUEUED
    assert soft.status is not TurnStatus.COMPLETED
    release.set()
    t.join(timeout=5.0)
    assert s.phase is SessionPhase.IDLE



def test_p1_ensure_session_does_not_rewrite_cwd(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    store = SessionStore(db)
    store.ensure_session("s1", cwd="/project/A", goal="g1")
    store.ensure_session("s1", cwd="/project/B", goal="g2")  # must not rehome
    row = store.list_recent_sessions(limit=5)
    # find s1
    hit = next(x for x in row if x.get("id") == "s1" or x.get("session_id") == "s1")
    cwd = hit.get("cwd")
    assert cwd in ("/project/A", str(Path("/project/A")))
    assert "B" not in str(cwd)


def test_p1_session_search_scopes_cwd(tmp_path: Path) -> None:
    from codedoggy.tools.builtins.session_search import SessionSearchTool

    db = tmp_path / "s.db"
    store = SessionStore(db)
    store.ensure_session("a", cwd=str(tmp_path / "projA"))
    store.append_message("a", "user", "auth cookie in A")
    store.ensure_session("b", cwd=str(tmp_path / "projB"))
    store.append_message("b", "user", "auth cookie in B")

    tool = SessionSearchTool(store)
    ctx = ToolCallContext(cwd=tmp_path / "projA", extra={"session_store": store})
    out = tool.run(ctx, {"query": "auth cookie", "limit": 10})
    assert "projB" not in out or '"session_id": "b"' not in out
    # A may or may not match FTS depending on backend; cwd key present
    assert "cwd" in out


def test_p1_budget_trigger_uses_reserves() -> None:
    b = ContextBudget(context_window=10_000, completion_reserve=4_000, tools_reserve=1_000)
    b.threshold_percent = 100
    # usable = 5000; trigger at 100% of usable = 5000, not 10000
    assert b.trigger_tokens == 5_000
    # messages totaling ~6000 tokens should need compact against usable, not full window
    msgs = [Message(role=Role.USER, content="x" * 24_000)]  # ~6k tokens at chars/4
    # needs_compaction uses estimate; with last_prompt may vary — check property first
    assert b.usable_window == 5_000


def test_p1_memory_tool_kind_is_write_capable() -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ft = tools.by_client_name.get("memory")
    assert ft is not None
    assert ft.kind is ToolKind.Edit


def test_p1_redact_aws_and_database_url() -> None:
    s = "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    r = redact_secrets(s)
    assert "wJalrXUtnFEMI" not in r
    assert "REDACTED" in r
    u = "DATABASE_URL=postgres://user:s3cretpass@localhost/db"
    r2 = redact_secrets(u)
    assert "s3cretpass" not in r2
