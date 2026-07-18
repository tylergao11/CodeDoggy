"""Regression tests for RuntimeKernel + audit Critical/High fixes."""

from __future__ import annotations

from pathlib import Path

from codedoggy.context.select import sanitize_tool_pairs
from codedoggy.tools.policy import WorkspacePolicy
from codedoggy.tools.util.write_detect import detect_shell_write_paths
from codedoggy.turn.types import Message, Role, ToolCall


def test_policy_case_insensitive_deny(tmp_path: Path) -> None:
    pol = WorkspacePolicy(cwd=tmp_path)
    d1 = pol.check_write(".GIT/config")
    d2 = pol.check_write(".Env")
    d3 = pol.check_write("Node_Modules/x")
    assert not d1.allowed
    assert not d2.allowed
    assert not d3.allowed


def test_detect_remove_item_and_literal_path() -> None:
    paths = detect_shell_write_paths(
        "Set-Content -LiteralPath .git\\config -Value x"
    )
    assert any("git" in p.lower() and "config" in p.lower() for p in paths)
    paths2 = detect_shell_write_paths("Remove-Item .git\\config")
    assert any("git" in p.lower() for p in paths2)
    paths3 = detect_shell_write_paths("del .env")
    assert any("env" in p.lower() for p in paths3)


def test_shell_policy_blocks_protected_via_detector(tmp_path: Path) -> None:
    pol = WorkspacePolicy(cwd=tmp_path)
    d = pol.check_shell("Remove-Item .git\\config -Force")
    assert not d.allowed


def test_sanitize_fills_missing_tool_results() -> None:
    msgs = [
        Message(role=Role.USER, content="hi"),
        Message(
            role=Role.ASSISTANT,
            content="ok",
            tool_calls=[
                ToolCall(id="a", name="t1", arguments={}),
                ToolCall(id="b", name="t2", arguments={}),
            ],
        ),
        Message(role=Role.TOOL, content="done a", tool_call_id="a", name="t1"),
    ]
    out = sanitize_tool_pairs(msgs)
    tool_ids = [m.tool_call_id for m in out if m.role is Role.TOOL]
    assert "a" in tool_ids
    assert "b" in tool_ids
    b_msg = next(m for m in out if m.tool_call_id == "b")
    assert "never executed" in (b_msg.content or "") or "cancel" in (
        b_msg.content or ""
    ).lower()


def test_set_goal_updates_config_and_system(tmp_path: Path) -> None:
    from codedoggy.bootstrap import build_session
    from codedoggy.model import CompletionResult
    from tests.test_bootstrap import ScriptClient

    main = ScriptClient([CompletionResult(content="ok", model="m")], name="m")
    s = build_session(
        tmp_path,
        goal="old goal",
        main_client=main,
        audit_client=main,
        enable_audit=False,
        enable_memory=False,
        enable_session_store=False,
        enable_graph=False,
    )
    try:
        s.set_goal("new goal")
        assert s.goal == "new goal"
        assert s.config.goal == "new goal"
        sp = s.extensions.turn_runner.system_prompt  # type: ignore[union-attr]
        assert "new goal" in (sp or "")
        assert "old goal" not in (sp or "")
    finally:
        s.close()


def test_hydrate_session_restores_live(tmp_path: Path) -> None:
    from codedoggy.bootstrap import build_session
    from codedoggy.memory.session_store import SessionStore
    from codedoggy.model import CompletionResult
    from tests.test_bootstrap import ScriptClient

    db = tmp_path / "s.db"
    store = SessionStore(db)
    sid = "hydrate-test-1"
    store.ensure_session(sid, cwd=str(tmp_path), goal="g")
    store.append_message(sid, "user", "hello from archive")
    store.append_message(sid, "assistant", "hi back")
    store.close()

    main = ScriptClient([CompletionResult(content="ok", model="m")], name="m")
    s = build_session(
        tmp_path,
        session_id=sid,
        session_db=db,
        main_client=main,
        audit_client=main,
        enable_audit=False,
        enable_memory=False,
        enable_graph=False,
    )
    try:
        live = s.extensions.turn_runner.live_messages  # type: ignore[union-attr]
        texts = [m.content for m in live]
        assert any("hello from archive" in (t or "") for t in texts)
    finally:
        s.close()


def test_diff_hint_keeps_after_on_large_before() -> None:
    from codedoggy.audit.types import MutationEvent

    before = "B" * 10_000
    after = "UNIQUE_AFTER_TAIL_XYZ"
    ev = MutationEvent(
        path="f.py",
        tool_name="search_replace",
        call_id="1",
        before=before,
        after=after,
        is_create=False,
    )
    hint = ev.unified_diff_hint(max_chars=500)
    # Real unified diff (Grok hunk spirit) — after content must remain visible
    assert "UNIQUE_AFTER_TAIL_XYZ" in hint
    assert "+++" in hint or "@@" in hint


def test_central_gate_blocks_write_without_tool_checking(tmp_path: Path) -> None:
    """Policy enforced at FinalizedToolset.call even if tool omits check."""
    from codedoggy.tools import ToolCallContext
    from codedoggy.tools.registry import ToolRegistryBuilder
    from codedoggy.tools.runtime import ToolError

    tools = ToolRegistryBuilder.new().finalize()
    pol = WorkspacePolicy(cwd=tmp_path)
    ctx = ToolCallContext(cwd=tmp_path, extra={"policy": pol})
    try:
        tools.call(
            "search_replace",
            {
                "file_path": ".env",
                "old_string": "",
                "new_string": "SECRET=1",
            },
            ctx,
        )
        raise AssertionError("expected policy deny")
    except ToolError as e:
        assert e.code == "policy_denied" or "denied" in e.message.lower()
