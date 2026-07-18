"""shadow: trajectory, soft feedback, session goal, memory select interface."""

from __future__ import annotations

from pathlib import Path

from codedoggy.audit import (
    AuditServices,
    CuratedMemorySelector,
    MemorySelectRequest,
    MutationEvent,
    MutationTrajectory,
    NoopMemorySelector,
    ScriptedAuditor,
)
from codedoggy.memory import MemoryStore
from codedoggy.session import Session, SessionExtensions
from codedoggy.tools import ToolRegistryBuilder
from codedoggy.turn import (
    AgentTurnRunner,
    Role,
    SampleResult,
    ToolCall,
    run_agent_loop,
)
from codedoggy.turn.types import Message


class ScriptedSampler:
    def __init__(self, script: list[SampleResult]) -> None:
        self.script = list(script)
        self.n = 0
        self.seen: list[list[Message]] = []

    def sample(self, messages, tools):
        self.seen.append(list(messages))
        if self.n >= len(self.script):
            return SampleResult(content="done")
        out = self.script[self.n]
        self.n += 1
        return out


def test_session_goal_field(tmp_path: Path) -> None:
    s = Session.create(tmp_path, goal="Fix the login bug only")
    assert s.goal == "Fix the login bug only"
    s.set_goal("Also add tests")
    assert s.goal == "Also add tests"
    s.close()


def test_search_replace_emits_before_after(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    traj = MutationTrajectory()
    from codedoggy.audit import PassThroughAuditor

    audit = AuditServices.create(auditor=PassThroughAuditor(), trajectory=traj)

    sampler = ScriptedSampler(
        [
            SampleResult(
                content="edit",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="search_replace",
                        arguments={
                            "file_path": "a.txt",
                            "old_string": "hello",
                            "new_string": "hi",
                        },
                    )
                ],
            ),
            SampleResult(content="done"),
        ]
    )
    from codedoggy.audit.hooks import ResidentAuditHooks

    result = run_agent_loop(
        user_text="edit",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        hooks=ResidentAuditHooks(audit),
        session=Session.create(tmp_path, goal="rename greeting"),
    )
    assert result.completed
    assert len(traj) == 1
    ev = traj.events()[0]
    assert ev.path == "a.txt"
    assert "hello" in (ev.before or "")
    assert "hi" in (ev.after or "")
    assert ev.goal_snapshot == "rename greeting"


def test_pass_is_silent(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    traj = MutationTrajectory()
    from codedoggy.audit import PassThroughAuditor
    from codedoggy.audit.hooks import ResidentAuditHooks

    audit = AuditServices.create(auditor=PassThroughAuditor(), trajectory=traj)
    sampler = ScriptedSampler(
        [
            SampleResult(
                content="c",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="search_replace",
                        arguments={
                            "file_path": "n.txt",
                            "old_string": "",
                            "new_string": "ok",
                        },
                    )
                ],
            ),
            SampleResult(content="done"),
        ]
    )
    result = run_agent_loop(
        user_text="x",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        hooks=ResidentAuditHooks(audit),
        session=Session.create(tmp_path, goal="write ok file"),
    )
    tool_msg = next(m for m in result.messages if m.role is Role.TOOL)
    assert "shadow" not in (tool_msg.content or "")


def test_fail_important_deferred_not_mid_tool(tmp_path: Path) -> None:
    """important findings are not mid-turn red cards; flushed at turn end."""
    tools = ToolRegistryBuilder.new().finalize()
    traj = MutationTrajectory()
    auditor = ScriptedAuditor(rules=[("bad.txt", "This path is off-goal")])
    audit = AuditServices.create(auditor=auditor, trajectory=traj)
    from codedoggy.audit.hooks import ResidentAuditHooks

    sampler = ScriptedSampler(
        [
            SampleResult(
                content="c",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="search_replace",
                        arguments={
                            "file_path": "bad.txt",
                            "old_string": "",
                            "new_string": "nope",
                        },
                    )
                ],
            ),
            SampleResult(content="rethinking"),
        ]
    )
    result = run_agent_loop(
        user_text="x",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        hooks=ResidentAuditHooks(audit),
        session=Session.create(tmp_path, goal="only touch good.txt"),
    )
    tool_msg = next(m for m in result.messages if m.role is Role.TOOL)
    assert "P0" not in (tool_msg.content or "")
    assert "off-goal" not in (tool_msg.content or "")
    assert result.completed
    assert "off-goal" in result.metadata.get("audit_deferred", "")
    assert "off-goal" in (result.final_text or "")
    # Non-P0 closed loop: deferred note lands in transcript for SessionStore/FTS.
    assert any(
        m.role is Role.USER
        and "end-of-turn notes" in (m.content or "")
        and "off-goal" in (m.content or "")
        for m in result.messages
    )


def test_memory_selector_interface(tmp_path: Path) -> None:
    store = MemoryStore(memory_dir=tmp_path / "mem")
    store.load_from_disk()
    store.add("memory", "Project prefers small diffs")
    store.load_from_disk()
    sel = CuratedMemorySelector(store)
    ev = MutationEvent(
        path="x.py",
        tool_name="search_replace",
        call_id="c",
        after="print(1)",
        is_create=True,
        goal_snapshot="keep diffs small",
    )
    res = sel.select(
        MemorySelectRequest(
            goal="keep diffs small",
            mutation=ev,
            trajectory_summary="(none)",
        )
    )
    assert res.curated_blocks
    assert "small diffs" in res.combined_text()
    # Extension point for Hermes remains empty until wired
    assert res.session_hits == []
    assert res.provider_hits == []


def test_session_auto_wires_audit_hooks(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    traj = MutationTrajectory()
    auditor = ScriptedAuditor(rules=[("x.txt", "stop and rethink")])
    audit = AuditServices.create(auditor=auditor, trajectory=traj)
    sampler = ScriptedSampler(
        [
            SampleResult(
                content="c",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="search_replace",
                        arguments={
                            "file_path": "x.txt",
                            "old_string": "",
                            "new_string": "1",
                        },
                    )
                ],
            ),
            SampleResult(content="fixed"),
        ]
    )
    runner = AgentTurnRunner(sampler=sampler, tools=tools)
    s = Session.create(tmp_path, goal="avoid x.txt")
    s.bind_extensions(
        SessionExtensions(turn_runner=runner, tools=tools, audit=audit)
    )
    r = s.handle_prompt("write")
    assert r.status.value == "completed"
    assert len(traj) == 1
    s.close()


def test_noop_memory_selector() -> None:
    sel = NoopMemorySelector()
    res = sel.select(
        MemorySelectRequest(
            goal=None,
            mutation=MutationEvent(path="a", tool_name="t", call_id="1"),
            trajectory_summary="",
        )
    )
    assert res.combined_text() == ""
