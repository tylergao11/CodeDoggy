"""Incomplete-work gate — premature COMPLETED regressions."""

from __future__ import annotations

from pathlib import Path

from codedoggy.orchestration.incomplete_work import (
    incomplete_work_reasons,
    open_todo_ids,
    running_subagent_ids,
)
from codedoggy.orchestration.subagent import SubagentSnapshot
from codedoggy.tools.grok_build.todo_logic import TodoItem, TodoState
from codedoggy.turn.loop import run_agent_loop
from codedoggy.turn.types import Message, Role, SampleResult, ToolCall


class _Scripted:
    def __init__(self, script: list[SampleResult]) -> None:
        self.script = list(script)
        self.i = 0

    def sample(self, messages, tools):
        if self.i >= len(self.script):
            return SampleResult(content="(done)")
        out = self.script[self.i]
        self.i += 1
        return out


def test_open_todo_ids() -> None:
    st = TodoState()
    st.push("a", TodoItem(content="x", status="pending"))
    st.push("b", TodoItem(content="y", status="completed"))
    st.push("c", TodoItem(content="z", status="in_progress"))
    assert open_todo_ids(st) == ["a", "c"]


def test_main_incomplete_ignores_child_only_todo_bag() -> None:
    """MAIN gate uses kernel.todo_state; stray child list on bag is not enough alone.

    When kernel has no open MAIN todos, bag pollution from a child checklist
    must not block MAIN if kernel.todo_state is empty completed-only.
    """
    main = TodoState()
    main.push("m", TodoItem(content="done", status="completed"))
    child = TodoState()
    child.push("c", TodoItem(content="child open", status="pending"))

    class _K:
        todo_state = main
        session_id = "parent"
        task_manager = None

    # MAIN path: kernel has no open todos → no todo reason even if bag has child list
    reasons = incomplete_work_reasons(
        {
            "kernel": _K(),
            "todo_state": child,  # should be ignored for MAIN when kernel present
            "session_id": "parent",
        }
    )
    assert not any("todo" in r.lower() or "待办" in r or "未完成 todo" in r for r in reasons)

    # Child path: only child todos count
    reasons_child = incomplete_work_reasons(
        {
            "is_subagent": True,
            "todo_state": child,
            "kernel": _K(),
            "session_id": "parent:child",
        }
    )
    assert any("未完成 todo" in r and "c" in r for r in reasons_child)


def test_child_incomplete_does_not_wait_on_sibling_subagents() -> None:
    class _Coord:
        def list_for_parent(self, _sid: str) -> list[SubagentSnapshot]:
            return [
                SubagentSnapshot(
                    subagent_id="sib",
                    subagent_type="general-purpose",
                    status="running",
                    description="other",
                )
            ]

    reasons = incomplete_work_reasons(
        {
            "is_subagent": True,
            "todo_state": TodoState(),
            "subagent_coordinator": _Coord(),
            "session_id": "parent:me",
            "parent_session_id": "parent",
        }
    )
    assert not any("子 agent 仍在运行" in r or "subagents still running" in r for r in reasons)


def test_running_subagent_ids_uses_subagent_id_field() -> None:
    """Regression: SubagentSnapshot has subagent_id, not id — empty meant no gate."""

    class _Coord:
        def list_for_parent(self, _sid: str) -> list[SubagentSnapshot]:
            return [
                SubagentSnapshot(
                    subagent_id="child-a",
                    subagent_type="general-purpose",
                    status="running",
                    description="slice",
                ),
                SubagentSnapshot(
                    subagent_id="child-b",
                    subagent_type="explore",
                    status="completed",
                    description="done",
                ),
            ]

    extra = {
        "subagent_coordinator": _Coord(),
        "session_id": "parent-sess",
    }
    assert running_subagent_ids(extra) == ["child-a"]
    reasons = incomplete_work_reasons(extra)
    assert any("子 agent 仍在运行" in r and "child-a" in r for r in reasons)


def test_prose_stop_with_running_subagent_is_nudged(tmp_path: Path) -> None:
    """wait=false style: prose stop must not COMPLETE while children run."""
    from codedoggy.tools import ToolRegistryBuilder

    class _Coord:
        def __init__(self) -> None:
            self.n = 0

        def list_for_parent(self, _sid: str) -> list[SubagentSnapshot]:
            self.n += 1
            if self.n <= 1:
                return [
                    SubagentSnapshot(
                        subagent_id="bg-1",
                        subagent_type="general-purpose",
                        status="running",
                        description="fan-out",
                    )
                ]
            return []

    tools = ToolRegistryBuilder.new().finalize()
    sampler = _Scripted(
        [
            SampleResult(content="Children are running, I am done."),
            SampleResult(content="Actually waiting done now."),
        ]
    )
    result = run_agent_loop(
        user_text="fan out and finish",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        max_turns=6,
        session_id="s-parallel",
        tool_extra={"subagent_coordinator": _Coord()},
    )
    assert result.completed is True
    nudge_msgs = [
        m
        for m in result.messages
        if m.role is Role.USER and "incomplete_work" in (m.content or "")
    ]
    assert nudge_msgs, "expected incomplete_work when subagents still running"
    assert any(
        "子 agent" in (m.content or "") or "subagents" in (m.content or "")
        for m in nudge_msgs
    )


def test_prose_stop_with_no_open_work_completes(tmp_path: Path) -> None:
    from codedoggy.tools import ToolRegistryBuilder

    tools = ToolRegistryBuilder.new().finalize()
    sampler = _Scripted([SampleResult(content="Here is a short answer.")])
    result = run_agent_loop(
        user_text="what is 2+2?",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        max_turns=3,
        session_id="s-compose",
        tool_extra={},
    )
    assert result.completed is True
    assert result.rounds == 1
    nudge_msgs = [
        m
        for m in result.messages
        if m.role is Role.USER and "incomplete_work" in (m.content or "")
    ]
    assert not nudge_msgs


def test_loop_nudges_instead_of_completed(tmp_path: Path) -> None:
    from codedoggy.tools import ToolRegistryBuilder

    tools = ToolRegistryBuilder.new().finalize()
    todos = TodoState()
    todos.push("work", TodoItem(content="finish feature", status="pending"))
    sampler = _Scripted(
        [
            SampleResult(content="I think we are done."),  # would early-complete
            SampleResult(
                content="updating todos",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="todo_write",
                        arguments={
                            "merge": True,
                            "todos": [
                                {
                                    "id": "work",
                                    "content": "finish feature",
                                    "status": "completed",
                                }
                            ],
                        },
                    )
                ],
            ),
            SampleResult(content="Now finished."),
        ]
    )
    result = run_agent_loop(
        user_text="do the feature",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        max_turns=8,
        session_id="s1",
        tool_extra={"todo_state": todos},
    )
    assert result.completed is True
    assert result.rounds >= 2
    # First prose stop must have been nudged (USER incomplete_work in messages)
    nudge_msgs = [
        m
        for m in result.messages
        if m.role is Role.USER and "incomplete_work" in (m.content or "")
    ]
    assert nudge_msgs, "expected incomplete_work nudge before final complete"


def test_update_goal_completed_refuses_open_work(tmp_path: Path) -> None:
    from codedoggy.tools.builtins.update_goal import UpdateGoalTool
    from codedoggy.tools.runtime import ToolCallContext

    todos = TodoState()
    todos.push("work", TodoItem(content="still open", status="pending"))
    tool = UpdateGoalTool()
    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={"todo_state": todos},
    )
    out = tool.run(ctx, {"completed": True, "message": "done"})
    assert "incomplete_work" in out
    open_ids = [tid for tid, item in todos.todo_items_with_ids() if item.status == "pending"]
    assert "work" in open_ids
