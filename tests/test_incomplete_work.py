"""Incomplete-work gate — premature COMPLETED regressions."""

from __future__ import annotations

from pathlib import Path

from codedoggy.orchestration.incomplete_work import (
    incomplete_work_reasons,
    open_todo_ids,
)
from codedoggy.orchestration.plan_first import PlanFirstGate
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


def test_reasons_plan_first_and_todos() -> None:
    st = TodoState()
    st.push("t1", TodoItem(content="do", status="pending"))
    gate = PlanFirstGate(require_plan_artifact=True)
    reasons = incomplete_work_reasons(
        {"todo_state": st, "plan_first_gate": gate}
    )
    assert any("todos" in r for r in reasons)
    assert any("record_plan" in r for r in reasons)


def test_loop_nudges_instead_of_completed(tmp_path: Path) -> None:
    from codedoggy.tools import ToolRegistryBuilder

    tools = ToolRegistryBuilder.new().finalize()
    todos = TodoState()
    todos.push("work", TodoItem(content="finish feature", status="pending"))
    gate = PlanFirstGate(require_plan_artifact=False)
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
        tool_extra={"todo_state": todos, "plan_first_gate": gate},
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
