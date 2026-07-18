"""Agentic turn loop tests (scripted sampler, no live model)."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.session import Session, SessionExtensions, TurnStatus
from codedoggy.tools import ToolRegistryBuilder
from codedoggy.turn import (
    AgentTurnRunner,
    HookDecision,
    Message,
    Role,
    SampleResult,
    ToolCall,
    run_agent_loop,
)
from codedoggy.turn.hooks import HookContext, NoopHooks
from codedoggy.turn.types import ToolResultRecord


class ScriptedSampler:
    """Returns predetermined samples in order."""

    def __init__(self, script: list[SampleResult]) -> None:
        self.script = list(script)
        self.calls = 0
        self.seen_messages: list[list[Message]] = []

    def sample(self, messages: list[Message], tools) -> SampleResult:
        self.seen_messages.append(list(messages))
        if self.calls >= len(self.script):
            return SampleResult(content="(script exhausted)")
        out = self.script[self.calls]
        self.calls += 1
        return out


class RecordingHooks(NoopHooks):
    def __init__(self) -> None:
        self.tools: list[str] = []
        self.mutations: list[str] = []

    def after_tool(self, record: ToolResultRecord, ctx: HookContext) -> HookDecision | None:
        self.tools.append(record.call.name)
        return None

    def after_mutation(
        self, record: ToolResultRecord, ctx: HookContext
    ) -> HookDecision | None:
        if record.mutation:
            self.mutations.append(record.mutation.path)
        return HookDecision(append_observation="[quality: ok]")


def test_loop_final_answer_no_tools(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    sampler = ScriptedSampler([SampleResult(content="hello")])
    result = run_agent_loop(
        user_text="hi",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        max_turns=5,
    )
    assert result.completed
    assert result.final_text == "hello"
    assert result.rounds == 1
    assert result.tools_called == []
    assert result.messages[0].role is Role.USER
    assert result.messages[-1].role is Role.ASSISTANT


def test_loop_empty_final_after_tools_gets_fallback(tmp_path: Path) -> None:
    """Local models often end with empty content after tools — never silent ""."""
    (tmp_path / "a.txt").write_text("payload-here\n", encoding="utf-8")
    tools = ToolRegistryBuilder.new().finalize()
    sampler = ScriptedSampler(
        [
            SampleResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="read_file",
                        arguments={"target_file": "a.txt"},
                    )
                ],
            ),
            SampleResult(content=""),  # empty final
        ]
    )
    result = run_agent_loop(
        user_text="read a",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        max_turns=5,
    )
    assert result.completed
    assert result.final_text
    assert result.final_text.strip()
    assert "harness summary" in result.final_text.lower() or "read_file" in result.final_text


def test_loop_tool_then_final(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("line1\n", encoding="utf-8")
    tools = ToolRegistryBuilder.new().finalize()
    sampler = ScriptedSampler(
        [
            SampleResult(
                content="reading",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="read_file",
                        arguments={"target_file": "a.txt"},
                    )
                ],
            ),
            SampleResult(content="done"),
        ]
    )
    result = run_agent_loop(
        user_text="read a",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        max_turns=5,
    )
    assert result.completed
    assert result.final_text == "done"
    assert result.tools_called == ["read_file"]
    assert result.rounds == 2
    # Tool observation written back before second sample
    roles = [m.role for m in result.messages]
    assert Role.TOOL in roles
    tool_msg = next(m for m in result.messages if m.role is Role.TOOL)
    assert "line1" in (tool_msg.content or "")
    # Second sample saw the tool message
    assert any(m.role is Role.TOOL for m in sampler.seen_messages[1])


def test_loop_max_turns(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    # Always request another tool — never finish.
    infinite = SampleResult(
        content="again",
        tool_calls=[
            ToolCall(id="c", name="read_file", arguments={"target_file": "a.txt"})
        ],
    )
    sampler = ScriptedSampler([infinite, infinite, infinite])
    result = run_agent_loop(
        user_text="loop",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        max_turns=2,
    )
    assert not result.completed
    assert result.max_turns_reached
    assert result.rounds == 2
    assert sampler.calls == 2


def test_loop_max_turns_none_unlimited(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    sampler = ScriptedSampler(
        [
            SampleResult(
                content="t",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="read_file",
                        arguments={"target_file": "missing.txt"},
                    )
                ],
            ),
            SampleResult(content="ok after miss"),
        ]
    )
    result = run_agent_loop(
        user_text="x",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        max_turns=None,
    )
    assert result.completed
    assert result.final_text == "ok after miss"


def test_loop_tool_error_is_observation(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    sampler = ScriptedSampler(
        [
            SampleResult(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="read_file",
                        arguments={"target_file": "nope.txt"},
                    )
                ],
            ),
            SampleResult(content="handled"),
        ]
    )
    result = run_agent_loop(
        user_text="x",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
    )
    assert result.completed
    tool_msg = next(m for m in result.messages if m.role is Role.TOOL)
    assert "Error" in (tool_msg.content or "")
    assert "not_found" in (tool_msg.content or "")


def test_loop_cancel(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    # Cancel as soon as the first sample has completed (calls >= 1).
    # Checked at the top of the next loop iteration.
    sampler = ScriptedSampler(
        [
            SampleResult(
                content="t",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="list_dir",
                        arguments={"target_directory": "."},
                    )
                ],
            ),
            SampleResult(content="never"),
        ]
    )
    result = run_agent_loop(
        user_text="x",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        max_turns=5,
        is_cancelled=lambda: sampler.calls >= 1,
    )
    assert result.cancelled
    assert not result.completed
    assert sampler.calls == 1


def test_loop_mutation_hook(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    hooks = RecordingHooks()
    sampler = ScriptedSampler(
        [
            SampleResult(
                content="edit",
                tool_calls=[
                    ToolCall(
                        id="e1",
                        name="search_replace",
                        arguments={
                            "file_path": "n.txt",
                            "old_string": "",
                            "new_string": "hi",
                        },
                    )
                ],
            ),
            SampleResult(content="done"),
        ]
    )
    result = run_agent_loop(
        user_text="create",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        hooks=hooks,
    )
    assert result.completed
    assert (tmp_path / "n.txt").read_text(encoding="utf-8") == "hi"
    assert hooks.mutations == ["n.txt"]
    tool_msg = next(m for m in result.messages if m.role is Role.TOOL)
    assert "[quality: ok]" in (tool_msg.content or "")


def test_parse_tool_arguments_json_string() -> None:
    from codedoggy.turn.executor import parse_tool_arguments

    assert parse_tool_arguments('{"target_file": "a.txt"}') == {"target_file": "a.txt"}
    assert parse_tool_arguments({}) == {}
    assert parse_tool_arguments(None) == {}
    assert "_raw" in parse_tool_arguments("not-json{")


def test_agent_turn_runner_via_session(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    sampler = ScriptedSampler([SampleResult(content="from runner")])
    runner = AgentTurnRunner(sampler=sampler, tools=tools)
    s = Session.create(tmp_path, max_turns=3)
    s.bind_extensions(
        SessionExtensions(turn_runner=runner, tools=tools)
    )
    r = s.handle_prompt("hello")
    assert r.status is TurnStatus.COMPLETED
    assert r.final_text == "from runner"
    assert r.metadata.get("rounds") == 1
    s.close()


def test_agent_turn_runner_max_turns_status(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    step = SampleResult(
        content="t",
        tool_calls=[
            ToolCall(id="1", name="read_file", arguments={"target_file": "a.txt"})
        ],
    )
    sampler = ScriptedSampler([step, step, step])
    runner = AgentTurnRunner(sampler=sampler, tools=tools)
    s = Session.create(tmp_path, max_turns=1)
    s.bind_turn_runner(runner)
    r = s.handle_prompt("go")
    assert r.status is TurnStatus.MAX_TURNS_REACHED
    assert r.tools_called == ["read_file"]
    s.close()


def test_multi_tool_batch_writeback(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B\n", encoding="utf-8")
    tools = ToolRegistryBuilder.new().finalize()
    sampler = ScriptedSampler(
        [
            SampleResult(
                content="two reads",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="read_file",
                        arguments={"target_file": "a.txt"},
                    ),
                    ToolCall(
                        id="2",
                        name="read_file",
                        arguments={"target_file": "b.txt"},
                    ),
                ],
            ),
            SampleResult(content="both ok"),
        ]
    )
    result = run_agent_loop(
        user_text="read both",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
    )
    assert result.completed
    assert result.tools_called == ["read_file", "read_file"]
    tool_msgs = [m for m in result.messages if m.role is Role.TOOL]
    assert len(tool_msgs) == 2
    assert "A" in (tool_msgs[0].content or "")
    assert "B" in (tool_msgs[1].content or "")


def test_abort_after_tool_stops_loop_but_batch_already_dispatched(tmp_path: Path) -> None:
    """Grok path-lock batch: phase-2 runs approved tools first; hooks see results after.

    after_tool abort ends the *loop* (no next sample) and can pause further
    turns, but does not un-run tools already dispatched in the same batch.
    """
    tools = ToolRegistryBuilder.new().finalize()
    seen: list[str] = []

    class AbortAfterFirstWriteback(NoopHooks):
        def after_tool(
            self, record: ToolResultRecord, ctx: HookContext
        ) -> HookDecision | None:
            seen.append(record.call.id)
            if record.call.id == "1":
                return HookDecision(abort=True, abort_reason="stop after first writeback")
            return None

    sampler = ScriptedSampler(
        [
            SampleResult(
                content="batch",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="search_replace",
                        arguments={
                            "file_path": "first.txt",
                            "old_string": "",
                            "new_string": "one",
                        },
                    ),
                    ToolCall(
                        id="2",
                        name="search_replace",
                        arguments={
                            "file_path": "second.txt",
                            "old_string": "",
                            "new_string": "two",
                        },
                    ),
                ],
            ),
            SampleResult(content="should not sample"),
        ]
    )
    result = run_agent_loop(
        user_text="edit two",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        hooks=AbortAfterFirstWriteback(),
    )
    assert result.aborted
    assert result.tools_called == ["search_replace", "search_replace"]
    assert (tmp_path / "first.txt").is_file()
    assert (tmp_path / "second.txt").is_file()  # phase-2 already ran both
    tool_msgs = [m for m in result.messages if m.role is Role.TOOL]
    assert len(tool_msgs) == 2
    assert tool_msgs[0].tool_call_id == "1"
    assert tool_msgs[1].tool_call_id == "2"
    asst = next(m for m in result.messages if m.role is Role.ASSISTANT)
    assert asst.tool_calls is not None and len(asst.tool_calls) == 2
    # No further sample after abort
    assert sampler.calls == 1


def test_unknown_tool_observation(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    sampler = ScriptedSampler(
        [
            SampleResult(
                content=None,
                tool_calls=[ToolCall(id="x", name="no_such_tool", arguments={})],
            ),
            SampleResult(content="recovered"),
        ]
    )
    result = run_agent_loop(
        user_text="x",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
    )
    assert result.completed
    tool_msg = next(m for m in result.messages if m.role is Role.TOOL)
    assert "not_found" in (tool_msg.content or "")
