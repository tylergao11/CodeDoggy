"""Live activity board — boss-visible tool status (effect layer)."""

from __future__ import annotations

from codedoggy.tui.activity import (
    LiveActivityBoard,
    format_tool_done,
    format_tools_running,
    looks_like_tool_failure,
)
from codedoggy.turn.types import Message, Role, ToolCall


def test_format_helpers() -> None:
    assert "shell" in format_tools_running(["shell"])
    assert "调用中" in format_tools_running(["shell", "read_file"])
    assert format_tool_done("grep", failed=False, still_open=[]).startswith("✓")
    assert format_tool_done("grep", failed=True, still_open=[]).startswith("✗")
    assert looks_like_tool_failure("Error: boom")
    assert not looks_like_tool_failure("ok result")


def test_observe_tool_call_then_result() -> None:
    board = LiveActivityBoard()
    tid, aid = "task_001", "task_001:main"
    board.observe(
        tid,
        aid,
        Message(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "ls"})],
        ),
    )
    assert board.line(tid, aid).startswith("→ shell")
    assert "调用中" in board.main_line(tid)

    board.observe(
        tid,
        aid,
        Message(role=Role.TOOL, name="shell", tool_call_id="c1", content="file.txt\n"),
    )
    assert board.line(tid, aid).startswith("✓ shell")
    assert "完成" in board.line(tid, aid)


def test_observe_parallel_tools() -> None:
    board = LiveActivityBoard()
    tid, aid = "t", "t:main"
    board.observe(
        tid,
        aid,
        Message(
            role=Role.ASSISTANT,
            tool_calls=[
                ToolCall(id="a", name="read_file", arguments={}),
                ToolCall(id="b", name="grep", arguments={}),
            ],
        ),
    )
    line = board.line(tid, aid)
    assert "read_file" in line and "grep" in line
    board.observe(
        tid,
        aid,
        Message(role=Role.TOOL, tool_call_id="a", content="ok"),
    )
    assert "仍在" in board.line(tid, aid) or "grep" in board.line(tid, aid)


def test_assistant_text_without_tools() -> None:
    board = LiveActivityBoard()
    board.observe(
        "t",
        "t:main",
        Message(role=Role.ASSISTANT, content="我先检查登录链路。"),
    )
    assert board.line("t", "t:main").startswith("…")
    assert "登录" in board.line("t", "t:main")


def test_clear_task() -> None:
    board = LiveActivityBoard()
    board.observe(
        "t1",
        "t1:main",
        Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="1", name="shell", arguments={})],
        ),
    )
    board.clear_task("t1")
    assert board.line("t1", "t1:main") == ""
