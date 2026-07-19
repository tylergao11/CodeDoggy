"""P1: subagent mid-run live message push (same tier as MAIN archive)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from codedoggy.orchestration.subagent import (
    SubagentCoordinator,
    SubagentRequest,
    SubagentSnapshot,
)
from codedoggy.tui.activity import LiveActivityBoard
from codedoggy.turn.types import Message, Role, ToolCall


def test_publish_live_message_updates_snapshot_and_listeners() -> None:
    coord = SubagentCoordinator(max_workers=2)
    req = SubagentRequest(
        subagent_type="explore",
        prompt="look around",
        description="scan",
        parent_session_id="sess-1",
        id="sub_live_1",
    )
    gate = threading.Event()
    saw: list[tuple[str, str]] = []

    def listener(snap: SubagentSnapshot, message: object) -> None:
        role = getattr(message, "role", None)
        role_s = getattr(role, "value", role) if role is not None else ""
        if isinstance(message, dict):
            role_s = message.get("role", "")
        tools = getattr(message, "tool_calls", None) or []
        if isinstance(message, dict):
            tools = message.get("tool_calls") or []
        name = ""
        if tools:
            tc0 = tools[0]
            name = getattr(tc0, "name", None) or (
                tc0.get("name") if isinstance(tc0, dict) else ""
            )
        saw.append((snap.status, str(name or role_s)))

    coord.add_listener(listener)

    def run_fn(request: SubagentRequest, cancel: threading.Event) -> SubagentSnapshot:
        # Simulate archive mid-run.
        msg = Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="c1", name="grep", arguments={"pattern": "x"})],
        )
        coord.publish_live_message(request.id, msg)
        gate.wait(timeout=2.0)
        return SubagentSnapshot(
            subagent_id=request.id,
            subagent_type=request.subagent_type,
            status="completed",
            description=request.description,
            output="found it",
            live_messages=[{"role": "assistant", "content": "done"}],
        )

    snap0 = coord.spawn(req, run_fn=run_fn)
    assert snap0.status in {"pending", "running"}

    # Wait until listener saw the tool call publish.
    for _ in range(50):
        if saw:
            break
        gate.wait(0.02) if False else __import__("time").sleep(0.02)

    assert any(name == "grep" for _st, name in saw)

    live = coord.lookup(req.id)
    assert live is not None
    assert live.live_messages
    assert live.live_messages[0]["tool_calls"][0]["name"] == "grep"

    gate.set()
    # Drain pool job
    for _ in range(50):
        done = coord.lookup(req.id)
        if done and done.status == "completed":
            break
        __import__("time").sleep(0.02)
    final = coord.lookup(req.id)
    assert final is not None
    assert final.status == "completed"
    coord.remove_listener(listener)


def test_activity_board_accepts_serialized_dict_messages() -> None:
    board = LiveActivityBoard()
    board.observe(
        "t1",
        "sub_x",
        {
            "role": "assistant",
            "tool_calls": [{"id": "1", "name": "read_file", "arguments": {}}],
        },
    )
    assert "read_file" in board.line("t1", "sub_x")
    board.observe(
        "t1",
        "sub_x",
        {"role": "tool", "tool_call_id": "1", "name": "read_file", "content": "ok"},
    )
    assert board.line("t1", "sub_x").startswith("✓")
