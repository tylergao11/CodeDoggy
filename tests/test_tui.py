"""Focused contract tests for the task-first CLI surface."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.utils import get_cwidth

from codedoggy.session.types import TurnResult, TurnStatus
from codedoggy.tui.app import (
    CodeDoggyTUI,
    _render_doggy_empty,
    _task_activity_text,
    _task_stage_text,
    agent_text_from_messages,
    task_report_from_agent,
)
from codedoggy.tui.model import TaskLedger
from codedoggy.turn.types import Message, Role, ToolCall


class _Session:
    def __init__(self) -> None:
        self.id = "session-tui"
        self.cwd = Path("C:/workspace")
        self.subagents: list[SimpleNamespace] = []
        self.extensions = SimpleNamespace(
            turn_runner=SimpleNamespace(sampler=None, live_messages=[]),
            kernel=SimpleNamespace(
                subagent_coordinator=SimpleNamespace(
                    list_for_parent=lambda _: list(self.subagents)
                )
            ),
            context=SimpleNamespace(
                budget=SimpleNamespace(last_prompt_tokens=82000, context_window=500000)
            ),
        )
        self.cancelled = False

    def handle_prompt(self, prompt: str, **_: object) -> TurnResult:
        time.sleep(0.03)
        return TurnResult(
            status=TurnStatus.COMPLETED,
            final_text=f"已完成：{prompt}",
        )

    def interject(self, _: str, **__: object) -> None:
        return None

    def cancel(self) -> None:
        self.cancelled = True


def test_ledger_keeps_agents_under_their_task_and_exposes_parallel_stage() -> None:
    ledger = TaskLedger()
    task = ledger.create("实现 CLI 驾驶舱")
    ledger.update_agent(
        task.id,
        "sub_builder",
        label="builder",
        status="running",
        output="正在实现",
    )

    snapshot = ledger.snapshots()[0]
    assert [agent.label for agent in snapshot.agents] == ["MAIN", "BUILDER"]
    ledger.set_task_phase(task.id, "parallel")
    snapshot = ledger.snapshots()[0]
    assert _task_stage_text(snapshot) == "2 个 Agent 并行中"
    assert _task_activity_text(snapshot) == "2 个 Agent 正在并行…"

    ledger.update_agent(
        task.id,
        "sub_builder",
        label="builder",
        status="completed",
        output="实现完成",
    )
    ledger.set_task_phase(task.id, "reporting")
    assert _task_stage_text(ledger.snapshots()[0]) == "MAIN 汇总中"


def test_agent_detail_contains_assistant_output_not_tool_records() -> None:
    messages = [
        Message(role=Role.ASSISTANT, content="我先检查入口。"),
        Message(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[ToolCall(id="1", name="grep", arguments={"pattern": "cli"})],
        ),
        Message(role=Role.TOOL, content="raw grep output", tool_call_id="1"),
        Message(role=Role.ASSISTANT, content="入口已经确认，可以实现。"),
    ]

    output = agent_text_from_messages(messages)
    assert output == "我先检查入口。\n\n入口已经确认，可以实现。"
    assert "raw grep output" not in output
    assert "grep" not in output


def test_task_report_is_the_agents_first_brief_paragraph() -> None:
    text = "## 已完成\n入口已经接通。\n\n后面是只有点开 Agent 才需要看的大量细节。"
    assert task_report_from_agent(text) == "已完成 入口已经接通。"
    assert task_report_from_agent("x" * 400).endswith("…")


def test_empty_state_renders_speeding_frenchie_without_overflow() -> None:
    first = "".join(fragment[1] for fragment in _render_doggy_empty(36, now=0.0))
    moving = "".join(fragment[1] for fragment in _render_doggy_empty(36, now=0.25))

    assert "CODEDOGGY" not in first
    assert "风驰电掣" not in first and "并行开工" not in first
    assert "D   o   g   g   y" in first
    assert "ᴥ" in first and "════════" in first
    assert first != moving
    assert all(get_cwidth(line) <= 36 for line in first.splitlines())


def test_parallel_runtime_uses_child_descriptions_as_clickable_participants() -> None:
    session = _Session()
    session.subagents.extend(
        [
            SimpleNamespace(
                subagent_id="sub_api",
                subagent_type="general-purpose",
                description="API 入口",
                status="running",
                output="正在检查 API",
                error=None,
            ),
            SimpleNamespace(
                subagent_id="sub_test",
                subagent_type="general-purpose",
                description="交互验证",
                status="running",
                output="正在验证交互",
                error=None,
            ),
        ]
    )
    with create_pipe_input() as pipe_input:
        tui = CodeDoggyTUI(session, input=pipe_input, output=DummyOutput())
        task = tui.ledger.create("并行完成 CLI")
        tui._active_task_id = task.id
        tui._subagent_baselines[task.id] = set()
        tui._sync_runtime()

        snapshot = tui.ledger.snapshots()[0]
        assert [agent.label for agent in snapshot.agents] == [
            "MAIN",
            "API 入口",
            "交互验证",
        ]
        assert snapshot.phase == "parallel"
        assert _task_stage_text(snapshot) == "3 个 Agent 并行中"
        rendered = "".join(fragment[1] for fragment in tui._render_tasks())
        assert "API 入口" in rendered and "交互验证" in rendered


def test_full_screen_agent_window_is_opaque_and_interactive() -> None:
    with create_pipe_input() as pipe_input:
        tui = CodeDoggyTUI(
            _Session(),
            initial_prompt="实现 CLI",
            input=pipe_input,
            output=DummyOutput(),
        )
        float_layer = tui.app.layout.container.floats[0]
        assert float_layer.transparent() is False
        assert float_layer.top == 1 and float_layer.bottom == 1
        top = "".join(fragment[1] for fragment in tui._render_prompt_top())
        bottom = "".join(fragment[1] for fragment in tui._render_prompt_bottom())
        shortcuts = "".join(fragment[1] for fragment in tui._render_shortcuts())
        assert top.startswith("  ╭") and top.endswith("╮")
        assert bottom.startswith("  ╰") and "model · auto" in bottom
        assert "Enter:" in shortcuts and "Ctrl+Q:退出" in shortcuts
        assert tui._input.control.input_processors

        tui._set_feedback("MAIN 已汇总，任务完成", "success")
        feedback = "".join(fragment[1] for fragment in tui._render_turn_status())
        assert "任务完成" in feedback
        assert tui._prompt_border_class() == "class:prompt.border.success"

        thread = threading.Thread(target=tui.run, daemon=True)
        thread.start()
        assert _wait_until(lambda: bool(tui.ledger.snapshots()))
        assert _wait_until(lambda: tui.ledger.snapshots()[0].phase == "done")
        task_text = "".join(fragment[1] for fragment in tui._render_tasks())
        assert "已完成 · 1 个 Agent" in task_text
        assert "└─ MAIN" in task_text

        pipe_input.send_text("\t\r")
        assert _wait_until(lambda: tui._modal_open)
        assert "已完成：实现 CLI" in tui._agent_output.text

        pipe_input.send_text("\x1b")
        assert _wait_until(lambda: not tui._modal_open)
        pipe_input.send_text("\x11")
        assert _wait_until(lambda: tui._quit_armed_until > time.monotonic())
        pipe_input.send_text("\x11")
        thread.join(timeout=2)
        assert not thread.is_alive()


def _wait_until(predicate: object, *, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(0.02)
    return False
