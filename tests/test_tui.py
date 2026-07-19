"""Focused contract tests for the task-first CLI surface."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import codedoggy.tui.app as tui_app
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.utils import get_cwidth

from codedoggy.session import Session, SessionExtensions
from codedoggy.session.types import TurnResult, TurnStatus
from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tui.app import (
    CodeDoggyTUI,
    _compact_task_stage_text,
    _render_doggy_empty,
    _task_activity_text,
    _task_status_style,
    _task_stage_text,
    agent_summary_text_from_messages,
    task_report_from_agent,
)
from codedoggy.tui.model import TaskLedger
from codedoggy.turn import AgentTurnRunner
from codedoggy.turn.types import Message, Role, SampleResult, ToolCall


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
    assert _compact_task_stage_text(snapshot) == "2 并行"
    assert _task_activity_text(snapshot) == "2 个 Agent 正在并行…"
    assert _task_status_style(snapshot) == "class:task.status.running"

    ledger.update_agent(
        task.id,
        "sub_builder",
        label="builder",
        status="completed",
        output="实现完成",
    )
    ledger.set_task_phase(task.id, "reporting")
    snapshot = ledger.snapshots()[0]
    assert _task_stage_text(snapshot) == "MAIN 汇总中"
    assert _task_status_style(snapshot) == "class:task.status.reporting"


def test_overview_summary_contains_assistant_output_not_tool_noise() -> None:
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

    output = agent_summary_text_from_messages(messages)
    assert output == "我先检查入口。\n\n入口已经确认，可以实现。"
    assert "raw grep output" not in output
    assert "grep" not in output


def test_open_agent_uses_full_turn_transcript_including_tool_details() -> None:
    class _TranscriptSession(_Session):
        def handle_prompt(self, prompt: str, **_: object) -> TurnResult:
            messages = [
                Message(role=Role.ASSISTANT, content="我先运行定向测试。"),
                Message(
                    role=Role.ASSISTANT,
                    tool_calls=[
                        ToolCall(
                            id="shell-1",
                            name="shell",
                            arguments={"command": "pytest tests/test_tui.py -q"},
                        )
                    ],
                ),
                Message(
                    role=Role.TOOL,
                    name="shell",
                    tool_call_id="shell-1",
                    content="12 passed in 0.84s",
                ),
                Message(role=Role.ASSISTANT, content="测试完成。"),
            ]
            observer = getattr(self.extensions.turn_runner, "on_live_message", None)
            for message in messages:
                if callable(observer):
                    observer(message)
            self.extensions.turn_runner.live_messages.extend(messages)
            return TurnResult(status=TurnStatus.COMPLETED, final_text=f"已完成：{prompt}")

    with create_pipe_input() as pipe_input:
        tui = CodeDoggyTUI(
            _TranscriptSession(),
            input=pipe_input,
            output=DummyOutput(),
        )
        tui._start_task("实现全量详情")
        assert _wait_until(lambda: not tui._is_running())
        task = tui.ledger.snapshots()[0]
        tui._open_agent(task.id, task.agents[0].id)

        detail = "".join(fragment[1] for fragment in tui._render_modal_body())
        assert "我先运行定向测试" in detail
        assert "$ pytest tests/test_tui.py -q" in detail
        assert "12 passed in 0.84s" in detail
        filters = "".join(fragment[1] for fragment in tui._render_modal_filters())
        assert "F1 全部" in filters and "F5 测试" in filters

        tui._set_detail_filter("test")
        filtered = "".join(fragment[1] for fragment in tui._render_modal_body())
        assert "pytest tests/test_tui.py -q" in filtered
        assert "我先运行定向测试" not in filtered


def test_real_session_runner_flows_into_clicked_main_detail(tmp_path: Path) -> None:
    class _ScriptedSampler:
        def __init__(self) -> None:
            self.calls = 0

        def sample(self, _: list[Message], __: object) -> SampleResult:
            self.calls += 1
            if self.calls == 1:
                return SampleResult(
                    content="我先读取真实文件。",
                    tool_calls=[
                        ToolCall(
                            id="real-read",
                            name="read_file",
                            arguments={"target_file": "detail.txt"},
                        )
                    ],
                )
            return SampleResult(content="真实链路读取完成。")

    (tmp_path / "detail.txt").write_text("full transcript payload\n", encoding="utf-8")
    tools = ToolRegistryBuilder.new().finalize()
    runner = AgentTurnRunner(sampler=_ScriptedSampler(), tools=tools)
    session = Session.create(tmp_path, max_turns=3)
    session.bind_extensions(SessionExtensions(turn_runner=runner, tools=tools))
    try:
        with create_pipe_input() as pipe_input:
            tui = CodeDoggyTUI(session, input=pipe_input, output=DummyOutput())
            tui._start_task("读取 detail.txt")
            assert _wait_until(lambda: not tui._is_running())
            task = tui.ledger.snapshots()[0]
            tui._open_agent(task.id, task.agents[0].id)

            detail = "".join(fragment[1] for fragment in tui._render_modal_body())
            assert "我先读取真实文件" in detail
            assert "target_file: detail.txt" in detail
            assert "full transcript payload" in detail
            assert "真实链路读取完成" in detail
    finally:
        session.close()


def test_task_report_is_the_agents_first_brief_paragraph() -> None:
    text = "## 已完成\n入口已经接通。\n\n后面是只有点开 Agent 才需要看的大量细节。"
    assert task_report_from_agent(text) == "已完成 入口已经接通。"
    assert task_report_from_agent("x" * 400).endswith("…")


def test_startup_brand_is_neon_couple_art_without_overflow() -> None:
    narrow_fragments = _render_doggy_empty(36, now=0.0)
    pulse_fragments = _render_doggy_empty(36, now=0.25)
    wide_fragments = _render_doggy_empty(80, now=0.0)
    narrow = "".join(fragment[1] for fragment in narrow_fragments)
    wide = "".join(fragment[1] for fragment in wide_fragments)

    assert "DOGGY DRIVE" not in narrow
    assert "D   o   g   g   y" not in narrow
    for accidental_label in (
        "SPEED",
        "GEAR",
        "HEAT",
        "SWAG",
        "[A/D]",
        "[W]",
        "[S]",
        "[Q]",
        "风驰电掣",
        "并行开工",
    ):
        assert accidental_label not in narrow
    styles = " ".join(fragment[0] for fragment in wide_fragments)
    # Neon + figure colors vary by scale/frame; require at least one brand accent
    # and one fur tone so we still catch "empty black" regressions.
    neon = ("#16dfe8", "#5af0f7", "#0b6670", "#ff2d9a", "#ff5ab3", "#ee4b8d", "#8f1b58")
    fur = ("#f0c7a4", "#e8b878", "#c9a978", "#75644a")
    assert any(c in styles for c in neon)
    assert any(c in styles for c in fur)
    assert narrow_fragments != pulse_fragments
    assert all(get_cwidth(line) <= 36 for line in narrow.splitlines())
    assert all(get_cwidth(line) <= 80 for line in wide.splitlines())


def test_startup_brand_is_one_shot_and_never_returns_after_first_task() -> None:
    """Brand art is launch-only: first task dismisses it for the process."""
    with create_pipe_input() as pipe_input:
        tui = CodeDoggyTUI(_Session(), input=pipe_input, output=DummyOutput())
        assert tui._showing_startup_brand() is True
        body = "".join(fragment[1] for fragment in tui._render_tasks())
        assert body.strip()  # neon doggy scene, not blank

        tui._start_task("first job")
        assert tui._startup_brand is False
        assert tui._showing_startup_brand() is False
        assert _wait_until(lambda: not tui._is_running())

        # Even if the ledger were empty again, splash must not return.
        tui.ledger = TaskLedger()
        assert tui._showing_startup_brand() is False
        assert "".join(fragment[1] for fragment in tui._render_tasks()).strip() == ""

    with create_pipe_input() as pipe_input:
        boot = CodeDoggyTUI(
            _Session(),
            initial_prompt="skip splash",
            input=pipe_input,
            output=DummyOutput(),
        )
        assert boot._startup_brand is False
        assert boot._showing_startup_brand() is False


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
        assert rendered.count("╭") >= 3
        assert "正在检查 API" in rendered
        assert "正在验证交互" in rendered


def test_child_agent_detail_uses_serialized_runtime_transcript() -> None:
    session = _Session()
    session.subagents.append(
        SimpleNamespace(
            subagent_id="sub_builder",
            subagent_type="general-purpose",
            description="详情构建",
            status="completed",
            output="实现完成",
            error=None,
            live_messages=[
                {"role": "assistant", "content": "我先读取详情入口。"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "read-child",
                            "name": "read_file",
                            "arguments": {"path": "src/codedoggy/tui/app.py"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "read_file",
                    "tool_call_id": "read-child",
                    "content": "1238 def _open_agent(...):",
                },
            ],
        )
    )
    with create_pipe_input() as pipe_input:
        tui = CodeDoggyTUI(session, input=pipe_input, output=DummyOutput())
        task = tui.ledger.create("接入子 Agent 详情")
        tui._active_task_id = task.id
        tui._subagent_baselines[task.id] = set()
        tui._sync_runtime()
        tui._open_agent(task.id, "sub_builder")

        detail = "".join(fragment[1] for fragment in tui._render_modal_body())
        assert "我先读取详情入口" in detail
        assert "path: src/codedoggy/tui/app.py" in detail
        assert "def _open_agent" in detail


def test_running_child_detail_does_not_fake_unavailable_tool_history() -> None:
    session = _Session()
    session.subagents.append(
        SimpleNamespace(
            subagent_id="sub_running",
            subagent_type="general-purpose",
            description="交互验证",
            status="running",
            output="正在验证",
            error=None,
        )
    )
    with create_pipe_input() as pipe_input:
        tui = CodeDoggyTUI(session, input=pipe_input, output=DummyOutput())
        task = tui.ledger.create("等待子 Agent 完整记录")
        tui._active_task_id = task.id
        tui._subagent_baselines[task.id] = set()
        tui._sync_runtime()
        tui._open_agent(task.id, "sub_running")

        detail = "".join(fragment[1] for fragment in tui._render_modal_body())
        assert "当前运行时只在本轮结束后同步完整工具记录" in detail
        assert "正在验证" not in detail


def test_detail_interjection_cannot_cross_from_historical_task() -> None:
    class _RecordingSession(_Session):
        def __init__(self) -> None:
            super().__init__()
            self.interjections: list[tuple[str, str | None]] = []

        def interject(self, text: str, **kwargs: object) -> None:
            self.interjections.append((text, kwargs.get("prompt_id")))  # type: ignore[arg-type]

    session = _RecordingSession()
    with create_pipe_input() as pipe_input:
        tui = CodeDoggyTUI(session, input=pipe_input, output=DummyOutput())
        historical = tui.ledger.create("历史任务")
        current = tui.ledger.create("当前运行任务")
        tui._active_task_id = current.id
        tui._open_agent(historical.id, historical.agents[0].id)
        tui._is_running = lambda: True  # type: ignore[method-assign]
        buffer = SimpleNamespace(text="不要串到当前任务")

        assert tui._accept_detail_prompt(buffer) is True
        assert session.interjections == []
        assert tui._feedback_text == "只能向当前运行任务补充指令"


def test_detail_prompt_prefix_leaves_input_room_on_narrow_terminals(
    monkeypatch: object,
) -> None:
    with create_pipe_input() as pipe_input:
        tui = CodeDoggyTUI(_Session(), input=pipe_input, output=DummyOutput())
        task = tui.ledger.create("窄屏输入")
        tui.ledger.update_agent(
            task.id,
            "long-child",
            label="VERY LONG CHILD AGENT",
            status="running",
        )
        tui._open_agent(task.id, "long-child")

        for width in (12, 20, 36, 40, 80, 120):
            monkeypatch.setattr(  # type: ignore[attr-defined]
                tui_app,
                "_terminal_width",
                lambda width=width: width,
            )
            prefix = "".join(piece[1] for piece in tui._render_detail_prompt_prefix())
            assert get_cwidth(prefix) <= max(4, width - 20)


def test_task_panel_keeps_reference_layout_across_terminal_widths(
    monkeypatch: object,
) -> None:
    with create_pipe_input() as pipe_input:
        tui = CodeDoggyTUI(_Session(), input=pipe_input, output=DummyOutput())
        task = tui.ledger.create("CLI 内部任务面板美术设计")
        tui.ledger.update_agent(
            task.id,
            "sub_builder",
            label="builder",
            status="running",
            output="正在实现任务面板与状态样式。",
        )
        tui.ledger.update_agent(
            task.id,
            "sub_tester",
            label="tester",
            status="pending",
            output="测试计划已准备，等待可执行版本。",
        )
        tui.ledger.set_task_phase(task.id, "parallel")

        monkeypatch.setattr(tui_app, "_terminal_height", lambda: 36)  # type: ignore[attr-defined]
        for width in (20, 24, 32, 36, 48, 80, 120):
            monkeypatch.setattr(  # type: ignore[attr-defined]
                tui_app,
                "_terminal_width",
                lambda width=width: width,
            )
            rendered = "".join(fragment[1] for fragment in tui._render_tasks())
            assert all(get_cwidth(line) <= width for line in rendered.splitlines())
            header = "".join(fragment[1] for fragment in tui._render_header())
            assert get_cwidth(header) <= width
            assert header.startswith("  ==DOGGY==")
            assert "main ·" not in header
            assert "◆" in rendered
            expected_stage = "3 并行" if width < 36 else "3 个 Agent 并行中"
            assert expected_stage in rendered
            if width >= 36:
                assert "BUILDER" in rendered and "TESTER" in rendered
            else:
                assert "BUI" in rendered and "TES" in rendered
            assert "正在实现" in rendered
            assert "测试计划" in rendered

        assert "正在实现任务面板与状态样式。" in rendered
        assert "测试计划已准备，等待可执行版本。" in rendered


def test_running_status_and_feedback_fit_narrow_terminals(monkeypatch: object) -> None:
    with create_pipe_input() as pipe_input:
        tui = CodeDoggyTUI(_Session(), input=pipe_input, output=DummyOutput())
        task = tui.ledger.create("并行状态栏宽度验证")
        tui._active_task_id = task.id
        tui._task_started_at = time.monotonic() - 2.0
        monkeypatch.setattr(tui, "_is_running", lambda: True)  # type: ignore[attr-defined]

        for width in (20, 24, 32, 36, 48, 80, 120):
            monkeypatch.setattr(  # type: ignore[attr-defined]
                tui_app,
                "_terminal_width",
                lambda width=width: width,
            )
            status = "".join(fragment[1] for fragment in tui._render_turn_status())
            assert get_cwidth(status) <= width
            expected_stop = "[停]" if width < 36 else "[停止]"
            assert expected_stop in status

        monkeypatch.setattr(tui, "_is_running", lambda: False)  # type: ignore[attr-defined]
        tui._set_feedback("MAIN 已完成一段非常长的任务汇总反馈", "success")
        for width in (20, 24, 32, 36):
            monkeypatch.setattr(  # type: ignore[attr-defined]
                tui_app,
                "_terminal_width",
                lambda width=width: width,
            )
            feedback = "".join(fragment[1] for fragment in tui._render_turn_status())
            assert get_cwidth(feedback) <= width


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
        assert "Ctrl+L:登录" in shortcuts or "登录" in shortcuts
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
        assert "╭ MAIN  › ╮" in task_text
        assert "已完成：实现 CLI" in task_text

        pipe_input.send_text("\t\r")
        assert _wait_until(lambda: tui._modal_open)
        detail_text = "".join(fragment[1] for fragment in tui._render_modal_body())
        assert "已完成：实现 CLI" in detail_text
        assert tui.app.layout.has_focus(tui._detail_window)

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
