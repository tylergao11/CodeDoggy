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
    _task_list_summary,
    _task_status_style,
    _task_stage_text,
    agent_summary_text_from_messages,
    task_report_from_agent,
)
from codedoggy.tui import surface as session_surface
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
                ),
                todo_state=None,
                tool_extra={},
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


def test_apply_agent_status_refuses_revive_after_terminal() -> None:
    ledger = TaskLedger()
    task = ledger.create("fence")
    ledger.apply_agent_status(
        task.id, "child", label="child", status="completed", output="done"
    )
    ledger.finish_task(task.id, "completed")
    assert (
        ledger.apply_agent_status(
            task.id, "child", label="child", status="running", output="late"
        )
        is False
    )
    snap = ledger.snapshots()[0]
    child = next(a for a in snap.agents if a.id == "child")
    assert child.status == "completed"


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
        filters = "".join(fragment[1] for fragment in tui._render_modal_filters())
        assert "消息" in filters and "工具" in filters
        assert "全部" not in filters and "F1" not in filters

        tui._set_detail_filter("tool")
        filtered = "".join(fragment[1] for fragment in tui._render_modal_body())
        # Grok collapsed default: one-line tool headline, no body dump.
        assert "Ran" in filtered and "pytest" in filtered
        assert "12 passed in 0.84s" not in filtered
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
            assert "真实链路读取完成" in detail
            tui._set_detail_filter("tool")
            tools = "".join(fragment[1] for fragment in tui._render_modal_body())
            # Collapsed: Grok-style "Read detail.txt", details on expand.
            assert "Read" in tools and "detail.txt" in tools
            assert "full transcript payload" not in tools
    finally:
        session.close()


def test_task_report_is_the_agents_first_brief_paragraph() -> None:
    text = "## 已完成\n入口已经接通。\n\n后面是只有点开 Agent 才需要看的大量细节。"
    assert task_report_from_agent(text) == "已完成 入口已经接通。"
    # Full first paragraph kept; optional soft cap only when caller asks.
    long = task_report_from_agent("x" * 400)
    assert len(long) == 400
    capped = task_report_from_agent("x" * 400, max_chars=260)
    assert len(capped) == 260
    assert not long.endswith("…")
    assert not long.endswith("...")


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
        after = "".join(fragment[1] for fragment in tui._render_tasks())
        # Rounded idle plate is fine; full couple splash art must not return.
        assert "散步" in after or "DOGGY" in after or "∪" in after
        assert "FFFF" not in after  # pixel-art palette rows never in idle plate

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
        # Homepage cards stay compact — no ↳ MAIN / child agent rows (avoids jiggle).
        assert "↳" not in rendered
        assert "正在检查 API" not in rendered
        assert "正在验证交互" not in rendered
        assert "调用中" not in rendered


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
        tui._set_detail_filter("tool")
        tools = "".join(fragment[1] for fragment in tui._render_modal_body())
        assert "Read" in tools and "app.py" in tools
        assert "def _open_agent" not in tools


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
        assert "进行中" in detail and "结束后会同步" in detail
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
        tui.ledger.set_report(task.id, "MAIN", "面板样式与并行状态已对齐。")

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
            assert "DOGGY" in header or "DOG" in header
            assert header.strip().startswith("╭") or "DOGGY" in header
            assert "main ·" not in header
            # Compact list: title + stage + one human summary (no dog-ear separators).
            assert "∪" not in rendered
            assert "CLI" in rendered or "任务面板" in rendered
            expected_stage = "3 并行" if width < 36 else "3 个 Agent 并行中"
            assert expected_stage in rendered
            # No ↳ agent rows on homepage cards.
            assert "↳" not in rendered
            compact = rendered.replace("\n", "").replace(" ", "")
            assert "面板" in compact or "对齐" in compact

        # No tool-call live noise in the list body.
        assert "调用中" not in "".join(
            fragment[1] for fragment in tui._render_tasks()
        )


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


def test_reload_client_deferred_until_idle(monkeypatch: object) -> None:
    """OAuth finish while a turn runs must queue apply, not drop it."""
    with create_pipe_input() as pipe_input:
        tui = CodeDoggyTUI(_Session(), input=pipe_input, output=DummyOutput())
        applied: list[str | None] = []

        monkeypatch.setattr(tui, "_is_running", lambda: True)  # type: ignore[attr-defined]
        tui._queue_or_apply_reload(provider="grok", message="登录成功")
        assert tui._pending_reload is not None
        assert tui._pending_reload["provider"] == "grok"
        assert applied == []

        monkeypatch.setattr(  # type: ignore[attr-defined]
            tui,
            "_apply_reload_client",
            lambda **kwargs: applied.append(kwargs.get("provider")),
        )
        monkeypatch.setattr(tui, "_is_running", lambda: False)  # type: ignore[attr-defined]
        tui._flush_pending_reload()
        assert tui._pending_reload is None
        assert applied == ["grok"]


def test_budget_text_keeps_last_used_while_awaiting_usage() -> None:
    """Reasoning / compact clears last_prompt_tokens — UI must not flash '—'."""
    session_surface._budget_sticky.clear()

    class _Budget:
        last_prompt_tokens: int | None = 12_000
        context_window = 200_000

    budget = _Budget()
    session = SimpleNamespace(
        id="budget-sticky",
        extensions=SimpleNamespace(context=SimpleNamespace(budget=budget)),
    )
    assert session_surface.budget_text(session) == "12k / 200k"
    budget.last_prompt_tokens = None
    assert session_surface.budget_text(session) == "12k / 200k"
    assert "—" not in session_surface.budget_text(session)


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
        prefix = "".join(fragment[1] for fragment in tui._render_prompt_prefix())
        header = "".join(fragment[1] for fragment in tui._render_header())
        assert top.startswith("  ╭") and top.endswith("╮")
        assert "交代任务" in top
        assert "∪･ω･∪" not in top
        assert "∪･ω･∪" not in header
        assert "›" in prefix
        assert "∪" not in prefix
        assert "∪" not in header
        assert bottom.startswith("  ╰") and "未选择模型" in bottom
        # Default chrome: Tab=最新任务, S-Tab=Plan/Auto, ^Enter=换行; no arrows.
        tui.app.layout.focus(tui._input)
        shortcuts_input = "".join(fragment[1] for fragment in tui._render_shortcuts())
        assert "Tab:最新任务" in shortcuts_input
        assert "S-Tab:Plan/Auto" in shortcuts_input
        assert "^Enter:换行" in shortcuts_input
        assert "Ctrl+Q:退出" in shortcuts_input
        assert "↑↓" not in shortcuts_input and "←→" not in shortcuts_input

        # Ctrl+Enter / Ctrl+J inserts a hard newline (does not submit).
        tui._input.buffer.text = "line1"
        tui._input.buffer.cursor_position = len("line1")

        class _E:
            def __init__(self, app: object, buf: object) -> None:
                self.app = app
                self.current_buffer = buf

        CodeDoggyTUI._insert_buffer_newline(
            _E(tui.app, tui._input.buffer), max_lines=8
        )
        assert tui._input.buffer.text == "line1\n"
        # Cap: already max lines → no further insert
        tui._input.buffer.text = "\n".join(f"L{i}" for i in range(8))
        before = tui._input.buffer.text
        CodeDoggyTUI._insert_buffer_newline(
            _E(tui.app, tui._input.buffer), max_lines=8
        )
        assert tui._input.buffer.text == before
        tui.app.layout.focus(tui._task_window)
        shortcuts_tasks = "".join(fragment[1] for fragment in tui._render_shortcuts())
        assert shortcuts_tasks.strip().startswith("Space:输入")
        assert "Tab:进入" in shortcuts_tasks
        assert "S-Tab:Plan/Auto" in shortcuts_tasks
        assert tui._input.control.input_processors

        tui._set_feedback("任务完成", "info")
        feedback = "".join(fragment[1] for fragment in tui._render_turn_status())
        assert "任务完成" in feedback
        # Feedback must not recolor the prompt border (no green flash).
        assert tui._prompt_border_class() in {
            "class:prompt.border",
            "class:prompt.border.focus",
        }

        thread = threading.Thread(target=tui.run, daemon=True)
        thread.start()
        assert _wait_until(lambda: bool(tui.ledger.snapshots()))
        assert _wait_until(lambda: tui.ledger.snapshots()[0].phase == "done")
        task_text = "".join(fragment[1] for fragment in tui._render_tasks())
        assert "已完成 · 1 个 Agent" in task_text
        # Compact card: frame + title; summary is MAIN report prose.
        assert "╭" in task_text and "╮" in task_text
        assert "实现 CLI" in task_text
        assert "已完成：实现 CLI" in task_text

        # Tab → latest task → Tab enter detail (idle Esc no longer closes).
        pipe_input.send_text("\t\t")
        assert _wait_until(lambda: tui._modal_open)
        detail_text = "".join(fragment[1] for fragment in tui._render_modal_body())
        assert "已完成：实现 CLI" in detail_text
        assert tui.app.layout.has_focus(tui._detail_window)

        # Tab exits detail (Esc is cancel-only when running).
        pipe_input.send_text("\t")
        assert _wait_until(lambda: not tui._modal_open)
        pipe_input.send_text("\x11")
        assert _wait_until(lambda: tui._quit_armed_until > time.monotonic())
        pipe_input.send_text("\x11")
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_todo_badge_and_pane_render() -> None:
    """Grok-style 完成/总数 badge in header; click toggles expandable list."""
    from codedoggy.tools.grok_build.todo_logic import TodoItem, TodoState

    with create_pipe_input() as pipe_input:
        session = _Session()
        todos = TodoState()
        todos.push("1", TodoItem("调研", status="completed"))
        todos.push("2", TodoItem("实现", status="in_progress"))
        todos.push("3", TodoItem("测试", status="pending"))
        session.extensions.kernel.todo_state = todos
        tui = CodeDoggyTUI(
            session,
            input=pipe_input,
            output=DummyOutput(),
        )
        assert tui._todo_badge_label() == "1/3"
        header = "".join(fragment[1] for fragment in tui._render_header())
        assert "1/3" in header
        assert "✓" in header
        assert "计划" in header

        # Closed by default
        assert tui._render_todo_pane() == []
        tui._toggle_todo_pane()
        assert tui._todo_pane_open
        pane = "".join(fragment[1] for fragment in tui._render_todo_pane())
        assert "计划 1/3" in pane
        assert "调研" in pane and "实现" in pane and "测试" in pane
        assert "▶" in pane or "○" in pane or "✓" in pane

        # Scroll window: pad many items, ensure offset works
        for i in range(12):
            todos.push(f"n{i}", TodoItem(f"extra-{i}", status="pending"))
        tui._todo_scroll = 0
        tui._scroll_todo_pane(3)
        assert tui._todo_scroll == 3
        pane2 = "".join(fragment[1] for fragment in tui._render_todo_pane())
        assert "滚动" in pane2 or "/" in pane2

        tui._toggle_todo_pane()
        assert not tui._todo_pane_open


def test_plan_detail_does_not_embed_noisy_todo_progress() -> None:
    """Plan/agent detail no longer dumps MAIN 进度 checklist into the body."""
    from codedoggy.tools.grok_build.todo_logic import TodoItem, TodoState

    with create_pipe_input() as pipe_input:
        session = _Session()
        todos = TodoState()
        todos.push("1", TodoItem("step-one", status="completed"))
        todos.push("2", TodoItem("step-two", status="pending"))
        session.extensions.kernel.todo_state = todos
        tui = CodeDoggyTUI(session, input=pipe_input, output=DummyOutput())
        tui.ledger.create("plan-task")
        task = tui.ledger.snapshots()[0]
        tui._modal_open = True
        tui._modal_kind = "agent"
        tui._modal_ref = (task.id, f"{task.id}:main")
        tui._detail_filter = "plan"
        body = "".join(fragment[1] for fragment in tui._render_modal_body())
        assert "step-one" not in body
        assert "todo · MAIN" not in body
        # Header badge still owns progress chrome.
        assert tui._todo_badge_label() == "1/2"


def test_open_active_main_plan_tab() -> None:
    from codedoggy.tools.grok_build.todo_logic import TodoItem, TodoState

    with create_pipe_input() as pipe_input:
        session = _Session()
        todos = TodoState()
        todos.push("1", TodoItem("a", status="pending"))
        session.extensions.kernel.todo_state = todos
        tui = CodeDoggyTUI(session, input=pipe_input, output=DummyOutput())
        tui.ledger.create("plan-task")
        task = tui.ledger.snapshots()[0]
        tui._active_task_id = task.id
        tui._open_active_main_plan_tab()
        assert tui._modal_open
        assert tui._detail_filter == "plan"
        assert tui._modal_ref is not None
        assert tui._modal_ref[0] == task.id
        assert str(tui._modal_ref[1]).endswith(":main")
        assert tui._todo_pane_open


def test_sync_task_plan_with_session_planning() -> None:
    from codedoggy.orchestration.session_mode import SessionModeState

    with create_pipe_input() as pipe_input:
        session = _Session()
        mode = SessionModeState()
        mode.enter_plan(".grok/plan.md")
        session.extensions.kernel.session_mode_state = mode
        tui = CodeDoggyTUI(session, input=pipe_input, output=DummyOutput())
        tui.ledger.create("sync-me")
        tid = tui.ledger.snapshots()[0].id
        tui._active_task_id = tid
        assert tui.ledger.snapshots()[0].plan_state == "none"
        tui._sync_task_plan_with_session()
        assert tui.ledger.snapshots()[0].plan_state == "planning"


def test_ask_user_fn_opens_modal_and_accepts_selection() -> None:
    """ask_user_question uses a dedicated float — not the plan/agent card."""
    with create_pipe_input() as pipe_input:
        tui = CodeDoggyTUI(_Session(), input=pipe_input, output=DummyOutput())
        tui.ledger.create("plan-task")
        tui._active_task_id = tui.ledger.snapshots()[0].id
        questions = [
            {
                "question": "用 JWT 还是 session？",
                "options": [
                    {"label": "JWT (Recommended)", "description": "无状态"},
                    {"label": "Session", "description": "有状态"},
                ],
            }
        ]

        def answer_soon() -> None:
            time.sleep(0.05)
            assert tui._ask_active and tui._modal_kind == "ask"
            # Dedicated float: agent modal shell stays closed.
            assert tui._modal_open is False
            body = "".join(p[1] for p in tui._render_ask_body())
            assert "JWT" in body
            assert "用 JWT" in body
            title = "".join(p[1] for p in tui._render_ask_dialog_title())
            assert "问卷" in title
            # Larger questionnaire styles (bordered Frame host).
            styles = [p[0] for p in tui._render_ask_body() if p[0]]
            assert any("ask.question" in s or "ask.option" in s for s in styles)
            tui._ask_opt_index = 0
            tui._ask_confirm_current()

        threading.Thread(target=answer_soon, daemon=True).start()
        result = tui._ask_user_fn(questions)
        assert result.get("outcome") == "accepted"
        answers = result.get("answers") or {}
        assert "JWT (Recommended)" in answers.get("用 JWT 还是 session？", [])


def test_ask_user_tab_exits_without_modal_shell() -> None:
    """Tab exits questionnaire (dedicated float; _modal_open stays False)."""
    with create_pipe_input() as pipe_input:
        tui = CodeDoggyTUI(_Session(), input=pipe_input, output=DummyOutput())
        questions = [
            {
                "question": "Q?",
                "options": [{"label": "A", "description": "a"}],
            }
        ]

        def cancel_soon() -> None:
            time.sleep(0.05)
            assert tui._ask_active and tui._modal_kind == "ask"
            assert tui._modal_open is False
            # Same path as Tab binding.
            tui._resolve_ask({"outcome": "cancelled"})

        threading.Thread(target=cancel_soon, daemon=True).start()
        result = tui._ask_user_fn(questions)
        assert result.get("outcome") == "cancelled"


def test_ask_user_fn_up_down_moves_option() -> None:
    with create_pipe_input() as pipe_input:
        tui = CodeDoggyTUI(_Session(), input=pipe_input, output=DummyOutput())
        tui._ask_questions = [
            {
                "question": "Q?",
                "options": [
                    {"label": "A", "description": "a"},
                    {"label": "B", "description": "b"},
                ],
            }
        ]
        tui._ask_q_index = 0
        tui._ask_opt_index = 0
        tui._ask_move_option(1)
        assert tui._ask_opt_index == 1
        tui._ask_move_option(1)
        # wraps to Other (index 2) then A
        assert tui._ask_opt_index == 2
        tui._ask_move_option(1)
        assert tui._ask_opt_index == 0


def test_finished_task_clears_plan_draft_chrome() -> None:
    """After the agent finishes talking, card must not stick on「计划起草中」."""
    ledger = TaskLedger()
    task = ledger.create("plan then done")
    ledger.set_plan_state(task.id, "planning")
    ledger.update_agent(
        task.id,
        f"{task.id}:main",
        label="MAIN",
        status="completed",
        output="方案已说明完毕，可以开工。",
    )
    ledger.set_report(task.id, "MAIN", "方案已说明完毕，可以开工。")
    ledger.finish_task(task.id, "completed")
    snap = ledger.snapshots()[0]
    assert snap.phase == "done"
    assert snap.plan_state == "none"
    assert "起草" not in _task_stage_text(snap)
    assert "已完成" in _task_stage_text(snap)
    summary = _task_list_summary(snap)
    assert "起草" not in summary
    assert "方案已说明完毕" in summary


def test_live_plan_prefers_main_prose_over_draft_placeholder() -> None:
    ledger = TaskLedger()
    task = ledger.create("planning live")
    ledger.set_plan_state(task.id, "planning")
    ledger.update_agent(
        task.id,
        f"{task.id}:main",
        label="MAIN",
        status="running",
        output="我先读入口，再写 plan.md。",
    )
    snap = ledger.snapshots()[0]
    # Still planning phase → stage can say 起草中, but summary shows live prose.
    assert "起草" in _task_stage_text(snap)
    assert "我先读入口" in _task_list_summary(snap)


def test_toggle_todo_pane_does_not_steal_task_selection() -> None:
    from codedoggy.tools.grok_build.todo_logic import TodoItem, TodoState

    with create_pipe_input() as pipe_input:
        session = _Session()
        todos = TodoState()
        todos.push("1", TodoItem("a", status="pending"))
        session.extensions.kernel.todo_state = todos
        tui = CodeDoggyTUI(session, input=pipe_input, output=DummyOutput())
        tui.ledger.create("first")
        tui.ledger.create("second-active")
        tui._active_task_id = tui.ledger.snapshots()[-1].id
        tui._selected_task = 0
        tui._task_selection_active = True
        tui._todo_pane_open = False
        tui._toggle_todo_pane()
        assert tui._todo_pane_open
        # Opening plan checklist must not jump selection to the active/latest task.
        assert tui._selected_task == 0


def test_incomplete_work_status_hint_zh() -> None:
    from codedoggy.tools.grok_build.todo_logic import TodoItem, TodoState

    with create_pipe_input() as pipe_input:
        session = _Session()
        todos = TodoState()
        todos.push("1", TodoItem("a", status="pending"))
        todos.push("2", TodoItem("b", status="in_progress"))
        session.extensions.kernel.todo_state = todos
        session.extensions.kernel.session_id = "s1"
        session.extensions.kernel.subagent_coordinator = SimpleNamespace(
            list_for_parent=lambda _sid: []
        )
        session.extensions.kernel.task_manager = None
        tui = CodeDoggyTUI(session, input=pipe_input, output=DummyOutput())
        hint = tui._incomplete_work_status_hint()
        assert hint is not None
        assert "待办" in hint
        assert "2" in hint


def test_main_todo_chip_and_agent_chip() -> None:
    from codedoggy.tools.grok_build.todo_logic import TodoItem, TodoState

    with create_pipe_input() as pipe_input:
        session = _Session()
        todos = TodoState()
        todos.push("1", TodoItem("a", status="completed"))
        todos.push("2", TodoItem("b", status="pending"))
        session.extensions.kernel.todo_state = todos
        session.extensions.kernel.subagent_coordinator = SimpleNamespace(
            todo_state_for=lambda _id: None,
            lookup=lambda _id: None,
        )
        tui = CodeDoggyTUI(session, input=pipe_input, output=DummyOutput())
        assert tui._main_todo_chip() == "1/2"
        assert tui._agent_todo_chip("task_001:main") == "1/2"
        # Child falls back to None without coordinator data
        assert tui._agent_todo_chip("child-x") is None


def test_todo_state_for_open_agent_uses_main_by_default() -> None:
    from codedoggy.tools.grok_build.todo_logic import TodoItem, TodoState

    with create_pipe_input() as pipe_input:
        session = _Session()
        main_todos = TodoState()
        main_todos.push("m1", TodoItem("main-only", status="pending"))
        session.extensions.kernel.todo_state = main_todos
        tui = CodeDoggyTUI(session, input=pipe_input, output=DummyOutput())
        st = tui._todo_state_for_open_agent()
        assert st is main_todos
        # Simulated open MAIN agent
        tui._modal_ref = ("task_001", "task_001:main")
        assert tui._todo_state_for_open_agent() is main_todos


def test_shift_tab_plan_exits_goal_mode() -> None:
    """Shift+Tab toggles Plan/Auto (same helper as the binding)."""
    from codedoggy.orchestration.session_mode import SessionModeState

    with create_pipe_input() as pipe_input:
        session = _Session()
        mode = SessionModeState()
        mode.enter_goal()
        session.extensions.kernel.session_mode_state = mode
        session.extensions.kernel.cwd = Path("C:/workspace")
        session.extensions.kernel.enter_plan_mode_pending = (
            lambda p=None: mode.enter_plan_pending(p or ".grok/plan.md")
        )
        session.extensions.kernel.enter_plan_mode = lambda p=None: mode.enter_plan(
            p or ".grok/plan.md"
        )
        session.extensions.kernel.persist_plan_mode_state = lambda: None
        tui = CodeDoggyTUI(session, input=pipe_input, output=DummyOutput())
        assert mode.is_goal()
        tui._toggle_session_plan_mode()
        assert not mode.is_goal()
        assert mode.is_plan_ui() or mode.is_plan() or mode.plan_phase == "pending"


def test_kernel_close_flushes_todo(tmp_path: Path) -> None:
    from codedoggy.session.kernel import RuntimeKernel
    from codedoggy.tools.grok_build.todo_logic import TodoItem, TodoState, load_todo_state

    k = RuntimeKernel(cwd=tmp_path, session_id="flush1")
    st = TodoState()
    st.push("a", TodoItem("flush-me", status="pending"))
    k.todo_state = st
    k.close()
    restored = load_todo_state(cwd=tmp_path, session_id="flush1")
    assert restored is not None
    assert restored.get("a") is not None
    assert restored.get("a").content == "flush-me"


def _wait_until(predicate: object, *, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(0.02)
    return False
