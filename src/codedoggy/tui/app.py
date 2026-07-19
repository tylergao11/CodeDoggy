"""Prompt-toolkit boss cockpit: tasks first, Agent detail on demand."""

from __future__ import annotations

import re
import shutil
import threading
import time
from collections.abc import Callable
from itertools import groupby
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    ConditionalContainer,
    Float,
    FloatContainer,
    FormattedTextControl,
    HSplit,
    Layout,
    ScrollOffsets,
    VSplit,
    Window,
)
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.layout.screen import Point
from prompt_toolkit.layout.processors import AfterInput, ConditionalProcessor
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.output.color_depth import ColorDepth
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import TextArea

from codedoggy.session.types import TurnStatus
from codedoggy.tui.agent_detail import (
    DETAIL_FILTERS,
    DETAIL_FILTER_LABELS,
    DETAIL_STYLE_RULES,
    AgentDetailSnapshot,
    DetailFilter,
    render_detail_body,
    snapshot_from_messages,
)
from codedoggy.tui.model import AgentView, TaskLedger, TaskView
from codedoggy.turn.types import Message, Role


STATUS_TEXT = {
    "waiting": "等待",
    "pending": "准备中",
    "running": "推进中",
    "completed": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
    "max_turns": "需继续",
}


CODEDOGGY_DARK = Style.from_dict(
    {
        "root": "bg:#0b0b0d #f5f5f7",
        "header": "bg:#0b0b0d #dce9e9",
        "brand": "#ff2d9a bold",
        "brand.edge.pink": "#ff2d9a bold",
        "brand.edge.cyan": "#16dfe8 bold",
        "header.rule.dim": "#10373e",
        "header.rule.pink": "#ff2d9a bold",
        "header.rule.cyan": "#16dfe8 bold",
        "header.rule.scan": "#d9ffff bold",
        "meta": "#6f8791",
        "separator": "#123b43",
        "task.spine": "#123b43",
        "task.spine.active": "#16dfe8 bold",
        "task.marker": "#ff2d9a bold",
        "task.marker.active": "#ffb13b bold",
        "task.marker.idle": "#49636c",
        "task.title": "#f5f5f7 bold",
        "task.divider": "#123b43",
        "task.divider.pink": "#8f1b58",
        "task.divider.cyan": "#0b6670",
        "task.status": "#6f8791",
        "task.status.running": "#16dfe8 bold",
        "task.status.reporting": "#ff2d9a bold",
        "task.status.completed": "#ffb13b bold",
        "task.status.failed": "#ff2d9a bold",
        "doggy.wordmark": "#ff2d9a bold",
        "agent.border": "#16dfe8",
        "agent.border.selected": "#ff2d9a bold",
        "agent.label": "#16dfe8 bold",
        "agent.label.selected": "#ff9a3c bold",
        "reporter.running": "#16dfe8 bold",
        "reporter.completed": "#ffb13b bold",
        "reporter.waiting": "#6f8791 bold",
        "reporter.failed": "#ff2d9a bold",
        "report": "#dce9e9",
        "input": "bg:#071014 #f5f5f7",
        "input.placeholder": "bg:#071014 #536b75",
        "prompt": "bg:#071014 #ff2d9a bold",
        "prompt.border": "bg:#0b0b0d #16dfe8",
        "prompt.border.focus": "bg:#0b0b0d #ff2d9a bold",
        "prompt.border.dim": "bg:#0b0b0d #401a31",
        "prompt.corner.cyan": "bg:#0b0b0d #16dfe8 bold",
        "prompt.border.info": "bg:#0b0b0d #16dfe8",
        "prompt.border.success": "bg:#0b0b0d #ff9a3c",
        "prompt.border.warning": "bg:#0b0b0d #ff2d9a",
        "prompt.caption": "bg:#0b0b0d #16dfe8 bold",
        "turn.status": "bg:#0b0b0d #16dfe8",
        "turn.elapsed": "bg:#0b0b0d #78909c",
        "turn.stop": "bg:#0b0b0d #ff2d9a bold",
        "feedback.info": "bg:#0b0b0d #16dfe8",
        "feedback.success": "bg:#0b0b0d #ff9a3c",
        "feedback.warning": "bg:#0b0b0d #ff2d9a",
        "shortcut.key": "bg:#0b0b0d #ff2d9a bold",
        "shortcut.label": "bg:#0b0b0d #16dfe8",
        "shortcut.separator": "bg:#0b0b0d #15515a",
        "shortcut.pending": "bg:#0b0b0d #ff9a3c",
        "agent-window": "bg:#0b0b0d #f5f5f7",
        "agent-window.header": "bg:#0b0b0d #ff2d9a bold",
        "agent-window.close": "bg:#40102d #ff5ab3 bold",
        "agent-window.hint": "bg:#0b0b0d #16dfe8",
        "modal.border.left": "bg:#0b0b0d #ff2d9a bold",
        "modal.border.right": "bg:#0b0b0d #16dfe8 bold",
        "modal.border.dim": "bg:#0b0b0d #123b43",
        "modal.border.scan": "bg:#0b0b0d #d9ffff bold",
        "detail.input": "bg:#071014 #f5f5f7",
        "detail.input.prompt": "bg:#071014 #ff9a3c bold",
        **DETAIL_STYLE_RULES,
    }
)


class CodeDoggyTUI:
    """Interactive owner view over one real :class:`codedoggy.Session`."""

    def __init__(
        self,
        session: Any,
        *,
        initial_prompt: str | None = None,
        input: Any | None = None,
        output: Any | None = None,
    ) -> None:
        self.session = session
        self.initial_prompt = initial_prompt
        self.ledger = TaskLedger()
        self._worker: threading.Thread | None = None
        self._active_task_id: str | None = None
        self._agent_refs: list[tuple[str, str]] = []
        self._selected_agent = 0
        self._selected_line = 0
        self._modal_open = False
        self._modal_ref: tuple[str, str] | None = None
        self._detail_messages: dict[tuple[str, str], list[Any]] = {}
        self._detail_filter: DetailFilter = "all"
        self._detail_cursor_line = 0
        self._detail_line_count = 1
        self._closing = False
        self._task_started_at: float | None = None
        self._quit_armed_until = 0.0
        self._feedback_text = ""
        self._feedback_kind = "info"
        self._feedback_until = 0.0
        self._subagent_task: dict[str, str] = {}
        self._subagent_baselines: dict[str, set[str]] = {}

        self._task_control = FormattedTextControl(
            text=self._render_tasks,
            focusable=True,
            show_cursor=False,
            get_cursor_position=lambda: Point(x=0, y=self._selected_line),
        )
        self._task_window = Window(
            content=self._task_control,
            wrap_lines=True,
            scroll_offsets=ScrollOffsets(top=1, bottom=2),
            style="class:root",
        )
        self._input = TextArea(
            height=1,
            multiline=False,
            prompt=self._render_prompt_prefix,
            style="class:input",
            accept_handler=self._accept_prompt,
            input_processors=[
                ConditionalProcessor(
                    AfterInput("交代一个任务…", style="class:input.placeholder"),
                    Condition(
                        lambda: not getattr(self, "_input", None)
                        or not self._input.text
                    ),
                )
            ],
        )
        self._detail_control = FormattedTextControl(
            text=self._render_modal_body,
            focusable=True,
            show_cursor=False,
            get_cursor_position=lambda: Point(x=0, y=self._detail_cursor_line),
        )
        self._detail_window = Window(
            content=self._detail_control,
            wrap_lines=False,
            scroll_offsets=ScrollOffsets(top=1, bottom=2),
            right_margins=[ScrollbarMargin(display_arrows=True)],
            style="class:agent-window",
        )
        self._detail_input = TextArea(
            height=1,
            multiline=False,
            prompt=self._render_detail_prompt_prefix,
            style="class:detail.input",
            accept_handler=self._accept_detail_prompt,
            input_processors=[
                ConditionalProcessor(
                    AfterInput(
                        "补充要求…",
                        style="class:input.placeholder",
                    ),
                    Condition(
                        lambda: not getattr(self, "_detail_input", None)
                        or not self._detail_input.text
                    ),
                )
            ],
        )

        header = Window(
            FormattedTextControl(self._render_header),
            height=1,
            style="class:header",
        )
        separator = Window(
            FormattedTextControl(self._render_header_rule),
            height=1,
            style="class:header",
        )
        turn_status = Window(
            FormattedTextControl(self._render_turn_status),
            height=1,
            style="class:root",
        )
        prompt_top = Window(
            FormattedTextControl(self._render_prompt_top),
            height=1,
            style="class:root",
        )
        prompt_right = Window(
            FormattedTextControl(self._render_prompt_right),
            width=3,
            height=1,
            style="class:root",
        )
        prompt_bottom = Window(
            FormattedTextControl(self._render_prompt_bottom),
            height=1,
            style="class:root",
        )
        shortcuts = Window(
            FormattedTextControl(self._render_shortcuts),
            height=1,
            style="class:root",
        )
        prompt_box = HSplit(
            [
                prompt_top,
                VSplit([self._input, prompt_right]),
                prompt_bottom,
            ],
            style="class:root",
        )
        body = HSplit(
            [
                header,
                separator,
                self._task_window,
                turn_status,
                Window(height=1, style="class:root"),
                prompt_box,
                shortcuts,
            ],
            style="class:root",
        )

        close_control = FormattedTextControl(
            [("class:agent-window.close", "  ×  ", self._close_mouse)],
            focusable=False,
        )
        modal_header = VSplit(
            [
                Window(
                    FormattedTextControl(self._render_modal_title),
                    height=1,
                    style="class:agent-window.header",
                ),
                Window(close_control, width=5, height=1),
            ],
            style="class:agent-window",
        )
        modal_panel = HSplit(
            [
                modal_header,
                Window(height=1, char="─", style="class:separator"),
                Window(
                    FormattedTextControl(self._render_modal_filters),
                    height=1,
                    style="class:agent-window",
                ),
                Window(height=1, char="─", style="class:separator"),
                self._detail_window,
                self._detail_input,
                Window(
                    FormattedTextControl(self._render_modal_hint),
                    height=1,
                    style="class:agent-window.hint",
                ),
            ],
            style="class:agent-window",
        )
        modal_content = ConditionalContainer(
            HSplit(
                [
                    Window(
                        FormattedTextControl(
                            lambda: self._render_modal_border(top=True)
                        ),
                        height=1,
                        style="class:agent-window",
                    ),
                    VSplit(
                        [
                            Window(
                                width=1,
                                char="│",
                                style="class:modal.border.left",
                            ),
                            modal_panel,
                            Window(
                                width=1,
                                char="│",
                                style="class:modal.border.right",
                            ),
                        ],
                        style="class:agent-window",
                    ),
                    Window(
                        FormattedTextControl(
                            lambda: self._render_modal_border(top=False)
                        ),
                        height=1,
                        style="class:agent-window",
                    ),
                ],
                style="class:agent-window",
            ),
            filter=Condition(lambda: self._modal_open),
        )
        root = FloatContainer(
            content=body,
            floats=[
                Float(
                    top=1,
                    bottom=1,
                    left=2,
                    right=2,
                    content=modal_content,
                    transparent=False,
                    z_index=10,
                )
            ],
        )
        self._keys = self._build_key_bindings()
        self.app: Application[None] = Application(
            layout=Layout(root, focused_element=self._input),
            key_bindings=self._keys,
            style=CODEDOGGY_DARK,
            full_screen=True,
            mouse_support=True,
            color_depth=ColorDepth.TRUE_COLOR,
            refresh_interval=0.10,
            before_render=lambda _: self._sync_runtime(),
            input=input,
            output=output,
        )

    def run(self) -> None:
        def pre_run() -> None:
            if self.initial_prompt:
                self._start_task(self.initial_prompt)

        try:
            self.app.run(pre_run=pre_run)
        finally:
            self._closing = True
            if self._worker is not None and self._worker.is_alive():
                self.session.cancel()
                self._worker.join(timeout=3)

    def _build_key_bindings(self) -> KeyBindings:
        keys = KeyBindings()
        modal = Condition(lambda: self._modal_open)
        tasks_focused = Condition(
            lambda: not self._modal_open and get_app().layout.has_focus(self._task_window)
        )
        detail_focused = Condition(
            lambda: self._modal_open
            and get_app().layout.has_focus(self._detail_window)
        )

        @keys.add("tab", filter=~modal)
        def _next_agent(event: Any) -> None:
            self._move_agent(1)
            event.app.layout.focus(self._task_window)

        @keys.add("s-tab", filter=~modal)
        def _previous_agent(event: Any) -> None:
            self._move_agent(-1)
            event.app.layout.focus(self._task_window)

        @keys.add("tab", filter=modal)
        @keys.add("s-tab", filter=modal)
        def _toggle_detail_focus(event: Any) -> None:
            if event.app.layout.has_focus(self._detail_input):
                event.app.layout.focus(self._detail_window)
            else:
                event.app.layout.focus(self._detail_input)

        @keys.add("up", filter=detail_focused)
        def _detail_up(_: Any) -> None:
            self._move_detail_cursor(-1)

        @keys.add("down", filter=detail_focused)
        def _detail_down(_: Any) -> None:
            self._move_detail_cursor(1)

        @keys.add("pageup", filter=detail_focused)
        def _detail_page_up(_: Any) -> None:
            self._move_detail_cursor(-max(4, _terminal_height() - 10))

        @keys.add("pagedown", filter=detail_focused)
        def _detail_page_down(_: Any) -> None:
            self._move_detail_cursor(max(4, _terminal_height() - 10))

        @keys.add("home", filter=detail_focused)
        def _detail_home(_: Any) -> None:
            self._detail_cursor_line = 0
            self.app.invalidate()

        @keys.add("end", filter=detail_focused)
        def _detail_end(_: Any) -> None:
            self._detail_cursor_line = max(0, self._detail_line_count - 1)
            self.app.invalidate()

        for key, detail_filter in zip(
            ("f1", "f2", "f3", "f4", "f5"),
            DETAIL_FILTERS,
            strict=True,
        ):

            @keys.add(key, filter=modal)
            def _set_filter(_: Any, value: DetailFilter = detail_filter) -> None:
                self._set_detail_filter(value)

        @keys.add("enter", filter=tasks_focused)
        def _open_selected(_: Any) -> None:
            self._open_selected_agent()

        @keys.add("space", filter=tasks_focused)
        def _focus_prompt(event: Any) -> None:
            event.app.layout.focus(self._input)

        @keys.add("escape")
        def _escape(event: Any) -> None:
            if self._modal_open:
                self._close_modal()
            else:
                event.app.layout.focus(self._input)

        @keys.add("c-c")
        def _cancel(event: Any) -> None:
            if self._modal_open:
                self._close_modal()
                return
            if self._is_running():
                self._cancel_current()
                return
            if self._input.text:
                self._input.text = ""
                event.app.invalidate()
                return
            self._request_quit()

        @keys.add("c-q")
        def _quit(_: Any) -> None:
            self._request_quit()

        return keys

    def _accept_prompt(self, buffer: Any) -> bool:
        prompt = buffer.text.strip()
        buffer.text = ""
        if not prompt:
            return True
        if self._worker is not None and self._worker.is_alive():
            self.session.interject(prompt, prompt_id=self._active_task_id)
            self._set_feedback("补充指令已送达 MAIN", "info")
            self.app.invalidate()
            return True
        self._start_task(prompt)
        return True

    def _accept_detail_prompt(self, buffer: Any) -> bool:
        prompt = buffer.text.strip()
        buffer.text = ""
        if not prompt or self._modal_ref is None:
            return True
        if not self._is_running():
            self._set_feedback("任务已结束，无法继续插话", "warning", duration=2.2)
            self.app.invalidate()
            return True
        task_id, agent_id = self._modal_ref
        if task_id != self._active_task_id:
            self._set_feedback(
                "只能向当前运行任务补充指令",
                "warning",
                duration=2.2,
            )
            self.app.invalidate()
            return True
        agent = self.ledger.get_agent(task_id, agent_id)
        label = "AGENT" if agent is None else agent.label
        routed = prompt if label == "MAIN" else f"请转交给 {label}：{prompt}"
        self.session.interject(routed, prompt_id=task_id)
        self._set_feedback(f"补充指令已交给 MAIN · {label}", "info")
        self.app.layout.focus(self._detail_window)
        self.app.invalidate()
        return True

    def _start_task(self, prompt: str) -> None:
        task = self.ledger.create(prompt)
        self._active_task_id = task.id
        self._detail_messages[(task.id, f"{task.id}:main")] = []
        self._task_started_at = time.monotonic()
        self._subagent_baselines[task.id] = {
            item.subagent_id for item in self._subagents()
        }
        self._set_feedback("任务已交给 MAIN", "info")
        worker = threading.Thread(
            target=self._run_task,
            args=(task.id, prompt),
            name=f"codedoggy-{task.id}",
            daemon=True,
        )
        self._worker = worker
        worker.start()
        self.app.invalidate()

    def _run_task(self, task_id: str, prompt: str) -> None:
        runner = getattr(self.session.extensions, "turn_runner", None)
        sampler = getattr(runner, "sampler", None)
        detail_key = (task_id, f"{task_id}:main")
        turn_messages = self._detail_messages.setdefault(detail_key, [])
        streamed: list[str] = []
        old_stream = getattr(sampler, "stream", None)
        old_delta = getattr(sampler, "on_delta", None)
        old_live_message = getattr(runner, "on_live_message", None)

        def on_live_message(message: Any) -> None:
            turn_messages.append(message)
            if callable(old_live_message):
                old_live_message(message)
            self.app.invalidate()

        def on_delta(piece: str) -> bool:
            streamed.append(str(piece or ""))
            self.ledger.update_agent(
                task_id,
                f"{task_id}:main",
                label="MAIN",
                status="running",
                output="".join(streamed),
            )
            self.app.invalidate()
            return not self._closing

        if sampler is not None:
            sampler.stream = True
            sampler.on_delta = on_delta
        if runner is not None:
            runner.on_live_message = on_live_message

        try:
            result = self.session.handle_prompt(
                prompt,
                prompt_id=task_id,
                metadata={"tui_task_id": task_id},
            )
            messages = list(turn_messages)
            output = agent_summary_text_from_messages(messages)
            if not output:
                output = (result.final_text or "".join(streamed) or result.error or "").strip()
            status = _turn_status(result.status)
            self.ledger.update_agent(
                task_id,
                f"{task_id}:main",
                label="MAIN",
                status=status,
                output=output,
            )
            report = task_report_from_agent(
                result.final_text or result.error or "任务已结束。"
            )
            self.ledger.set_report(task_id, "MAIN", report)
            self._sync_runtime()

            task = next(
                (item for item in self.ledger.snapshots() if item.id == task_id),
                None,
            )
            children = [] if task is None else task.agents[1:]
            open_children = [
                agent for agent in children if agent.status in {"pending", "running"}
            ]
            failed_children = [
                agent for agent in children if agent.status in {"failed", "cancelled"}
            ]
            if open_children:
                final_status = "failed"
                self._set_feedback("MAIN 未完成并行收口", "warning", duration=2.2)
            elif failed_children:
                final_status = "failed"
                self._set_feedback("子 Agent 未全部成功", "warning", duration=2.2)
            elif status == "completed":
                final_status = "completed"
                self._set_feedback("MAIN 已汇总，任务完成", "success")
            else:
                final_status = status
                self._set_feedback("任务未能完成", "warning", duration=2.2)
            self.ledger.finish_task(task_id, final_status)
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            self.ledger.update_agent(
                task_id,
                f"{task_id}:main",
                label="MAIN",
                status="failed",
                output=message,
            )
            self.ledger.set_report(task_id, "MAIN", message)
            self.ledger.finish_task(task_id, "failed")
            self._set_feedback("任务执行失败", "warning", duration=2.2)
        finally:
            if sampler is not None:
                if old_stream is not None:
                    sampler.stream = old_stream
                sampler.on_delta = old_delta
            if runner is not None:
                runner.on_live_message = old_live_message
            self._sync_runtime()
            if self._active_task_id == task_id:
                self._task_started_at = None
            self.app.invalidate()

    def _sync_runtime(self) -> None:
        snapshots = self._subagents()
        active = self._active_task_id
        if active is not None:
            baseline = self._subagent_baselines.get(active, set())
            for snap in snapshots:
                if snap.subagent_id not in self._subagent_task and snap.subagent_id not in baseline:
                    self._subagent_task[snap.subagent_id] = active

        label_counts: dict[tuple[str, str], int] = {}
        for snap in snapshots:
            task_id = self._subagent_task.get(snap.subagent_id)
            if task_id is None:
                continue
            description = str(snap.description or "").strip()
            raw_label = description or str(snap.subagent_type or "agent")
            base = _truncate_display(raw_label, 18).upper()
            key = (task_id, base)
            label_counts[key] = label_counts.get(key, 0) + 1
            label = base if label_counts[key] == 1 else f"{base} {label_counts[key]}"
            output = subagent_text(snap)
            self.ledger.update_agent(
                task_id,
                snap.subagent_id,
                label=label,
                status=str(snap.status or "waiting"),
                output=output,
                description=description,
            )
            live_messages = getattr(snap, "live_messages", None)
            if live_messages is not None:
                self._detail_messages[(task_id, snap.subagent_id)] = list(live_messages)

        for task in self.ledger.snapshots():
            if task.phase in {"done", "failed", "cancelled"}:
                continue
            children = task.agents[1:]
            if any(agent.status in {"pending", "running"} for agent in children):
                self.ledger.set_task_phase(task.id, "parallel")
            elif children:
                self.ledger.set_task_phase(task.id, "reporting")
            else:
                self.ledger.set_task_phase(task.id, "dispatching")

        if self._modal_open:
            self._detail_cursor_line = min(
                self._detail_cursor_line,
                max(0, self._detail_line_count - 1),
            )

    def _subagents(self) -> list[Any]:
        kernel = getattr(self.session.extensions, "kernel", None)
        coordinator = getattr(kernel, "subagent_coordinator", None)
        if coordinator is None:
            return []
        try:
            return list(coordinator.list_for_parent(str(self.session.id)))
        except Exception:  # noqa: BLE001
            return []

    def _is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _cancel_current(self) -> None:
        if not self._is_running():
            return
        self.session.cancel()
        if self._active_task_id:
            self.ledger.set_task_status(self._active_task_id, "cancelled")
            self.ledger.set_task_phase(self._active_task_id, "cancelled")
        self._set_feedback("已请求停止当前任务", "warning")
        self.app.invalidate()

    def _set_feedback(
        self,
        text: str,
        kind: str = "info",
        *,
        duration: float = 1.6,
    ) -> None:
        """Show a short, event-backed acknowledgement without creating a log."""
        self._feedback_text = text
        self._feedback_kind = kind if kind in {"info", "success", "warning"} else "info"
        self._feedback_until = time.monotonic() + duration

    def _feedback_active(self) -> bool:
        return bool(self._feedback_text) and self._feedback_until > time.monotonic()

    def _request_quit(self) -> None:
        now = time.monotonic()
        if self._quit_armed_until > now:
            self.app.exit()
            return
        self._quit_armed_until = now + 2.0
        self.app.invalidate()

    def _render_turn_status(self) -> StyleAndTextTuples:
        width = max(1, _terminal_width())
        if not self._is_running():
            if self._feedback_active():
                icon = {"info": "●", "success": "✓", "warning": "!"}[
                    self._feedback_kind
                ]
                prefix = f"  {icon} "
                if get_cwidth(prefix) >= width:
                    return [
                        (
                            f"class:feedback.{self._feedback_kind}",
                            _truncate_display(prefix, width),
                        )
                    ]
                return [
                    (f"class:feedback.{self._feedback_kind}", prefix),
                    (
                        "class:turn.status",
                        _truncate_display(
                            self._feedback_text,
                            width - get_cwidth(prefix),
                        ),
                    ),
                ]
            return [("class:turn.status", "")]
        elapsed = max(0.0, time.monotonic() - (self._task_started_at or time.monotonic()))
        spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(elapsed * 8) % 10]
        active = next(
            (task for task in self.ledger.snapshots() if task.id == self._active_task_id),
            None,
        )
        label = _task_activity_text(active) if active is not None else "等待响应…"
        budget = _budget_text(self.session)
        stop = "[停]" if width < 36 else "[停止]"
        trailing = "  " if width >= 12 else ""
        prefix = f"  {spinner} "
        minimum_gap = 1
        fixed = (
            get_cwidth(prefix)
            + minimum_gap
            + get_cwidth(stop)
            + get_cwidth(trailing)
        )
        if width <= fixed:
            compact = _truncate_display(f"{spinner} {stop}", width)
            return [("class:turn.stop", compact, self._stop_mouse)]

        elapsed_piece = ""
        budget_piece = ""
        label_budget = width - fixed
        elapsed_candidate = f" {_format_elapsed(elapsed)}"
        if width >= 28 and label_budget - get_cwidth(elapsed_candidate) >= 4:
            elapsed_piece = elapsed_candidate
            label_budget -= get_cwidth(elapsed_piece)
        budget_candidate = f"{budget}  " if budget else ""
        if (
            width >= 56
            and budget_candidate
            and label_budget - get_cwidth(budget_candidate) >= 4
        ):
            budget_piece = budget_candidate
            label_budget -= get_cwidth(budget_piece)

        label = _truncate_display(label, label_budget)
        left = prefix + label
        gap = max(
            1,
            width
            - get_cwidth(left)
            - get_cwidth(elapsed_piece)
            - get_cwidth(budget_piece)
            - get_cwidth(stop)
            - get_cwidth(trailing),
        )
        return [
            ("class:turn.status", left),
            ("class:turn.elapsed", elapsed_piece),
            ("class:turn.elapsed", " " * gap + budget_piece),
            ("class:turn.stop", stop, self._stop_mouse),
            ("class:turn.elapsed", trailing),
        ]

    def _render_prompt_prefix(self) -> StyleAndTextTuples:
        border = self._prompt_border_class()
        return [(border, "  │ "), ("class:prompt", "› ")]

    def _render_prompt_top(self) -> StyleAndTextTuples:
        width = max(16, _terminal_width())
        border = self._prompt_border_class()
        rail_width = width - 4
        if border != "class:prompt.border.focus" or rail_width < 8:
            return [(border, "  ╭" + "─" * rail_width + "╮")]

        scan = int(time.monotonic() * 14) % rail_width
        styles = ["class:prompt.border.dim"] * rail_width
        styles[0] = border
        for offset in range(3):
            styles[(scan + offset) % rail_width] = border
        fragments: StyleAndTextTuples = [(border, "  ╭")]
        for style, cells in groupby(styles):
            fragments.append((style, "─" * sum(1 for _ in cells)))
        fragments.append(("class:prompt.corner.cyan", "╮"))
        return fragments

    def _render_prompt_right(self) -> StyleAndTextTuples:
        return [("class:prompt.corner.cyan", "│  ")]

    def _render_prompt_bottom(self) -> StyleAndTextTuples:
        width = max(16, _terminal_width())
        caption_text = _truncate_display(_model_and_mode_text(self.session), width - 7)
        caption = f" {caption_text} "
        fill = max(1, width - 4 - get_cwidth(caption))
        border = self._prompt_border_class()
        rail = (
            "class:prompt.border.dim"
            if border == "class:prompt.border.focus"
            else border
        )
        return [
            (border, "  ╰"),
            (rail, "─" * fill),
            ("class:prompt.caption", caption),
            ("class:prompt.corner.cyan", "╯"),
        ]

    def _prompt_border_class(self) -> str:
        if self._feedback_active():
            return f"class:prompt.border.{self._feedback_kind}"
        try:
            focused = get_app().layout.has_focus(self._input)
        except Exception:  # noqa: BLE001
            focused = False
        return "class:prompt.border.focus" if focused else "class:prompt.border"

    def _render_shortcuts(self) -> StyleAndTextTuples:
        now = time.monotonic()
        if self._quit_armed_until and self._quit_armed_until <= now:
            self._quit_armed_until = 0.0
        if self._quit_armed_until > now:
            return [
                ("class:shortcut.pending", "  "),
                ("class:shortcut.key", "Ctrl+Q", self._shortcut_mouse("quit")),
                ("class:shortcut.label", ":再按一次退出", self._shortcut_mouse("quit")),
            ]

        if self._modal_open:
            items = [
                ("PgUp/PgDn", "滚动", "noop", False),
                ("Esc", "关闭", "close", False),
                ("Ctrl+Q", "退出", "quit", True),
            ]
        else:
            try:
                input_focused = get_app().layout.has_focus(self._input)
            except Exception:  # noqa: BLE001
                input_focused = True
            if input_focused:
                items = [
                    ("Enter", "补充" if self._is_running() else "开工", "prompt", False),
                    ("Tab", "Agent", "next", False),
                ]
                if self._is_running():
                    items.append(("Ctrl+C", "取消", "cancel", False))
                items.append(("Ctrl+Q", "退出", "quit", True))
            else:
                items = [
                    ("Tab", "下一个", "next", False),
                    ("Shift+Tab", "上一个", "previous", False),
                    ("Enter", "打开", "open", False),
                    ("Space", "输入", "input", False),
                    ("Ctrl+Q", "退出", "quit", True),
                ]
        return self._fit_shortcuts(items, max(20, _terminal_width() - 4))

    def _fit_shortcuts(
        self,
        items: list[tuple[str, str, str, bool]],
        width: int,
    ) -> StyleAndTextTuples:
        pinned = next((item for item in items if item[3]), None)
        regular = [item for item in items if not item[3]]

        def item_width(item: tuple[str, str, str, bool]) -> int:
            return get_cwidth(item[0]) + 1 + get_cwidth(item[1])

        chosen: list[tuple[str, str, str, bool]] = []
        used = 2
        reserved = item_width(pinned) + (5 if pinned else 0) if pinned else 0
        for item in regular:
            extra = item_width(item) + (5 if chosen else 0)
            if used + extra + reserved > width:
                break
            chosen.append(item)
            used += extra
        if pinned is not None:
            chosen.append(pinned)

        fragments: StyleAndTextTuples = [("", "  ")]
        for index, (key, label, action, _) in enumerate(chosen):
            if index:
                fragments.append(("class:shortcut.separator", "  │  "))
            handler = self._shortcut_mouse(action)
            fragments.append(("class:shortcut.key", key, handler))
            fragments.append(("class:shortcut.label", f":{label}", handler))
        return fragments

    def _shortcut_mouse(self, action: str) -> Callable[[MouseEvent], None]:
        def handler(event: MouseEvent) -> None:
            if event.event_type is not MouseEventType.MOUSE_UP:
                return
            if action == "quit":
                self._request_quit()
            elif action == "cancel":
                self._cancel_current()
            elif action == "close":
                self._close_modal()
            elif action == "next":
                self._move_agent(1)
                self.app.layout.focus(self._task_window)
            elif action == "previous":
                self._move_agent(-1)
                self.app.layout.focus(self._task_window)
            elif action == "open":
                self._open_selected_agent()
            elif action == "input":
                self.app.layout.focus(self._input)
            elif action == "prompt":
                self.app.layout.focus(self._input)
            self.app.invalidate()

        return handler

    def _stop_mouse(self, event: MouseEvent) -> None:
        if event.event_type is MouseEventType.MOUSE_UP:
            self._cancel_current()

    def _render_header(self) -> StyleAndTextTuples:
        width = max(1, _terminal_width())
        left = "  ==DOGGY=="
        right = _budget_text(self.session)
        if width < get_cwidth(left):
            return [("class:brand", _truncate_display(left, width))]

        pulse = int(time.monotonic() * 2) % 2
        edge_left = (
            "class:brand.edge.pink" if pulse == 0 else "class:brand.edge.cyan"
        )
        edge_right = (
            "class:brand.edge.cyan" if pulse == 0 else "class:brand.edge.pink"
        )
        fragments: StyleAndTextTuples = [
            ("class:header", "  "),
            (edge_left, "=="),
            ("class:brand", "DOGGY"),
            (edge_right, "=="),
        ]
        if not right or get_cwidth(left) + get_cwidth(right) + 2 > width:
            return fragments
        gap = width - get_cwidth(left) - get_cwidth(right) - 1
        fragments.append(("class:meta", " " * gap + right + " "))
        return fragments

    def _render_header_rule(self) -> StyleAndTextTuples:
        width = max(1, _terminal_width())
        styles = ["class:header.rule.dim"] * width
        for index in range(min(4, width)):
            styles[index] = "class:header.rule.pink"
        for index in range(max(0, width - 4), width):
            styles[index] = "class:header.rule.cyan"
        scan = int(time.monotonic() * 18) % width
        for offset in range(min(3, width)):
            styles[(scan + offset) % width] = "class:header.rule.scan"
        fragments: StyleAndTextTuples = []
        for style, cells in groupby(styles):
            fragments.append((style, "─" * sum(1 for _ in cells)))
        return fragments

    def _render_modal_border(self, *, top: bool) -> StyleAndTextTuples:
        width = max(4, _terminal_width() - 4)
        rail_width = width - 2
        styles = ["class:modal.border.dim"] * rail_width
        scan = int(time.monotonic() * 10) % rail_width
        for offset in range(min(3, rail_width)):
            styles[(scan + offset) % rail_width] = "class:modal.border.scan"
        fragments: StyleAndTextTuples = [
            ("class:modal.border.left", "╭" if top else "╰")
        ]
        for style, cells in groupby(styles):
            fragments.append((style, "─" * sum(1 for _ in cells)))
        fragments.append(
            ("class:modal.border.right", "╮" if top else "╯")
        )
        return fragments

    def _render_tasks(self) -> StyleAndTextTuples:
        tasks = self.ledger.snapshots()
        fragments: StyleAndTextTuples = []
        refs: list[tuple[str, str]] = []
        selected_line = 0
        line = 0
        width = max(1, _terminal_width() - 2)

        if not tasks:
            return _render_doggy_empty(width)

        for task_index, task in enumerate(tasks):
            active = task.phase in {"dispatching", "parallel", "reporting"}
            spine_style = "class:task.spine.active" if active else "class:task.spine"
            prefix = "  │  " if active else "     "
            status = (
                _compact_task_stage_text(task)
                if width < 34
                else _task_stage_text(task)
            )
            marker = "◆" if active else "•"
            marker_style = (
                "class:task.marker.active"
                if active and int(time.monotonic() * 4) % 2 == 0
                else (
                    "class:task.marker"
                    if active
                    else "class:task.marker.idle"
                )
            )
            minimum_gap = 1 if width < 34 else 2
            fixed_width = get_cwidth(prefix) + 1 + 2 + minimum_gap + 2
            title_budget = max(1, width - get_cwidth(status) - fixed_width)
            title = _truncate_display(task.title, title_budget)
            left = f"{prefix}{marker}  {title}"
            gap = max(minimum_gap, width - get_cwidth(left) - get_cwidth(status) - 2)
            fragments.extend(
                [
                    (spine_style, prefix),
                    (marker_style, marker),
                    ("class:task.title", f"  {title}"),
                    (_task_status_style(task), " " * gap + status + "  \n"),
                ]
            )
            line += 1
            fragments.extend([(spine_style, prefix), ("", "\n")])
            line += 1

            boxes, line, selected_line = self._render_agent_boxes(
                task,
                width,
                refs,
                line,
                selected_line,
                prefix,
                spine_style,
            )
            fragments.extend(boxes)

            divider_width = max(1, width - get_cwidth(prefix) - 4)
            fragments.extend(
                [
                    (spine_style, prefix),
                    ("class:task.divider.pink", "  ╾"),
                    (
                        "class:task.divider",
                        "┈" * max(1, divider_width - 2),
                    ),
                    ("class:task.divider.cyan", "╼\n"),
                ]
            )
            line += 1
            for reporter, report, agent_status in _task_briefs(task):
                available = max(2, width - get_cwidth(prefix))
                label_width = min(14, max(1, available // 3))
                label = _truncate_display(reporter, label_width)
                padded_label = label + " " * max(0, label_width - get_cwidth(label))
                report_width = max(1, available - label_width)
                fragments.extend(
                    [
                        (spine_style, prefix),
                        (_reporter_style(agent_status), padded_label),
                        (
                            "class:report",
                            _truncate_display(report, report_width) + "\n",
                        ),
                    ]
                )
                line += 1
            if task_index != len(tasks) - 1:
                fragments.append(
                    ("class:separator", "  " + "─" * max(1, width - 4) + "\n")
                )
                line += 1

        mascot = _render_doggy_corner(width) if width >= 72 else []
        mascot_lines = sum(fragment[1].count("\n") for fragment in mascot)
        task_height = max(8, _terminal_height() - 8)
        if mascot and line + mascot_lines <= task_height:
            padding = task_height - line - mascot_lines
            fragments.append(("", "\n" * padding))
            fragments.extend(mascot)

        self._agent_refs = refs
        if refs:
            self._selected_agent %= len(refs)
        else:
            self._selected_agent = 0
        self._selected_line = selected_line
        return fragments

    def _render_agent_boxes(
        self,
        task: TaskView,
        width: int,
        refs: list[tuple[str, str]],
        line: int,
        selected_line: int,
        prefix: str,
        spine_style: str,
    ) -> tuple[StyleAndTextTuples, int, int]:
        content_width = max(1, width - get_cwidth(prefix) - 2)
        chips: list[tuple[int, str, int]] = []
        for agent in task.agents:
            index = len(refs)
            refs.append((task.id, agent.id))
            label = _truncate_display(agent.label, max(1, min(14, content_width - 7)))
            inner = f" {label}  › "
            chips.append((index, inner, get_cwidth(inner) + 2))

        groups: list[list[tuple[int, str, int]]] = []
        current: list[tuple[int, str, int]] = []
        used = 0
        for chip in chips:
            extra = chip[2] + (2 if current else 0)
            if current and used + extra > content_width:
                groups.append(current)
                current = []
                used = 0
                extra = chip[2]
            current.append(chip)
            used += extra
        if current:
            groups.append(current)

        fragments: StyleAndTextTuples = []
        for group in groups:
            fragments.extend([(spine_style, prefix), ("", "  ")])
            for chip_index, (index, inner, _box_width) in enumerate(group):
                selected = index == self._selected_agent
                border = (
                    "class:agent.border.selected"
                    if selected
                    else "class:agent.border"
                )
                label_style = (
                    "class:agent.label.selected"
                    if selected
                    else "class:agent.label"
                )
                handler = self._agent_mouse(index)
                if chip_index:
                    fragments.append(("", "  "))
                fragments.extend(
                    [
                        (border, "╭", handler),
                        (label_style, inner, handler),
                        (border, "╮", handler),
                    ]
                )
                if selected:
                    selected_line = line
            fragments.append(("", "\n"))
            line += 1
        return fragments, line, selected_line

    def _render_modal_title(self) -> StyleAndTextTuples:
        if not self._modal_ref:
            return []
        task_id, agent_id = self._modal_ref
        agent = self.ledger.get_agent(task_id, agent_id)
        task = next((item for item in self.ledger.snapshots() if item.id == task_id), None)
        if agent is None or task is None:
            return []
        width = max(12, _terminal_width() - 9)
        left = f"  ‹ {agent.label} · {task.title}"
        right = STATUS_TEXT.get(agent.status, agent.status)
        if agent.status in {"pending", "running"}:
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[
                int(time.monotonic() * 8) % 10
            ]
            right = f"{spinner} {right}"
        if get_cwidth(left) + get_cwidth(right) + 2 <= width:
            gap = width - get_cwidth(left) - get_cwidth(right)
            return [
                ("class:agent-window.header", left),
                ("", " " * gap),
                ("class:detail.active", right),
            ]
        return [("class:agent-window.header", _truncate_display(left, width))]

    def _render_modal_filters(self) -> StyleAndTextTuples:
        width = max(12, _terminal_width() - 8)
        fragments: StyleAndTextTuples = [("", "  ")]
        used = 2
        for index, detail_filter in enumerate(DETAIL_FILTERS, start=1):
            active = detail_filter == self._detail_filter
            base_label = f"F{index} {DETAIL_FILTER_LABELS[detail_filter]}"
            label = f"╾ {base_label} ╼" if active else base_label
            piece_width = get_cwidth(label) + (3 if used > 2 else 0)
            if used + piece_width > width:
                break
            if used > 2:
                fragments.append(("class:detail.meta", " · "))
            style = (
                "class:detail.active"
                if active
                else "class:detail.meta"
            )
            fragments.append(
                (style, label, self._detail_filter_mouse(detail_filter))
            )
            used += piece_width
        return fragments

    def _render_modal_body(self) -> StyleAndTextTuples:
        snapshot = self._current_detail_snapshot()
        if snapshot is None:
            self._detail_line_count = 1
            return [("class:detail.meta", "当前 Agent 没有可用记录。\n")]
        width = max(12, _terminal_width() - 8)
        fragments = render_detail_body(
            snapshot,
            width,
            active_filter=self._detail_filter,
        )
        self._detail_line_count = max(
            1,
            sum(fragment[1].count("\n") for fragment in fragments),
        )
        self._detail_cursor_line = min(
            self._detail_cursor_line,
            self._detail_line_count - 1,
        )
        return fragments

    def _current_detail_snapshot(self) -> AgentDetailSnapshot | None:
        if self._modal_ref is None:
            return None
        task_id, agent_id = self._modal_ref
        task = next((item for item in self.ledger.snapshots() if item.id == task_id), None)
        agent = self.ledger.get_agent(task_id, agent_id)
        if task is None or agent is None:
            return None
        messages = list(self._detail_messages.get((task_id, agent_id), []))
        if not messages:
            is_main = agent_id == f"{task_id}:main"
            if not is_main and agent.status in {"pending", "running"}:
                fallback = (
                    "子 Agent 正在执行。当前运行时只在本轮结束后同步完整工具记录；"
                    "完成后此页会显示真实消息、工具参数与返回结果。"
                )
            else:
                fallback = agent.output.strip() or agent.description.strip()
            if fallback:
                messages = [Message(role=Role.ASSISTANT, content=fallback)]
        return snapshot_from_messages(
            messages,
            task_id=task_id,
            agent_id=agent_id,
            agent_label=agent.label,
            task_title=task.title,
            status=agent.status,
        )

    def _render_detail_prompt_prefix(self) -> StyleAndTextTuples:
        label = "MAIN"
        if self._modal_ref:
            task_id, agent_id = self._modal_ref
            agent = self.ledger.get_agent(task_id, agent_id)
            if agent is not None:
                label = agent.label
        terminal_width = max(1, _terminal_width())
        if terminal_width < 40:
            text = "  › "
        else:
            if label == "MAIN":
                text = "  › 给 MAIN 补充指令  "
            else:
                text = f"  › 请 MAIN 转交给 {label}  "
            text = _truncate_display(text, max(4, min(36, terminal_width - 20)))
        return [("class:detail.input.prompt", text)]

    def _render_modal_hint(self) -> StyleAndTextTuples:
        text = "  ↑↓/PgUp/PgDn 滚动 · F1-F5 筛选 · Tab 补充指令 · Esc 返回"
        return [
            (
                "class:agent-window.hint",
                _truncate_display(text, max(1, _terminal_width() - 6)),
            )
        ]

    def _detail_filter_mouse(
        self, detail_filter: DetailFilter
    ) -> Callable[[MouseEvent], None]:
        def handler(event: MouseEvent) -> None:
            if event.event_type is MouseEventType.MOUSE_UP:
                self._set_detail_filter(detail_filter)

        return handler

    def _set_detail_filter(self, detail_filter: DetailFilter) -> None:
        self._detail_filter = detail_filter
        self._detail_cursor_line = 0
        self.app.layout.focus(self._detail_window)
        self.app.invalidate()

    def _move_detail_cursor(self, delta: int) -> None:
        self._detail_cursor_line = min(
            max(0, self._detail_cursor_line + delta),
            max(0, self._detail_line_count - 1),
        )
        self.app.invalidate()

    def _move_agent(self, delta: int) -> None:
        self._render_tasks()
        if not self._agent_refs:
            return
        self._selected_agent = (self._selected_agent + delta) % len(self._agent_refs)
        self.app.invalidate()

    def _open_selected_agent(self) -> None:
        self._render_tasks()
        if not self._agent_refs:
            return
        task_id, agent_id = self._agent_refs[self._selected_agent]
        self._open_agent(task_id, agent_id)

    def _open_agent(self, task_id: str, agent_id: str) -> None:
        agent = self.ledger.get_agent(task_id, agent_id)
        if agent is None:
            return
        self._modal_ref = (task_id, agent_id)
        self._detail_filter = "all"
        self._detail_cursor_line = 0
        self._detail_input.text = ""
        self._modal_open = True
        self.app.layout.focus(self._detail_window)
        self.app.invalidate()

    def _close_modal(self) -> None:
        self._modal_open = False
        self._modal_ref = None
        self._detail_input.text = ""
        self.app.layout.focus(self._task_window)
        self.app.invalidate()

    def _agent_mouse(self, index: int) -> Callable[[MouseEvent], None]:
        def handler(event: MouseEvent) -> None:
            if event.event_type is MouseEventType.MOUSE_UP:
                self._selected_agent = index
                if 0 <= index < len(self._agent_refs):
                    self._open_agent(*self._agent_refs[index])

        return handler

    def _close_mouse(self, event: MouseEvent) -> None:
        if event.event_type is MouseEventType.MOUSE_UP:
            self._close_modal()


def run_tui(session: Any, *, initial_prompt: str | None = None) -> None:
    CodeDoggyTUI(session, initial_prompt=initial_prompt).run()


def agent_summary_text_from_messages(messages: list[Any]) -> str:
    """Return assistant prose for the compact overview, not the detail page."""
    parts: list[str] = []
    for message in messages:
        role = getattr(message, "role", None)
        if role is not Role.ASSISTANT and getattr(role, "value", role) != "assistant":
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts)


def subagent_text(snapshot: Any) -> str:
    text = str(getattr(snapshot, "output", "") or "").strip()
    if text.startswith("[subagent:") and "\n" in text:
        text = text.split("\n", 1)[1].strip()
    if text:
        return text
    error = str(getattr(snapshot, "error", "") or "").strip()
    if error:
        return error
    if str(getattr(snapshot, "status", "")) in {"pending", "running"}:
        return "Agent 正在工作，完成后会在这里给出完整输出。"
    return "Agent 已结束，没有留下文字输出。"


def task_report_from_agent(text: str, *, max_chars: int = 260) -> str:
    """Keep the boss view brief while preserving MAIN's own final wording."""
    clean = text.strip()
    if not clean:
        return "任务已结束。"
    paragraphs = [" ".join(part.split()) for part in re.split(r"\n\s*\n", clean)]
    report = next((part for part in paragraphs if part), clean)
    report = re.sub(r"^#{1,6}\s+", "", report)
    if len(report) <= max_chars:
        return report
    return report[: max_chars - 1].rstrip() + "…"


def _turn_status(status: TurnStatus | Any) -> str:
    value = getattr(status, "value", status)
    if value == TurnStatus.COMPLETED.value:
        return "completed"
    if value == TurnStatus.CANCELLED.value:
        return "cancelled"
    if value == TurnStatus.MAX_TURNS_REACHED.value:
        return "max_turns"
    return "failed"


def _terminal_width() -> int:
    try:
        return get_app().output.get_size().columns
    except Exception:  # noqa: BLE001
        return shutil.get_terminal_size(fallback=(100, 30)).columns


def _terminal_height() -> int:
    try:
        return get_app().output.get_size().rows
    except Exception:  # noqa: BLE001
        return shutil.get_terminal_size(fallback=(100, 30)).lines


def _budget_text(session: Any) -> str:
    context = getattr(session.extensions, "context", None)
    budget = getattr(context, "budget", None)
    used = getattr(budget, "last_prompt_tokens", None)
    total = getattr(budget, "context_window", None)
    if not total:
        return ""
    used_text = "—" if used is None else _compact_number(int(used))
    return f"{used_text} / {_compact_number(int(total))}"


def _compact_number(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(value)


def _model_and_mode_text(session: Any) -> str:
    runner = getattr(session.extensions, "turn_runner", None)
    sampler = getattr(runner, "sampler", None)
    client = getattr(sampler, "client", None)
    config = getattr(client, "config", None)
    model = str(getattr(config, "model", "") or "model")
    kernel = getattr(session.extensions, "kernel", None)
    mode_state = getattr(kernel, "session_mode_state", None)
    raw_mode = getattr(getattr(mode_state, "mode", None), "value", None)
    mode = {"normal": "auto", "goal": "goal", "plan": "plan"}.get(
        str(raw_mode or "normal"), str(raw_mode or "auto")
    )
    return f"{model} · {mode}"


def _format_elapsed(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    remain = int(seconds % 60)
    return f"{minutes}m{remain:02d}s"


def _task_stage_text(task: TaskView) -> str:
    if task.phase == "dispatching":
        return "MAIN 拆解中"
    if task.phase == "parallel":
        active = sum(
            agent.status in {"pending", "running"} for agent in task.agents
        )
        return f"{max(1, active)} 个 Agent 并行中"
    if task.phase == "reporting":
        return "MAIN 汇总中"
    if task.phase == "done":
        return f"已完成 · {len(task.agents)} 个 Agent"
    return STATUS_TEXT.get(task.status, task.status)


def _compact_task_stage_text(task: TaskView) -> str:
    if task.phase == "dispatching":
        return "拆解中"
    if task.phase == "parallel":
        active = sum(
            agent.status in {"pending", "running"} for agent in task.agents
        )
        return f"{max(1, active)} 并行"
    if task.phase == "reporting":
        return "汇总中"
    if task.phase == "done":
        return f"完成·{len(task.agents)}"
    return STATUS_TEXT.get(task.status, task.status)


def _task_status_style(task: TaskView) -> str:
    if task.status in {"failed", "max_turns"}:
        return "class:task.status.failed"
    if task.phase == "done":
        return "class:task.status.completed"
    if task.phase == "reporting":
        return "class:task.status.reporting"
    if task.phase in {"dispatching", "parallel"}:
        return "class:task.status.running"
    return "class:task.status"


def _task_briefs(task: TaskView) -> list[tuple[str, str, str]]:
    """Return one boss-readable first paragraph per reporting Agent."""
    briefs: list[tuple[str, str, str]] = []
    report_matched = False
    for agent in task.agents:
        raw = agent.output
        if task.report and agent.label == task.reporter:
            raw = task.report
            report_matched = True
        if raw.strip():
            briefs.append(
                (agent.label, task_report_from_agent(raw), agent.status)
            )
    if task.report and not report_matched:
        briefs.append((task.reporter, task_report_from_agent(task.report), task.status))
    if not briefs:
        main = task.agents[0] if task.agents else None
        briefs.append(
            (
                main.label if main is not None else "MAIN",
                _task_activity_text(task),
                main.status if main is not None else task.status,
            )
        )
    return briefs


def _reporter_style(status: str) -> str:
    if status in {"running", "pending"}:
        return "class:reporter.running"
    if status == "completed":
        return "class:reporter.completed"
    if status in {"failed", "max_turns"}:
        return "class:reporter.failed"
    return "class:reporter.waiting"


_DOGGY_CITY_ART = (
    "........................FFF......FF.............................",
    "........................FMFF....FFF.....SSS.....................",
    "........................FMMF....FMFF...FSSS.....................",
    "............MM..........FMMFFFFFFMMF..SSSS......................",
    "............MM..........FMMFFFFFFFF...SSS.......................",
    ".......M....MM..........FFFFFFFFFFF...SSSSS.....................",
    ".......M.....M...........FFFFFWFFFFW...SSSS.....................",
    ".......M....MMM...........FF.WWFFFWW..SSS.......................",
    "..M...MMM..MMMM..........FFFFFFFFFFF..SFS.......................",
    "..M...MMM..MMM...........FFFFFFDDDFF.SSS........................",
    "M.M...MMM..M.M...........GFFFFDDDDGWWM..........................",
    "MMM...MMMMMM.M.MM.....CC.GGGFDDDDDCCCC..........................",
    "MMM.M.MMM.MM.F.M.CCCCCCCFFGGGFFFGGFFCCCCCC......................",
    "MMM.M..MM.M..CCCCC..CCC.FFFGGGFGGGFFF....CCC....................",
    "MMM.M.....CCCCCCCCCCCCCCFFFFFGGGGDD..DD.CC.CCCC.................",
    "MMM.M.CCCCCCCC.C.CCCCC.CCCFFFCCGD.DDD.D.CCCC..CCCC..............",
    "M...CCCBCCCCCCCCCCCCCCCCCCCC..CCCCCCCCCCCCCCCCCCCCCCCC..........",
    "M..CCCCCBCCCCBCCCCCCCCCCCBCCCC....CC..........CCC...CCCC........",
    "MM.CCCCCCCC.CCCC...CCCCDCCDDCCCCCC.CCC..........CC....CCC.......",
    "MM.CCCCCCCC..CCB....CCCCCCCBBCCCWCCCCCC...........CC.CCWCC......",
    "MMMCCDCC.CC..CC.....CCCDCCCCCBCCWWW.CCCC.......CC..CCCCWWCC.....",
    "...CCGCCC.CC.CC.....CCDDDCCCCCCCCCWCCC.CCCCCCCCCCCCCCCCCCCC.....",
    "MMMCCGCCC..CCCCCC...CCDGDCCDDD..CCCCC...CC.....GGG....C..CCC....",
    ".MM.CDCCCC..CCCCCCCCCCDG.CCBB....CCCCCCCCCCCCCCCCCCCCCCCCCC.....",
    "MMM.CDCCCC......CCCCCCDDDCCB....CCD......CC..........CC...C.....",
    "....CCC.CCCCCCCCCC..CCCDCCCCC..CCD......CCCCCCCCCCCCCC...CC.....",
    "MMMM.DD..MM...CCCCCCCCCCCDC.CC.CCC.....CCCCCCCCCCCCCCCCCCCC.....",
    "..MMMMMM..MMMMMM......CCC.CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC.....",
    "......MMMMMM...M..MMM..D........................................",
    "..........MMMMMM....MMMM........................................",
    ".............MMMMMMM....MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM...",
    ".................MMMMMM.........................................",
)

_DOGGY_CORNER_ART = (
    "........................",
    "...GG..........GG.......",
    "..GDDG........GDDG......",
    "..GFDG........GDFG......",
    "..GGGGGGGGGGGGGGGG......",
    "...GGDDDDGGDDDDGGG......",
    "...GDWDDGGDDWDGGG.......",
    "...GGDDDDDDDDDDGGG......",
    "....GGGGDFFFDGGGG.......",
    ".....GGFFFFFFFFGG.......",
    "......GGFFDDFFGG........",
    "....GGGGGGGGGGGGGG......",
    "...GGGGGGGGGGGGGGGG.....",
    "...GGGGCCCCCCCCGGGG.....",
    "...GGGGGGCCCCGGGGGG.....",
    "..GGGGGFFFFFFFFGGGG.D...",
    "..GGGGFFFFFFFFFFGGG.DD..",
    "...GGGFFFFFFFFFFGG.GG...",
    "...GGGG...FFFF...GGGG...",
    "..GGGG....FFFF....GGGG..",
    "..GG......G..G......GG..",
    "........................",
)

_DOGGY_ART_PALETTE = {
    ".": "#0b0b0d",
    "C": "#16dfe8",
    "M": "#ff2d9a",
    "c": "#0b6670",
    "m": "#8f1b58",
    "G": "#ff7a32",
    "Y": "#ffd43b",
    "T": "#ff9a3c",
    "P": "#ff2d9a",
    "R": "#071014",
    "F": "#f0c7a4",
    "D": "#2c2c2e",
    "S": "#8e8e93",
    "W": "#f5f5f7",
    "B": "#1c1c1e",
}

_DOGGY_ART_PRIORITY = {
    ".": 0,
    "B": 1,
    "D": 2,
    "S": 3,
    "c": 4,
    "m": 4,
    "F": 4,
    "C": 5,
    "M": 6,
    "G": 7,
    "Y": 8,
    "T": 8,
    "P": 8,
    "R": 1,
    "W": 9,
}

# The large idle art is still portable ANSI/Unicode, but these overlays add
# the details that matter at terminal resolution: sunglasses, gold chain,
# moving cigarette smoke and a separate exhaust plume behind the car.
_DOGGY_GLASSES_PIXELS = {
    (26, 6): "D", (27, 6): "W", (28, 6): "D", (29, 6): "D", (30, 6): "D",
    (32, 6): "D", (33, 6): "W", (34, 6): "D", (35, 6): "D", (36, 6): "D",
    (26, 7): "D", (27, 7): "D", (28, 7): "D", (29, 7): "D", (30, 7): "D",
    (31, 7): "D",
    (32, 7): "D", (33, 7): "D", (34, 7): "D", (35, 7): "D", (36, 7): "D",
    (27, 8): "D", (28, 8): "D", (29, 8): "D", (30, 8): "D",
    (32, 8): "D", (33, 8): "D", (34, 8): "D", (35, 8): "D",
}

_DOGGY_CHAIN_PIXELS = (
    (25, 10), (34, 10),
    (25, 11), (26, 11), (27, 11),
    (26, 12), (27, 12), (28, 12), (32, 12), (33, 12),
    (27, 13), (28, 13), (29, 13), (31, 13), (32, 13), (33, 13),
    (29, 14), (30, 14), (31, 14), (32, 14),
    (31, 15),
)

_DOGGY_SMOKE_FRAMES = (
    ((38, 9), (38, 8), (39, 7), (39, 6), (40, 5), (40, 4), (39, 3), (40, 2), (41, 1)),
    ((38, 9), (39, 8), (39, 7), (40, 6), (41, 5), (40, 4), (41, 3), (42, 2), (43, 1)),
    ((38, 9), (38, 8), (40, 7), (41, 6), (41, 5), (42, 4), (41, 3), (43, 2), (44, 1)),
    ((38, 9), (39, 8), (40, 7), (39, 6), (40, 5), (41, 4), (42, 3), (42, 2), (44, 1)),
)

_DOGGY_EXHAUST_FRAMES = (
    ((4, 23), (3, 24), (2, 25), (1, 25), (0, 26)),
    ((4, 24), (3, 24), (2, 24), (1, 25), (0, 25)),
    ((4, 23), (3, 24), (2, 24), (1, 24), (0, 25)),
    ((4, 24), (3, 25), (2, 25), (1, 26), (0, 26)),
)

_DOGGY_WHEEL_CENTERS = ((5, 22), (23, 22))
_DOGGY_WHEEL_SPOKES = (
    ((0, -1), (0, 1)),
    ((-1, -1), (1, 1)),
    ((-1, 0), (1, 0)),
    ((1, -1), (-1, 1)),
)


def _animate_doggy_city(rows: tuple[str, ...], frame: int) -> tuple[str, ...]:
    """Return one refined animation frame without mutating the source art."""

    canvas = [list(row.replace("S", ".")) for row in rows]
    # Remove the two legacy hot-pink masses called out in the visual review:
    # the solid skyline behind the rear half of the car and the horizontal
    # under-car trail. Procedural skyline, palms and diagonal track stay intact.
    for y, row in enumerate(canvas):
        for x, value in enumerate(row):
            if value == "M" and ((x < 24 and y < 28) or y >= 27):
                canvas[y][x] = "."
    for (x, y), value in _DOGGY_GLASSES_PIXELS.items():
        canvas[y][x] = value
    canvas[6][27] = "D"
    canvas[6][33] = "D"
    glint_shift = frame % 3
    canvas[6][27 + glint_shift] = "W"
    canvas[6][33 + glint_shift] = "W"
    for x, y in _DOGGY_CHAIN_PIXELS:
        canvas[y][x] = "Y"
    for x, y in _DOGGY_SMOKE_FRAMES[frame % len(_DOGGY_SMOKE_FRAMES)]:
        canvas[y][x] = "S"
    for x, y in _DOGGY_EXHAUST_FRAMES[frame % len(_DOGGY_EXHAUST_FRAMES)]:
        canvas[y][x] = "S"
    ring = (
        (-1, -2), (0, -2), (1, -2),
        (-2, -1), (2, -1),
        (-2, 0), (2, 0),
        (-2, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2),
    )
    for center_x, center_y in _DOGGY_WHEEL_CENTERS:
        for dx, dy in ring:
            canvas[center_y + dy][center_x + dx] = "D"
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                canvas[center_y + dy][center_x + dx] = "B"
        for dx, dy in _DOGGY_WHEEL_SPOKES[frame % len(_DOGGY_WHEEL_SPOKES)]:
            canvas[center_y + dy][center_x + dx] = "W"
        canvas[center_y][center_x] = "Y"
    # Cigarette ember flickers independently from the smoke ribbon.
    canvas[10][37] = "Y" if frame % 2 else "M"
    return tuple("".join(row) for row in canvas)


def _compose_doggy_scene(
    art_rows: tuple[str, ...],
    width: int,
    scene_time: float,
) -> tuple[str, ...]:
    """Compose the diagonal road behind the foreground dog-and-car art."""

    height = len(art_rows)
    scene_width = max(1, width)
    flicker_tick = int(scene_time * 4)
    canvas = [["."] * scene_width for _ in range(height)]

    def put(x: int, y: int, value: str, *, overwrite: bool = True) -> None:
        if 0 <= x < scene_width and 0 <= y < height:
            if overwrite or canvas[y][x] == ".":
                canvas[y][x] = value

    # A diagonal road runs between two parallel rails in exactly the
    # bottom-right -> top-left direction indicated by the reference arrows.
    road_top = 3
    road_span = max(1, height - road_top - 1)
    road_width = max(24, int(scene_width * 0.68))
    for y in range(road_top, height):
        progress = (y - road_top) / road_span
        left = int(scene_width * (-0.18 + 0.48 * progress))
        right = left + road_width
        for x in range(left + 1, min(scene_width, right)):
            put(x, y, "R")
        put(left, y, "M")
        put(left + 1, y, "M")
        put(right - 1, y, "C")
        put(right, y, "C")
        lane = left + road_width // 2
        if (y + flicker_tick) % 6 < 3:
            put(lane, y, "Y")
            put(lane + 1, y, "Y")

    art_width = len(art_rows[0]) if art_rows else 0

    # Foreground art wins over the procedural road; transparent cells leave
    # the track visible around and through the vehicle silhouette.
    art_left = max(0, (scene_width - art_width) // 2)
    for y, row in enumerate(art_rows):
        for x, value in enumerate(row):
            if value != ".":
                put(art_left + x, y, value)
    return tuple("".join(row) for row in canvas)


def _render_doggy_empty(
    width: int,
    *,
    now: float | None = None,
) -> StyleAndTextTuples:
    """Render the idle Frenchie cockpit as opaque true-colour terminal pixels."""
    clock = time.monotonic() if now is None else now
    art_tick = int(clock * 4)
    frame = art_tick % len(_DOGGY_SMOKE_FRAMES)
    rows = _animate_doggy_city(_DOGGY_CITY_ART, frame)
    target_width = max(1, min(len(rows[0]), width - 4))
    if target_width < len(rows[0]):
        rows = tuple(_fit_art_row(row, target_width) for row in rows)

    rows = _compose_doggy_scene(rows, width, clock)
    art_width = len(rows[0])
    outer = max(0, width - art_width)
    tick = frame % 2
    palette = dict(_DOGGY_ART_PALETTE)
    if tick:
        palette["M"] = "#ff5ab3"
        palette["S"] = "#aeaeb2"

    fragments: StyleAndTextTuples = [("", "\n")]
    for top, bottom in zip(rows[::2], rows[1::2], strict=True):
        fragments.append(("", " " * outer))
        pairs = zip(top, bottom, strict=True)
        for pair, cells in groupby(pairs):
            count = sum(1 for _ in cells)
            style, glyph = _half_block(pair[0], pair[1], palette)
            fragments.append((style, glyph * count))
        fragments.append(("", "\n"))
    runway_height = max(0, min(16, _terminal_height() - 25))
    fragments.extend(_render_neon_track(width, runway_height, int(clock * 6)))
    return fragments


def _render_neon_track(
    width: int,
    height: int,
    frame: int,
) -> StyleAndTextTuples:
    """Continue the scene's parallel diagonal rails toward the lower-right."""

    if height <= 0 or width < 12:
        return []
    fragments: StyleAndTextTuples = []
    base_style = "bg:#0b0b0d"
    road_style = "bg:#071014"
    edge_pink = "fg:#ff2d9a bg:#071014 bold"
    edge_cyan = "fg:#16dfe8 bg:#071014 bold"
    glow_pink = "fg:#8f1b58 bg:#071014"
    glow_cyan = "fg:#0b6670 bg:#071014"
    lane_orange = "fg:#ff9a3c bg:#071014 bold"
    road_width = max(24, int(width * 0.68))
    left_start = int(width * 0.30)
    diagonal_step = max(2, int(width * 0.035))
    phase = frame % 10
    hud_left = 2
    hud_width = min(44, max(0, left_start - 4))
    hud_right = hud_left + hud_width - 1

    for y in range(height):
        chars = [" "] * width
        styles = [base_style] * width

        def put(x: int, glyph: str, style: str) -> None:
            if 0 <= x < width:
                chars[x] = glyph
                styles[x] = style

        def write(x: int, text: str, style: str) -> None:
            for offset, glyph in enumerate(text):
                put(x + offset, glyph, style)

        left = left_start + y * diagonal_step
        right = left + road_width
        for x in range(max(0, left + 1), min(width, right)):
            styles[x] = road_style
            if (x + y * 2 + phase // 2) % 17 == 0:
                put(x, "·", "fg:#0b6670 bg:#071014")

        put(left, "\\", edge_pink)
        put(left + 1, "\\", edge_pink)
        put(right - 1, "\\", edge_cyan)
        put(right, "\\", edge_cyan)

        lane = left + road_width // 2
        if (y + phase - 2) % 8 == 0:
            put(lane, "╲", lane_orange)

        # Sparse roadside reflectors retreat up-left beneath the right-facing
        # car. Deriving x from y follows the road perspective instead of
        # sweeping sideways or piling up into barcode-like clusters.
        pink_tail = (y + phase) % 10
        if pink_tail in (0, 1):
            marker = left + road_width // 3
            put(marker, "╲", edge_pink if pink_tail == 0 else glow_pink)
            if pink_tail == 0:
                put(marker + 1, "╲", edge_pink)

        cyan_tail = (y + phase - 5) % 10
        if cyan_tail in (0, 1):
            marker = left + road_width * 2 // 3
            put(marker, "╲", edge_cyan if cyan_tail == 0 else glow_cyan)
            if cyan_tail == 0:
                put(marker + 1, "╲", edge_cyan)

        if height >= 5 and hud_width >= 28 and y < 5:
            hud_bg = "bg:#071014"
            hud_pink = "fg:#ff2d9a bg:#071014 bold"
            hud_cyan = "fg:#16dfe8 bg:#071014 bold"
            hud_orange = "fg:#ff9a3c bg:#071014 bold"
            hud_dim = "fg:#0b6670 bg:#071014"
            for x in range(hud_left, hud_right + 1):
                styles[x] = hud_bg

            if y == 0:
                for x in range(hud_left + 1, hud_right):
                    put(x, "─", hud_pink)
                put(hud_left, "╭", hud_pink)
                put(hud_right, "╮", hud_cyan)
                title = " STREET MODE "
                write(hud_left + 2, title, hud_pink)
            elif y == 4:
                for x in range(hud_left + 1, hud_right):
                    put(x, "─", hud_cyan)
                put(hud_left, "╰", hud_pink)
                put(hud_right, "╯", hud_cyan)
            else:
                put(hud_left, "│", hud_pink)
                put(hud_right, "│", hud_cyan)
                if y == 1:
                    speed = 274 + (frame * 11) % 25
                    write(hud_left + 2, "SPEED", hud_cyan)
                    write(hud_left + 9, f"{speed:03d}", hud_orange)
                    write(hud_left + 13, "KM/H", hud_cyan)
                elif y == 2:
                    write(hud_left + 2, "GEAR", hud_cyan)
                    write(hud_left + 7, "5", hud_orange)
                    write(hud_left + 10, "BOOST", hud_cyan)
                    bar_width = max(3, hud_width - 19)
                    active = min(bar_width, max(1, bar_width - 2 + frame % 3))
                    for index in range(bar_width):
                        put(
                            hud_left + 16 + index,
                            "■" if index < active else "·",
                            hud_pink if index < active else hud_dim,
                        )
                else:
                    write(hud_left + 2, "NO BRAKES", hud_pink)
                    swag = "SWAG MAX"
                    write(hud_right - len(swag) - 2, swag, hud_orange)

        for (style, glyph), cells in groupby(zip(styles, chars, strict=True)):
            count = sum(1 for _ in cells)
            fragments.append((style, glyph * count))
        fragments.append((base_style, "\n"))
    return fragments


def _render_doggy_corner(width: int) -> StyleAndTextTuples:
    """Render the small decorative Doggy at the lower-right of a task canvas."""
    rows = _DOGGY_CORNER_ART
    art_width = len(rows[0])
    outer = max(0, width - art_width - 4)
    palette = dict(_DOGGY_ART_PALETTE)
    if int(time.monotonic() * 3) % 2:
        palette["G"] = "#ff9a5a"
    fragments: StyleAndTextTuples = []
    for top, bottom in zip(rows[::2], rows[1::2], strict=True):
        fragments.append(("", " " * outer))
        for pair, cells in groupby(zip(top, bottom, strict=True)):
            count = sum(1 for _ in cells)
            style, glyph = _half_block(pair[0], pair[1], palette)
            fragments.append((style, glyph * count))
        fragments.append(("", "\n"))
    return fragments


def _fit_art_row(row: str, width: int) -> str:
    """Keep bright silhouette pixels while fitting art to the terminal width."""
    fitted: list[str] = []
    for index in range(width):
        start = index * len(row) // width
        end = max(start + 1, (index + 1) * len(row) // width)
        fitted.append(max(row[start:end], key=_DOGGY_ART_PRIORITY.__getitem__))
    return "".join(fitted)


def _half_block(
    top: str,
    bottom: str,
    palette: dict[str, str],
) -> tuple[str, str]:
    background = palette["."]
    if top == bottom == ".":
        return f"bg:{background}", " "
    if top == bottom:
        return f"fg:{palette[top]} bg:{background}", "█"
    if top == ".":
        return f"fg:{palette[bottom]} bg:{background}", "▄"
    if bottom == ".":
        return f"fg:{palette[top]} bg:{background}", "▀"
    return f"fg:{palette[top]} bg:{palette[bottom]}", "▀"


def _task_activity_text(task: TaskView) -> str:
    if task.phase == "dispatching":
        return "MAIN 正在拆解任务…"
    if task.phase == "parallel":
        active = sum(
            agent.status in {"pending", "running"} for agent in task.agents
        )
        return f"{max(1, active)} 个 Agent 正在并行…"
    if task.phase == "reporting":
        return "MAIN 正在汇总结果…"
    return "MAIN 正在推进…"


def _truncate_display(text: str, width: int) -> str:
    if get_cwidth(text) <= width:
        return text
    if width <= 1:
        return "…"
    out: list[str] = []
    used = 0
    for char in text:
        char_width = get_cwidth(char)
        if used + char_width > width - 1:
            break
        out.append(char)
        used += char_width
    return "".join(out).rstrip() + "…"
