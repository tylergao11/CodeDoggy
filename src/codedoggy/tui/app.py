"""Prompt-toolkit boss cockpit: tasks first, Agent detail on demand."""

from __future__ import annotations

import re
import shutil
import threading
import time
from collections.abc import Callable
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
from prompt_toolkit.layout.screen import Point
from prompt_toolkit.layout.processors import AfterInput, ConditionalProcessor
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.output.color_depth import ColorDepth
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import TextArea

from codedoggy.session.types import TurnStatus
from codedoggy.tui.model import AgentView, TaskLedger, TaskView
from codedoggy.turn.types import Role


STATUS_TEXT = {
    "waiting": "等待",
    "pending": "准备中",
    "running": "推进中",
    "completed": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
    "max_turns": "需继续",
}


APPLE_DARK = Style.from_dict(
    {
        "root": "bg:#0b0b0d #f5f5f7",
        "header": "bg:#0b0b0d #f5f5f7",
        "brand": "#f5f5f7 bold",
        "meta": "#8e8e93",
        "separator": "#2c2c2e",
        "tree": "#3a3a3c",
        "task.title": "#f5f5f7 bold",
        "task.status": "#8e8e93",
        "status.running": "#64d2ff",
        "status.completed": "#30d158",
        "status.waiting": "#636366",
        "status.failed": "#ff9f0a bold",
        "doggy.coat": "#ff9f0a bold",
        "doggy.detail": "#f5f5f7 bold",
        "doggy.speed": "#64d2ff bold",
        "doggy.wordmark": "#d1d1d6 bold",
        "agent": "bg:#2c2c2e #e5e5ea",
        "agent.selected": "bg:#f2f2f7 #1c1c1e bold",
        "reporter": "#f2f2f7 bold",
        "report": "#d1d1d6",
        "empty": "#8e8e93",
        "input": "bg:#1c1c1e #f5f5f7",
        "input.placeholder": "bg:#1c1c1e #636366",
        "prompt": "bg:#1c1c1e #f5f5f7 bold",
        "prompt.border": "bg:#0b0b0d #48484a",
        "prompt.border.focus": "bg:#0b0b0d #8e8e93",
        "prompt.border.info": "bg:#0b0b0d #64d2ff",
        "prompt.border.success": "bg:#0b0b0d #30d158",
        "prompt.border.warning": "bg:#0b0b0d #ff9f0a",
        "prompt.caption": "bg:#0b0b0d #8e8e93",
        "turn.status": "bg:#0b0b0d #d1d1d6",
        "turn.elapsed": "bg:#0b0b0d #636366",
        "turn.stop": "bg:#0b0b0d #8e8e93",
        "feedback.info": "bg:#0b0b0d #64d2ff",
        "feedback.success": "bg:#0b0b0d #30d158",
        "feedback.warning": "bg:#0b0b0d #ff9f0a",
        "shortcut.key": "bg:#0b0b0d #d1d1d6 bold",
        "shortcut.label": "bg:#0b0b0d #636366",
        "shortcut.separator": "bg:#0b0b0d #3a3a3c",
        "shortcut.pending": "bg:#0b0b0d #d1d1d6",
        "agent-window": "bg:#1c1c1e #f5f5f7",
        "agent-window.header": "bg:#1c1c1e #f5f5f7 bold",
        "agent-window.close": "bg:#3a3a3c #f2f2f7 bold",
        "agent-output": "bg:#1c1c1e #f5f5f7",
        "agent-window.hint": "bg:#1c1c1e #8e8e93",
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
        self._agent_output = TextArea(
            text="",
            read_only=True,
            focusable=True,
            focus_on_click=True,
            scrollbar=True,
            wrap_lines=True,
            style="class:agent-output",
        )

        header = Window(
            FormattedTextControl(self._render_header),
            height=1,
            style="class:header",
        )
        separator = Window(height=1, char="─", style="class:separator")
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
        modal_content = ConditionalContainer(
            HSplit(
                [
                    modal_header,
                    Window(height=1, char="─", style="class:separator"),
                    self._agent_output,
                    Window(
                        FormattedTextControl(
                            [("class:agent-window.hint", "Esc 关闭")]
                        ),
                        height=1,
                        style="class:agent-window.hint",
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
            style=APPLE_DARK,
            full_screen=True,
            mouse_support=True,
            color_depth=ColorDepth.TRUE_COLOR,
            refresh_interval=0.25,
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

        @keys.add("tab", filter=~modal)
        def _next_agent(event: Any) -> None:
            self._move_agent(1)
            event.app.layout.focus(self._task_window)

        @keys.add("s-tab", filter=~modal)
        def _previous_agent(event: Any) -> None:
            self._move_agent(-1)
            event.app.layout.focus(self._task_window)

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

    def _start_task(self, prompt: str) -> None:
        task = self.ledger.create(prompt)
        self._active_task_id = task.id
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
        before_messages = len(getattr(runner, "live_messages", []) or [])
        streamed: list[str] = []
        old_stream = getattr(sampler, "stream", None)
        old_delta = getattr(sampler, "on_delta", None)

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

        try:
            result = self.session.handle_prompt(
                prompt,
                prompt_id=task_id,
                metadata={"tui_task_id": task_id},
            )
            messages = list(getattr(runner, "live_messages", []) or [])[before_messages:]
            output = agent_text_from_messages(messages)
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

        if self._modal_open and self._modal_ref:
            task_id, agent_id = self._modal_ref
            agent = self.ledger.get_agent(task_id, agent_id)
            if agent is not None:
                self._agent_output.text = _display_agent_output(agent)

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
        if not self._is_running():
            if self._feedback_active():
                icon = {"info": "●", "success": "✓", "warning": "!"}[
                    self._feedback_kind
                ]
                return [
                    (f"class:feedback.{self._feedback_kind}", f"  {icon} "),
                    ("class:turn.status", self._feedback_text),
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
        right_prefix = f"{budget}  " if budget else ""
        stop = "[停止]"
        right = f"{right_prefix}{stop}  "
        width = max(36, _terminal_width())
        label_budget = max(8, width - get_cwidth(right) - 18)
        label = _truncate_display(label, label_budget)
        left = f"  {spinner} {label}"
        elapsed_text = _format_elapsed(elapsed)
        gap = max(
            1,
            width
            - get_cwidth(left)
            - get_cwidth(elapsed_text)
            - get_cwidth(right)
            - 1,
        )
        return [
            ("class:turn.status", left),
            ("class:turn.elapsed", f" {elapsed_text}"),
            ("class:turn.elapsed", " " * gap + right_prefix),
            ("class:turn.stop", stop, self._stop_mouse),
            ("class:turn.elapsed", "  "),
        ]

    def _render_prompt_prefix(self) -> StyleAndTextTuples:
        border = self._prompt_border_class()
        return [(border, "  │ "), ("class:prompt", "› ")]

    def _render_prompt_top(self) -> StyleAndTextTuples:
        width = max(16, _terminal_width())
        return [(self._prompt_border_class(), "  ╭" + "─" * (width - 4) + "╮")]

    def _render_prompt_right(self) -> StyleAndTextTuples:
        return [(self._prompt_border_class(), "│  ")]

    def _render_prompt_bottom(self) -> StyleAndTextTuples:
        width = max(16, _terminal_width())
        caption_text = _truncate_display(_model_and_mode_text(self.session), width - 7)
        caption = f" {caption_text} "
        fill = max(1, width - 4 - get_cwidth(caption))
        return [
            (self._prompt_border_class(), "  ╰" + "─" * fill),
            ("class:prompt.caption", caption),
            (self._prompt_border_class(), "╯"),
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
        width = max(36, _terminal_width())
        cwd = str(getattr(self.session, "cwd", ""))
        left = f"  CodeDoggy  main · {cwd}"
        right = _budget_text(self.session)
        gap = max(1, width - get_cwidth(left) - get_cwidth(right) - 1)
        return [
            ("class:brand", "  CodeDoggy"),
            ("class:meta", f"  main · {cwd}"),
            ("class:meta", " " * gap + right + " "),
        ]

    def _render_tasks(self) -> StyleAndTextTuples:
        tasks = self.ledger.snapshots()
        fragments: StyleAndTextTuples = []
        refs: list[tuple[str, str]] = []
        selected_line = 0
        line = 0
        width = max(36, _terminal_width() - 2)

        if not tasks:
            return _render_doggy_empty(width)

        for task_index, task in enumerate(tasks):
            status = _task_stage_text(task)
            task_icon, task_icon_style = _status_icon(task.status, ambient=True)
            title_budget = max(8, width - get_cwidth(status) - 8)
            title = _truncate_display(task.title, title_budget)
            left = f"  {task_icon} {title}"
            gap = max(2, width - get_cwidth(left) - get_cwidth(status))
            fragments.extend(
                [
                    ("", "  "),
                    (task_icon_style, task_icon),
                    ("class:task.title", f" {title}"),
                    ("class:task.status", " " * gap + status + "  \n"),
                ]
            )
            line += 1
            fragments.append(("class:tree", "  │  "))
            for agent in task.agents:
                index = len(refs)
                refs.append((task.id, agent.id))
                style = "class:agent.selected" if index == self._selected_agent else "class:agent"
                if index == self._selected_agent:
                    selected_line = line
                agent_icon, _ = _status_icon(agent.status, ambient=False)
                fragments.append(
                    (style, f" {agent_icon} {agent.label}  › ", self._agent_mouse(index))
                )
                fragments.append(("", "  "))
            fragments.append(("", "\n"))
            line += 1
            if task.report:
                fragments.extend(
                    [
                        ("class:tree", "  └─ "),
                        ("class:reporter", f"{task.reporter:<8}"),
                        ("class:report", task.report.strip() + "\n"),
                    ]
                )
                line += max(1, task.report.count("\n") + 1)
            if task_index != len(tasks) - 1:
                fragments.append(
                    ("class:separator", "  " + "─" * max(1, width - 2) + "\n")
                )
                line += 1

        self._agent_refs = refs
        if refs:
            self._selected_agent %= len(refs)
        else:
            self._selected_agent = 0
        self._selected_line = selected_line
        return fragments

    def _render_modal_title(self) -> StyleAndTextTuples:
        if not self._modal_ref:
            return []
        task_id, agent_id = self._modal_ref
        agent = self.ledger.get_agent(task_id, agent_id)
        task = next((item for item in self.ledger.snapshots() if item.id == task_id), None)
        if agent is None or task is None:
            return []
        return [("class:agent-window.header", f"  {agent.label} · {task.title}")]

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
        self._agent_output.text = _display_agent_output(agent)
        self._agent_output.buffer.cursor_position = 0
        self._modal_open = True
        self.app.layout.focus(self._agent_output)
        self.app.invalidate()

    def _close_modal(self) -> None:
        self._modal_open = False
        self._modal_ref = None
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


def agent_text_from_messages(messages: list[Any]) -> str:
    """Return only normal assistant prose; tool records stay hidden."""
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


def _display_agent_output(agent: AgentView) -> str:
    if agent.output.strip():
        return agent.output.strip()
    if agent.status in {"pending", "running"}:
        return "Agent 正在工作，完成后会在这里给出完整输出。"
    return agent.description.strip() or "Agent 没有留下文字输出。"


def _terminal_width() -> int:
    try:
        return get_app().output.get_size().columns
    except Exception:  # noqa: BLE001
        return shutil.get_terminal_size(fallback=(100, 30)).columns


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


def _render_doggy_empty(
    width: int,
    *,
    now: float | None = None,
) -> StyleAndTextTuples:
    """Render the launch mascot as fixed-width terminal art without jitter."""
    tick = int((time.monotonic() if now is None else now) * 4) % 4
    wind = ("≋≋≋━━━", "━≋≋≋━━", "━━≋≋≋━", "≋━━≋≋≋")[tick]
    wheel = ("◉", "●", "◎", "●")[tick]
    canvas_width = 23
    lines: list[list[tuple[str, str]]] = [
        [("class:doggy.coat", "       ╱╲       ╱╲")],
        [("class:doggy.coat", "      ╱  ╲_____╱  ╲")],
        [("class:doggy.detail", "     │  ╭━━╮ ╭━━╮  │")],
        [("class:doggy.detail", "     │  ┃● ┃━┃ ●┃  │")],
        [("class:doggy.detail", "     │  ╰━━╯ ╰━━╯  │")],
        [("class:doggy.coat", "     ╰╮     ᴥ     ╭╯")],
        [("class:doggy.coat", "      ╰──╮ ╰━╯ ╭──╯")],
        [
            ("class:doggy.speed", wind),
            ("class:doggy.coat", "╯ │     │ ╰━━"),
        ],
        [("class:doggy.coat", "          ╲_____╱")],
        [("class:doggy.speed", "    ═══════════════════")],
        [("class:doggy.speed", f"       {wheel}           {wheel}")],
    ]

    outer = max(0, (width - canvas_width) // 2)
    fragments: StyleAndTextTuples = [("", "\n")]
    for parts in lines:
        fragments.append(("", " " * outer))
        fragments.extend(parts)
        fragments.append(("", "\n"))
    fragments.extend(
        [
            ("", "\n"),
            ("", " " * max(0, (width - 17) // 2)),
            ("class:doggy.coat", "D"),
            ("", "   "),
            ("class:doggy.wordmark", "o"),
            ("", "   "),
            ("class:doggy.wordmark", "g"),
            ("", "   "),
            ("class:doggy.wordmark", "g"),
            ("", "   "),
            ("class:doggy.speed", "y\n"),
        ]
    )
    return fragments


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


def _status_icon(status: str, *, ambient: bool) -> tuple[str, str]:
    if status in {"running", "pending"}:
        if ambient:
            frames = ("○", "◎", "◉", "◎")
            frame = frames[int(time.monotonic() * 3.75) % len(frames)]
        else:
            frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            frame = frames[int(time.monotonic() * 7.5) % len(frames)]
        return frame, "class:status.running"
    if status == "completed":
        return "✓", "class:status.completed"
    if status in {"failed", "max_turns"}:
        return "!", "class:status.failed"
    if status == "cancelled":
        return "×", "class:status.waiting"
    return "○", "class:status.waiting"


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
