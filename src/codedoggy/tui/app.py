"""Prompt-toolkit boss cockpit: tasks first, Agent detail on demand."""

from __future__ import annotations

import re
import shutil
import threading
import time
from collections.abc import Callable
from itertools import groupby
from pathlib import Path
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
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.layout.screen import Point
from prompt_toolkit.layout.processors import (
    AfterInput,
    ConditionalProcessor,
    PasswordProcessor,
)
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.output.color_depth import ColorDepth
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import TextArea

from codedoggy.session.types import SessionPhase, TurnStatus
from codedoggy.tui.clipboard_image import (
    get_system_clipboard_text,
    insert_path_token,
    save_clipboard_image,
)
from codedoggy.tui.agent_detail import (
    DETAIL_FILTERS,
    DETAIL_FILTER_LABELS,
    DETAIL_STYLE_RULES,
    AgentDetailSnapshot,
    DetailFilter,
    render_detail_body,
    snapshot_from_messages,
)
from codedoggy.tui.activity import LiveActivityBoard
from codedoggy.tui.login_wizard import AuthWizard, WizardStep, run_browser_login
from codedoggy.tui.model import AgentView, TaskLedger, TaskView
from codedoggy.tui import surface as session_surface
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

_STREAM_PREVIEW_LIMIT = 2_000
_STREAM_REFRESH_INTERVAL_S = 0.04


def _append_stream_preview(
    chunks: list[str],
    state: dict[str, Any],
    piece: Any,
) -> tuple[str, bool]:
    """Append losslessly while keeping the task-card preview bounded and smooth."""
    chunk = str(piece or "")
    chunks.append(chunk)
    preview = f"{state.get('preview', '')}{chunk}"
    if len(preview) > _STREAM_PREVIEW_LIMIT:
        preview = preview[-_STREAM_PREVIEW_LIMIT:]
    state["preview"] = preview
    now = time.monotonic()
    last_emit = float(state.get("last_emit", 0.0) or 0.0)
    should_emit = last_emit == 0.0 or now - last_emit >= _STREAM_REFRESH_INTERVAL_S
    if should_emit:
        state["last_emit"] = now
    return preview, should_emit


def _live_messages_signature(messages: list[Any]) -> tuple[Any, ...]:
    """Cheap append-oriented fingerprint for the TUI's polling fallback."""
    if not messages:
        return (0,)
    last = messages[-1]
    if isinstance(last, dict):
        role = last.get("role")
        call_id = last.get("tool_call_id") or last.get("id")
        name = last.get("name")
        content = last.get("content")
        tool_calls = last.get("tool_calls") or []
    else:
        role = getattr(last, "role", None)
        call_id = getattr(last, "tool_call_id", None) or getattr(last, "id", None)
        name = getattr(last, "name", None)
        content = getattr(last, "content", None)
        tool_calls = getattr(last, "tool_calls", None) or []
    if isinstance(content, str):
        content_sig: tuple[Any, ...] = (len(content), content[-64:])
    elif content is None:
        content_sig = (0, "")
    else:
        try:
            content_sig = (len(content), type(content).__name__)
        except TypeError:
            content_sig = (-1, type(content).__name__)
    last_tool = tool_calls[-1] if tool_calls else None
    if isinstance(last_tool, dict):
        tool_sig = (last_tool.get("id"), last_tool.get("name"))
    else:
        tool_sig = (
            getattr(last_tool, "id", None),
            getattr(last_tool, "name", None),
        )
    return (
        len(messages),
        str(getattr(role, "value", role) or ""),
        str(call_id or ""),
        str(name or ""),
        content_sig,
        len(tool_calls),
        tool_sig,
    )


# Box + doggy motif (rounded frames ∪･ω･∪ are the visual signature).
_DOG_FACE = "∪･ω･∪"
_DOG_EAR = "∪"
_INPUT_MAX_LINES = 8
_DETAIL_INPUT_MAX_LINES = 6
_TASK_BRIEF_LINES = 2  # brief agent output rows under each task card
_DOG_PAW = "ᕱ"


def _rounded_title(title: str, width: int, *, fill: str = "─") -> str:
    """Build ``╭─ title ─╮`` clipped to ``width`` display cells."""
    core = f" {title} "
    # corners + at least one fill each side
    min_w = get_cwidth(core) + 4
    if width < min_w:
        return _truncate_display(f"╭{core}╮", width)
    fill_cells = width - get_cwidth(core) - 2
    left = fill_cells // 2
    right = fill_cells - left
    return "╭" + fill * left + core + fill * right + "╮"


CODEDOGGY_DARK = Style.from_dict(
    {
        # State colors: yellow = selected, cyan = active, gray = inactive.
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
        "task.marker": "#16dfe8 bold",
        "task.marker.active": "#16dfe8 bold",
        "task.marker.selected": "#f2ca55 bold",
        "task.marker.idle": "#49636c",
        "task.title": "#f5f5f7 bold",
        "task.divider": "#123b43",
        "task.divider.pink": "#8f1b58",
        "task.divider.cyan": "#0b6670",
        "task.selection.border": "#f2ca55 bold",
        "task.status": "#6f8791",
        "task.status.running": "#16dfe8 bold",
        "task.status.reporting": "#16dfe8 bold",
        "task.status.completed": "#6f8791 bold",
        "task.status.failed": "#ff2d9a bold",
        "doggy.wordmark": "#ff2d9a bold",
        "agent.border": "#16dfe8",
        "agent.border.selected": "#f2ca55 bold",
        "agent.border.inactive": "#6f8791",
        "agent.label": "#16dfe8 bold",
        "agent.label.selected": "#f2ca55 bold",
        "agent.label.inactive": "#6f8791",
        "reporter.running": "#16dfe8 bold",
        "reporter.completed": "#6f8791 bold",
        "reporter.waiting": "#6f8791 bold",
        "reporter.failed": "#ff2d9a bold",
        "report": "#dce9e9",
        "input": "bg:#071014 #f5f5f7",
        "input.placeholder": "bg:#071014 #536b75",
        "prompt": "bg:#071014 #f2ca55 bold",
        "prompt.border": "bg:#0b0b0d #16dfe8",
        "prompt.border.focus": "bg:#0b0b0d #f2ca55 bold",
        "prompt.border.dim": "bg:#0b0b0d #6f8791",
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
        "auth.item": "bg:#0b0b0d #dce9e9",
        "auth.item.selected": "bg:#12262c #f2ca55 bold",
        "auth.item.active": "bg:#0b0b0d #f2ca55 bold",
        "auth.item.active.selected": "bg:#12262c #f2ca55 bold",
        "auth.item.logged": "bg:#0b0b0d #16dfe8 bold",
        "auth.item.logged.selected": "bg:#12262c #f2ca55 bold",
        "auth.item.offline": "bg:#0b0b0d #6f8791",
        "auth.item.offline.selected": "bg:#12262c #6f8791 bold",
        "auth.item.accent": "bg:#0b0b0d #16dfe8",
        "auth.item.ok": "bg:#0b0b0d #ff9a3c bold",
        "auth.item.danger": "bg:#0b0b0d #ff2d9a bold",
        "auth.item.muted": "bg:#0b0b0d #6f8791",
        "auth.cursor": "bg:#0b0b0d #ff2d9a bold",
        "auth.hint": "bg:#0b0b0d #6f8791",
        "auth.note": "bg:#0b0b0d #16dfe8",
        "hud.title": "fg:#ff2d9a bg:#071014 bold",
        "hud.ok": "fg:#ff9a3c bg:#071014 bold",
        "hud.warn": "fg:#ff2d9a bg:#071014 bold",
        "hud.cyan": "fg:#16dfe8 bg:#071014 bold",
        "hud.dim": "fg:#0b6670 bg:#071014",
        "hud.bg": "bg:#071014",
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
        self._task_refs: list[str] = []
        self._selected_task = 0
        self._selected_agent_by_task: dict[str, int] = {}
        self._agent_refs: list[tuple[str, str]] = []
        self._selected_line = 0
        self._task_line_count = 1  # clamp cursor y — PT crashes if y >= line_count
        self._pinned_task_for_line: int | None = None  # re-pin only on task change
        self._modal_open = False
        self._modal_kind: str = "agent"  # agent | auth
        self._modal_ref: tuple[str, str] | None = None
        self._detail_messages: dict[tuple[str, str], list[Any]] = {}
        self._detail_filter: DetailFilter = "all"
        self._detail_cursor_line = 0
        self._detail_line_count = 1  # clamp detail cursor y — same class as task crash
        self._redraw_pending = False
        self._closing = False
        self._task_started_at: float | None = None
        self._quit_armed_until = 0.0
        self._feedback_text = ""
        self._feedback_kind = "info"
        self._feedback_until = 0.0
        self._subagent_task: dict[str, str] = {}
        self._subagent_baselines: dict[str, set[str]] = {}
        self._subagent_live_signatures: dict[
            tuple[str, str], tuple[Any, ...]
        ] = {}
        self._auth_wizard = AuthWizard()
        self._auth_login_worker: threading.Thread | None = None
        self._auth_login_cancel: threading.Event | None = None
        self._pending_prompt: str | None = None
        # One-shot startup brand (concept art). Dismissed forever on first task;
        # not "empty ledger" — finished tasks never bring the splash back.
        self._startup_brand = not bool(
            initial_prompt and str(initial_prompt).strip()
        )
        # before_render throttle + splash cache (ESC/modal close snappiness)
        self._last_sync_runtime_at = 0.0
        self._doggy_empty_cache: tuple[tuple[Any, ...], StyleAndTextTuples] | None = None
        # Keep the newest task card fully in view until the user scrolls away.
        self._follow_latest_task = True
        # Live tool/activity lines from on_live_message (effect layer, not truth).
        self._activity = LiveActivityBoard()
        self._subagent_listener_bound = False
        self._session_listener_bound = False
        self._external_turn_views: dict[int, dict[str, Any]] = {}
        self._view_lock = threading.RLock()

        self._task_control = FormattedTextControl(
            text=self._render_tasks,
            focusable=True,
            show_cursor=False,
            get_cursor_position=self._task_cursor_position,
        )
        self._task_window = Window(
            content=self._task_control,
            wrap_lines=False,  # line y == content row; wrap broke scroll/cursor map
            scroll_offsets=ScrollOffsets(top=1, bottom=3),
            right_margins=[ScrollbarMargin(display_arrows=True)],
            style="class:root",
            dont_extend_height=False,
        )
        self._input = TextArea(
            # Grow with wrapped content so long prompts stay readable.
            height=self._main_input_height,
            multiline=True,
            wrap_lines=True,
            scrollbar=True,
            dont_extend_height=True,
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
        # Height follows content; re-layout on every edit.
        self._input.buffer.on_text_changed += lambda _buf: self._invalidate_safe()
        self._detail_control = FormattedTextControl(
            text=self._render_modal_body,
            focusable=True,
            show_cursor=False,
            get_cursor_position=self._detail_cursor_position,
        )
        self._detail_window = Window(
            content=self._detail_control,
            wrap_lines=False,
            scroll_offsets=ScrollOffsets(top=1, bottom=2),
            right_margins=[ScrollbarMargin(display_arrows=True)],
            style="class:agent-window",
        )
        self._detail_input = TextArea(
            height=self._detail_input_height,
            multiline=True,
            wrap_lines=True,
            scrollbar=True,
            dont_extend_height=True,
            prompt=self._render_detail_prompt_prefix,
            style="class:detail.input",
            accept_handler=self._accept_detail_prompt,
            input_processors=[
                ConditionalProcessor(
                    PasswordProcessor(),
                    Condition(
                        lambda: self._modal_kind == "auth"
                        and self._auth_wizard.step == WizardStep.PASTE
                        and self._auth_wizard.paste_kind != "model"
                    ),
                ),
                ConditionalProcessor(
                    AfterInput(
                        "补充要求…",
                        style="class:input.placeholder",
                    ),
                    Condition(
                        lambda: self._modal_kind == "agent"
                        and (
                            not getattr(self, "_detail_input", None)
                            or not self._detail_input.text
                        )
                    ),
                ),
                ConditionalProcessor(
                    AfterInput(
                        "粘贴 Token / API Key…",
                        style="class:input.placeholder",
                    ),
                    Condition(
                        lambda: self._modal_kind == "auth"
                        and self._auth_wizard.step == WizardStep.PASTE
                        and self._auth_wizard.paste_kind != "model"
                        and (
                            not getattr(self, "_detail_input", None)
                            or not self._detail_input.text
                        )
                    ),
                ),
                ConditionalProcessor(
                    AfterInput(
                        "输入 model id…",
                        style="class:input.placeholder",
                    ),
                    Condition(
                        lambda: self._modal_kind == "auth"
                        and self._auth_wizard.step == WizardStep.PASTE
                        and self._auth_wizard.paste_kind == "model"
                        and (
                            not getattr(self, "_detail_input", None)
                            or not self._detail_input.text
                        )
                    ),
                ),
            ],
        )
        self._detail_input.buffer.on_text_changed += (
            lambda _buf: self._invalidate_safe()
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
            # Match multi-line input height (no fixed height=1).
            style="class:root",
            dont_extend_width=True,
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
                ConditionalContainer(
                    self._detail_input,
                    filter=Condition(
                        lambda: self._modal_kind == "agent"
                        or (
                            self._modal_kind == "auth"
                            and self._auth_wizard.step == WizardStep.PASTE
                        )
                    ),
                ),
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
        street_hud = ConditionalContainer(
            Window(
                FormattedTextControl(self._render_street_hud),
                width=44,
                height=5,
                style="class:root",
            ),
            filter=Condition(
                lambda: (
                    not self._modal_open
                    and self._showing_startup_brand()
                    and _terminal_width() >= 48
                    and _terminal_height() >= 16
                )
            ),
        )
        root = FloatContainer(
            content=body,
            floats=[
                # Modal first so existing tests and z-order treat it as primary float.
                Float(
                    top=1,
                    bottom=1,
                    left=2,
                    right=2,
                    content=modal_content,
                    transparent=False,
                    z_index=10,
                ),
                Float(
                    bottom=7,
                    left=2,
                    width=44,
                    height=5,
                    content=street_hud,
                    transparent=False,
                    z_index=5,
                ),
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
            # 10Hz while busy; idle refresh still runs but _sync_runtime self-throttles.
            refresh_interval=0.10,
            before_render=lambda _: self._before_render(),
            input=input,
            output=output,
        )
        # Esc vs CSI (arrows) ambiguity: prompt_toolkit defaults ttimeoutlen=0.5s,
        # so Esc feels half a second slower than clicking × (mouse skips this wait).
        # 50ms is enough to disambiguate real escape sequences on modern terminals.
        self.app.ttimeoutlen = 0.05

    def run(self) -> None:
        def pre_run() -> None:
            self._bind_subagent_listener()
            self._bind_session_listener()
            if self.initial_prompt:
                if self._ensure_auth_ready():
                    self._start_task(self.initial_prompt)
                else:
                    self._pending_prompt = self.initial_prompt
                    self._open_auth_wizard()
                    self._set_feedback("先完成登录，再开工", "warning")

        try:
            self.app.run(pre_run=pre_run)
        finally:
            self._closing = True
            if self._auth_login_cancel is not None:
                self._auth_login_cancel.set()
            kernel = getattr(self.session.extensions, "kernel", None)
            scheduler_runtime = (
                (getattr(kernel, "tool_extra", None) or {}).get("scheduler_runtime")
                if kernel is not None
                else None
            )
            stop_scheduler = getattr(scheduler_runtime, "stop", None)
            if callable(stop_scheduler):
                try:
                    stop_scheduler()
                except Exception:  # noqa: BLE001
                    pass
            stop_ingress = getattr(self.session, "stop_prompt_ingress", None)
            if callable(stop_ingress):
                try:
                    stop_ingress(clear_queue=True)
                except Exception:  # noqa: BLE001
                    pass
            if getattr(self.session, "phase", None) is SessionPhase.TURN_RUNNING:
                self.session.cancel()
            if self._worker is not None and self._worker.is_alive():
                self._worker.join(timeout=3)
            wait_for_turn = getattr(self.session, "wait_for_turn", None)
            if callable(wait_for_turn):
                try:
                    wait_for_turn(timeout_s=5.0)
                except Exception:  # noqa: BLE001
                    pass
            if (
                self._auth_login_worker is not None
                and self._auth_login_worker.is_alive()
            ):
                self._auth_login_worker.join(timeout=6)
            self._unbind_subagent_listener()
            self._unbind_session_listener()

    def _subagent_coordinator(self) -> Any | None:
        kernel = getattr(self.session.extensions, "kernel", None)
        return getattr(kernel, "subagent_coordinator", None)

    def _bind_session_listener(self) -> None:
        if self._session_listener_bound:
            return
        add = getattr(self.session, "add_turn_listener", None)
        if callable(add):
            add(self._on_session_turn)
            self._session_listener_bound = True

    def _unbind_session_listener(self) -> None:
        if not self._session_listener_bound:
            return
        remove = getattr(self.session, "remove_turn_listener", None)
        if callable(remove):
            try:
                remove(self._on_session_turn)
            except Exception:  # noqa: BLE001
                pass
        self._session_listener_bound = False

    def _on_session_turn(self, event: str, request: Any, result: Any = None) -> None:
        """Project scheduler/host turns into the same boss-view data path.

        Direct TUI prompts already carry ``tui_task_id`` and own their view in
        ``_run_task``.  Synthetic/queued prompts have no worker of their own,
        so this listener creates the missing ledger entry and injects scoped
        stream callbacks into that exact TurnRequest.
        """
        metadata = getattr(request, "metadata", None)
        if not isinstance(metadata, dict):
            return
        key = id(request)
        if event == "start":
            if metadata.get("tui_task_id"):
                return
            task = self.ledger.create(str(getattr(request, "text", "") or ""))
            task_id = task.id
            main_id = f"{task_id}:main"
            messages: list[Any] = []
            streamed: list[str] = []
            stream_state: dict[str, Any] = {}
            callback_state = {"active": True}
            prior_live = metadata.get("on_live_message")
            prior_delta = metadata.get("on_sample_delta")

            def on_live_message(message: Any) -> None:
                if not callback_state["active"] or self._closing:
                    return
                # Ledger/activity are lock-safe; message list + paint are UI-only.
                line = self._activity.observe(task_id, main_id, message)
                if line:
                    self.ledger.update_live_agent(
                        task_id,
                        main_id,
                        label="MAIN",
                        status="running",
                        output=line,
                    )
                if (
                    callable(prior_live)
                    and callback_state["active"]
                    and not self._closing
                ):
                    prior_live(message)

                def apply() -> None:
                    if not callback_state["active"] or self._closing:
                        return
                    messages.append(message)
                    self._request_redraw()

                self._call_in_ui_thread(apply)

            def on_delta(piece: str) -> bool:
                if not callback_state["active"] or self._closing:
                    return False
                preview, should_emit = _append_stream_preview(
                    streamed,
                    stream_state,
                    piece,
                )
                if should_emit and not self._activity.line(task_id, main_id).startswith("→"):
                    self.ledger.update_live_agent(
                        task_id,
                        main_id,
                        label="MAIN",
                        status="running",
                        output=preview,
                    )
                if not callback_state["active"] or self._closing:
                    return False
                prior_result = prior_delta(piece) if callable(prior_delta) else True
                if should_emit and callback_state["active"] and not self._closing:
                    self._request_redraw()
                return (
                    prior_result is not False
                    and callback_state["active"]
                    and not self._closing
                )

            def apply_start() -> None:
                if self._closing:
                    return
                with self._view_lock:
                    previous_active = self._active_task_id
                    previous_started = self._task_started_at
                    self._external_turn_views[key] = {
                        "task_id": task_id,
                        "messages": messages,
                        "streamed": streamed,
                        "callback_state": callback_state,
                        "previous_active": previous_active,
                        "previous_started": previous_started,
                    }
                    self._active_task_id = task_id
                    self._task_started_at = time.monotonic()
                self._selected_task = max(0, len(self.ledger.snapshots()) - 1)
                self._follow_latest_task = True
                self._detail_messages[(task_id, main_id)] = messages
                self._activity.clear_task(task_id)
                self._subagent_baselines[task_id] = {
                    item.subagent_id for item in self._subagents()
                }
                self._dismiss_startup_brand()
                self._set_feedback("后台任务已进入 MAIN", "info")
                self._invalidate_safe()

            # Stream callbacks must be installed before the turn body runs.
            # View bookkeeping (selection / detail map / invalidate) is UI-only.
            metadata.update(
                {
                    "tui_task_id": task_id,
                    "stream_sample": True,
                    "on_sample_delta": on_delta,
                    "on_live_message": on_live_message,
                }
            )
            self._call_in_ui_thread(apply_start)
            return

        if event != "end":
            return
        with self._view_lock:
            state = self._external_turn_views.pop(key, None)
        if state is None:
            return
        state["callback_state"]["active"] = False
        task_id = str(state["task_id"])
        main_id = f"{task_id}:main"
        messages = list(state["messages"])
        streamed = list(state["streamed"])
        previous_active = state["previous_active"]
        previous_started = state["previous_started"]
        output = agent_summary_text_from_messages(messages)
        if not output:
            output = str(
                getattr(result, "final_text", None)
                or "".join(streamed)
                or getattr(result, "error", None)
                or ""
            ).strip()
        status = _turn_status(getattr(result, "status", TurnStatus.ERROR))
        self.ledger.update_agent(
            task_id,
            main_id,
            label="MAIN",
            status=status,
            output=output,
        )
        self.ledger.set_report(
            task_id,
            "MAIN",
            task_report_from_agent(
                getattr(result, "final_text", None)
                or getattr(result, "error", None)
                or "任务已结束。"
            ),
        )
        self.ledger.finish_task(
            task_id,
            "completed" if status == "completed" else status,
        )

        def apply_end() -> None:
            if self._closing:
                return
            self._sync_runtime()
            with self._view_lock:
                if self._active_task_id == task_id:
                    self._active_task_id = previous_active
                    self._task_started_at = previous_started
            self._set_feedback(
                "后台任务已完成" if status == "completed" else "后台任务未完成",
                "success" if status == "completed" else "warning",
            )
            self._invalidate_safe()

        self._call_in_ui_thread(apply_end)

    def _bind_subagent_listener(self) -> None:
        if self._subagent_listener_bound:
            return
        coord = self._subagent_coordinator()
        if coord is None or not hasattr(coord, "add_listener"):
            return
        coord.add_listener(self._on_subagent_live)
        self._subagent_listener_bound = True

    def _unbind_subagent_listener(self) -> None:
        if not self._subagent_listener_bound:
            return
        coord = self._subagent_coordinator()
        if coord is not None and hasattr(coord, "remove_listener"):
            try:
                coord.remove_listener(self._on_subagent_live)
            except Exception:  # noqa: BLE001
                pass
        self._subagent_listener_bound = False

    def _on_subagent_live(self, snap: Any, message: Any = None) -> None:
        """Worker-thread callback: schedule UI apply on the prompt_toolkit thread."""
        if self._closing:
            return

        def apply() -> None:
            self._apply_subagent_live(snap, message)

        self._call_in_ui_thread(apply)

    def _call_in_ui_thread(self, callback: Callable[[], None]) -> None:
        """Schedule a state mutation on prompt_toolkit's asyncio loop."""
        if self._closing:
            return
        loop = getattr(self.app, "loop", None)
        loop_thread = getattr(self.app, "_loop_thread", None)
        if loop is not None and not loop.is_closed():
            if loop_thread is threading.current_thread():
                callback()
            else:
                loop.call_soon_threadsafe(callback)
            return
        # Unit tests and pre-run callbacks have no application loop yet.
        callback()

    def _request_redraw(self) -> None:
        """Coalesce cross-thread invalidates onto the UI loop (one per turn)."""
        if self._closing:
            return

        def schedule() -> None:
            if self._closing or self._redraw_pending:
                return
            self._redraw_pending = True

            def flush() -> None:
                self._redraw_pending = False
                if not self._closing:
                    self._invalidate_safe()

            loop = getattr(self.app, "loop", None)
            if loop is not None and not loop.is_closed():
                loop.call_soon(flush)
            else:
                flush()

        self._call_in_ui_thread(schedule)

    def _apply_subagent_live(self, snap: Any, message: Any = None) -> None:
        """Same-tier push path for child agents (mirrors MAIN on_live_message)."""
        sub_id = str(getattr(snap, "subagent_id", "") or "")
        if not sub_id:
            return
        task_id = self._subagent_task.get(sub_id)
        if task_id is None:
            active = self._active_task_id
            if active is None:
                return
            baseline = self._subagent_baselines.get(active, set())
            if sub_id in baseline:
                return
            self._subagent_task[sub_id] = active
            task_id = active

        status = str(getattr(snap, "status", "") or "running")
        description = str(getattr(snap, "description", "") or "").strip()
        raw_label = description or str(getattr(snap, "subagent_type", "") or "agent")
        label = _truncate_display(raw_label, 18).upper()

        live = getattr(snap, "live_messages", None)
        if live is not None:
            msgs = list(live)
            detail_key = (task_id, sub_id)
            self._detail_messages[detail_key] = msgs
            self._subagent_live_signatures[detail_key] = _live_messages_signature(msgs)

        activity = ""
        if message is not None:
            activity = self._activity.observe(task_id, sub_id, message)
        elif live:
            activity = self._activity.rebuild(task_id, sub_id, list(live))

        output = activity or subagent_text(snap)
        self.ledger.update_agent(
            task_id,
            sub_id,
            label=label,
            status=status,
            output=output,
            description=description,
        )
        if status in {"pending", "running"}:
            self.ledger.set_task_phase(task_id, "parallel")
        try:
            self.app.invalidate()
        except Exception:  # noqa: BLE001
            pass

    def _build_key_bindings(self) -> KeyBindings:
        keys = KeyBindings()
        modal = Condition(lambda: self._modal_open)
        auth_modal = Condition(lambda: self._modal_open and self._modal_kind == "auth")
        agent_modal = Condition(lambda: self._modal_open and self._modal_kind == "agent")
        tasks_focused = Condition(
            lambda: not self._modal_open and get_app().layout.has_focus(self._task_window)
        )
        detail_focused = Condition(
            lambda: self._modal_open
            and self._modal_kind == "agent"
            and get_app().layout.has_focus(self._detail_window)
        )
        auth_list_focused = Condition(
            lambda: self._modal_open
            and self._modal_kind == "auth"
            and self._auth_wizard.step != WizardStep.PASTE
            and get_app().layout.has_focus(self._detail_window)
        )
        auth_paste = Condition(
            lambda: self._modal_open
            and self._modal_kind == "auth"
            and self._auth_wizard.step == WizardStep.PASTE
        )
        # Main prompt or detail interject (not auth token paste).
        prompt_paste = Condition(
            lambda: (
                get_app().layout.has_focus(self._input)
                or (
                    self._modal_open
                    and self._modal_kind == "agent"
                    and get_app().layout.has_focus(self._detail_input)
                )
            )
        )
        main_input_focused = Condition(
            lambda: (not self._modal_open) and get_app().layout.has_focus(self._input)
        )
        detail_input_focused = Condition(
            lambda: self._modal_open and get_app().layout.has_focus(self._detail_input)
        )

        @keys.add("c-v", filter=prompt_paste, eager=True)
        @keys.add("s-insert", filter=prompt_paste, eager=True)
        def _paste_image_or_text(event: Any) -> None:
            """Clipboard image → save file → insert path; else OS/text paste."""
            self._paste_into_buffer(event)

        # Multiline TextArea defaults to Enter=newline; product wants Enter=send.
        # Hard newline: Esc then Enter (prompt_toolkit multiline convention).
        @keys.add("enter", filter=main_input_focused, eager=True)
        def _submit_main_input(event: Any) -> None:
            self._input.buffer.validate_and_handle()

        @keys.add("escape", "enter", filter=main_input_focused, eager=True)
        def _newline_main_input(event: Any) -> None:
            event.current_buffer.newline(copy_margin=False)

        @keys.add("enter", filter=detail_input_focused, eager=True)
        def _submit_detail_input(event: Any) -> None:
            self._detail_input.buffer.validate_and_handle()

        @keys.add("escape", "enter", filter=detail_input_focused, eager=True)
        def _newline_detail_input(event: Any) -> None:
            event.current_buffer.newline(copy_margin=False)

        @keys.add("tab", filter=~modal)
        def _next_task_or_focus(event: Any) -> None:
            if event.app.layout.has_focus(self._input):
                self._render_tasks()
                if self._task_refs:
                    event.app.layout.focus(self._task_window)
                    event.app.invalidate()
            else:
                self._move_task(1)
                event.app.layout.focus(self._task_window)
                event.app.invalidate()

        @keys.add("s-tab", filter=~modal)
        def _previous_task_or_focus(event: Any) -> None:
            if event.app.layout.has_focus(self._input):
                self._render_tasks()
                if self._task_refs:
                    event.app.layout.focus(self._task_window)
                    event.app.invalidate()
            else:
                self._move_task(-1)
                event.app.layout.focus(self._task_window)
                event.app.invalidate()

        @keys.add("tab", filter=agent_modal)
        @keys.add("s-tab", filter=agent_modal)
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

        @keys.add("up", filter=auth_list_focused)
        @keys.add("k", filter=auth_list_focused)
        def _auth_up(_: Any) -> None:
            self._auth_wizard.move(-1)
            self.app.invalidate()

        @keys.add("down", filter=auth_list_focused)
        @keys.add("j", filter=auth_list_focused)
        def _auth_down(_: Any) -> None:
            self._auth_wizard.move(1)
            self.app.invalidate()

        @keys.add("enter", filter=auth_list_focused)
        def _auth_enter(_: Any) -> None:
            self._dispatch_wizard_action(self._auth_wizard.activate())

        for digit, idx in zip("123456789", range(9), strict=False):

            @keys.add(digit, filter=auth_list_focused)
            def _auth_digit(_: Any, index: int = idx) -> None:
                self._auth_wizard.set_cursor(index)
                self._dispatch_wizard_action(self._auth_wizard.activate())

        for key, detail_filter in zip(
            ("f1", "f2", "f3", "f4", "f5"),
            DETAIL_FILTERS,
            strict=True,
        ):

            @keys.add(key, filter=agent_modal)
            def _set_filter(_: Any, value: DetailFilter = detail_filter) -> None:
                self._set_detail_filter(value)

        @keys.add("enter", filter=tasks_focused)
        def _open_selected(_: Any) -> None:
            self._open_selected_task()

        @keys.add("up", filter=tasks_focused)
        def _previous_task(_: Any) -> None:
            self._move_task(-1)

        @keys.add("down", filter=tasks_focused)
        def _next_task(_: Any) -> None:
            self._move_task(1)

        @keys.add("left", filter=tasks_focused)
        def _previous_task_agent(_: Any) -> None:
            self._move_task_agent(-1)

        @keys.add("right", filter=tasks_focused)
        def _next_task_agent(_: Any) -> None:
            self._move_task_agent(1)

        @keys.add("space", filter=tasks_focused)
        def _focus_prompt(event: Any) -> None:
            event.app.layout.focus(self._input)
            # Drop yellow selection chrome until Tab returns to the task list.
            event.app.invalidate()

        @keys.add("pageup", filter=tasks_focused)
        def _tasks_page_up(_: Any) -> None:
            self._scroll_tasks(-max(4, _terminal_height() // 2))

        @keys.add("pagedown", filter=tasks_focused)
        def _tasks_page_down(_: Any) -> None:
            self._scroll_tasks(max(4, _terminal_height() // 2))

        @keys.add("c-l", filter=~modal)
        def _open_auth(_: Any) -> None:
            self._open_auth_wizard()

        @keys.add("escape")
        def _escape(event: Any) -> None:
            if self._modal_open and self._modal_kind == "auth":
                action = self._auth_wizard.go_back()
                self._dispatch_wizard_action(action)
                return
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

    def _paste_into_buffer(self, event: Any) -> None:
        """If OS clipboard holds an image, dump it and insert the path.

        Otherwise paste OS / prompt_toolkit text. Intercepting Ctrl+V means we
        must read the *system* clipboard ourselves — PT's pad is often empty.
        """
        buffer = event.current_buffer
        cwd = getattr(self.session, "cwd", None) or Path.cwd()
        try:
            saved = save_clipboard_image(cwd)
        except Exception:  # noqa: BLE001
            saved = None
        if saved is not None:
            token = insert_path_token(saved)
            pos = buffer.cursor_position
            before = buffer.text[:pos]
            lead = " " if before and not before[-1].isspace() else ""
            buffer.insert_text(f"{lead}{token} ")
            self._set_feedback(f"已粘贴图片 → {saved.name}", "success", duration=2.5)
            event.app.invalidate()
            return

        text: str | None = None
        try:
            data = event.app.clipboard.get_data()
            raw = getattr(data, "text", None) if data is not None else None
            if isinstance(raw, str) and raw:
                text = raw
        except Exception:  # noqa: BLE001
            text = None
        if not text:
            try:
                text = get_system_clipboard_text()
            except Exception:  # noqa: BLE001
                text = None
        if text:
            buffer.insert_text(text)
        event.app.invalidate()

    def _accept_prompt(self, buffer: Any) -> bool:
        prompt = buffer.text.strip()
        buffer.text = ""
        if not prompt:
            return True
        self._dismiss_startup_brand()
        if self._worker is not None and self._worker.is_alive():
            self.session.interject(prompt, prompt_id=self._active_task_id)
            self._set_feedback("补充指令已送达 MAIN", "info")
            self.app.invalidate()
            return True
        if not self._ensure_auth_ready():
            self._pending_prompt = prompt
            self._open_auth_wizard()
            self._set_feedback("先完成登录，再开工", "warning")
            return True
        self._start_task(prompt)
        return True

    def _accept_detail_prompt(self, buffer: Any) -> bool:
        if self._modal_kind == "auth" and self._auth_wizard.step == WizardStep.PASTE:
            text = buffer.text.strip()
            buffer.text = ""
            self._auth_wizard.paste_buffer = text
            self._dispatch_wizard_action(self._auth_wizard.submit_paste_text(text))
            return True

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
        self._dismiss_startup_brand()
        self._bind_subagent_listener()
        task = self.ledger.create(prompt)
        self._selected_task = max(0, len(self.ledger.snapshots()) - 1)
        self._follow_latest_task = True
        self._active_task_id = task.id
        self._detail_messages[(task.id, f"{task.id}:main")] = []
        self._activity.clear_task(task.id)
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

    def _dismiss_startup_brand(self) -> None:
        """Hide the launch splash for the rest of this process."""
        self._startup_brand = False

    def _showing_startup_brand(self) -> bool:
        return self._startup_brand and not self.ledger.snapshots()

    def _run_task(self, task_id: str, prompt: str) -> None:
        runner = getattr(self.session.extensions, "turn_runner", None)
        legacy_runner_events = bool(
            runner is not None
            and not getattr(runner, "supports_turn_host_events", False)
        )
        old_live_message = (
            getattr(runner, "on_live_message", None) if legacy_runner_events else None
        )
        detail_key = (task_id, f"{task_id}:main")
        turn_messages = self._detail_messages.setdefault(detail_key, [])
        streamed: list[str] = []
        stream_state: dict[str, Any] = {}
        callback_state = {"active": True}

        main_agent_id = f"{task_id}:main"

        def on_live_message(message: Any) -> None:
            if not callback_state["active"] or self._closing:
                return
            activity = self._activity.observe(task_id, main_agent_id, message)
            if activity:
                # Prefer tool/activity line on the boss list while running.
                self.ledger.update_live_agent(
                    task_id,
                    main_agent_id,
                    label="MAIN",
                    status="running",
                    output=activity,
                )
            if (
                callable(old_live_message)
                and callback_state["active"]
                and not self._closing
            ):
                old_live_message(message)

            def apply() -> None:
                if not callback_state["active"] or self._closing:
                    return
                # UI-thread only: shared list is also read by modal paint.
                turn_messages.append(message)
                self._request_redraw()

            self._call_in_ui_thread(apply)

        def on_delta(piece: str) -> bool:
            if not callback_state["active"] or self._closing:
                return False
            preview, should_emit = _append_stream_preview(
                streamed,
                stream_state,
                piece,
            )
            # Text stream fills the card only when no open tool activity.
            if should_emit and not self._activity.line(task_id, main_agent_id).startswith("→"):
                self.ledger.update_live_agent(
                    task_id,
                    main_agent_id,
                    label="MAIN",
                    status="running",
                    output=preview,
                )
            if should_emit and callback_state["active"] and not self._closing:
                self._request_redraw()
            return callback_state["active"] and not self._closing

        if legacy_runner_events:
            runner.on_live_message = on_live_message

        try:
            result = self.session.handle_prompt(
                prompt,
                prompt_id=task_id,
                metadata={
                    "tui_task_id": task_id,
                    "stream_sample": True,
                    "on_sample_delta": on_delta,
                    "on_live_message": on_live_message,
                },
            )
            callback_state["active"] = False
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
                feedback = ("MAIN 未完成并行收口", "warning", 2.2)
            elif failed_children:
                final_status = "failed"
                feedback = ("子 Agent 未全部成功", "warning", 2.2)
            elif status == "completed":
                final_status = "completed"
                feedback = ("MAIN 已汇总，任务完成", "success", None)
            else:
                final_status = status
                feedback = ("任务未能完成", "warning", 2.2)
            self.ledger.finish_task(task_id, final_status)

            def apply_success() -> None:
                if feedback[2] is None:
                    self._set_feedback(feedback[0], feedback[1])
                else:
                    self._set_feedback(
                        feedback[0], feedback[1], duration=float(feedback[2])
                    )

            self._call_in_ui_thread(apply_success)
        except Exception as exc:  # noqa: BLE001
            callback_state["active"] = False
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

            def apply_fail() -> None:
                self._set_feedback("任务执行失败", "warning", duration=2.2)

            self._call_in_ui_thread(apply_fail)
        finally:
            callback_state["active"] = False
            if legacy_runner_events:
                runner.on_live_message = old_live_message

            def apply_finish() -> None:
                self._sync_runtime()
                if self._active_task_id == task_id:
                    self._task_started_at = None
                self._invalidate_safe()

            self._call_in_ui_thread(apply_finish)

    def _before_render(self) -> None:
        """prompt_toolkit before_render hook — keep off the hot path when idle."""
        self._sync_runtime()

    def _sync_runtime(self) -> None:
        now = time.monotonic()
        running = self._is_running()
        # Agent detail needs live transcript; auth wizard / idle do not.
        hot = running or (self._modal_open and self._modal_kind == "agent")
        if not hot and (now - self._last_sync_runtime_at) < 0.35:
            return
        self._last_sync_runtime_at = now

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
                msgs = list(live_messages)
                detail_key = (task_id, snap.subagent_id)
                signature = _live_messages_signature(msgs)
                if self._subagent_live_signatures.get(detail_key) != signature:
                    self._subagent_live_signatures[detail_key] = signature
                    self._detail_messages[detail_key] = msgs
                    # Effect: rebuild only after the transcript changed.
                    line = self._activity.rebuild(task_id, snap.subagent_id, msgs)
                    if line and str(snap.status or "") in {"pending", "running"}:
                        self.ledger.update_agent(
                            task_id,
                            snap.subagent_id,
                            label=label,
                            status=str(snap.status or "waiting"),
                            output=line,
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

        if self._modal_open:
            self._detail_cursor_line = max(
                0,
                min(int(self._detail_cursor_line), max(0, int(self._detail_line_count) - 1)),
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
        worker_running = self._worker is not None and self._worker.is_alive()
        return worker_running or getattr(self.session, "phase", None) is SessionPhase.TURN_RUNNING

    def _cancel_current(self) -> None:
        if not self._is_running():
            return
        self.session.cancel()
        task_id = self._active_task_id
        coordinator = self._subagent_coordinator()
        if task_id and coordinator is not None:
            for subagent_id, owner_task_id in list(self._subagent_task.items()):
                if owner_task_id != task_id:
                    continue
                try:
                    coordinator.cancel(subagent_id)
                except Exception:  # noqa: BLE001
                    pass
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
        if active is not None:
            live = self._activity.main_line(active.id)
            if live:
                label = live
        budget = session_surface.budget_text(self.session)
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
        # Rounded well + dog ear caret.
        return [(border, "  │ "), ("class:prompt", f"{_DOG_EAR}› ")]

    def _render_prompt_top(self) -> StyleAndTextTuples:
        width = max(16, _terminal_width())
        border = self._prompt_border_class()
        rail_width = width - 4
        # Title plate in the top rail when wide enough.
        plate = " ∪･ω･∪  交代任务 "
        plate_w = get_cwidth(plate)
        if border != "class:prompt.border.focus" or rail_width < 8:
            if rail_width >= plate_w + 4:
                left = max(1, (rail_width - plate_w) // 2)
                right = max(1, rail_width - plate_w - left)
                return [
                    (border, "  ╭" + "─" * left),
                    ("class:prompt.caption", plate),
                    (border, "─" * right + "╮"),
                ]
            return [(border, "  ╭" + "─" * rail_width + "╮")]

        scan = int(time.monotonic() * 14) % rail_width
        styles = ["class:prompt.border.dim"] * rail_width
        styles[0] = border
        for offset in range(3):
            styles[(scan + offset) % rail_width] = border
        fragments: StyleAndTextTuples = [(border, "  ╭")]
        if rail_width >= plate_w + 4:
            left = max(1, (rail_width - plate_w) // 2)
            right = max(1, rail_width - plate_w - left)
            for style, cells in groupby(styles[:left]):
                fragments.append((style, "─" * sum(1 for _ in cells)))
            fragments.append(("class:prompt.caption", plate))
            for style, cells in groupby(styles[left + plate_w : left + plate_w + right]):
                fragments.append((style, "─" * sum(1 for _ in cells)))
            # pad if scan groups short
            used = left + plate_w + right
            if used < rail_width:
                fragments.append((border, "─" * (rail_width - used)))
        else:
            for style, cells in groupby(styles):
                fragments.append((style, "─" * sum(1 for _ in cells)))
        fragments.append(("class:prompt.corner.cyan", "╮"))
        return fragments

    def _render_prompt_right(self) -> StyleAndTextTuples:
        # One │ per visual input row so the frame grows with wrap height.
        rows = 1
        try:
            rows = max(
                1,
                min(
                    _INPUT_MAX_LINES,
                    self._estimate_buffer_display_lines(self._input.buffer.text),
                ),
            )
        except Exception:  # noqa: BLE001
            rows = 1
        fragments: StyleAndTextTuples = []
        for i in range(rows):
            fragments.append(("class:prompt.corner.cyan", "│  "))
            if i + 1 < rows:
                fragments.append(("", "\n"))
        return fragments

    def _invalidate_safe(self) -> None:
        """Invalidate when already on the UI path; swallow closed-app races."""
        try:
            self.app.invalidate()
        except Exception:  # noqa: BLE001
            pass

    def _main_input_height(self) -> Dimension:
        try:
            preferred = self._estimate_buffer_display_lines(
                self._input.buffer.text, max_lines=_INPUT_MAX_LINES
            )
        except Exception:  # noqa: BLE001
            preferred = 1
        return Dimension(min=1, max=_INPUT_MAX_LINES, preferred=preferred)

    def _detail_input_height(self) -> Dimension:
        try:
            preferred = self._estimate_buffer_display_lines(
                self._detail_input.buffer.text,
                prefix_cols=10,
                max_lines=_DETAIL_INPUT_MAX_LINES,
            )
        except Exception:  # noqa: BLE001
            preferred = 1
        return Dimension(min=1, max=_DETAIL_INPUT_MAX_LINES, preferred=preferred)

    def _estimate_buffer_display_lines(
        self,
        text: str | None,
        *,
        prefix_cols: int = 8,
        max_lines: int | None = None,
    ) -> int:
        """Count soft-wrapped display rows for dynamic TextArea height."""
        cap = max_lines if max_lines is not None else _INPUT_MAX_LINES
        raw = text or ""
        # Input sits left of a 3-col right rail; leave a little slack.
        avail = max(8, _terminal_width() - prefix_cols - 6)
        total = 0
        parts = raw.split("\n") if raw else [""]
        for part in parts:
            w = get_cwidth(part)
            if w <= 0:
                total += 1
            else:
                total += max(1, (w + avail - 1) // avail)
        return max(1, min(cap, total))

    def _render_prompt_bottom(self) -> StyleAndTextTuples:
        width = max(16, _terminal_width())
        caption_text = _truncate_display(
            session_surface.model_and_mode_text(self.session), width - 10
        )
        caption = f" {_DOG_PAW} {caption_text} "
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

        if self._modal_open and self._modal_kind == "auth":
            items = [
                ("↑↓", "选择", "noop", False),
                ("Enter", "确认", "noop", False),
                ("Esc", "返回", "close", False),
                ("Ctrl+Q", "退出", "quit", True),
            ]
        elif self._modal_open:
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
                    ("Ctrl+L", "登录", "login", False),
                    ("Tab", "任务", "tasks", False),
                ]
                if self._is_running():
                    items.append(("Ctrl+C", "取消", "cancel", False))
                items.append(("Ctrl+Q", "退出", "quit", True))
            else:
                items = [
                    ("Enter", "打开", "open", False),
                    ("Ctrl+L", "登录", "login", False),
                    ("Tab", "下一任务", "next", False),
                    ("↑↓", "切任务", "noop", False),
                    ("←→", "Agent", "noop", False),
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

    def _shortcut_mouse(self, action: str) -> Callable[[MouseEvent], object]:
        def _on_up(event: MouseEvent) -> None:
            if action == "quit":
                self._request_quit()
            elif action == "cancel":
                self._cancel_current()
            elif action == "close":
                self._close_modal()
            elif action == "login":
                self._open_auth_wizard()
            elif action == "tasks":
                self._render_tasks()
                if self._task_refs:
                    self.app.layout.focus(self._task_window)
            elif action == "next":
                self._move_task(1)
                self.app.layout.focus(self._task_window)
            elif action == "previous":
                self._move_task(-1)
                self.app.layout.focus(self._task_window)
            elif action == "open":
                self._open_selected_task()
            elif action == "input":
                self.app.layout.focus(self._input)
            elif action == "prompt":
                self.app.layout.focus(self._input)
            self.app.invalidate()

        # Footer chrome must not steal wheel from the focused pane.
        return self._only_mouse_up(_on_up, scroll_target="none")

    def _stop_mouse(self, event: MouseEvent) -> object:
        if event.event_type is not MouseEventType.MOUSE_UP:
            return NotImplemented
        self._cancel_current()
        return None

    def _render_header(self) -> StyleAndTextTuples:
        width = max(1, _terminal_width())
        # Rounded brand plate: ╭ ∪ DOGGY ∪ ╮
        brand_inner = f" {_DOG_EAR} DOGGY {_DOG_EAR} "
        left = "  ╭" + brand_inner + "╮"
        right = session_surface.budget_text(self.session)
        if width < get_cwidth(left):
            compact = f" ╭{_DOG_EAR}DOG{_DOG_EAR}╮"
            return [("class:brand", _truncate_display(compact, width))]

        pulse = int(time.monotonic() * 2) % 2
        edge_left = (
            "class:brand.edge.pink" if pulse == 0 else "class:brand.edge.cyan"
        )
        edge_right = (
            "class:brand.edge.cyan" if pulse == 0 else "class:brand.edge.pink"
        )
        fragments: StyleAndTextTuples = [
            ("class:header", "  "),
            (edge_left, "╭"),
            ("class:brand", brand_inner),
            (edge_right, "╮"),
        ]
        if not right or get_cwidth(left) + get_cwidth(right) + 2 > width:
            return fragments
        gap = width - get_cwidth(left) - get_cwidth(right) - 1
        # Tiny dog face on the budget side when there is room.
        face = f" {_DOG_FACE} "
        if gap > get_cwidth(face) + 2:
            mid = gap - get_cwidth(face)
            left_gap = mid // 2
            right_gap = mid - left_gap
            fragments.append(("class:meta", " " * left_gap))
            fragments.append(("class:brand.edge.cyan", face))
            fragments.append(("class:meta", " " * right_gap + right + " "))
        else:
            fragments.append(("class:meta", " " * gap + right + " "))
        return fragments

    def _render_street_hud(self) -> StyleAndTextTuples:
        """Auth gate surface (login entry). Click / Ctrl+L opens wizard."""
        width = 44
        snap = session_surface.hud_projection(self.session)
        frame = int(time.monotonic() * 4)
        pulse = frame % 2
        open_handler = self._hud_open_mouse

        def line(style: str, text: str) -> StyleAndTextTuples:
            padded = text + " " * max(0, width - get_cwidth(text))
            return [(style, padded[:width] if len(padded) > width else padded, open_handler)]

        title_style = "class:hud.title"
        ok_style = "class:hud.ok"
        warn_style = "class:hud.warn"
        cyan = "class:hud.cyan"
        dim = "class:hud.dim"
        bg = "class:hud.bg"

        cur = str(snap.get("provider") or "—")
        cur_model = str(snap.get("model") or "")
        cur_ok = bool(snap.get("current_ok"))
        status_word = "AUTH ON" if cur_ok else ("AUTH OFF" if pulse else "LOGIN ›")
        status_style = ok_style if cur_ok else warn_style
        now_label = f"{cur}/{cur_model}" if cur_model else cur

        fragments: StyleAndTextTuples = []
        # row 0 border title — rounded plate + dog face
        top = _rounded_title(f"{_DOG_FACE} STREET AUTH", width)
        fragments.extend(line(title_style, top[:width]))
        fragments.append((bg, "\n", open_handler))

        # row 1 status — provider/model from connection truth
        mid1 = f"│ {status_word:<10}  NOW {now_label[:18]:<18}"
        mid1 = mid1[: width - 1] + "│"
        fragments.append((status_style, mid1[:14], open_handler))
        fragments.append((cyan if cur_ok else dim, mid1[14:], open_handler))
        fragments.append((bg, "\n", open_handler))

        # row 2 tri-state
        bits = []
        for row in snap.get("rows") or []:
            mark = "✓" if row.get("logged_in") else "○"
            bits.append(f"{mark}{str(row.get('id') or '')[:5]}")
        mcp = snap.get("mcp") or {}
        if mcp.get("configured"):
            if int(mcp.get("bad") or 0):
                bits.append(f"!MCP:{int(mcp.get('bad') or 0)}")
            elif int(mcp.get("connecting") or 0):
                bits.append(f"…MCP:{int(mcp.get('connecting') or 0)}")
            else:
                bits.append(f"✓MCP:{int(mcp.get('ready') or 0)}")
        mid2 = "│ " + "  ".join(bits)
        mid2 = (mid2 + " " * width)[: width - 1] + "│"
        fragments.extend(line(cyan, mid2))
        fragments.append((bg, "\n", open_handler))

        # row 3 action
        action = f"│ {_DOG_EAR} Enter/Click · Ctrl+L 打开登录向导"
        action = (action + " " * width)[: width - 1] + "│"
        fragments.extend(line(dim, action))
        fragments.append((bg, "\n", open_handler))

        # row 4 bottom
        bot = "╰" + "─" * (width - 2) + "╯"
        fragments.extend(line(title_style, bot[:width]))
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
        line = 0
        width = max(1, _terminal_width() - 2)

        # Modal float covers this pane — skip expensive splash/task paint while open
        # so ESC close is not stuck behind a full truecolor doggy recompute.
        if self._modal_open:
            if self._showing_startup_brand() or not tasks:
                self._task_refs = []
                self._agent_refs = []
                empty: StyleAndTextTuples = [("", "\n")]
                self._set_task_line_count(empty)
                return empty
            # Agent detail on top: still keep task_refs in sync for selection keys,
            # but avoid re-walking every agent line under the float each frame.
            self._task_refs = [task.id for task in tasks]
            self._agent_refs = []  # no underlay chip handlers while modal owns input
            if tasks:
                if self._follow_latest_task:
                    self._selected_task = len(tasks) - 1
                self._selected_task %= len(tasks)
            empty = [("", "\n")]
            self._set_task_line_count(empty)
            return empty

        # Launch splash only — first task dismisses it for the whole session.
        if self._showing_startup_brand():
            fr = self._render_doggy_empty_cached(width)
            self._set_task_line_count(fr)
            return fr

        if not tasks:
            self._task_refs = []
            self._agent_refs = []
            self._selected_task = 0
            fr = _render_doggy_idle_panel(width)
            self._set_task_line_count(fr)
            return fr

        self._task_refs = [task.id for task in tasks]
        # Pin selection to the newest task while following.
        if self._follow_latest_task:
            self._selected_task = len(tasks) - 1
        self._selected_task %= len(tasks)
        # Yellow = task list focus + cursor row. Card borders stay always so
        # Space→input only recolors chrome (no layout jump / reflow).
        list_focused = self._task_list_has_focus()
        # Keep a constant card geometry whenever width allows.
        has_frame = width >= 20
        selected_line_start = 0
        selected_line_end = 0

        for task_index, task in enumerate(tasks):
            selected = list_focused and task_index == self._selected_task
            # Always track cursor anchor for the selected index (even when the
            # list is unfocused) so auto-follow can keep the latest card in view.
            is_cursor_task = task_index == self._selected_task
            # Yellow border only when this card is the focused cursor.
            framed = selected and has_frame
            inner_width = max(1, width - 2) if has_frame else width
            border_style = (
                "class:task.selection.border"
                if framed
                else "class:task.spine"
            )
            card_start = line
            if is_cursor_task:
                selected_line_start = card_start
            if has_frame:
                if framed:
                    plate = _rounded_title(f"{_DOG_FACE} selected", inner_width + 2)
                    fragments.append((border_style, plate + "\n"))
                else:
                    fragments.append(
                        (border_style, "╭" + "─" * inner_width + "╮\n")
                    )
                line += 1

            def append_task_line(parts: StyleAndTextTuples) -> None:
                nonlocal line
                if has_frame:
                    fragments.append((border_style, "│"))
                used = sum(get_cwidth(part[1]) for part in parts)
                fragments.extend(parts)
                if used < inner_width:
                    fragments.append(("", " " * (inner_width - used)))
                if has_frame:
                    fragments.append((border_style, "│"))
                fragments.append(("", "\n"))
                line += 1

            active = task.phase in {"dispatching", "parallel", "reporting"}
            spine_style = "class:task.spine.active" if active else "class:task.spine"
            prefix = "  │  " if active else "     "
            status = (
                _compact_task_stage_text(task)
                if width < 34
                else _task_stage_text(task)
            )
            # Left gutter mark only (1 cell + space). Never put ∪･ω･∪ in the
            # title row — it ate 5 columns and squeezed user text.
            if selected:
                marker = _DOG_EAR
            elif active:
                marker = _DOG_EAR
            else:
                marker = "·"
            marker_style = (
                "class:task.marker.selected"
                if selected
                else (
                    "class:task.marker.active"
                    if active
                    else "class:task.marker.idle"
                )
            )
            # Fixed-width left gutter so selected/idle recolor does not reflow.
            gutter = f"{marker} "
            gutter_w = get_cwidth(gutter)
            # Wrap full task text across lines (scroll list to read all) —
            # never leave the user with a half-eaten single truncated row.
            text_cols = max(1, inner_width - get_cwidth(prefix) - gutter_w)
            title_lines = _wrap_display_lines(task.title, text_cols)
            if not title_lines:
                title_lines = [""]
            # First line: gutter + title (+ status if it fits after wrap).
            first = title_lines[0]
            status_w = get_cwidth(status)
            if status_w + 1 < text_cols and get_cwidth(first) + 1 + status_w <= text_cols:
                gap = max(1, text_cols - get_cwidth(first) - status_w)
                append_task_line(
                    [
                        (spine_style, prefix),
                        (marker_style, gutter),
                        ("class:task.title", first),
                        (_task_status_style(task), " " * gap + status),
                    ]
                )
            else:
                append_task_line(
                    [
                        (spine_style, prefix),
                        (marker_style, gutter),
                        ("class:task.title", first),
                    ]
                )
                # Status on its own row when title is long.
                append_task_line(
                    [
                        (spine_style, prefix),
                        ("", " " * gutter_w),
                        (_task_status_style(task), _truncate_display(status, text_cols)),
                    ]
                )
            for cont in title_lines[1:]:
                append_task_line(
                    [
                        (spine_style, prefix),
                        ("", " " * gutter_w),
                        ("class:task.title", cont),
                    ]
                )
            append_task_line([(spine_style, prefix)])

            box_lines = self._render_agent_box_lines(
                task,
                inner_width,
                refs,
                prefix,
                spine_style,
                list_focused=list_focused,
            )
            for box_line in box_lines:
                append_task_line(box_line)

            divider_width = max(1, inner_width - get_cwidth(prefix) - 6)
            append_task_line(
                [
                    (spine_style, prefix),
                    ("class:task.divider.pink", "  ╭"),
                    (
                        "class:task.divider",
                        "┈" * max(1, divider_width - 2),
                    ),
                    ("class:task.divider.cyan", f"╯{_DOG_PAW}"),
                ]
            )
            for reporter, report, agent_status, agent_id in _task_briefs_with_ids(task):
                if agent_status in {"pending", "running"}:
                    live = self._activity.line(task.id, agent_id)
                    if live:
                        report = live
                available = max(2, inner_width - get_cwidth(prefix))
                minimum_label = 4 if available >= 8 else 1
                label_width = min(14, max(minimum_label, available // 3))
                label = _truncate_display(reporter, label_width)
                padded_label = label + " " * max(0, label_width - get_cwidth(label))
                report_width = max(1, available - label_width)
                # First line longer, second a bit shorter — easier to scan.
                brief_lines = _brief_two_lines(report, report_width)
                append_task_line(
                    [
                        (spine_style, prefix),
                        (_reporter_style(agent_status), padded_label),
                        ("class:report", brief_lines[0]),
                    ]
                )
                blank_label = " " * label_width
                if len(brief_lines) > 1 and brief_lines[1]:
                    append_task_line(
                        [
                            (spine_style, prefix),
                            ("", blank_label),
                            ("class:report", brief_lines[1]),
                        ]
                    )

            if has_frame:
                fragments.append((border_style, "╰" + "─" * inner_width + "╯\n"))
                line += 1
            if is_cursor_task:
                # Cursor on the *last* row of the card so the full card can sit
                # above the scroll bottom margin (not clipped mid-card).
                selected_line_end = max(selected_line_start, line - 1)
            if task_index != len(tasks) - 1:
                sep = f"  ╭{'┈' * max(1, width - 8)}╯ {_DOG_EAR}"
                fragments.append(
                    ("class:separator", _truncate_display(sep, width) + "\n")
                )
                line += 1

        self._agent_refs = refs
        # Re-pin selection only when follow is on or the selected task changed.
        # Every-paint pin fights free scroll (wheel moves y, paint snaps it back).
        pin_task = int(self._selected_task)
        if self._follow_latest_task:
            self._selected_line = selected_line_end
            self._pinned_task_for_line = pin_task
        elif self._pinned_task_for_line != pin_task:
            self._selected_line = selected_line_start
            self._pinned_task_for_line = pin_task
        self._set_task_line_count(fragments)
        return fragments

    def _task_list_has_focus(self) -> bool:
        """True only when the task pane owns keyboard focus (not the prompt)."""
        if self._modal_open:
            return False
        try:
            return bool(get_app().layout.has_focus(self._task_window))
        except Exception:  # noqa: BLE001
            return False

    def _task_cursor_position(self) -> Point:
        """Cursor y must always be in-range for FormattedTextControl lines."""
        max_y = max(0, int(self._task_line_count) - 1)
        y = max(0, min(int(self._selected_line), max_y))
        return Point(x=0, y=y)

    def _count_fragment_lines(self, fragments: StyleAndTextTuples) -> int:
        """Match prompt_toolkit split_lines / UIContent.line_count.

        PT always yields ``text.count("\\n") + 1`` lines (a trailing empty row
        after a final newline). Undercounting made clamps fail-safe but broke
        End/scroll-into-view and invited overcount "fixes" that re-crash.
        """
        if not fragments:
            return 1
        n = 0
        for item in fragments:
            text = item[1] if len(item) > 1 else ""
            n += str(text).count("\n")
        return max(1, n + 1)

    @staticmethod
    def _ensure_fragments(
        fragments: StyleAndTextTuples | None,
    ) -> StyleAndTextTuples:
        """Never hand focusable FormattedTextControl an empty fragment list."""
        if fragments:
            return fragments
        return [("", "\n")]

    def _set_task_line_count(self, fragments: StyleAndTextTuples) -> None:
        """Derive line count from rendered fragments and clamp selection."""
        n = self._count_fragment_lines(fragments)
        self._task_line_count = n
        self._selected_line = max(0, min(int(self._selected_line), n - 1))

    def _detail_cursor_position(self) -> Point:
        max_y = max(0, int(self._detail_line_count) - 1)
        y = max(0, min(int(self._detail_cursor_line), max_y))
        return Point(x=0, y=y)

    def _set_detail_line_count(
        self,
        fragments: StyleAndTextTuples,
        *,
        preferred_cursor: int | None = None,
    ) -> None:
        n = self._count_fragment_lines(fragments)
        self._detail_line_count = n
        if preferred_cursor is not None:
            self._detail_cursor_line = int(preferred_cursor)
        self._detail_cursor_line = max(0, min(int(self._detail_cursor_line), n - 1))

    def _render_agent_box_lines(
        self,
        task: TaskView,
        width: int,
        refs: list[tuple[str, str]],
        prefix: str,
        spine_style: str,
        *,
        list_focused: bool = True,
    ) -> list[StyleAndTextTuples]:
        content_width = max(1, width - get_cwidth(prefix) - 2)
        selected_task = (
            self._task_refs[self._selected_task]
            if self._task_refs
            else ""
        )
        selected_agent = self._selected_agent_index(task)
        chips: list[tuple[int, int, str, int, bool]] = []
        for agent_index, agent in enumerate(task.agents):
            index = len(refs)
            refs.append((task.id, agent.id))
            label = _truncate_display(agent.label, max(1, min(12, content_width - 9)))
            inner = f" {_DOG_EAR}{label} › "
            chips.append(
                (
                    index,
                    agent_index,
                    inner,
                    get_cwidth(inner) + 2,
                    agent.status in {"pending", "running"},
                )
            )

        groups: list[list[tuple[int, int, str, int, bool]]] = []
        current: list[tuple[int, int, str, int, bool]] = []
        used = 0
        for chip in chips:
            extra = chip[3] + (2 if current else 0)
            if current and used + extra > content_width:
                groups.append(current)
                current = []
                used = 0
                extra = chip[3]
            current.append(chip)
            used += extra
        if current:
            groups.append(current)

        lines: list[StyleAndTextTuples] = []
        for group in groups:
            fragments: StyleAndTextTuples = [(spine_style, prefix), ("", "  ")]
            for chip_index, (index, agent_index, inner, _box_width, active) in enumerate(group):
                selected = (
                    list_focused
                    and task.id == selected_task
                    and agent_index == selected_agent
                )
                if selected:
                    border = "class:agent.border.selected"
                    label_style = "class:agent.label.selected"
                elif active:
                    border = "class:agent.border"
                    label_style = "class:agent.label"
                else:
                    border = "class:agent.border.inactive"
                    label_style = "class:agent.label.inactive"
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
            lines.append(fragments)
        return lines


    def _render_modal_title(self) -> StyleAndTextTuples:
        width = max(12, _terminal_width() - 9)
        if self._modal_kind == "auth":
            left = f"  ╭ {_DOG_EAR} {self._auth_wizard.title} "
            right = "AUTH ╮"
            if self._auth_wizard.busy:
                spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(time.monotonic() * 8) % 10]
                right = f"{spinner} WAIT ╮"
            gap = max(1, width - get_cwidth(left) - get_cwidth(right))
            return [
                ("class:agent-window.header", left),
                ("", "─" * gap),
                ("class:detail.active", right),
            ]
        if not self._modal_ref:
            return [
                (
                    "class:agent-window.header",
                    _truncate_display(f"  ╭ {_DOG_EAR} Agent ╮", width),
                )
            ]
        task_id, agent_id = self._modal_ref
        agent = self.ledger.get_agent(task_id, agent_id)
        task = next((item for item in self.ledger.snapshots() if item.id == task_id), None)
        if agent is None or task is None:
            return [
                (
                    "class:agent-window.header",
                    _truncate_display(f"  ╭ {_DOG_EAR} 记录不可用 ╮", width),
                )
            ]
        left = f"  ╭ {_DOG_EAR} {agent.label} · {task.title} "
        right = STATUS_TEXT.get(agent.status, agent.status)
        if agent.status in {"pending", "running"}:
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[
                int(time.monotonic() * 8) % 10
            ]
            right = f"{spinner} {right}"
        right = f" {right} ╮"
        if get_cwidth(left) + get_cwidth(right) + 2 <= width:
            gap = width - get_cwidth(left) - get_cwidth(right)
            return [
                ("class:agent-window.header", left),
                ("class:modal.border.dim", "─" * gap),
                ("class:detail.active", right),
            ]
        return [("class:agent-window.header", _truncate_display(left + right, width))]

    def _render_modal_filters(self) -> StyleAndTextTuples:
        if self._modal_kind == "auth":
            text = f"  ╭ {_DOG_EAR} {self._auth_wizard.subtitle} "
            return [("class:auth.note", _truncate_display(text, max(12, _terminal_width() - 8)))]
        width = max(12, _terminal_width() - 8)
        fragments: StyleAndTextTuples = [("", "  ")]
        used = 2
        for index, detail_filter in enumerate(DETAIL_FILTERS, start=1):
            active = detail_filter == self._detail_filter
            base_label = f"F{index} {DETAIL_FILTER_LABELS[detail_filter]}"
            # Rounded chips: ╭ label ╮ for active filter
            label = f"╭ {base_label} ╮" if active else f" {base_label} "
            piece_width = get_cwidth(label) + (2 if used > 2 else 0)
            if used + piece_width > width:
                break
            if used > 2:
                fragments.append(("class:detail.meta", " "))
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
        if self._modal_kind == "auth":
            return self._render_auth_body()
        snapshot = self._current_detail_snapshot()
        if snapshot is None:
            w = max(12, _terminal_width() - 8)
            empty = [
                _rounded_title(f"{_DOG_FACE} empty", w),
                f"│ {_DOG_EAR} 当前 Agent 没有可用记录  │",
                "╰" + "─" * max(1, w - 2) + "╯",
            ]
            frags: StyleAndTextTuples = []
            for i, raw in enumerate(empty):
                text = _truncate_display(raw, w)
                style = "class:brand" if i == 0 else "class:detail.meta"
                frags.append((style, text + "\n"))
            frags = self._with_detail_scroll(self._ensure_fragments(frags))
            self._set_detail_line_count(frags, preferred_cursor=0)
            return frags
        width = max(12, _terminal_width() - 8)
        fragments = render_detail_body(
            snapshot,
            width,
            active_filter=self._detail_filter,
            path_mouse=self._image_path_mouse,
        )
        fragments = self._with_detail_scroll(self._ensure_fragments(fragments))
        self._set_detail_line_count(fragments)
        return fragments

    def _with_detail_scroll(
        self, fragments: StyleAndTextTuples
    ) -> StyleAndTextTuples:
        """Attach wheel→_scroll_detail on plain body text (not just links).

        Unhandled prose used Window-only scroll; ScrollOffsets then snapped
        the viewport back to a stale _detail_cursor_line on every invalidate.
        """
        handler = self._only_mouse_up(lambda _e: None, scroll_target="detail")
        out: StyleAndTextTuples = []
        for item in fragments:
            if len(item) >= 3 and item[2] is not None:
                out.append(item)
            else:
                style = item[0] if item else ""
                text = item[1] if len(item) > 1 else ""
                out.append((style, text, handler))
        return out if out else [("", "\n", handler)]

    def _render_auth_body(self) -> StyleAndTextTuples:
        width = max(20, _terminal_width() - 10)
        fragments: StyleAndTextTuples = []
        wiz = self._auth_wizard
        if wiz.body_note:
            note = _truncate_display(f"  {wiz.body_note}", width)
            fragments.append(("class:auth.note", note + "\n"))
            fragments.append(("class:auth.hint", "\n"))
        if wiz.step == WizardStep.WAITING:
            spin = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(time.monotonic() * 8) % 10]
            wait_w = min(width, max(28, width - 2))
            fragments.append(
                ("class:auth.item.accent", _rounded_title(f"{spin} {_DOG_FACE} auth", wait_w) + "\n")
            )
            fragments.append(
                ("class:auth.hint", f"│ {_DOG_EAR} 等待浏览器授权完成…\n")
            )
            fragments.append(
                ("class:auth.hint", "│ 完成后回到结果页 · Esc 取消\n")
            )
            fragments.append(
                ("class:agent.border", "╰" + "─" * max(1, wait_w - 2) + "╯\n")
            )
        # First visual line of each menu item (for scroll-into-view).
        item_line_starts: list[int] = []
        for index, item in enumerate(wiz.items):
            # Next line index == total newlines so far when every fragment ends with \n.
            start_y = sum(str(it[1]).count("\n") for it in fragments)
            item_line_starts.append(start_y)
            selected = index == wiz.cursor
            marker = f"╭{_DOG_EAR} " if selected else "│  "
            style = {
                "accent": "class:auth.item.accent",
                "ok": "class:auth.item.ok",
                "danger": "class:auth.item.danger",
                "muted": "class:auth.item.muted",
                "active": "class:auth.item.active",
                "logged": "class:auth.item.logged",
                "offline": "class:auth.item.offline",
            }.get(item.style, "class:auth.item")
            if selected:
                style = {
                    "active": "class:auth.item.active.selected",
                    "logged": "class:auth.item.logged.selected",
                    "offline": "class:auth.item.offline.selected",
                }.get(item.style, "class:auth.item.selected")
            tail = " ╮" if selected else ""
            label = _truncate_display(f"{marker}{item.label}{tail}", width)
            handler = self._auth_item_mouse(index)
            fragments.append((style, label + "\n", handler))
            if item.hint:
                hint = _truncate_display(f"    {item.hint}", width)
                fragments.append(("class:auth.hint", hint + "\n", handler))
            fragments.append(("", "\n", handler))
        preferred = 0
        if 0 <= wiz.cursor < len(item_line_starts):
            preferred = item_line_starts[wiz.cursor]
        fragments = self._with_detail_scroll(self._ensure_fragments(fragments))
        self._set_detail_line_count(fragments, preferred_cursor=preferred)
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
        if self._modal_kind == "auth":
            text = f"  ╭{_DOG_EAR}› 凭证  "
            return [("class:detail.input.prompt", text)]
        label = "MAIN"
        if self._modal_ref:
            task_id, agent_id = self._modal_ref
            agent = self.ledger.get_agent(task_id, agent_id)
            if agent is not None:
                label = agent.label
        terminal_width = max(1, _terminal_width())
        budget = max(4, terminal_width - 20)
        if terminal_width < 40:
            text = "  › " if budget < 6 else f"  {_DOG_EAR}› "
        else:
            if label == "MAIN":
                text = f"  ╭{_DOG_EAR}› 给 MAIN 补充  "
            else:
                text = f"  ╭{_DOG_EAR}› 转交 {label}  "
        text = _truncate_display(text, budget)
        return [("class:detail.input.prompt", text)]

    def _render_modal_hint(self) -> StyleAndTextTuples:
        if self._modal_kind == "auth":
            text = f"  ╰ {_DOG_PAW} ↑↓ 选择 · Enter 确认 · Esc 返回 · Ctrl+L ╮"
        else:
            text = f"  ╰ {_DOG_EAR} ↑↓ 滚动 · F1-F5 筛选 · Tab 补充 · Esc 返回 ╮"
        return [
            (
                "class:agent-window.hint",
                _truncate_display(text, max(1, _terminal_width() - 6)),
            )
        ]

    def _detail_filter_mouse(
        self, detail_filter: DetailFilter
    ) -> Callable[[MouseEvent], object]:
        return self._only_mouse_up(
            lambda _e: self._set_detail_filter(detail_filter),
            scroll_target="detail",
        )

    def _set_detail_filter(self, detail_filter: DetailFilter) -> None:
        self._detail_filter = detail_filter
        self._detail_cursor_line = 0
        self.app.layout.focus(self._detail_window)
        self.app.invalidate()

    def _move_detail_cursor(self, delta: int) -> None:
        max_y = max(0, int(self._detail_line_count) - 1)
        self._detail_cursor_line = min(
            max(0, int(self._detail_cursor_line) + delta),
            max_y,
        )
        self.app.invalidate()

    def _scroll_tasks(self, delta_lines: int) -> None:
        """Manual scroll: move content + stop auto-follow of the latest task."""
        self._follow_latest_task = False
        win = self._task_window
        info = getattr(win, "render_info", None)
        if info is not None:
            max_scroll = max(0, int(info.content_height) - int(info.window_height))
            win.vertical_scroll = max(
                0, min(max_scroll, int(win.vertical_scroll) + int(delta_lines))
            )
        else:
            win.vertical_scroll = max(0, int(win.vertical_scroll) + int(delta_lines))
        # Keep cursor y near the new viewport so the next paint does not snap back.
        max_y = max(0, int(self._task_line_count) - 1)
        self._selected_line = max(
            0, min(max_y, int(self._selected_line) + int(delta_lines))
        )
        self.app.invalidate()

    def _only_mouse_up(
        self,
        action: Callable[[MouseEvent], None],
        *,
        scroll_target: str = "auto",
    ) -> Callable[[MouseEvent], object]:
        """Fragment mouse handlers must not swallow wheel incorrectly.

        Returning None for non-UP events marks the event handled. Wheel over
        task chips scrolls the task list; wheel over modal/detail must let the
        Window scroll (or scroll the detail pane), never the underlay tasks.
        """

        def handler(event: MouseEvent) -> object:
            if event.event_type is not MouseEventType.MOUSE_UP:
                if event.event_type in {
                    MouseEventType.SCROLL_UP,
                    MouseEventType.SCROLL_DOWN,
                }:
                    step = -3 if event.event_type is MouseEventType.SCROLL_UP else 3
                    target = scroll_target
                    if target == "none":
                        return NotImplemented
                    if target == "auto":
                        target = "detail" if self._modal_open else "tasks"
                    if target == "tasks" and not self._modal_open:
                        self._scroll_tasks(step)
                        return None
                    if target == "detail" and self._modal_open:
                        self._scroll_detail(step)
                        return None
                    # Let Window handle default scroll when unsure.
                    return NotImplemented
                return NotImplemented
            action(event)
            return None

        return handler

    def _scroll_detail(self, delta_lines: int) -> None:
        """Scroll agent/auth detail pane without touching task list state."""
        win = self._detail_window
        info = getattr(win, "render_info", None)
        if info is not None:
            max_scroll = max(0, int(info.content_height) - int(info.window_height))
            win.vertical_scroll = max(
                0, min(max_scroll, int(win.vertical_scroll) + int(delta_lines))
            )
        else:
            win.vertical_scroll = max(0, int(win.vertical_scroll) + int(delta_lines))
        max_y = max(0, int(self._detail_line_count) - 1)
        self._detail_cursor_line = max(
            0, min(max_y, int(self._detail_cursor_line) + int(delta_lines))
        )
        self.app.invalidate()

    def _selected_task_view(self) -> TaskView | None:
        tasks = self.ledger.snapshots()
        if not tasks:
            return None
        self._selected_task %= len(tasks)
        return tasks[self._selected_task]

    def _selected_agent_index(self, task: TaskView) -> int:
        if not task.agents:
            self._selected_agent_by_task[task.id] = 0
            return 0
        index = self._selected_agent_by_task.get(task.id, 0) % len(task.agents)
        self._selected_agent_by_task[task.id] = index
        return index

    def _move_task(self, delta: int) -> None:
        tasks = self.ledger.snapshots()
        if not tasks:
            return
        self._selected_task = (self._selected_task + delta) % len(tasks)
        # Browsing older tasks pauses follow; landing on the last resumes it.
        self._follow_latest_task = self._selected_task == len(tasks) - 1
        self.app.invalidate()

    def _move_task_agent(self, delta: int) -> None:
        task = self._selected_task_view()
        if task is None or not task.agents:
            return
        index = self._selected_agent_index(task)
        self._selected_agent_by_task[task.id] = (index + delta) % len(task.agents)
        self.app.invalidate()

    def _open_selected_task(self) -> None:
        task = self._selected_task_view()
        if task is None or not task.agents:
            return
        agent = task.agents[self._selected_agent_index(task)]
        self._open_agent(task.id, agent.id)

    def _reset_detail_cursor_state(self) -> None:
        """Drop stale detail height before the next body paint measures it."""
        self._detail_cursor_line = 0
        self._detail_line_count = 1

    def _open_agent(self, task_id: str, agent_id: str) -> None:
        agent = self.ledger.get_agent(task_id, agent_id)
        if agent is None:
            return
        self._modal_kind = "agent"
        self._modal_ref = (task_id, agent_id)
        self._detail_filter = "all"
        self._reset_detail_cursor_state()
        self._detail_input.text = ""
        self._modal_open = True
        self.app.layout.focus(self._detail_window)
        self.app.invalidate()

    def _open_auth_wizard(self) -> None:
        self._modal_kind = "auth"
        self._modal_ref = None
        self._reset_detail_cursor_state()
        self._auth_wizard.open(
            active_provider=session_surface.provider_id(self.session),
            active_model=session_surface.model_id(self.session),
        )
        self._detail_input.text = ""
        self._modal_open = True
        self.app.layout.focus(self._detail_window)
        self.app.invalidate()

    def _close_modal(self) -> None:
        was_auth = self._modal_kind == "auth"
        had_tasks = bool(self.ledger.snapshots())
        self._modal_open = False
        self._modal_kind = "agent"
        self._modal_ref = None
        self._reset_detail_cursor_state()
        self._detail_input.text = ""
        # Focus the prompt caret immediately — focusing the task window on empty
        # ledgers forced a heavy underlay paint before the user saw the close.
        try:
            if was_auth or not had_tasks:
                self.app.layout.focus(self._input)
            else:
                self.app.layout.focus(self._task_window)
        except Exception:  # noqa: BLE001
            try:
                self.app.layout.focus(self._input)
            except Exception:  # noqa: BLE001
                pass
        self.app.invalidate()
        if was_auth and self._pending_prompt and self._ensure_auth_ready():
            prompt = self._pending_prompt
            self._pending_prompt = None
            self._start_task(prompt)

    def _render_doggy_empty_cached(self, width: int) -> StyleAndTextTuples:
        """Cache splash by animation frame + terminal size (skip recompute)."""
        clock = time.monotonic()
        frame = int(clock * 5) % max(1, _DOGGY_COUPLE_FRAMES)
        key = (frame, width, _terminal_height())
        cached = self._doggy_empty_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        fragments = _render_doggy_empty(width, now=clock)
        self._doggy_empty_cache = (key, fragments)
        return fragments

    def _agent_mouse(self, index: int) -> Callable[[MouseEvent], object]:
        def _on_up(_event: MouseEvent) -> None:
            if not 0 <= index < len(self._agent_refs):
                return
            task_id, agent_id = self._agent_refs[index]
            if task_id in self._task_refs:
                self._selected_task = self._task_refs.index(task_id)
                self._follow_latest_task = (
                    self._selected_task == len(self._task_refs) - 1
                )
            task = self._selected_task_view()
            if task is not None:
                for agent_index, agent in enumerate(task.agents):
                    if agent.id == agent_id:
                        self._selected_agent_by_task[task.id] = agent_index
                        break
            self._open_agent(task_id, agent_id)

        return self._only_mouse_up(_on_up, scroll_target="tasks")

    def _auth_item_mouse(self, index: int) -> Callable[[MouseEvent], object]:
        def _on_up(_event: MouseEvent) -> None:
            if self._modal_kind != "auth":
                return
            if self._auth_wizard.busy and self._auth_wizard.step is not WizardStep.WAITING:
                return
            # Disabled rows must not activate the previously selected item.
            if not self._auth_wizard.set_cursor(index):
                return
            self._dispatch_wizard_action(self._auth_wizard.activate())

        return self._only_mouse_up(_on_up, scroll_target="detail")

    def _hud_open_mouse(self, event: MouseEvent) -> object:
        if event.event_type is not MouseEventType.MOUSE_UP:
            return NotImplemented
        if not self._modal_open:
            self._open_auth_wizard()
        return None

    def _close_mouse(self, event: MouseEvent) -> object:
        if event.event_type is not MouseEventType.MOUSE_UP:
            return NotImplemented
        self._close_modal()
        return None

    def _current_provider_id(self) -> str:
        return session_surface.provider_id(self.session)

    def _ensure_auth_ready(self) -> bool:
        """True when connection truth says we can sample."""
        return session_surface.ready_to_sample(self.session)

    def _dispatch_wizard_action(self, action: Any) -> None:
        kind = getattr(action, "kind", "none")
        message = str(getattr(action, "message", "") or "")
        fb = str(getattr(action, "feedback_kind", "info") or "info")
        if message:
            self._set_feedback(message, fb)

        if kind == "close":
            self._close_modal()
            return
        if kind == "focus_input":
            self.app.layout.focus(self._detail_input)
            self.app.invalidate()
            return
        if kind == "blur_input":
            self.app.layout.focus(self._detail_window)
            self.app.invalidate()
            return
        if kind == "reload_client":
            if self._is_running():
                self._set_feedback(
                    "任务运行中，结束后再切换模型",
                    "warning",
                    duration=2.2,
                )
                self.app.invalidate()
                return
            try:
                snap = self._reload_model_client(
                    getattr(action, "provider", None),
                    model=getattr(action, "model", None),
                )
                if snap is not None:
                    self._auth_wizard.active_provider = snap.provider
                    self._auth_wizard.active_model = snap.model
                    if self._auth_wizard.step in {
                        WizardStep.PROVIDER,
                        WizardStep.MODEL,
                        WizardStep.RESULT,
                        WizardStep.HOME,
                    }:
                        self._auth_wizard._rebuild()
                self._set_feedback(
                    message or f"已连接 {snap.label if snap else ''}".strip(),
                    "success",
                )
            except Exception as exc:  # noqa: BLE001
                self._set_feedback(f"切换失败: {exc}", "warning")
            self.app.invalidate()
            return
        if kind == "start_login":
            provider = str(getattr(action, "provider", None) or "grok")
            self._start_browser_login(provider)
            self.app.invalidate()
            return
        if kind == "cancel_login":
            if self._auth_login_cancel is not None:
                self._auth_login_cancel.set()
            self.app.invalidate()
            return
        self.app.invalidate()

    def _start_browser_login(self, provider: str) -> None:
        if self._auth_login_worker is not None and self._auth_login_worker.is_alive():
            self._set_feedback("已有登录在进行", "warning")
            return

        cancel_event = threading.Event()
        self._auth_login_cancel = cancel_event

        def worker() -> None:
            try:
                status = run_browser_login(provider, cancel_event=cancel_event)
            except Exception as exc:  # noqa: BLE001
                from codedoggy.model.auth.base import AuthStatus, AUTH_OAUTH

                status = AuthStatus(
                    provider=provider,
                    kind=AUTH_OAUTH,
                    logged_in=False,
                    detail=str(exc),
                )
            if self._closing or cancel_event.is_set():
                return

            def finish() -> None:
                action = self._auth_wizard.on_login_finished(status)
                self._dispatch_wizard_action(action)

            self._call_in_ui_thread(finish)

        self._auth_login_worker = threading.Thread(
            target=worker, name="auth-login", daemon=True
        )
        self._auth_login_worker.start()

    def _reload_model_client(
        self,
        provider: str | None = None,
        *,
        model: str | None = None,
    ) -> Any:
        """Apply provider/model through ConnectionService only."""
        return session_surface.apply_connection(
            self.session,
            provider=provider,
            model=model,
            require_auth=True,
            source="panel",
        )

    def _image_path_mouse(self, path: str) -> Callable[[MouseEvent], object]:
        """Click-to-open for image_gen/image_edit paths (OS default viewer)."""

        def _on_up(_event: MouseEvent) -> None:
            from codedoggy.tui.open_path import open_local_path

            cwd = getattr(self.session, "cwd", None)
            ok, message = open_local_path(path, cwd=cwd)
            self._set_feedback(
                message,
                "success" if ok else "warning",
                duration=2.0,
            )
            try:
                self.app.invalidate()
            except Exception:  # noqa: BLE001
                pass

        return self._only_mouse_up(_on_up, scroll_target="detail")


def _render_doggy_idle_panel(width: int) -> StyleAndTextTuples:
    """Post-splash empty task area — rounded plate + dog face, not a blank void."""
    w = max(12, width)
    face = _DOG_FACE
    lines = [
        _rounded_title(f"{face} DOGGY", w),
        f"│ {face}  散步完了 · 等你下一句  │",
        f"│ {_DOG_EAR} 在下方输入框交代任务…   │",
        "╰" + "─" * max(1, w - 2) + "╯",
    ]
    # Fit lines to width
    out: StyleAndTextTuples = []
    for i, raw in enumerate(lines):
        text = _truncate_display(raw, w)
        pad = max(0, w - get_cwidth(text))
        style = "class:brand" if i == 0 else (
            "class:agent.border" if i in {0, 3} or text.startswith(("╭", "╰", "│")) else "class:meta"
        )
        if i == 0:
            style = "class:brand"
        elif i == 3:
            style = "class:agent.border"
        elif text.startswith("│"):
            style = "class:meta"
        out.append((style, text + " " * pad + "\n"))
    return out

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
    return [
        (label, report, status)
        for label, report, status, _agent_id in _task_briefs_with_ids(task)
    ]


def _task_briefs_with_ids(task: TaskView) -> list[tuple[str, str, str, str]]:
    """Briefs plus agent id for live activity overlay."""
    briefs: list[tuple[str, str, str, str]] = []
    report_matched = False
    for agent in task.agents:
        raw = agent.output
        if task.report and agent.label == task.reporter:
            raw = task.report
            report_matched = True
        if raw.strip():
            briefs.append(
                (
                    agent.label,
                    task_report_from_agent(raw),
                    agent.status,
                    agent.id,
                )
            )
    if task.report and not report_matched:
        main_id = task.agents[0].id if task.agents else ""
        briefs.append(
            (
                task.reporter,
                task_report_from_agent(task.report),
                task.status,
                main_id,
            )
        )
    if not briefs:
        main = task.agents[0] if task.agents else None
        briefs.append(
            (
                main.label if main is not None else "MAIN",
                _task_activity_text(task),
                main.status if main is not None else task.status,
                main.id if main is not None else "",
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


# Startup brand: neon street couple (concept image) — black void, not car city.
# Keys: F fur · H golden fur · D dark cloth · B shades · M pink · C cyan ·
# Y gold · W white · P hot pink · N nose · L cream shoe · S soft · . void
_DOGGY_COUPLE_ART = (
    "....................................................",
    "........HHH.........HHHH............................",
    "........HHSF.......HHHHF............................",
    "........H..HH......FHHSH............................",
    "........HSH.HH.HHHHHSFHF............................",
    "........HFF.HFHHHHFFHHHF............................",
    "........HFF.HFFFFFFFFFH.............................",
    "........HHHHFFFFHHFFFHHH............................",
    "........FHHFFHHHHSHS....FH..........................",
    "........SHHHS.....H.....SS........HHHH.MMM..........",
    "........HHHHHH....FFFFFSHHHS....HHHHHHHMMM.M........",
    "........HHFFFH...HFLLFF...F....HHHFFFFHMMMMMM.......",
    ".......HFFFFFLFFFLLLLLFF..F...HHHFFFFFHHHHMM........",
    ".......HHFFFFLLLFFLLLLLLFFF...HHHFFFFFFFFHSSH.......",
    "........HFFFLLLLFHFFFFFFHFF...FHHHFFFHFFFFHHH.......",
    "..........HFFLLLLLFFFFFFFH...HHHFFFFHHHFFHFFFH......",
    ".........MSHFFLLLLLLLFFFFSM..HHFF.HFFFFFFHHFFH......",
    ".......MMMS..FFFFFFFF...MMMMSFHFFSFFFFLFHHFFFFH.....",
    ".......MMMS.SFFFFFFFF..MMMMCCSHFFHFFHFLFFHFFFFHH....",
    ".......MMHMMHFFFFFFFH.MMS.....HHFFFFFLLFHHFFFFFH....",
    ".......CC.SM.HFFFFFH.MM..........HFFFLFH..HFFFFH....",
    ".......C...MSHHFFF.HSS............HFFFS....HSHH.....",
    "......C.....SSH...HHS........CC..MSHSMMM.HFFFS......",
    ".....CC.....S.HH.HF.S..MM...CCHSSHHHMSHSHSHHHS......",
    ".....CC.....S..HHH.SS..MMM..CSH.HFHHHFH.FHHFFFH.....",
    ".....C..C...S..SSS.CC.MMMMS.CHH.HFFFFFHHFHFHFFF.....",
    "....C...CC..M..HSH.CC..MMM..CHS.HHHHHH.FH.HHFHF.....",
    "...CC...CC..M..HHH.CC..SSS..CCS..SSM....H..HHHH.....",
    "..CC....CC..M......CC.......CC...MMMM...HC.HH.C.....",
    "..C.....C..S.......CC.......CCC..MMM.....CC...CC....",
    "..C..CCCC..S.......CC.......C.C...M.....CCC....C....",
    "..C....C...M.......CC.......C.C.........C.C....CC...",
    "..C.......SM.......CC.......C.CSHHHHHHHHC.CC....C...",
    "..CC..MM..S.FFFF...CC.......C.CHFFFFFFFFSC.C....CC..",
    "..CC.SMSHSS.FFFFHFHCC......SS.CCSSSSHHHSCC.C.....C..",
    "...C.MHFH...........M...MSSS..CCCCCCCCCCCC.CC....C..",
    "....CSHFS....C.C....SS..MSSS.C...CCC.....C..C...CC..",
    ".....CSH.....CC.......SMS....C.C...C..C..CC.CCCCC...",
    ".......C......C......CC.C...CC.C...C..S...C..CSSH...",
    ".......C.....CC........CC..CC.S...C....C..CC..HHFH..",
    ".......CC....C.........CC..S..S...S....S...SS.FHFH..",
    "........C....C........CC..SSS.S...S....M..SSM.HSFH..",
    "........CC..C......S..C...SMMMMMMSMMMMSMMMMMM...HH..",
    ".........C..C......C..C......MMHMSSHHHHSMHM..H......",
    ".........CCCC.....CC.C.........H..FFFFFF....HH......",
    "..........CC.........CC........HHSFFFFFH.HHHFH......",
    "..........SC......C.C.CC..HH...HHHFFFFFS.HFFFS......",
    "..........CC.....C.CC..CFFFFH...HHFFFFH.C.HFH.......",
    "..........C......C.C...CCHFH.....FFFFFH.SS..........",
    ".........CC.....C..C....CHH......FFFFF...CS.........",
    "........CCCCC...C.CC...CSH......SFFFFH..CCSH........",
    "........C....CC...C...CC..H....CSSHFFH.CC..H........",
    ".......FFFFH..SC.CC.CCC..FF.....FCCSH.SC...F........",
    "......FFHHFFH..CSC..SH...F...HFFF..CS..H..HH........",
    "......F.....FF..CC.HH...FF...F..FF..S.HS..H.....M...",
    "......HS....FF..C.HS....F....F...HF.SS...HH.....M...",
    "CCCCC..HH....HLHS..FFFFFH.SS.HH...FHSSFHFH.CCSSCCCCC",
    "........SH....HF...SSSSS......SH...FF.SSS...........",
    "MM.C.C...HFFFFFF..CCCCCCCCC....SFFFF...S...MM.SSS.MM",
    "....................................................",
)

_DOGGY_ART_PALETTE = {
    ".": "#0b0b0d",
    "C": "#00bac5",
    "M": "#ee4b8d",
    "c": "#0b6670",
    "m": "#8f1b58",
    "G": "#ff7a32",
    "Y": "#d9ad32",
    "T": "#f2ca55",
    "P": "#ff68ad",
    "R": "#071014",
    "F": "#e1d2ae",
    "H": "#c9a978",
    "D": "#2c2c2e",
    "S": "#75644a",
    "W": "#f5f5f7",
    "B": "#1c1c1e",
    "N": "#3a2a22",
    "L": "#f0e6cc",
    "K": "#050507",
    "E": "#3b2a20",
}

_DOGGY_COUPLE_FRAMES = 12

# High-priority facial pixels survive terminal resizing instead of dissolving
# into the surrounding tan fur.
_DOGGY_FEMALE_EYE_DETAILS = (
    (32, 13, "K"), (33, 13, "K"), (34, 13, "K"),
    (32, 14, "K"), (33, 14, "W"), (34, 14, "K"),
    (38, 13, "K"), (39, 13, "K"), (40, 13, "K"),
    (38, 14, "K"), (39, 14, "W"), (40, 14, "K"),
)

_DOGGY_FEMALE_MASK_SPANS = (
    (16, 33, 40),
    (17, 31, 42),
    (18, 31, 42),
    (19, 31, 42),
    (20, 33, 40),
)

_DOGGY_FEMALE_MASK_HIGHLIGHTS = (
    (29, 17, "m"), (30, 17, "m"),
    (43, 17, "m"), (44, 17, "m"),
    (34, 18, "P"), (35, 18, "P"), (36, 18, "P"),
    (37, 18, "P"), (38, 18, "P"), (39, 18, "P"),
)

_DOGGY_FEMALE_CROWN_SPANS = (
    (7, 36, 42, "H"),
    (8, 34, 44, "H"),
    (9, 32, 45, "H"),
)

_DOGGY_FEMALE_BOW_DETAILS = (
    (42, 7, "M"), (43, 7, "M"), (46, 7, "M"), (47, 7, "M"),
    (42, 8, "M"), (43, 8, "M"), (44, 8, "M"),
    (45, 8, "P"),
    (46, 8, "M"), (47, 8, "M"), (48, 8, "M"),
    (43, 9, "M"), (44, 9, "M"), (45, 9, "P"),
    (46, 9, "M"), (47, 9, "M"),
)

_DOGGY_CHAIN_DETAILS = (
    (17, 20), (18, 21), (19, 22),
    (23, 20), (22, 21), (21, 22),
    (20, 23), (36, 23),
)


def _animate_doggy_couple(rows: tuple[str, ...], frame: int) -> tuple[str, ...]:
    """Keep the portrait still while tiny jewellery and bow highlights breathe."""

    canvas = [list(row) for row in rows]
    height = len(canvas)
    width = len(canvas[0]) if canvas else 0
    phase = frame % _DOGGY_COUPLE_FRAMES

    for y, start, end, value in _DOGGY_FEMALE_CROWN_SPANS:
        if 0 <= y < height:
            for x in range(start, min(end + 1, width)):
                canvas[y][x] = value

    for x, y, value in _DOGGY_FEMALE_BOW_DETAILS:
        if 0 <= y < height and 0 <= x < width:
            canvas[y][x] = value

    for x, y, value in _DOGGY_FEMALE_EYE_DETAILS:
        if 0 <= y < height and 0 <= x < width:
            canvas[y][x] = value

    for y, start, end in _DOGGY_FEMALE_MASK_SPANS:
        if 0 <= y < height:
            for x in range(start, min(end + 1, width)):
                canvas[y][x] = "m" if x in {start, end} else "M"

    for x, y, value in _DOGGY_FEMALE_MASK_HIGHLIGHTS:
        if 0 <= y < height and 0 <= x < width:
            canvas[y][x] = value

    for index, (x, y) in enumerate(_DOGGY_CHAIN_DETAILS):
        if 0 <= y < height and 0 <= x < width:
            canvas[y][x] = "T" if index == phase % len(_DOGGY_CHAIN_DETAILS) else "Y"

    bow_pixels = [
        (x, y)
        for y in range(min(14, height))
        for x in range(38, width)
        if canvas[y][x] == "M"
    ]
    if bow_pixels:
        x, y = bow_pixels[(phase // 2) % len(bow_pixels)]
        canvas[y][x] = "P"

    return tuple("".join(row) for row in canvas)


def _compose_doggy_night(
    art_rows: tuple[str, ...],
    width: int,
    scene_time: float,
) -> tuple[str, ...]:
    """Place the locked portrait in the reference image's sparse neon night."""

    height = max(len(art_rows), 2)
    if height % 2:
        height += 1
    scene_width = max(1, width)
    tick = int(scene_time * 5)
    canvas = [["."] * scene_width for _ in range(height)]

    def put(x: int, y: int, value: str, *, soft: bool = False) -> None:
        if 0 <= x < scene_width and 0 <= y < height:
            if soft and canvas[y][x] != ".":
                return
            canvas[y][x] = value

    # Pink crescent from the reference, left of the taller dog.
    moon_x = max(2, round(scene_width * 0.25))
    moon_y = max(1, round(height * 0.09))
    for dy, span in ((0, (1, 2)), (1, (0, 3)), (2, (0, 3)), (3, (1, 2))):
        for dx in range(span[0], span[1] + 1):
            put(moon_x + dx, moon_y + dy, "M" if (tick // 3) % 2 == 0 else "m")
    put(moon_x + 2, moon_y + 1, ".")
    put(moon_x + 2, moon_y + 2, ".")

    # Sparse pink/cyan stars; only their intensity changes, never their position.
    spark_seed = (
        (0.20, 0.25, "C", True),
        (0.29, 0.17, "M", False),
        (0.75, 0.13, "M", False),
        (0.72, 0.29, "C", True),
        (0.14, 0.38, "C", False),
        (0.76, 0.48, "M", False),
        (0.18, 0.62, "M", False),
        (0.70, 0.70, "M", False),
        (0.24, 0.78, "C", False),
        (0.83, 0.58, "C", False),
    )
    for i, (fx, fy, color, cross) in enumerate(spark_seed):
        if (tick + i) % 5 == 0:
            continue
        x = int(fx * (scene_width - 1))
        y = int(fy * (height - 1))
        dim = "c" if color == "C" else "m"
        sparkle = color if (tick + i) % 2 == 0 else dim
        put(x, y, sparkle, soft=True)
        if cross and (tick + i) % 3:
            put(x - 1, y, sparkle, soft=True)
            put(x + 1, y, sparkle, soft=True)
            put(x, y - 1, sparkle, soft=True)
            put(x, y + 1, sparkle, soft=True)

    # The couple never bobs: its approved 52x60 height and pose stay locked.
    art_width = len(art_rows[0]) if art_rows else 0
    art_height = len(art_rows)
    art_left = max(0, (scene_width - art_width) // 2)
    art_top = max(0, (height - art_height) // 2)
    for y, row in enumerate(art_rows):
        ty = art_top + y
        if ty >= height:
            break
        for x, value in enumerate(row):
            if value != ".":
                put(art_left + x, ty, value)

    return tuple("".join(row) for row in canvas)


_DOGGY_DESIGN_WIDTH = 120
_DOGGY_DESIGN_TOP_MARGIN = 1
_DOGGY_DESIGN_BOTTOM_MARGIN = 1


def _render_doggy_empty(
    width: int,
    *,
    now: float | None = None,
) -> StyleAndTextTuples:
    """Render the locked 52x60 neon couple portrait and sparse night field."""
    try:
        clock = time.monotonic() if now is None else now
        art_tick = int(clock * 5)
        frame = art_tick % _DOGGY_COUPLE_FRAMES
        rows = _animate_doggy_couple(_DOGGY_COUPLE_ART, frame)
        terminal_height = _terminal_height()

        task_height = max(1, terminal_height - 8)
        # Keep the portrait on its native pixel grid. Fractional nearest-neighbour
        # scaling changes which eye/mask rows survive as the terminal is resized.
        stage_width = max(
            1,
            min(width, _DOGGY_DESIGN_WIDTH),
        )

        target_width = max(1, min(len(rows[0]), stage_width - 4))
        if target_width < len(rows[0]):
            crop_left = max(0, (len(rows[0]) - target_width) // 2)
            rows = tuple(row[crop_left : crop_left + target_width] for row in rows)

        rows = _compose_doggy_night(rows, stage_width, clock)
        if len(rows) % 2:
            rows = rows + (rows[-1] if rows else "." * max(1, stage_width),)
        art_width = len(rows[0])
        outer = max(0, (width - art_width) // 2)
        palette = dict(_DOGGY_ART_PALETTE)

        art_height = len(rows) // 2
        top_margin = _DOGGY_DESIGN_TOP_MARGIN
        bottom_margin = _DOGGY_DESIGN_BOTTOM_MARGIN
        scaled_frame_height = top_margin + art_height + bottom_margin
        vertical_slack = max(0, task_height - scaled_frame_height)
        top_padding = top_margin + vertical_slack // 2

        fragments: StyleAndTextTuples = [("", "\n" * top_padding)]
        for top, bottom in zip(rows[::2], rows[1::2]):
            fragments.append(("", " " * outer))
            pairs = zip(top, bottom)
            for pair, cells in groupby(pairs):
                count = sum(1 for _ in cells)
                style, glyph = _half_block(pair[0], pair[1], palette)
                fragments.append((style, glyph * count))
            fragments.append(("", "\n"))
        return fragments if fragments else [("", "\n")]
    except Exception:  # noqa: BLE001
        # Splash must never take down the whole TUI paint path.
        return _render_doggy_idle_panel(max(1, width))


def _half_block(
    top: str,
    bottom: str,
    palette: dict[str, str],
) -> tuple[str, str]:
    background = palette.get(".", "#000000")
    top_color = palette.get(top, background)
    bottom_color = palette.get(bottom, background)
    if top == bottom == ".":
        return f"bg:{background}", " "
    if top == bottom:
        return f"fg:{top_color} bg:{background}", "█"
    if top == ".":
        return f"fg:{bottom_color} bg:{background}", "▄"
    if bottom == ".":
        return f"fg:{top_color} bg:{background}", "▀"
    return f"fg:{top_color} bg:{bottom_color}", "▀"


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


def _wrap_display_lines(text: str, width: int, *, max_lines: int = 40) -> list[str]:
    """Wrap text to display-cell width (no ellipsis drop of trailing content)."""
    width = max(1, int(width))
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return [""]
    lines: list[str] = []
    for para in raw.split("\n"):
        if not para:
            lines.append("")
            if len(lines) >= max_lines:
                break
            continue
        buf: list[str] = []
        used = 0
        for char in para:
            cw = get_cwidth(char)
            if buf and used + cw > width:
                lines.append("".join(buf))
                if len(lines) >= max_lines:
                    return lines
                buf = [char]
                used = cw
            else:
                buf.append(char)
                used += cw
        if buf:
            lines.append("".join(buf))
            if len(lines) >= max_lines:
                break
    return lines or [""]


def _brief_two_lines(text: str, full_width: int) -> list[str]:
    """Task brief: longer first line, slightly shorter second (ragged right)."""
    full_width = max(8, int(full_width))
    # ~88% / ~72% of available report columns — first longer, second a bit shorter.
    w1 = max(6, min(full_width, int(full_width * 0.88)))
    w2 = max(5, min(full_width - 1, int(full_width * 0.72)))
    raw = " ".join((text or "").replace("\r", "\n").split())
    if not raw:
        return [""]
    if get_cwidth(raw) <= w1:
        return [raw]
    # Fill line 1 to w1, line 2 to w2 (ellipsis if more remains).
    line1_chars: list[str] = []
    used = 0
    rest = raw
    for i, ch in enumerate(raw):
        cw = get_cwidth(ch)
        if used + cw > w1:
            rest = raw[i:]
            break
        line1_chars.append(ch)
        used += cw
    else:
        return ["".join(line1_chars)]
    line1 = "".join(line1_chars).rstrip()
    rest = rest.lstrip()
    if not rest:
        return [line1]
    line2 = _truncate_display(rest, w2)
    return [line1, line2]
