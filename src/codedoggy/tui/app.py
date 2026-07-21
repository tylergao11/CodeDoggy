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
from prompt_toolkit.filters import Condition, has_selection
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import FileHistory, InMemoryHistory
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
from prompt_toolkit.layout.margins import Margin
from prompt_toolkit.layout.screen import Point
from prompt_toolkit.layout.processors import (
    AfterInput,
    ConditionalProcessor,
    PasswordProcessor,
)
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.output.color_depth import ColorDepth
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Frame, TextArea

from codedoggy.session.types import SessionPhase, TurnStatus
from codedoggy.tui.clipboard_image import (
    get_system_clipboard_text,
    insert_image_chip,
    insert_path_token,
    save_clipboard_image,
)
from codedoggy.tui.agent_detail import (
    DETAIL_FILTERS,
    DETAIL_FILTER_LABELS,
    AgentDetailSnapshot,
    DetailFilter,
    render_detail_body,
    snapshot_from_messages,
)
from codedoggy.tui.open_path import (
    VIEW_IMAGE_LABEL,
    open_local_path,
    path_under_cursor,
)
from codedoggy.tui.theme import build_style
from codedoggy.tui.activity import LiveActivityBoard
from codedoggy.tui.login_wizard import AuthWizard, WizardStep, run_browser_login
from codedoggy.tui.model import TaskLedger, TaskView
from codedoggy.tui import surface as session_surface
from codedoggy.turn.types import Message, Role


STATUS_TEXT = {
    "waiting": "等待",
    "pending": "准备中",
    "running": "推进中",
    "queued": "排队中",
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
        reasoning = last.get("reasoning_content")
    else:
        role = getattr(last, "role", None)
        call_id = getattr(last, "tool_call_id", None) or getattr(last, "id", None)
        name = getattr(last, "name", None)
        content = getattr(last, "content", None)
        tool_calls = getattr(last, "tool_calls", None) or []
        reasoning = getattr(last, "reasoning_content", None)
    if isinstance(content, str):
        content_sig: tuple[Any, ...] = (len(content), content[-64:])
    elif content is None:
        content_sig = (0, "")
    else:
        try:
            content_sig = (len(content), type(content).__name__)
        except TypeError:
            content_sig = (-1, type(content).__name__)
    if isinstance(reasoning, str):
        reasoning_sig: tuple[Any, ...] = (len(reasoning), reasoning[-64:])
    else:
        reasoning_sig = (0, "")
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
        reasoning_sig,
        len(tool_calls),
        tool_sig,
    )


# Box + doggy motif (rounded frames ∪･ω･∪ are the visual signature).
_DOG_FACE = "∪･ω･∪"
_DOG_EAR = "∪"
_INPUT_MAX_LINES = 8
_DETAIL_INPUT_MAX_LINES = 6
_DOUBLE_CLICK_S = 0.45  # task/plan card double-click window


class InteractiveScrollbarMargin(Margin):
    """Clickable/draggable scrollbar (stock PT ScrollbarMargin is paint-only).

    Important: prompt_toolkit's ``Window._copy_margin`` paints margin glyphs but
    never registers fragment mouse handlers. Use :class:`ScrollableWindow` so
    :meth:`install_mouse_handlers` wires the right-margin column (and expands
    to the full window while dragging — a 1-cell rail cannot receive MOVE).
    """

    def __init__(
        self,
        *,
        on_scroll: Callable[[int], None] | None = None,
        display_arrows: bool = True,
    ) -> None:
        self.on_scroll = on_scroll
        self.display_arrows = display_arrows
        self._dragging = False
        self._drag_grab_row = 0  # row within track where thumb was grabbed
        self._drag_start_scroll = 0

    def get_width(self, get_ui_content: Callable[[], Any]) -> int:
        return 1

    def _geometry(
        self, window_render_info: Any
    ) -> tuple[Any, int, int, int, int, int, int, bool]:
        content_height = int(window_render_info.content_height)
        window_height = int(window_render_info.window_height)
        scroll = int(window_render_info.vertical_scroll)
        win = window_render_info.window
        show_arrows = bool(self.display_arrows)
        track_height = max(1, window_height - (2 if show_arrows else 0))
        max_scroll = max(0, content_height - window_height)
        fraction_visible = min(
            1.0, window_height / float(max(1, content_height))
        )
        thumb_h = max(1, min(track_height, int(round(track_height * fraction_visible))))
        if max_scroll <= 0:
            thumb_top = 0
            thumb_h = track_height
        else:
            travel = max(0, track_height - thumb_h)
            thumb_top = int(round((scroll / float(max_scroll)) * travel))
            thumb_top = max(0, min(travel, thumb_top))
        return (
            win,
            scroll,
            track_height,
            max_scroll,
            thumb_h,
            thumb_top,
            content_height,
            show_arrows,
        )

    def _set_scroll(self, win: Any, max_scroll: int, value: int) -> None:
        new_scroll = max(0, min(max_scroll, int(value)))
        try:
            win.vertical_scroll = new_scroll
        except Exception:  # noqa: BLE001
            return
        if self.on_scroll is not None:
            try:
                self.on_scroll(new_scroll)
            except Exception:  # noqa: BLE001
                pass
        try:
            get_app().invalidate()
        except Exception:  # noqa: BLE001
            pass

    def create_margin(
        self, window_render_info: Any, width: int, height: int
    ) -> StyleAndTextTuples:
        (
            _win,
            _scroll,
            track_height,
            max_scroll,
            thumb_h,
            thumb_top,
            content_height,
            show_arrows,
        ) = self._geometry(window_render_info)
        if content_height <= 0:
            return [("", "\n") for _ in range(max(1, height))]

        # Fragments are paint-only unless ScrollableWindow installs handlers.
        result: StyleAndTextTuples = []
        if show_arrows:
            result.append(("class:scrollbar.arrow", "▴"))
            result.append(("", "\n"))

        for i in range(track_height):
            is_thumb = thumb_top <= i < thumb_top + thumb_h
            if is_thumb:
                style = "class:scrollbar.button"
                if i == thumb_top + thumb_h - 1:
                    style = "class:scrollbar.button,scrollbar.end"
                result.append((style, " "))
            else:
                style = "class:scrollbar.background"
                if i + 1 == thumb_top:
                    style = "class:scrollbar.background,scrollbar.start"
                result.append((style, " "))
            result.append(("", "\n"))

        if show_arrows:
            result.append(("class:scrollbar.arrow", "▾"))

        return result

    def install_mouse_handlers(
        self,
        mouse_handlers: Any,
        *,
        window_render_info: Any,
        bar_xpos: int,
        ypos: int,
        height: int,
        capture_x_min: int,
        capture_x_max: int,
    ) -> None:
        """Register click/drag handlers on the PT mouse raster (screen coords)."""
        (
            win,
            _scroll,
            track_height,
            max_scroll,
            thumb_h,
            thumb_top,
            content_height,
            show_arrows,
        ) = self._geometry(window_render_info)
        if content_height <= 0 or height <= 0:
            return

        set_scroll = lambda value: self._set_scroll(win, max_scroll, value)

        def track_row_from_screen_y(screen_y: int) -> int:
            local = int(screen_y) - int(ypos)
            if show_arrows:
                local -= 1
            return max(0, min(track_height - 1, local))

        def apply_thumb_drag(screen_y: int) -> None:
            if track_height <= 1 or max_scroll <= 0:
                return
            travel = max(1, track_height - thumb_h)
            row = track_row_from_screen_y(screen_y)
            delta = row - self._drag_grab_row
            set_scroll(self._drag_start_scroll + int(round((delta / travel) * max_scroll)))

        def jump_track(screen_y: int) -> None:
            if track_height <= 1 or max_scroll <= 0:
                set_scroll(0)
                return
            travel = max(1, track_height - thumb_h)
            row = track_row_from_screen_y(screen_y)
            target_top = max(0, min(travel, row - thumb_h // 2))
            set_scroll(int(round((target_top / travel) * max_scroll)))

        def cell_handler(kind: str) -> Callable[[MouseEvent], object]:
            def handler(event: MouseEvent) -> object:
                et = event.event_type
                if et is MouseEventType.SCROLL_UP:
                    set_scroll(int(getattr(win, "vertical_scroll", 0) or 0) - 3)
                    return None
                if et is MouseEventType.SCROLL_DOWN:
                    set_scroll(int(getattr(win, "vertical_scroll", 0) or 0) + 3)
                    return None
                if et is MouseEventType.MOUSE_DOWN:
                    if kind == "up":
                        set_scroll(int(getattr(win, "vertical_scroll", 0) or 0) - 1)
                    elif kind == "down":
                        set_scroll(int(getattr(win, "vertical_scroll", 0) or 0) + 1)
                    elif kind == "thumb":
                        self._dragging = True
                        self._drag_grab_row = track_row_from_screen_y(event.position.y)
                        self._drag_start_scroll = int(
                            getattr(win, "vertical_scroll", 0) or 0
                        )
                        # Re-paint so install_mouse_handlers expands capture
                        # beyond the 1-cell rail for subsequent MOVE events.
                        try:
                            get_app().invalidate()
                        except Exception:  # noqa: BLE001
                            pass
                    elif kind == "track":
                        jump_track(event.position.y)
                    return None
                if et is MouseEventType.MOUSE_MOVE and self._dragging:
                    apply_thumb_drag(event.position.y)
                    return None
                if et is MouseEventType.MOUSE_UP:
                    if self._dragging:
                        apply_thumb_drag(event.position.y)
                    self._dragging = False
                    return None
                return NotImplemented

            return handler

        def drag_capture(event: MouseEvent) -> object:
            """Full-window capture while thumb-dragging (cursor leaves the rail)."""
            et = event.event_type
            if not self._dragging:
                return NotImplemented
            if et is MouseEventType.MOUSE_MOVE:
                apply_thumb_drag(event.position.y)
                return None
            if et is MouseEventType.MOUSE_UP:
                apply_thumb_drag(event.position.y)
                self._dragging = False
                return None
            return NotImplemented

        # While dragging, own the whole window so MOVE/UP still reach us.
        if self._dragging:
            mouse_handlers.set_mouse_handler_for_range(
                x_min=capture_x_min,
                x_max=capture_x_max,
                y_min=ypos,
                y_max=ypos + height,
                handler=drag_capture,
            )

        # Always wire the 1-cell rail (arrows / track / thumb).
        for row in range(height):
            screen_y = ypos + row
            if show_arrows and row == 0:
                kind = "up"
            elif show_arrows and row == height - 1:
                kind = "down"
            else:
                track_i = row - (1 if show_arrows else 0)
                if thumb_top <= track_i < thumb_top + thumb_h:
                    kind = "thumb"
                else:
                    kind = "track"
            mouse_handlers.mouse_handlers[screen_y][bar_xpos] = cell_handler(kind)


class ScrollableWindow(Window):
    """Window whose right InteractiveScrollbarMargin receives real mouse events."""

    def write_to_screen(
        self,
        screen: Any,
        mouse_handlers: Any,
        write_position: Any,
        parent_style: str,
        erase_bg: bool,
        z_index: int | None,
    ) -> None:
        super().write_to_screen(
            screen,
            mouse_handlers,
            write_position,
            parent_style,
            erase_bg,
            z_index,
        )
        info = getattr(self, "render_info", None)
        if info is None:
            return
        right_widths = [self._get_margin_width(m) for m in self.right_margins]
        total_right = sum(right_widths)
        if total_right <= 0:
            return
        # Prefer the clamped rect Window actually painted into.
        painted = getattr(screen, "visible_windows_to_write_positions", {}).get(self)
        wp = painted if painted is not None else write_position
        xpos = int(wp.xpos)
        ypos = int(wp.ypos)
        width = int(wp.width)
        height = int(wp.height)
        bar_x = xpos + width - total_right
        for margin, mw in zip(self.right_margins, right_widths):
            if mw > 0 and isinstance(margin, InteractiveScrollbarMargin):
                margin.install_mouse_handlers(
                    mouse_handlers,
                    window_render_info=info,
                    bar_xpos=bar_x,
                    ypos=ypos,
                    height=height,
                    capture_x_min=xpos,
                    capture_x_max=xpos + width,
                )
            bar_x += mw


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


# Default ``fresh``; override with CODEDOGGY_THEME=groknight|dark|quiet|cute.
CODEDOGGY_DARK = build_style()


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
        # -1 = no intentional selection (never treat 0 as "default first card").
        self._selected_task = -1
        # False until user picks a card / keyboard-nav / follow-latest.
        # Prevents "move/click blank → yellow first task" (selection must be intentional).
        self._task_selection_active = False
        self._selected_agent_by_task: dict[str, int] = {}
        self._selected_line = 0
        self._task_line_count = 1  # clamp cursor y — PT crashes if y >= line_count
        self._pinned_task_for_line: int | None = None  # re-pin only on task change
        # Active-task interject flash for homepage card: task_id -> (until, preview)
        self._interject_flash: dict[str, tuple[float, str]] = {}
        self._modal_open = False
        self._modal_kind: str = "agent"  # agent | auth | ask
        self._modal_ref: tuple[str, str] | None = None
        # ask_user_question host (Grok AskUser) — blocks worker, UI inside modal.
        self._ask_event = threading.Event()
        self._ask_questions: list[dict[str, Any]] = []
        self._ask_q_index = 0
        self._ask_opt_index = 0
        self._ask_answers: dict[str, list[str]] = {}
        self._ask_result: dict[str, Any] | None = None
        self._ask_multi_picked: set[int] = set()
        self._ask_other_editing = False
        self._ask_active = False
        self._detail_messages: dict[tuple[str, str], list[Any]] = {}
        self._detail_filter: DetailFilter = "message"
        self._detail_collapsed: set[str] = set()
        self._detail_collapse_seeded_for: tuple[str, str] | None = None
        self._detail_known_fold_keys: set[str] = set()
        self._detail_cursor_line = 0
        self._detail_line_count = 1  # clamp detail cursor y — same class as task crash
        self._detail_scroll_syncing = False
        # Cache full detail body paint — plan-mode memory tools explode message
        # volume; re-render every 100ms freezes the UI into an "infinite loop".
        self._detail_body_cache: StyleAndTextTuples | None = None
        self._detail_body_cache_key: tuple[Any, ...] | None = None
        self._detail_scroll_handler: Callable[[MouseEvent], object] | None = None
        self._redraw_pending = False
        self._closing = False
        self._task_started_at: float | None = None
        self._quit_armed_until = 0.0
        self._feedback_text = ""
        self._feedback_kind = "info"
        # Plan mode host hooks (task-card driven; keyboard a/s/q + Enter/Esc).
        self._plan_ui: str | None = None  # None | "consent" | "review"
        self._plan_ui_task_id: str | None = None
        self._plan_consent_event = threading.Event()
        self._plan_consent_ok = False
        self._plan_exit_event = threading.Event()
        self._plan_exit_outcome = "approved"
        self._plan_exit_feedback = ""
        # True while exit_plan_mode host fn is blocked on a/s/q.
        self._plan_exit_waiting = False
        # Grok-style todo plan badge + expandable list pane.
        self._todo_pane_open = False
        self._todo_scroll = 0  # first visible item index when list is long
        # Task card: DOWN+UP same card selects; double-click / Ctrl+left opens.
        self._task_mouse_down_index: int | None = None
        self._task_card_last_click: tuple[int, float] | None = None
        self._todo_badge_last_click: float | None = None
        self._subagent_task: dict[str, str] = {}
        self._subagent_baselines: dict[str, set[str]] = {}
        self._subagent_live_signatures: dict[
            tuple[str, str], tuple[Any, ...]
        ] = {}
        self._auth_wizard = AuthWizard()
        self._auth_login_worker: threading.Thread | None = None
        self._auth_login_cancel: threading.Event | None = None
        # Login finished while a turn was running — apply after idle.
        self._pending_reload: dict[str, Any] | None = None
        self._pending_prompt: str | None = None
        # One-shot startup brand (concept art). Dismissed forever on first task;
        # not "empty ledger" — finished tasks never bring the splash back.
        self._startup_brand = not bool(
            initial_prompt and str(initial_prompt).strip()
        )
        # before_render throttle + splash cache (ESC/modal close snappiness)
        self._last_sync_runtime_at = 0.0
        self._paint_clock = 0.0  # one monotonic sample per paint (no extra invalidate)
        self._doggy_empty_cache: tuple[tuple[Any, ...], StyleAndTextTuples] | None = None
        # Full task-list fragment cache: skip card rebuild when content unchanged.
        self._task_paint_cache: tuple[Any, ...] | None = None
        # Scroll-to-latest only — does NOT auto-select a card (that caused
        # "blank → first/latest always selected" when follow defaulted True).
        self._follow_latest_task = False
        # Live tool/activity lines from on_live_message (effect layer, not truth).
        self._activity = LiveActivityBoard()
        self._subagent_listener_bound = False
        self._session_listener_bound = False
        self._external_turn_views: dict[int, dict[str, Any]] = {}
        self._view_lock = threading.RLock()
        self._prompt_history = self._make_prompt_history()
        self._last_pasted_path: str | None = None

        self._task_control = FormattedTextControl(
            text=self._render_tasks,
            focusable=True,
            show_cursor=False,
            get_cursor_position=self._task_cursor_position,
        )
        self._task_window = ScrollableWindow(
            content=self._task_control,
            wrap_lines=False,  # line y == content row; wrap broke scroll/cursor map
            scroll_offsets=ScrollOffsets(top=1, bottom=3),
            right_margins=[
                InteractiveScrollbarMargin(on_scroll=self._on_task_scrollbar)
            ],
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
            history=self._prompt_history,
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
        self._detail_window = ScrollableWindow(
            content=self._detail_control,
            wrap_lines=False,
            scroll_offsets=ScrollOffsets(top=1, bottom=2),
            right_margins=[
                InteractiveScrollbarMargin(on_scroll=self._on_detail_scrollbar)
            ],
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
                        "任务进行中可在这里补一句…",
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
        self._wire_buffer_ctrl_click(self._input)
        self._wire_buffer_ctrl_click(self._detail_input)

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
        todo_pane = Window(
            FormattedTextControl(self._render_todo_pane),
            height=self._todo_pane_height,
            style="class:root",
            wrap_lines=False,
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
                todo_pane,
                Window(height=1, style="class:root"),
                prompt_box,
                shortcuts,
            ],
            style="class:root",
        )

        close_control = FormattedTextControl(
            [
                (
                    "class:agent-window.close",
                    "  ×  ",
                    self._close_mouse,
                )
            ],
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
                    filter=Condition(self._detail_input_visible),
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
            # Agent detail + auth only — questionnaire is a separate float.
            filter=Condition(
                lambda: self._modal_open and self._modal_kind in {"agent", "auth"}
            ),
        )
        street_hud = ConditionalContainer(
            Window(
                FormattedTextControl(self._render_street_hud),
                width=44,
                height=5,
                style="class:root",
            ),
            # Only while empty startup chrome is up AND auth is still needed.
            filter=Condition(self._show_street_hud),
        )
        # Dedicated ask_user_question dialog — bordered float, not plan/agent shell.
        self._ask_control = FormattedTextControl(
            text=self._render_ask_dialog,
            focusable=True,
            show_cursor=False,
        )
        self._ask_window = Window(
            content=self._ask_control,
            wrap_lines=False,
            style="class:ask.dialog",
        )
        ask_dialog = ConditionalContainer(
            Frame(
                body=HSplit(
                    [
                        Window(
                            FormattedTextControl(self._render_ask_dialog_title),
                            height=1,
                            style="class:ask.header",
                        ),
                        self._ask_window,
                        Window(
                            FormattedTextControl(self._render_ask_dialog_hint),
                            height=1,
                            style="class:ask.hint",
                        ),
                    ],
                    style="class:ask.dialog",
                ),
                title=" 问卷 ",
                style="class:ask.border",
            ),
            filter=Condition(
                lambda: bool(self._ask_active and self._modal_kind == "ask")
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
                # Questionnaire: compact bordered float (not full-screen).
                Float(
                    top=4,
                    bottom=8,
                    left=10,
                    right=10,
                    content=ask_dialog,
                    transparent=False,
                    z_index=20,
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
            # GrokBuild animation.fps default = 30.
            refresh_interval=1.0 / 30.0,
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
            self._wire_plan_mode_hooks()
            # Ensure plan/todo disk state is visible even if hydrate skipped.
            kernel = getattr(self.session.extensions, "kernel", None)
            if kernel is not None:
                if hasattr(kernel, "load_todo_state"):
                    try:
                        kernel.load_todo_state()
                    except Exception:  # noqa: BLE001
                        pass
            self._maybe_restore_plan_approval_chrome()
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
                # Do not steal focus / selection after a turn starts — stay on input.
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
            # Terminal state is on the task card — only surface failures.
            if status != "completed":
                self._set_feedback("后台任务未完成", "warning")
            else:
                self._clear_feedback()
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
        self.ledger.apply_agent_status(
            task_id,
            sub_id,
            label=label,
            status=status,
            output=output,
            description=description,
        )
        if status in {"pending", "running"}:
            self.ledger.set_task_phase(task_id, "parallel")
        self._request_redraw()

    def _build_key_bindings(self) -> KeyBindings:
        keys = KeyBindings()
        modal = Condition(lambda: self._modal_open)
        auth_modal = Condition(lambda: self._modal_open and self._modal_kind == "auth")
        agent_modal = Condition(lambda: self._modal_open and self._modal_kind == "agent")
        ask_modal = Condition(
            lambda: bool(self._ask_active and self._modal_kind == "ask")
        )
        tasks_focused = Condition(
            lambda: not self._modal_open
            and not self._ask_active
            and get_app().layout.has_focus(self._task_window)
        )
        # Scroll detail with ↑↓ whenever agent modal is open and the interject
        # box is not focused (do not require detail_window focus — title/filter
        # clicks used to leave keys dead).
        detail_focused = Condition(
            lambda: self._modal_open
            and self._modal_kind == "agent"
            and not get_app().layout.has_focus(self._detail_input)
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

        # Mouse selection does not set shift_mode — stock Backspace won't cut.
        @keys.add("delete", filter=prompt_paste & has_selection, eager=True)
        @keys.add("backspace", filter=prompt_paste & has_selection, eager=True)
        def _cut_input_selection(event: Any) -> None:
            event.current_buffer.cut_selection()
            event.app.invalidate()

        # Enter = submit. Ctrl+Enter (c-j; on Windows: Esc then c-j) = hard newline.
        # When questionnaire is waiting for Other free-text, Enter finishes Other.
        @keys.add("enter", filter=main_input_focused, eager=True)
        def _submit_main_input(event: Any) -> None:
            if self._ask_active and self._ask_other_editing:
                self._ask_confirm_current()
                return
            self._input.buffer.validate_and_handle()

        @keys.add("enter", filter=detail_input_focused, eager=True)
        def _submit_detail_input(event: Any) -> None:
            self._detail_input.buffer.validate_and_handle()

        @keys.add("c-j", filter=main_input_focused, eager=True)
        @keys.add("escape", "c-j", filter=main_input_focused, eager=True)
        def _newline_main_input(event: Any) -> None:
            self._insert_buffer_newline(event, max_lines=_INPUT_MAX_LINES)

        @keys.add("c-j", filter=detail_input_focused, eager=True)
        @keys.add("escape", "c-j", filter=detail_input_focused, eager=True)
        def _newline_detail_input(event: Any) -> None:
            self._insert_buffer_newline(event, max_lines=_DETAIL_INPUT_MAX_LINES)

        @keys.add("c-u", filter=main_input_focused, eager=True)
        def _clear_main_input(event: Any) -> None:
            event.current_buffer.text = ""
            event.app.invalidate()

        @keys.add("c-u", filter=detail_input_focused, eager=True)
        def _clear_detail_input(event: Any) -> None:
            event.current_buffer.text = ""
            event.app.invalidate()

        # Tab = task cycle. Shift+Tab = Plan ↔ Auto. (Not while auth/ask.)
        tab_tasks_ok = Condition(
            lambda: not (
                self._ask_active
                or (self._modal_open and self._modal_kind in {"auth", "ask"})
            )
        )
        plan_toggle_ok = Condition(
            lambda: not self._modal_open
            and not self._ask_active
        )

        @keys.add("tab", filter=tab_tasks_ok, eager=True)
        def _tab_to_tasks(_: Any) -> None:
            self._tab_task_cycle()

        @keys.add("s-tab", filter=plan_toggle_ok, eager=True)
        def _shift_tab_plan(_: Any) -> None:
            self._toggle_session_plan_mode()

        # Ctrl+Space / Windows NUL (c-@): same as Tab cycle.
        @keys.add("c-space", filter=tab_tasks_ok, eager=True)
        @keys.add("c-@", filter=tab_tasks_ok, eager=True)
        def _ctrl_space_tasks(_: Any) -> None:
            self._tab_task_cycle()

        # Todo list open: ↑↓ scroll the plan checklist (not task cards).
        todo_pane_nav = Condition(
            lambda: self._todo_pane_open
            and not self._modal_open
            and not get_app().layout.has_focus(self._input)
        )

        @keys.add("up", filter=todo_pane_nav, eager=True)
        def _todo_scroll_up(_: Any) -> None:
            self._scroll_todo_pane(-1)

        @keys.add("down", filter=todo_pane_nav, eager=True)
        def _todo_scroll_down(_: Any) -> None:
            self._scroll_todo_pane(1)

        @keys.add("pageup", filter=todo_pane_nav, eager=True)
        def _todo_page_up(_: Any) -> None:
            self._scroll_todo_pane(-5)

        @keys.add("pagedown", filter=todo_pane_nav, eager=True)
        def _todo_page_down(_: Any) -> None:
            self._scroll_todo_pane(5)

        # Agent detail: Tab closes → tasks (via tab_tasks_ok). Detail input
        # focus stays via click / Space patterns elsewhere.

        @keys.add("up", filter=detail_focused, eager=True)
        def _detail_up(_: Any) -> None:
            self._move_detail_cursor(-1)

        @keys.add("down", filter=detail_focused, eager=True)
        def _detail_down(_: Any) -> None:
            self._move_detail_cursor(1)

        @keys.add("pageup", filter=detail_focused, eager=True)
        def _detail_page_up(_: Any) -> None:
            self._move_detail_cursor(-max(4, _terminal_height() - 10))

        @keys.add("pagedown", filter=detail_focused, eager=True)
        def _detail_page_down(_: Any) -> None:
            self._move_detail_cursor(max(4, _terminal_height() - 10))

        @keys.add("home", filter=detail_focused, eager=True)
        def _detail_home(_: Any) -> None:
            self._scroll_detail_to_line(0)

        @keys.add("end", filter=detail_focused, eager=True)
        def _detail_end(_: Any) -> None:
            self._scroll_detail_to_bottom()

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

        @keys.add("left", filter=detail_focused)
        def _detail_filter_prev(_: Any) -> None:
            self._cycle_detail_filter(-1)

        @keys.add("right", filter=detail_focused)
        def _detail_filter_next(_: Any) -> None:
            self._cycle_detail_filter(1)

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

        @keys.add("up", filter=ask_modal, eager=True)
        def _ask_up(_: Any) -> None:
            self._ask_move_option(-1)

        @keys.add("down", filter=ask_modal, eager=True)
        def _ask_down(_: Any) -> None:
            self._ask_move_option(1)

        @keys.add("enter", filter=ask_modal, eager=True)
        def _ask_enter(_: Any) -> None:
            self._ask_confirm_current()

        @keys.add("space", filter=ask_modal, eager=True)
        def _ask_space(_: Any) -> None:
            if self._ask_is_multi():
                self._ask_toggle_multi()
            else:
                self._ask_confirm_current()

        # Questionnaire: Tab exits (same “leave layer” as task detail). No s/h.
        @keys.add("tab", filter=ask_modal, eager=True)
        def _tab_exit_ask(_: Any) -> None:
            self._resolve_ask({"outcome": "cancelled"})

        # Auth wizard: Tab = back / close (not Esc).
        @keys.add("tab", filter=auth_modal, eager=True)
        def _tab_auth_back(_: Any) -> None:
            action = self._auth_wizard.go_back()
            self._dispatch_wizard_action(action)

        @keys.add("escape")
        def _escape(event: Any) -> None:
            # Esc = cancel running task only (not leave UI layers — use Tab).
            if self._is_running():
                self._cancel_current()
                return
            # Idle / auth / questionnaire: do nothing.

        # Input history (Grok-style): ↑/↓ on first/last line cycle prior prompts.
        @keys.add("up", filter=main_input_focused & ~ask_modal, eager=True)
        def _input_history_up(event: Any) -> None:
            buf = event.current_buffer
            doc = buf.document
            if doc.cursor_position_row > 0:
                buf.cursor_up(count=1)
                return
            buf.history_backward(count=1)
            event.app.invalidate()

        @keys.add("down", filter=main_input_focused & ~ask_modal, eager=True)
        def _input_history_down(event: Any) -> None:
            buf = event.current_buffer
            doc = buf.document
            if doc.cursor_position_row < doc.line_count - 1:
                buf.cursor_down(count=1)
                return
            buf.history_forward(count=1)
            event.app.invalidate()

        @keys.add("c-q")
        def _quit(_: Any) -> None:
            self._request_quit()

        plan_consent = Condition(lambda: self._plan_ui == "consent")
        plan_review = Condition(
            lambda: self._plan_ui == "review"
            or (
                self._modal_open
                and self._modal_kind == "agent"
                and self._detail_filter == "plan"
                and self._task_awaiting_plan_approval()
            )
        )

        @keys.add("enter", filter=plan_consent, eager=True)
        def _plan_consent_yes(_: Any) -> None:
            self._resolve_plan_consent(True)

        @keys.add("escape", filter=plan_consent, eager=True)
        def _plan_consent_no(_: Any) -> None:
            self._resolve_plan_consent(False)

        @keys.add("a", filter=plan_review, eager=True)
        def _plan_approve(_: Any) -> None:
            self._resolve_plan_exit("approved")

        @keys.add("s", filter=plan_review, eager=True)
        def _plan_revise(_: Any) -> None:
            self._resolve_plan_exit("revise")

        @keys.add("q", filter=plan_review, eager=True)
        def _plan_quit(_: Any) -> None:
            self._resolve_plan_exit("abandoned")

        return keys

    @staticmethod
    def _insert_buffer_newline(event: Any, *, max_lines: int) -> None:
        """Hard newline at caret (Ctrl+Enter / Ctrl+J). Caps at input max lines.

        Mutates ``Buffer`` text/cursor directly so offline tests (no event loop)
        and TUI path stay free of completer side-effects from ``insert_text``.
        """
        buffer = event.current_buffer
        # lines == newlines + 1; refuse when already at cap
        if buffer.text.count("\n") + 1 >= max(1, int(max_lines)):
            return
        pos = buffer.cursor_position
        text = buffer.text
        buffer.text = text[:pos] + "\n" + text[pos:]
        buffer.cursor_position = pos + 1
        try:
            event.app.invalidate()
        except Exception:  # noqa: BLE001
            pass

    def _paste_into_buffer(self, event: Any) -> None:
        """If OS clipboard holds an image, dump it and insert a「查看图片」chip.

        Otherwise paste OS / prompt_toolkit text. Intercepting Ctrl+V means we
        must read the *system* clipboard ourselves — PT's pad is often empty.
        """
        buffer = event.current_buffer
        if buffer.selection_state is not None:
            buffer.cut_selection()
        cwd = getattr(self.session, "cwd", None) or Path.cwd()
        try:
            saved = save_clipboard_image(cwd)
        except Exception:  # noqa: BLE001
            saved = None
        if saved is not None:
            token = insert_image_chip(saved, cwd=cwd)
            pos = buffer.cursor_position
            before = buffer.text[:pos]
            lead = " " if before and not before[-1].isspace() else ""
            insert = f"{lead}{token} "
            text = buffer.text
            buffer.text = text[:pos] + insert + text[pos:]
            buffer.cursor_position = pos + len(insert)
            self._last_pasted_path = str(saved)
            self._set_feedback(
                f"已粘贴{VIEW_IMAGE_LABEL} · Ctrl+点击路径可打开",
                "info",
            )
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
            pos = buffer.cursor_position
            body = buffer.text
            buffer.text = body[:pos] + text + body[pos:]
            buffer.cursor_position = pos + len(text)
        else:
            self._set_feedback("剪贴板没有可用图片或文字", "warning")
        event.app.invalidate()

    def _accept_prompt(self, buffer: Any) -> bool:
        prompt = buffer.text.strip()
        if not prompt:
            buffer.text = ""
            return True
        # Persist into ↑/↓ history before clearing (prompt_toolkit stores on accept
        # only when working_index is at the end — we also append explicitly).
        try:
            hist = getattr(buffer, "history", None)
            if hist is not None:
                hist.append_string(prompt)
        except Exception:  # noqa: BLE001
            pass
        buffer.text = ""
        self._clear_feedback()
        self._dismiss_startup_brand()
        if self._worker is not None and self._worker.is_alive():
            tid = self._active_task_id
            self.session.interject(prompt, prompt_id=tid)
            if tid:
                self._note_interject(tid, prompt)
            # Interject flash lives on the task card — no toast.
            self.app.invalidate()
            return True
        if not self._ensure_auth_ready():
            self._pending_prompt = prompt
            self._open_auth_wizard()
            # Keep text in the box so the user can edit; send resumes after login.
            self._input.text = prompt
            preview = _truncate_display(prompt.replace("\n", " "), 36)
            self._set_feedback(f"先登录 · 将发送：{preview}", "warning")
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
        if self._modal_kind == "ask":
            # Other free-text is confirmed via same path as option Enter.
            self._ask_confirm_current()
            return True

        prompt = buffer.text.strip()
        buffer.text = ""
        if not prompt or self._modal_ref is None:
            return True
        if not self._is_running():
            self._set_feedback("任务已结束，无法继续插话", "warning")
            self.app.invalidate()
            return True
        task_id, agent_id = self._modal_ref
        if task_id != self._active_task_id:
            self._set_feedback(
                "只能向当前运行任务补充指令",
                "warning",
            )
            self.app.invalidate()
            return True
        agent = self.ledger.get_agent(task_id, agent_id)
        label = "AGENT" if agent is None else agent.label
        routed = prompt if label == "MAIN" else f"请转交给 {label}：{prompt}"
        self.session.interject(routed, prompt_id=task_id)
        self._note_interject(task_id, prompt)
        self._set_feedback(f"补充指令已交给 MAIN · {label}", "info")
        self.app.layout.focus(self._detail_window)
        self.app.invalidate()
        return True

    def _note_interject(self, task_id: str, text: str) -> None:
        """Homepage card pulse: show 插入中 until flash expires."""
        preview = _truncate_display(
            " ".join((text or "").split()) or "…", 42
        )
        self._interject_flash[str(task_id)] = (
            time.monotonic() + 5.0,
            preview,
        )
        # Bump live line so cache key changes even if report text is stale.
        try:
            self.ledger.update_live_agent(
                task_id,
                f"{task_id}:main",
                label="MAIN",
                status="running",
                output=f"↩ 插入中 · {preview}",
            )
        except Exception:  # noqa: BLE001
            pass
        self._task_paint_cache = None

    def _interject_preview(self, task_id: str) -> str | None:
        item = self._interject_flash.get(str(task_id))
        if item is None:
            return None
        until, preview = item
        if time.monotonic() > until:
            self._interject_flash.pop(str(task_id), None)
            return None
        return preview

    def _start_task(self, prompt: str) -> None:
        self._dismiss_startup_brand()
        self._bind_subagent_listener()
        task = self.ledger.create(prompt)
        # Keep keyboard on the prompt — never auto-select task / plan after submit.
        self._active_task_id = task.id
        self._detail_messages[(task.id, f"{task.id}:main")] = []
        self._activity.clear_task(task.id)
        self._task_started_at = time.monotonic()
        self._subagent_baselines[task.id] = {
            item.subagent_id for item in self._subagents()
        }
        self._clear_feedback()
        worker = threading.Thread(
            target=self._run_task,
            args=(task.id, prompt),
            name=f"codedoggy-{task.id}",
            daemon=True,
        )
        self._worker = worker
        worker.start()
        try:
            self.app.layout.focus(self._input)
        except Exception:  # noqa: BLE001
            pass
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
            if status != "completed" and result.error:
                output = _friendly_failure_toast(str(result.error))
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
                # MAIN turn ended but children still run — not failure / not done.
                # _sync_runtime finishes the task when children become terminal.
                self.ledger.set_task_status(task_id, "running")
                self.ledger.set_task_phase(task_id, "parallel")
                feedback: tuple[str, str] | None = None
            elif failed_children:
                self.ledger.finish_task(task_id, "failed")
                feedback = ("子 Agent 未全部成功", "warning")
            elif status == "completed":
                self.ledger.finish_task(task_id, "completed")
                # Card already shows 已完成 + duration — no toast.
                feedback = None
            else:
                self.ledger.finish_task(task_id, status)
                feedback = (_friendly_failure_toast(result.error or output), "warning")

            def apply_success() -> None:
                if feedback is None:
                    self._clear_feedback()
                else:
                    self._set_feedback(feedback[0], feedback[1])

            self._call_in_ui_thread(apply_success)
        except Exception as exc:  # noqa: BLE001
            callback_state["active"] = False
            message = f"{type(exc).__name__}: {exc}"
            friendly = _friendly_failure_toast(message)
            self.ledger.update_agent(
                task_id,
                f"{task_id}:main",
                label="MAIN",
                status="failed",
                output=friendly,
            )
            self.ledger.set_report(task_id, "MAIN", friendly)
            task = next(
                (item for item in self.ledger.snapshots() if item.id == task_id),
                None,
            )
            open_kids = [
                a
                for a in ([] if task is None else task.agents[1:])
                if a.status in {"pending", "running"}
            ]
            if open_kids:
                self.ledger.set_task_status(task_id, "running")
                self.ledger.set_task_phase(task_id, "parallel")
            else:
                self.ledger.finish_task(task_id, "failed")

            def apply_fail() -> None:
                self._set_feedback(friendly, "warning")

            self._call_in_ui_thread(apply_fail)
        finally:
            callback_state["active"] = False
            if legacy_runner_events:
                runner.on_live_message = old_live_message

            def apply_finish() -> None:
                self._sync_runtime()
                if self._active_task_id == task_id:
                    self._task_started_at = None
                self._flush_pending_reload()
                # Never auto-focus task list after a turn ends — leave input alone.
                self._invalidate_safe()

            self._call_in_ui_thread(apply_finish)

    def _before_render(self) -> None:
        """prompt_toolkit before_render hook — keep off the hot path when idle."""
        # Single clock sample for this paint. Animations (spinners, ==>) read it;
        # they must never schedule their own invalidate / full redraw storms.
        self._paint_clock = time.monotonic()
        # Fixed 30fps (GrokBuild animation.fps default). No dynamic downclock.
        try:
            self.app.refresh_interval = 1.0 / 30.0
        except Exception:  # noqa: BLE001
            pass
        # Goal mode is exclusive of plan approval chrome.
        self._clear_plan_ui_if_goal()
        # Interactive scrollbar + Window wheel only touch vertical_scroll;
        # re-anchor cursor lines so get_cursor_position cannot snap back.
        self._sync_detail_scroll_from_window()
        self._sync_task_scroll_from_window()
        self._sync_runtime()

    def _clear_plan_ui_if_goal(self) -> None:
        """If session entered Goal, drop plan approval chrome."""
        if not self._plan_ui:
            return
        kernel = getattr(self.session.extensions, "kernel", None)
        state = getattr(kernel, "session_mode_state", None) if kernel else None
        if state is None or not getattr(state, "is_goal", lambda: False)():
            return
        self._plan_ui = None
        self._plan_exit_waiting = False
        try:
            self._plan_exit_event.set()
        except Exception:  # noqa: BLE001
            pass

    def _sync_task_plan_with_session(self) -> None:
        """Mirror session Plan onto the active task card.

        - Session plan active → task plan_state at least ``planning``
        - Session awaiting approval → ``awaiting_approval``
        - Session left plan / task terminal → clear draft chrome so cards do not
          stick on「计划起草中」after the turn finished talking
        Does not stomp execution phases once past planning/plan_review.
        """
        tid = self._active_task_id
        if not tid:
            return
        task = next((t for t in self.ledger.snapshots() if t.id == tid), None)
        if task is None:
            return
        # Terminal tasks: drop draft/review chrome (finish_task also clears).
        if task.phase in {"done", "failed", "cancelled"}:
            if task.plan_state in {"planning", "consent", "awaiting_approval"}:
                self.ledger.set_plan_state(tid, "none")
            return
        kernel = getattr(self.session.extensions, "kernel", None)
        state = getattr(kernel, "session_mode_state", None) if kernel else None
        plan_file = ""
        if state is not None:
            plan_file = str(getattr(state, "plan_file", "") or "")
        if state is not None and getattr(state, "awaiting_plan_approval", False):
            if task.plan_state != "awaiting_approval":
                self.ledger.set_plan_state(
                    tid, "awaiting_approval", plan_file=plan_file or None
                )
            return
        if state is not None and (
            getattr(state, "is_plan", lambda: False)()
            or getattr(state, "is_plan_ui", lambda: False)()
        ):
            if task.plan_state in {"none", "", "consent"}:
                self.ledger.set_plan_state(
                    tid, "planning", plan_file=plan_file or None
                )
            return
        # Session no longer in plan UI: clear stale draft chrome if we already
        # moved into real work (or never needed plan chrome).
        if task.plan_state in {"planning", "consent"} and task.phase not in {
            "planning",
            "plan_review",
        }:
            self.ledger.set_plan_state(tid, "none")

    def _show_street_hud(self) -> bool:
        if self._modal_open or not self._showing_startup_brand():
            return False
        if _terminal_width() < 48 or _terminal_height() < 16:
            return False
        snap = session_surface.hud_projection(self.session)
        return not bool(snap.get("current_ok"))

    def _sync_runtime(self) -> None:
        now = time.monotonic()
        running = self._is_running()
        # Agent detail open: throttle hard — full subagent + plan sync every
        # paint was freezing the UI when memory tools flooded the transcript.
        if self._modal_open and self._modal_kind == "agent":
            if (now - self._last_sync_runtime_at) < 0.45:
                return
        elif not running and (now - self._last_sync_runtime_at) < 0.35:
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
            # Live child checklist chip (isolated TodoState).
            try:
                from codedoggy.tools.grok_build.todo_logic import count_todos

                coord = getattr(
                    getattr(self.session.extensions, "kernel", None),
                    "subagent_coordinator",
                    None,
                )
                badge = None
                if coord is not None and hasattr(coord, "todo_state_for"):
                    badge = count_todos(
                        coord.todo_state_for(snap.subagent_id)
                    ).badge_text()
                if not badge:
                    meta = getattr(snap, "metadata", None) or {}
                    badge = meta.get("todo_badge")
                if badge:
                    label = f"{label} {badge}"
            except Exception:  # noqa: BLE001
                pass
            output = subagent_text(snap)
            status = str(snap.status or "waiting")
            self.ledger.apply_agent_status(
                task_id,
                snap.subagent_id,
                label=label,
                status=status,
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
                    if line and status in {"pending", "running"}:
                        self.ledger.apply_agent_status(
                            task_id,
                            snap.subagent_id,
                            label=label,
                            status=status,
                            output=line,
                            description=description,
                        )

        # Keep task card plan_state aligned with session Plan mode / open todos.
        self._sync_task_plan_with_session()

        for task in self.ledger.snapshots():
            if task.phase in {"done", "failed", "cancelled"}:
                continue
            children = task.agents[1:]
            open_kids = [
                a for a in children if a.status in {"pending", "running"}
            ]
            if open_kids:
                self.ledger.set_task_phase(task.id, "parallel")
                continue
            if children:
                self.ledger.set_task_phase(task.id, "reporting")
                # MAIN already returned (status left running while waiting);
                # children now terminal — close the task.
                main = task.agents[0] if task.agents else None
                main_done = main is not None and main.status not in {
                    "pending",
                    "running",
                    "waiting",
                }
                if task.status == "running" and main_done:
                    failed = any(
                        a.status in {"failed", "cancelled"} for a in children
                    )
                    self.ledger.finish_task(
                        task.id, "failed" if failed else "completed"
                    )
            else:
                self.ledger.set_task_phase(task.id, "dispatching")

        if self._modal_open:
            self._detail_cursor_line = max(
                0,
                min(int(self._detail_cursor_line), max(0, int(self._detail_line_count) - 1)),
            )
        # Children may be the last thing keeping _is_running true.
        self._flush_pending_reload()

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
        if worker_running or getattr(self.session, "phase", None) is SessionPhase.TURN_RUNNING:
            return True
        # Background subagents outlive MAIN's worker — still "running" for UX.
        task_id = self._active_task_id
        if task_id:
            task = next(
                (item for item in self.ledger.snapshots() if item.id == task_id),
                None,
            )
            if task is not None:
                for agent in task.agents:
                    if agent.status in {"pending", "running"}:
                        return True
        for snap in self._subagents():
            if getattr(snap, "status", None) in {"pending", "running"}:
                return True
        return False

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
            tid = self._active_task_id
            self.ledger.set_task_status(tid, "cancelled")
            self.ledger.set_task_phase(tid, "cancelled")
            task = next(
                (item for item in self.ledger.snapshots() if item.id == tid),
                None,
            )
            for agent in list(getattr(task, "agents", None) or []):
                aid = getattr(agent, "id", None)
                if not aid:
                    continue
                st = str(getattr(agent, "status", "") or "").lower()
                if st in {"completed", "failed", "cancelled"}:
                    continue
                self.ledger.apply_agent_status(
                    tid,
                    str(aid),
                    label=str(getattr(agent, "label", "") or "AGENT"),
                    status="cancelled",
                )
        self._clear_feedback()
        self.app.invalidate()

    def _set_feedback(self, text: str, kind: str = "info") -> None:
        """Sticky one-line status above the prompt until the next clear.

        Call ``_clear_feedback`` on submit / new turn / cancel — no TTL toast.
        """
        self._feedback_text = (text or "").strip()
        self._feedback_kind = kind if kind in {"info", "success", "warning"} else "info"

    def _clear_feedback(self) -> None:
        self._feedback_text = ""
        self._feedback_kind = "info"

    def _feedback_active(self) -> bool:
        return bool(self._feedback_text)

    def _request_quit(self) -> None:
        now = time.monotonic()
        if self._quit_armed_until > now:
            self.app.exit()
            return
        self._quit_armed_until = now + 2.0
        self.app.invalidate()

    def _render_turn_status(self) -> StyleAndTextTuples:
        width = max(1, _terminal_width())
        if self._plan_ui == "consent":
            title = "任务"
            tid = self._plan_ui_task_id or self._active_task_id
            if tid:
                for t in self.ledger.snapshots():
                    if t.id == tid:
                        title = t.title
                        break
            text = f"  ╭ {title} · 代理想进入 Plan · Enter 同意 · Esc 拒绝 ╮"
            return [
                (
                    "class:feedback.warning",
                    _truncate_display(text, width),
                    self._plan_consent_mouse(True),
                )
            ]
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
        # Surface incomplete-work (open MAIN todos / running kids) in Chinese.
        open_hint = self._incomplete_work_status_hint()
        if open_hint:
            label = f"{label} · {open_hint}"
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
        stop_style = "class:turn.stop"
        if width <= fixed:
            compact = _truncate_display(f"{spinner} {stop}", width)
            return [(stop_style, compact, self._stop_mouse)]

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
            (stop_style, stop, self._stop_mouse),
            ("class:turn.elapsed", trailing),
        ]

    def _focus_main_input_mouse(self) -> Callable[[MouseEvent], object]:
        """Click prompt chrome (or near the box) → focus the main TextArea.

        TextArea itself focuses on direct hits; top/side/bottom frame fragments
        used to swallow clicks without moving focus, which felt broken.
        """

        def handler(event: MouseEvent) -> object:
            if event.event_type not in {
                MouseEventType.MOUSE_DOWN,
                MouseEventType.MOUSE_UP,
            }:
                return NotImplemented
            if self._modal_open or self._ask_active:
                return None
            try:
                self.app.layout.focus(self._input)
            except Exception:  # noqa: BLE001
                pass
            self.app.invalidate()
            return None

        return handler

    def _with_input_focus_mouse(
        self, fragments: StyleAndTextTuples
    ) -> StyleAndTextTuples:
        h = self._focus_main_input_mouse()
        out: StyleAndTextTuples = []
        for item in fragments:
            if len(item) >= 3 and item[2] is not None:
                out.append(item)
            else:
                style = item[0] if item else ""
                text = item[1] if len(item) > 1 else ""
                out.append((style, text, h))
        return out

    def _render_prompt_prefix(self) -> StyleAndTextTuples:
        border = self._prompt_border_class()
        # Rounded well + simple caret (no dog motif — less visual noise).
        return self._with_input_focus_mouse(
            [(border, "  │ "), ("class:prompt", "› ")]
        )

    def _render_prompt_top(self) -> StyleAndTextTuples:
        width = max(16, _terminal_width())
        border = self._prompt_border_class()
        rail_width = width - 4
        # Title plate in the top rail when wide enough.
        plate = " 交代任务 "
        plate_w = get_cwidth(plate)
        if border != "class:prompt.border.focus" or rail_width < 8:
            if rail_width >= plate_w + 4:
                left = max(1, (rail_width - plate_w) // 2)
                right = max(1, rail_width - plate_w - left)
                fr = [
                    (border, "  ╭" + "─" * left),
                    ("class:prompt.caption", plate),
                    (border, "─" * right + "╮"),
                ]
            else:
                fr = [(border, "  ╭" + "─" * rail_width + "╮")]
            return self._with_input_focus_mouse(fr)

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
        fragments.append((border, "╮"))
        return self._with_input_focus_mouse(fragments)

    def _render_prompt_right(self) -> StyleAndTextTuples:
        # One │ per visual input row so the frame grows with wrap height.
        # Must use the same border class as top/bottom — never a hard-coded cyan.
        border = self._prompt_border_class()
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
            fragments.append((border, "│  "))
            if i + 1 < rows:
                fragments.append(("", "\n"))
        return self._with_input_focus_mouse(fragments)

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
        caption = f" {caption_text} "
        fill = max(1, width - 4 - get_cwidth(caption))
        border = self._prompt_border_class()
        rail = (
            "class:prompt.border.dim"
            if border == "class:prompt.border.focus"
            else border
        )
        return self._with_input_focus_mouse(
            [
                (border, "  ╰"),
                (rail, "─" * fill),
                ("class:prompt.caption", caption),
                (border, "╯"),
            ]
        )

    def _prompt_border_class(self) -> str:
        """Prompt chrome follows focus only — never flash success/warning colors.

        Feedback text still shows in the turn-status line; recoloring the whole
        input border green on task-done was leftover chrome noise.
        """
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
                ("Tab", "返回", "auth_back", False),
                ("Ctrl+Q", "退出", "quit", True),
            ]
        elif self._modal_open:
            items = [
                ("↑↓", "滚动", "noop", False),
                ("Tab", "返回", "tasks", False),
            ]
            # Esc cancels the running task (not close) — show when busy.
            if self._is_running():
                items.insert(0, ("Esc", "取消任务", "cancel", False))
            items.append(("Ctrl+Q", "退出", "quit", True))
        else:
            try:
                # Prefer self.app — get_app() may be another Application offline.
                input_focused = self.app.layout.has_focus(self._input)
            except Exception:  # noqa: BLE001
                input_focused = True
            if input_focused:
                items = [
                    ("Tab", "最新任务", "tasks", False),
                    ("S-Tab", "Plan/Auto", "plan_mode", False),
                    ("^Enter", "换行", "noop", False),
                    ("Ctrl+L", "登录", "login", False),
                ]
                if self._is_running():
                    items.insert(0, ("Esc", "取消任务", "cancel", False))
                items.append(("Ctrl+Q", "退出", "quit", True))
            else:
                # Task list: Space → input; Tab → enter; Esc → cancel running task.
                items = [
                    ("Space", "输入", "input", False),
                    ("Tab", "进入", "open", False),
                    ("S-Tab", "Plan/Auto", "plan_mode", False),
                ]
                if self._is_running():
                    items.insert(0, ("Esc", "取消任务", "cancel", False))
                items.append(("Ctrl+L", "登录", "login", False))
                items.append(("Ctrl+Q", "退出", "quit", True))
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

        # Prefer primary actions first; always keep pinned quit when possible.
        priority = {
            "Enter",
            "Esc",
            "^Enter",
            "↑↓",
            "Tab",
            "S-Tab",
            "Ctrl+L",
            "←→",
            "Space",
        }
        regular_sorted = sorted(
            regular,
            key=lambda it: (0 if it[0] in priority else 1, regular.index(it)),
        )
        chosen: list[tuple[str, str, str, bool]] = []
        used = 2
        reserved = item_width(pinned) + (5 if pinned else 0) if pinned else 0
        for item in regular_sorted:
            extra = item_width(item) + (5 if chosen else 0)
            if used + extra + reserved > width:
                continue  # skip middle items, keep trying later high-priority already sorted
            chosen.append(item)
            used += extra
        # Restore original relative order among chosen
        order = {id(it): i for i, it in enumerate(items)}
        chosen.sort(key=lambda it: order.get(id(it), 99))
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
            elif action == "auth_back":
                if self._modal_open and self._modal_kind == "auth":
                    self._dispatch_wizard_action(self._auth_wizard.go_back())
            elif action == "login":
                self._open_auth_wizard()
            elif action == "plan_mode":
                self._toggle_session_plan_mode()
            elif action == "tasks":
                self._tab_task_cycle()
            elif action == "next":
                self._move_task(1)
                self.app.layout.focus(self._task_window)
            elif action == "previous":
                self._move_task(-1)
                self.app.layout.focus(self._task_window)
            elif action == "open":
                self._tab_task_cycle()
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
        # Quiet brand plate — no ∪ ears / dog face.
        brand_inner = " DOGGY "
        left = "  ╭" + brand_inner + "╮"
        right = session_surface.budget_text(self.session)
        badge = self._todo_badge_label()
        if width < get_cwidth(left):
            return [("class:brand", _truncate_display(" DOGGY ", width))]

        fragments: StyleAndTextTuples = [
            ("class:header", "  "),
            ("class:brand.edge.pink", "╭"),
            ("class:brand", brand_inner),
            ("class:brand.edge.pink", "╮"),
        ]
        badge_handler = self._todo_badge_mouse() if badge else None
        badge_style = (
            "class:todo.badge.open" if self._todo_pane_open else "class:todo.badge"
        )
        # Layout: brand · [计划 n/m ✓] · gap · budget
        used = get_cwidth(left)
        mid = ""
        if badge:
            mid = f"  计划 {badge} ✓" if width >= 40 else f"  {badge} ✓"
            if used + get_cwidth(mid) + 2 <= width:
                fragments.append((badge_style, mid, badge_handler))
                used += get_cwidth(mid)
            else:
                mid = f"  {badge}"
                if used + get_cwidth(mid) + 2 <= width:
                    fragments.append((badge_style, mid, badge_handler))
                    used += get_cwidth(mid)
                else:
                    mid = ""
        if not right or used + get_cwidth(right) + 2 > width:
            return fragments
        gap = width - used - get_cwidth(right) - 1
        fragments.append(("class:meta", " " * max(1, gap) + right + " "))
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
        cur_reason = str(snap.get("reasoning") or "")
        cur_ok = bool(snap.get("current_ok"))
        status_word = "AUTH ON" if cur_ok else ("AUTH OFF" if pulse else "LOGIN ›")
        status_style = ok_style if cur_ok else warn_style
        now_label = f"{cur}/{cur_model}" if cur_model else cur
        if cur_reason:
            now_label = f"{now_label} {cur_reason}"

        fragments: StyleAndTextTuples = []
        # row 0 border title — rounded plate + dog face
        top = _rounded_title(f"{_DOG_FACE} STREET AUTH", width)
        fragments.extend(line(title_style, top[:width]))
        fragments.append((bg, "\n", open_handler))

        # row 1 status — provider/model/reasoning from connection truth
        mid1 = f"│ {status_word:<10}  NOW {now_label[:22]:<22}"
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
        # Quiet single-tone rule — no dual pink/cyan end-caps, no neon scan.
        return [("class:header.rule.dim", "─" * width)]

    # ── Grok-style todo plan badge + expandable pane ─────────────────

    def _session_todo_state(self) -> Any | None:
        """MAIN session TodoState (header badge / default plan tab)."""
        kernel = getattr(self.session.extensions, "kernel", None)
        if kernel is None:
            return None
        st = getattr(kernel, "todo_state", None)
        if st is not None:
            return st
        bag = getattr(kernel, "tool_extra", None)
        if isinstance(bag, dict):
            return bag.get("todo_state")
        return None

    def _todo_state_for_open_agent(self) -> Any | None:
        """Todo for open modal agent: MAIN → session; child → isolated list."""
        if self._modal_ref is None:
            return self._session_todo_state()
        _task_id, agent_id = self._modal_ref
        # MAIN agents are ``task_xxx:main``
        if str(agent_id).endswith(":main") or str(agent_id).upper() == "MAIN":
            return self._session_todo_state()
        # Child agent ids are coordinator subagent ids (not always task-prefixed).
        kernel = getattr(self.session.extensions, "kernel", None)
        coord = getattr(kernel, "subagent_coordinator", None) if kernel else None
        if coord is not None and hasattr(coord, "todo_state_for"):
            try:
                child = coord.todo_state_for(str(agent_id))
                if child is not None:
                    return child
            except Exception:  # noqa: BLE001
                pass
            # Fallback: terminal snapshot metadata
            try:
                snap = coord.lookup(str(agent_id)) if hasattr(coord, "lookup") else None
                if snap is not None:
                    meta = getattr(snap, "metadata", None) or {}
                    todos = meta.get("todos")
                    if isinstance(todos, list) and todos:
                        from codedoggy.tools.grok_build.todo_logic import (
                            TodoItem,
                            TodoState,
                        )

                        st = TodoState()
                        for row in todos:
                            if not isinstance(row, dict):
                                continue
                            tid = str(row.get("id") or "")
                            if not tid:
                                continue
                            st.push(
                                tid,
                                TodoItem(
                                    content=str(row.get("content") or tid),
                                    status=str(row.get("status") or "pending"),
                                ),
                            )
                        return st
            except Exception:  # noqa: BLE001
                pass
        return None

    def _todo_counts(self) -> Any:
        from codedoggy.tools.grok_build.todo_logic import count_todos

        return count_todos(self._session_todo_state())

    def _todo_badge_label(self) -> str | None:
        counts = self._todo_counts()
        return counts.badge_text()

    def _main_todo_chip(self) -> str | None:
        """Short ``n/m`` for task cards (MAIN session todos only)."""
        return self._todo_badge_label()

    def _incomplete_work_status_hint(self) -> str | None:
        """Compact Chinese hint for open todos / running children (MAIN)."""
        from codedoggy.orchestration.incomplete_work import (
            open_todo_ids,
            running_subagent_ids,
        )

        kernel = getattr(self.session.extensions, "kernel", None)
        bag: dict[str, Any] = {}
        if kernel is not None:
            bag["kernel"] = kernel
            bag["session_id"] = str(getattr(kernel, "session_id", "") or "")
            bag["subagent_coordinator"] = getattr(
                kernel, "subagent_coordinator", None
            )
            bag["todo_state"] = getattr(kernel, "todo_state", None)
            bag["task_manager"] = getattr(kernel, "task_manager", None)
        parts: list[str] = []
        todos = open_todo_ids(bag.get("todo_state") or getattr(kernel, "todo_state", None))
        if todos:
            parts.append(f"待办{len(todos)}")
        try:
            kids = running_subagent_ids(bag)
        except Exception:  # noqa: BLE001
            kids = []
        if kids:
            parts.append(f"{len(kids)}子任务")
        if not parts:
            return None
        return "未完·" + "·".join(parts)

    def _agent_todo_chip(self, agent_id: str) -> str | None:
        """Live or terminal todo badge for one agent on the task card."""
        from codedoggy.tools.grok_build.todo_logic import count_todos

        if str(agent_id).endswith(":main"):
            return self._main_todo_chip()
        kernel = getattr(self.session.extensions, "kernel", None)
        coord = getattr(kernel, "subagent_coordinator", None) if kernel else None
        if coord is not None and hasattr(coord, "todo_state_for"):
            try:
                st = coord.todo_state_for(str(agent_id))
                badge = count_todos(st).badge_text()
                if badge:
                    return badge
            except Exception:  # noqa: BLE001
                pass
            try:
                snap = coord.lookup(str(agent_id)) if hasattr(coord, "lookup") else None
                meta = getattr(snap, "metadata", None) or {}
                b = meta.get("todo_badge")
                if b:
                    return str(b)
            except Exception:  # noqa: BLE001
                pass
        return None

    _TODO_PANE_VISIBLE = 8  # body rows (not counting chrome)

    def _todo_pane_height(self) -> Dimension:
        if not self._todo_pane_open:
            return Dimension.exact(0)
        state = self._session_todo_state()
        if state is None or getattr(state, "is_empty", lambda: True)():
            return Dimension.exact(0)
        n = sum(1 for _ in state.todo_items_with_ids())
        # title + visible body + bottom (+ optional scroll hint row)
        body = min(self._TODO_PANE_VISIBLE, max(1, n))
        h = body + 2
        if n > self._TODO_PANE_VISIBLE:
            h += 1  # scroll hint line
        return Dimension(min=h, max=h, preferred=h)

    def _toggle_todo_pane(self) -> None:
        if self._todo_badge_label() is None:
            self._todo_pane_open = False
            self._set_feedback("暂无计划任务（todo）", "warning")
            return
        self._todo_pane_open = not self._todo_pane_open
        if self._todo_pane_open:
            self._todo_scroll = 0
            # Do not steal focus from the prompt when opening the checklist.
        self.app.invalidate()

    def _focus_active_or_latest_task(self) -> None:
        """Select active running task, else latest (explicit UI actions only)."""
        tasks = self.ledger.snapshots()
        if not tasks:
            return
        idx = len(tasks) - 1
        active_id = self._active_task_id
        if active_id:
            for i, t in enumerate(tasks):
                if t.id == active_id:
                    idx = i
                    break
        self._selected_task = idx
        self._task_selection_active = True
        self._follow_latest_task = idx == len(tasks) - 1
        self.app.invalidate()

    def _scroll_todo_pane(self, delta: int) -> None:
        state = self._session_todo_state()
        if state is None:
            return
        n = sum(1 for _ in state.todo_items_with_ids())
        max_scroll = max(0, n - self._TODO_PANE_VISIBLE)
        self._todo_scroll = max(0, min(max_scroll, self._todo_scroll + int(delta)))
        self.app.invalidate()

    def _todo_badge_mouse(self) -> Callable[[MouseEvent], object]:
        def _on_up(event: MouseEvent) -> None:
            from prompt_toolkit.mouse_events import MouseModifier

            mods = getattr(event, "modifiers", None) or ()
            now = time.monotonic()
            # Ctrl+left or double-click → open MAIN 计划 tab.
            if MouseModifier.CONTROL in mods:
                self._todo_badge_last_click = None
                self._open_active_main_plan_tab()
                return
            last = self._todo_badge_last_click
            if last is not None and (now - last) <= _DOUBLE_CLICK_S:
                self._todo_badge_last_click = None
                self._open_active_main_plan_tab()
                return
            self._todo_badge_last_click = now
            self._toggle_todo_pane()

        # Wheel over badge scrolls list when open; click toggles / double opens.
        return self._only_mouse_up(_on_up, scroll_target="todo")

    def _open_active_main_plan_tab(self) -> None:
        """Focus active/latest task and open MAIN detail on 计划 filter."""
        self._focus_active_or_latest_task()
        task = self._selected_task_view()
        if task is None:
            self._set_feedback("暂无任务可打开", "warning")
            return
        main_id = f"{task.id}:main"
        # Prefer real MAIN agent id if present.
        for a in task.agents:
            if str(a.id).endswith(":main") or str(a.label).upper() == "MAIN":
                main_id = a.id
                break
        self._todo_pane_open = True
        self._todo_scroll = 0
        self._open_agent(task.id, main_id)
        self._detail_filter = "plan"
        self._clear_feedback()
        self.app.invalidate()

    def _todo_pane_mouse(self) -> Callable[[MouseEvent], object]:
        """Wheel scrolls the checklist; click no-ops (use badge to close)."""
        return self._only_mouse_up(lambda _e: None, scroll_target="todo")

    def _render_todo_pane(self) -> StyleAndTextTuples:
        """Expandable plan checklist under turn status (click badge to open)."""
        if not self._todo_pane_open:
            return []
        state = self._session_todo_state()
        width = max(12, _terminal_width())
        if state is None or getattr(state, "is_empty", lambda: True)():
            self._todo_pane_open = False
            return []

        wheel = self._todo_pane_mouse()
        counts = self._todo_counts()
        badge = counts.badge_text() or "0/0"
        title = f"  计划 {badge}"
        close_hint = " ↑↓滚轮 · 单击关闭 · 双击/Ctrl 开计划 "
        # Top border with title
        inner = max(8, width - 4)
        title_disp = _truncate_display(title, max(4, inner - get_cwidth(close_hint) - 2))
        pad = max(0, inner - get_cwidth(title_disp) - get_cwidth(close_hint))
        top_mid = title_disp + " " * pad + close_hint
        fragments: StyleAndTextTuples = [
            ("class:todo.pane.border", "  ╭", wheel),
            ("class:todo.pane.title", top_mid[:inner], wheel),
            ("class:todo.pane.border", "╮\n", wheel),
        ]

        status_icon = {
            "pending": ("○", "class:todo.item.pending"),
            "in_progress": ("▶", "class:todo.item.progress"),
            "completed": ("✓", "class:todo.item.done"),
            "cancelled": ("×", "class:todo.item.cancelled"),
        }
        items = list(state.todo_items_with_ids())
        max_scroll = max(0, len(items) - self._TODO_PANE_VISIBLE)
        self._todo_scroll = max(0, min(max_scroll, self._todo_scroll))
        shown = items[self._todo_scroll : self._todo_scroll + self._TODO_PANE_VISIBLE]
        for _tid, item in shown:
            st = (item.status or "pending").lower()
            icon, style = status_icon.get(st, status_icon["pending"])
            body = f" {icon} {item.content or _tid}"
            line = _truncate_display(body, max(4, width - 6))
            pad_r = max(0, width - 4 - get_cwidth(line))
            fragments.append(("class:todo.pane.border", "  │", wheel))
            fragments.append((style, line + " " * pad_r, wheel))
            fragments.append(("class:todo.pane.border", "│\n", wheel))
        if max_scroll > 0:
            lo = self._todo_scroll + 1
            hi = self._todo_scroll + len(shown)
            more = f" …{lo}-{hi}/{len(items)} · ↑↓/滚轮"
            more = _truncate_display(more, max(4, width - 6))
            pad_r = max(0, width - 4 - get_cwidth(more))
            fragments.append(("class:todo.pane.border", "  │", wheel))
            fragments.append(("class:todo.pane", more + " " * pad_r, wheel))
            fragments.append(("class:todo.pane.border", "│\n", wheel))

        fragments.append(
            ("class:todo.pane.border", "  ╰" + "─" * inner + "╯", wheel)
        )
        return fragments

    def _render_modal_border(self, *, top: bool) -> StyleAndTextTuples:
        width = max(4, _terminal_width() - 4)
        rail_width = width - 2
        # Matched gray rails (no left-pink / right-cyan clash).
        return [
            ("class:modal.border.left", "╭" if top else "╰"),
            ("class:modal.border.dim", "─" * rail_width),
            ("class:modal.border.right", "╮" if top else "╯"),
        ]

    def _render_tasks(self) -> StyleAndTextTuples:
        tasks = self.ledger.snapshots()
        fragments: StyleAndTextTuples = []
        line = 0
        width = max(1, _terminal_width() - 2)

        # Modal float covers this pane — skip expensive splash/task paint while open
        # so ESC close is not stuck behind a full truecolor doggy recompute.
        if self._modal_open:
            if self._showing_startup_brand() or not tasks:
                self._task_refs = []
                empty: StyleAndTextTuples = [("", "\n")]
                # Underlay line count only — do not clamp free-scroll _selected_line.
                self._task_line_count = self._count_fragment_lines(empty)
                return empty
            # Agent detail on top: still keep task_refs in sync for selection keys,
            # but avoid re-walking every card under the float each frame.
            self._task_refs = [task.id for task in tasks]
            # Modal owns focus — do not advance selection via follow-latest.
            if tasks and self._task_selection_active and self._selected_task >= 0:
                self._selected_task = max(
                    0, min(int(self._selected_task), len(tasks) - 1)
                )
            elif not self._task_selection_active:
                self._selected_task = -1
            empty = [("", "\n")]
            self._task_line_count = self._count_fragment_lines(empty)
            return empty

        # Launch splash only — first task dismisses it for the whole session.
        if self._showing_startup_brand():
            fr = self._render_doggy_empty_cached(width)
            self._set_task_line_count(fr)
            return fr

        if not tasks:
            self._task_refs = []
            self._selected_task = -1
            self._task_selection_active = False
            self._task_paint_cache = None
            fr = _render_doggy_idle_panel(width)
            self._set_task_line_count(fr)
            return fr

        # Cheap path: reuse last card walk when ledger/selection/width are unchanged.
        cache_key = self._task_paint_cache_key(tasks, width)
        cached = self._task_paint_cache
        if cached is not None and cached[0] == cache_key:
            (
                _key,
                fr,
                task_refs,
                selected_line,
                line_count,
            ) = cached
            self._task_refs = task_refs
            self._selected_line = selected_line
            self._task_line_count = line_count
            return fr

        self._task_refs = [task.id for task in tasks]
        # Follow = scroll-to-bottom only. NEVER invent a selection here.
        # Selection only from click / keyboard / explicit focus helpers.
        if self._task_selection_active:
            if self._selected_task < 0:
                self._task_selection_active = False
            else:
                self._selected_task = max(
                    0, min(int(self._selected_task), len(tasks) - 1)
                )
        else:
            self._selected_task = -1
        # Selection chrome only when list is focused (not hover, not follow alone).
        list_focused = self._task_list_has_focus()
        # Keep a constant card geometry whenever width allows.
        has_frame = width >= 20
        selected_line_start = 0

        for task_index, task in enumerate(tasks):
            has_sel = self._task_selection_active and self._selected_task >= 0
            selected = (
                list_focused
                and has_sel
                and task_index == self._selected_task
            )
            # Anchor only when a real selection exists (user pick / focus helper).
            is_cursor_task = has_sel and task_index == self._selected_task
            # Focused card frame = same recipe as the main input prompt focus.
            framed = selected and has_frame
            inner_width = max(1, width - 2) if has_frame else width
            # prompt.border.focus when selected; quiet spine otherwise.
            side_style = (
                "class:prompt.border.focus"
                if framed
                else "class:task.spine"
            )
            card_start = line
            if is_cursor_task:
                selected_line_start = card_start
            card_mouse = self._task_card_mouse(task_index)
            if has_frame:
                # Whole card is the hit target (frame + body + padding).
                fragments.extend(
                    self._task_card_h_rail(
                        top=True,
                        inner_width=inner_width,
                        framed=framed,
                        mouse=card_mouse,
                    )
                )
                line += 1

            def append_task_line(parts: StyleAndTextTuples) -> None:
                """Paint one card row — entire row (incl. pad/rails) is clickable.

                Nested handlers (e.g. agent row) keep their own mouse; everything
                else uses the card handler (select / Ctrl+left open).
                """
                nonlocal line
                if has_frame:
                    fragments.append((side_style, "│", card_mouse))
                used = 0
                for part in parts:
                    style = part[0]
                    text = part[1]
                    used += get_cwidth(text)
                    if len(part) >= 3 and part[2] is not None:
                        fragments.append(part)
                    else:
                        fragments.append((style, text, card_mouse))
                if used < inner_width:
                    fragments.append(
                        ("", " " * (inner_width - used), card_mouse)
                    )
                if has_frame:
                    fragments.append((side_style, "│", card_mouse))
                fragments.append(("", "\n", card_mouse))
                line += 1

            # Compact chat-list density: title + status + wrapped summary.
            active = task.phase in {
                "dispatching",
                "parallel",
                "reporting",
                "planning",
                "plan_review",
            } or task.status == "running"
            is_latest = task_index == len(tasks) - 1
            spine_style = "class:task.spine.active" if active else "class:task.spine"
            prefix = "  "
            flash = self._interject_preview(task.id)
            status = (
                _compact_task_stage_text(task, interject=flash)
                if width < 34
                else _task_stage_text(task, interject=flash)
            )
            # Total task duration on the homepage card (live while running).
            elapsed_label = _task_elapsed_label(task)
            if elapsed_label:
                status = f"{status} · {elapsed_label}"
            if selected:
                marker = "›"
            elif flash:
                marker = "↩"
            elif is_latest and active:
                marker = "●"
            elif active:
                marker = "•"
            else:
                marker = "·"
            if selected:
                marker_style = "class:task.marker.selected"
            elif flash:
                marker_style = "class:task.marker.interject"
            elif active:
                marker_style = "class:task.marker.active"
            else:
                marker_style = "class:task.marker.idle"
            gutter = f"{marker} "
            gutter_w = get_cwidth(gutter)
            text_cols = max(1, inner_width - get_cwidth(prefix) - gutter_w)
            # Show full title by wrapping — do not hard-crop with "…" when it fits.
            title_lines = _wrap_display_lines(task.title, text_cols, max_lines=20)
            if not title_lines:
                title_lines = [""]
            status_w = get_cwidth(status)
            first = title_lines[0]
            status_style = (
                "class:task.interject"
                if flash
                else _task_status_style(task)
            )
            if (
                len(title_lines) == 1
                and status_w + 1 < text_cols
                and get_cwidth(first) + 1 + status_w <= text_cols
            ):
                gap = max(1, text_cols - get_cwidth(first) - status_w)
                append_task_line(
                    [
                        (spine_style, prefix),
                        (marker_style, gutter),
                        ("class:task.title", first),
                        (status_style, " " * gap + status),
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
                for cont in title_lines[1:]:
                    append_task_line(
                        [
                            (spine_style, prefix),
                            ("", " " * gutter_w),
                            ("class:task.title", cont),
                        ]
                    )
                # Status on its own row when title is multi-line or too wide.
                append_task_line(
                    [
                        (spine_style, prefix),
                        ("", " " * gutter_w),
                        (status_style, status),
                    ]
                )

            # Full summary via wrap — human prose only, never tool live names.
            summary = _task_list_summary(task, interject=flash)
            if summary:
                sum_cols = max(1, inner_width - get_cwidth(prefix) - gutter_w)
                for sum_line in _wrap_display_lines(summary, sum_cols, max_lines=12):
                    append_task_line(
                        [
                            (spine_style, prefix),
                            ("", " " * gutter_w),
                            ("class:report", sum_line),
                        ]
                    )

            # No ↳ MAIN / agent rows on the list — extra lines jiggle when
            # arrowing through cards. Open detail via Tab / Ctrl+click.

            if has_frame:
                fragments.extend(
                    self._task_card_h_rail(
                        top=False,
                        inner_width=inner_width,
                        framed=framed,
                        mouse=card_mouse,
                    )
                )
                line += 1
            # No blank spacer between cards — frames already separate them.

        # Pad remaining viewport with void hit-targets — click clears selection.
        void = self._task_void_mouse()
        try:
            win_h = int(getattr(self._task_window.render_info, "window_height", 0) or 0)
        except Exception:  # noqa: BLE001
            win_h = 0
        pad_lines = max(12, (win_h - line + 4) if win_h else 12)
        for _ in range(pad_lines):
            fragments.append(("", "\n", void))
            line += 1

        # Re-pin cursor only when a real selection exists — never jump list to end.
        if self._task_selection_active and self._selected_task >= 0:
            pin_task = int(self._selected_task)
            if self._pinned_task_for_line != pin_task:
                self._selected_line = selected_line_start
                self._pinned_task_for_line = pin_task
        else:
            self._pinned_task_for_line = None
        self._set_task_line_count(fragments)
        # Recompute key after pin so selected_line matches what we store.
        store_key = self._task_paint_cache_key(tasks, width)
        self._task_paint_cache = (
            store_key,
            fragments,
            list(self._task_refs),
            int(self._selected_line),
            int(self._task_line_count),
        )
        return fragments

    def _task_paint_cache_key(self, tasks: list[TaskView], width: int) -> tuple[Any, ...]:
        """Stable identity for task-list paint; excludes free-scroll noise."""
        rows: list[tuple[Any, ...]] = []
        for task in tasks:
            agents: list[tuple[Any, ...]] = []
            for agent in task.agents:
                live = ""
                if agent.status in {"pending", "running"}:
                    live = self._activity.line(task.id, agent.id)
                agents.append(
                    (
                        agent.id,
                        agent.label,
                        agent.status,
                        agent.output,
                        live,
                    )
                )
            flash = self._interject_preview(task.id)
            # Bucket elapsed by whole seconds so running cards refresh the timer.
            el = _task_elapsed_seconds(task)
            el_bucket = int(el) if el is not None else -1
            rows.append(
                (
                    task.id,
                    task.title,
                    task.phase,
                    task.status,
                    task.plan_state,
                    task.report,
                    task.reporter,
                    flash or "",
                    self._selected_agent_by_task.get(task.id, 0),
                    el_bucket,
                    tuple(agents),
                )
            )
        # Focused card top scan must tick; otherwise paint cache freezes it.
        scan_frame = 0
        if (
            self._task_selection_active
            and self._selected_task >= 0
            and self._task_list_has_focus()
        ):
            scan_frame = int(self._paint_clock * 14) % 16
        return (
            width,
            int(self._selected_task),
            bool(self._task_selection_active),
            bool(self._follow_latest_task),
            self._pinned_task_for_line,
            int(self._selected_line),
            self._task_list_has_focus(),
            scan_frame,
            tuple(rows),
        )

    def _task_card_h_rail(
        self,
        *,
        top: bool,
        inner_width: int,
        framed: bool,
        mouse: Callable[[MouseEvent], object],
    ) -> StyleAndTextTuples:
        """Card top/bottom edge — mirror ``_render_prompt_top`` / bottom exactly.

        Focused:
        - top: dim rail + 3-cell pink scan (prompt focus animation)
        - bottom: pink corners + dim fill (no scan)
        - corners and sides use ``prompt.border.focus``
        Unfocused: quiet ``task.spine`` box.
        """
        corner_l = "╭" if top else "╰"
        corner_r = "╮" if top else "╯"
        w = max(1, int(inner_width))
        if not framed:
            return [
                (
                    "class:task.spine",
                    f"{corner_l}{'─' * w}{corner_r}\n",
                    mouse,
                ),
            ]
        # Same style classes as the main input chrome.
        border = "class:prompt.border.focus"
        dim = "class:prompt.border.dim"
        if top:
            # Identical scan to _render_prompt_top (focused branch).
            scan = int(time.monotonic() * 14) % w
            styles = [dim] * w
            styles[0] = border
            for offset in range(3):
                styles[(scan + offset) % w] = border
            out: StyleAndTextTuples = [(border, corner_l, mouse)]
            for style, cells in groupby(styles):
                out.append((style, "─" * sum(1 for _ in cells), mouse))
            out.append((border, corner_r + "\n", mouse))
            return out
        # Bottom: same as _render_prompt_bottom — pink corners, dim mid rail.
        return [
            (border, corner_l, mouse),
            (dim, "─" * w, mouse),
            (border, corner_r + "\n", mouse),
        ]

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
        if self._modal_kind == "ask":
            n = max(1, len(self._ask_questions))
            i = min(self._ask_q_index + 1, n)
            left = f"  ╭ {_DOG_EAR} 计划提问 {i}/{n} "
            right = "ASK ╮"
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
        if self._modal_kind == "ask":
            multi = " · 多选空格勾选" if self._ask_is_multi() else ""
            return [
                (
                    "class:detail.active",
                    _truncate_display(
                        f"  问卷{multi} · ↑↓ · Enter · Tab 退出",
                        max(12, _terminal_width() - 8),
                    )
                    + "\n",
                )
            ]
        width = max(12, _terminal_width() - 8)
        fragments: StyleAndTextTuples = [("", "  ")]
        used = 2
        filters = self._detail_filters_for_open_task()
        for detail_filter in filters:
            active = detail_filter == self._detail_filter
            base_label = DETAIL_FILTER_LABELS.get(detail_filter, detail_filter)
            # Rounded chips: ╭ label ╮ for active filter (no F-keys)
            label = f"╭ {base_label} ╮" if active else f" {base_label} "
            piece_width = get_cwidth(label) + (2 if used > 2 else 0)
            if used + piece_width > width:
                break
            if used > 2:
                fragments.append(("class:detail.meta", " "))
            style = "class:detail.active" if active else "class:detail.meta"
            fragments.append(
                (style, label, self._detail_filter_mouse(detail_filter))
            )
            used += piece_width
        return fragments

    def _detail_messages_signature(
        self, messages: list[Any]
    ) -> tuple[Any, ...]:
        """Cheap fingerprint so we skip full detail re-layout when unchanged."""
        n = len(messages)
        if n == 0:
            return (0, 0, "")
        last = messages[-1]
        content = str(getattr(last, "content", None) or "")
        role = getattr(last, "role", None)
        role_s = str(getattr(role, "value", role) or "")
        # Tail sample only — full hash of multi-MB tool dumps freezes paint.
        return (n, role_s, len(content), content[:64], content[-32:] if content else "")

    def _invalidate_detail_body_cache(self) -> None:
        self._detail_body_cache = None
        self._detail_body_cache_key = None

    def _render_modal_body(self) -> StyleAndTextTuples:
        if self._modal_kind == "auth":
            return self._render_auth_body()
        if self._modal_kind == "ask":
            return self._render_ask_body()
        if self._detail_filter == "plan":
            return self._render_plan_detail_body()
        snapshot = self._current_detail_snapshot()
        width = max(12, _terminal_width() - 8)
        # Cache key: avoid re-walking huge tool transcripts every paint.
        if self._modal_ref is not None:
            task_id, agent_id = self._modal_ref
            msgs = self._detail_messages.get((task_id, agent_id), [])
            cache_key: tuple[Any, ...] = (
                self._modal_ref,
                self._detail_filter,
                width,
                self._detail_messages_signature(list(msgs)),
                self._todo_badge_label(),
            )
            if (
                cache_key == self._detail_body_cache_key
                and self._detail_body_cache is not None
            ):
                return self._detail_body_cache
        else:
            cache_key = ()

        if snapshot is None:
            empty = [
                _rounded_title(f"{_DOG_FACE} empty", width),
                f"│ {_DOG_EAR} 当前 Agent 没有可用记录  │",
                "╰" + "─" * max(1, width - 2) + "╯",
            ]
            frags: StyleAndTextTuples = []
            for i, raw in enumerate(empty):
                text = _truncate_display(raw, width)
                style = "class:brand" if i == 0 else "class:detail.meta"
                frags.append((style, text + "\n"))
            frags = self._with_detail_scroll(self._ensure_fragments(frags))
            self._set_detail_line_count(frags, preferred_cursor=0)
            if cache_key:
                self._detail_body_cache_key = cache_key
                self._detail_body_cache = frags
            return frags
        width = max(12, _terminal_width() - 8)
        # Grok default: tools collapsed to one line; click header expands body.
        from codedoggy.tui.agent_detail import default_collapsed_keys

        defaults = set(default_collapsed_keys(snapshot.records))
        # First paint for this agent: seed collapsed set.
        if self._detail_collapse_seeded_for != self._modal_ref:
            self._detail_collapsed = set(defaults)
            self._detail_known_fold_keys = set(defaults)
            self._detail_collapse_seeded_for = self._modal_ref
        else:
            # Live tools: fold new keys only; keep user expands.
            for k in defaults:
                if k not in self._detail_known_fold_keys:
                    self._detail_collapsed.add(k)
                    self._detail_known_fold_keys.add(k)
        fragments = render_detail_body(
            snapshot,
            width,
            active_filter=self._detail_filter,
            path_mouse=self._image_path_mouse,
            collapsed_keys=self._detail_collapsed,
            fold_mouse=self._detail_fold_mouse,
        )
        fragments = self._with_detail_scroll(self._ensure_fragments(fragments))
        self._set_detail_line_count(fragments)
        if cache_key:
            self._detail_body_cache_key = cache_key
            self._detail_body_cache = fragments
        return fragments

    def _with_detail_scroll(
        self, fragments: StyleAndTextTuples
    ) -> StyleAndTextTuples:
        """Attach wheel→_scroll_detail on plain body text (not just links).

        Unhandled prose used Window-only scroll; ScrollOffsets then snapped
        the viewport back to a stale _detail_cursor_line on every invalidate.
        """
        # Reuse one handler — allocating per fragment per paint GC-storms.
        if self._detail_scroll_handler is None:
            self._detail_scroll_handler = self._only_mouse_up(
                lambda _e: None, scroll_target="detail"
            )
        handler = self._detail_scroll_handler
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
                ("class:auth.item.active", _rounded_title(f"{spin} {_DOG_FACE} auth", wait_w) + "\n")
            )
            fragments.append(
                ("class:auth.hint", f"│ {_DOG_EAR} 等待浏览器授权完成…\n")
            )
            fragments.append(
                ("class:auth.hint", "│ 完成后回到结果页 · Tab 取消\n")
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
            # Selection is always blue; 使用中 = yellow; logged-in = white.
            # Up/down only moves the blue marker.
            if selected:
                style = "class:auth.item.selected"
            elif item.style == "active":
                style = "class:auth.item.active"
            elif item.style in {"logged", "ok"}:
                style = "class:auth.item.logged"
            elif item.style in {"muted", "offline", "danger"}:
                style = "class:auth.item.muted"
            else:
                style = "class:auth.item"
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
        # Cap live transcript for paint — plan-mode memory floods freeze open.
        _DETAIL_MSG_CAP = 120
        if len(messages) > _DETAIL_MSG_CAP:
            messages = messages[-_DETAIL_MSG_CAP:]
        if not messages:
            is_main = agent_id == f"{task_id}:main"
            if not is_main and agent.status in {"pending", "running"}:
                fallback = (
                    f"进行中 · {agent.label or '子任务'} 仍在跑。"
                    "结束后会同步完整消息、思考与工具记录；可先看 MAIN 或稍后再进。"
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
            text = "  › 凭证  "
            return [("class:detail.input.prompt", text)]
        if self._modal_kind == "ask":
            text = "  › Other  "
            return [("class:detail.input.prompt", text)]
        # Mid-turn interject while a task is open — plain language, no "MAIN" jargon.
        terminal_width = max(1, _terminal_width())
        budget = max(4, terminal_width - 20)
        if terminal_width < 40:
            text = "  › "
        else:
            text = "  › 继续说  "
        text = _truncate_display(text, budget)
        return [("class:detail.input.prompt", text)]

    def _render_modal_hint(self) -> StyleAndTextTuples:
        if self._modal_kind == "auth":
            text = "  ╰ ↑↓ 选择 · Enter 确认 · Tab 返回 · Ctrl+L ╮"
        elif self._modal_kind == "ask":
            if self._ask_other_editing:
                text = "  ╰ 输入自定义答案 · Enter 提交 · Tab 退出 ╮"
            elif self._ask_is_multi():
                text = "  ╰ ↑↓ 移动 · Space 勾选 · Enter 完成 · Tab 退出 ╮"
            else:
                text = "  ╰ ↑↓ 选择 · Enter 确认 · Tab 退出 ╮"
        elif self._task_awaiting_plan_approval() and self._detail_filter == "plan":
            text = "  ╰ a 批准开工 · s 要修改 · q 放弃 · ↑↓ 滚动 · Tab 返回 ╮"
        elif self._detail_input_visible():
            text = "  ╰ ←→ 切换 · ↑↓ 滚动 · Tab 返回 ╮"
        else:
            text = "  ╰ ←→ 切换 · ↑↓ 滚动 · Tab 返回 ╮"
        line = _truncate_display(text, max(1, _terminal_width() - 6))
        if self._task_awaiting_plan_approval() and self._detail_filter == "plan":
            return [
                (
                    "class:agent-window.hint",
                    line,
                    self._plan_action_mouse("approved"),
                ),
            ]
        return [("class:agent-window.hint", line)]

    def _detail_filter_mouse(
        self, detail_filter: DetailFilter
    ) -> Callable[[MouseEvent], object]:
        return self._only_mouse_up(
            lambda _e: self._set_detail_filter(detail_filter),
            scroll_target="detail",
        )

    def _detail_fold_mouse(
        self, collapse_key: str
    ) -> Callable[[MouseEvent], object]:
        """Toggle collapsed tool/thinking body (Grok: click header to expand)."""

        def _on_up(_event: MouseEvent) -> None:
            self._toggle_detail_fold(collapse_key)

        return self._only_mouse_up(_on_up, scroll_target="detail")

    def _toggle_detail_fold(self, collapse_key: str) -> None:
        # Tool records use record_id:index — expand/collapse all sibling blocks.
        base = collapse_key.rsplit(":", 1)[0]
        siblings = [
            k
            for k in list(self._detail_collapsed)
            if k == collapse_key or k.rsplit(":", 1)[0] == base
        ]
        if collapse_key in self._detail_collapsed or siblings:
            # Expand: drop this record's collapsed keys.
            self._detail_collapsed = {
                k
                for k in self._detail_collapsed
                if k.rsplit(":", 1)[0] != base
            }
        else:
            # Collapse: re-fold sibling indices 0..7 for this record.
            for i in range(8):
                self._detail_collapsed.add(f"{base}:{i}")
        self._invalidate_detail_body_cache()
        self.app.invalidate()

    def _set_detail_filter(self, detail_filter: DetailFilter) -> None:
        if detail_filter not in DETAIL_FILTERS:
            detail_filter = "message"
        self._detail_filter = detail_filter
        self._detail_cursor_line = 0
        self._invalidate_detail_body_cache()
        try:
            self._detail_window.vertical_scroll = 0
        except Exception:  # noqa: BLE001
            pass
        self.app.layout.focus(self._detail_window)
        self.app.invalidate()

    def _cycle_detail_filter(self, delta: int) -> None:
        filters = self._detail_filters_for_open_task()
        try:
            index = filters.index(self._detail_filter)
        except ValueError:
            index = 0
        self._set_detail_filter(filters[(index + delta) % len(filters)])

    def _detail_filters_for_open_task(self) -> tuple[DetailFilter, ...]:
        show_plan = False
        if self._modal_ref is not None:
            task_id, _ = self._modal_ref
            for t in self.ledger.snapshots():
                if t.id == task_id and t.plan_state in {
                    "planning",
                    "awaiting_approval",
                    "approved",
                    "consent",
                }:
                    show_plan = True
                    break
        if show_plan:
            return ("message", "tool", "plan")
        return ("message", "tool")

    def _detail_input_visible(self) -> bool:
        """Show interject/auth paste only when there is something to type for."""
        if not self._modal_open:
            return False
        if self._modal_kind == "auth":
            return self._auth_wizard.step == WizardStep.PASTE
        if self._modal_kind == "ask":
            return False  # dedicated ask float; Other uses main input
        if self._modal_kind != "agent":
            return False
        # Plan review: keep the box for "s" revision notes.
        if self._task_awaiting_plan_approval():
            return True
        if self._modal_ref is None:
            return False
        task_id, agent_id = self._modal_ref
        agent = self.ledger.get_agent(task_id, agent_id)
        if agent is not None and agent.status in {"pending", "running", "waiting"}:
            return True
        for task in self.ledger.snapshots():
            if task.id != task_id:
                continue
            if task.status == "running" or task.phase in {
                "dispatching",
                "parallel",
                "reporting",
                "planning",
                "plan_review",
            }:
                return True
            return False
        return bool(self._is_running() and self._active_task_id == task_id)

    def _task_awaiting_plan_approval(self) -> bool:
        if self._modal_ref is None:
            return False
        task_id, _ = self._modal_ref
        for t in self.ledger.snapshots():
            if t.id == task_id:
                return t.plan_state == "awaiting_approval"
        return False

    def _wire_plan_mode_hooks(self) -> None:
        kernel = getattr(self.session.extensions, "kernel", None)
        if kernel is None:
            return
        bag = getattr(kernel, "tool_extra", None)
        if not isinstance(bag, dict):
            bag = {}
            kernel.tool_extra = bag
        bag["plan_mode_consent_fn"] = self._plan_mode_consent_fn
        bag["plan_mode_exit_fn"] = self._plan_mode_exit_fn
        bag["todo_changed_fn"] = self._on_todo_changed
        # Override CLI stdin ask (which paints *outside* the TUI) with modal UI.
        bag["ask_user_fn"] = self._ask_user_fn

    def _on_todo_changed(self) -> None:
        """Called from todo_write worker thread after list mutates."""
        self._call_in_ui_thread(self.app.invalidate)

    def _ask_user_fn(self, questions: list[dict[str, Any]]) -> dict[str, Any]:
        """Host hook for ask_user_question — park worker; answers via plan-card modal.

        Replaces stdin CLI (ask_user_cli) so options render *inside* the TUI
        task/plan modal with ↑↓ selection, not as a stray terminal popup.
        """
        if self._closing:
            return {"outcome": "cancelled"}
        if not isinstance(questions, list) or not questions:
            return {"outcome": "accepted", "answers": {}}

        self._ask_questions = [q for q in questions if isinstance(q, dict)]
        if not self._ask_questions:
            return {"outcome": "accepted", "answers": {}}
        self._ask_q_index = 0
        self._ask_opt_index = 0
        self._ask_answers = {}
        self._ask_result = None
        self._ask_multi_picked = set()
        self._ask_other_editing = False
        self._ask_active = True
        self._ask_event.clear()

        def open_ui() -> None:
            self._open_ask_modal()

        self._call_in_ui_thread(open_ui)
        signaled = self._ask_event.wait(timeout=600)
        self._ask_active = False
        if not signaled or self._ask_result is None:
            self._call_in_ui_thread(
                lambda: self._set_feedback("问卷超时 · 已取消", "warning")
            )
            self._call_in_ui_thread(self._close_ask_modal_only)
            return {"outcome": "cancelled"}
        result = dict(self._ask_result)
        self._ask_result = None
        self._call_in_ui_thread(self._close_ask_modal_only)
        return result

    def _open_ask_modal(self) -> None:
        """Show dedicated questionnaire float — not the plan/agent card."""
        # Do not hijack agent detail: close plan/agent shell so ask is a real popup.
        if self._modal_open and self._modal_kind in {"agent", "auth"}:
            # Keep ref only as context; close shell so z-order shows ask float alone.
            saved_ref = self._modal_ref
            self._modal_open = False
            self._modal_ref = saved_ref
        self._modal_kind = "ask"
        self._ask_other_editing = False
        self._detail_input.text = ""
        try:
            self.app.layout.focus(self._ask_window)
        except Exception:  # noqa: BLE001
            try:
                self.app.layout.focus(self._task_window)
            except Exception:  # noqa: BLE001
                pass
        self._clear_feedback()
        self.app.invalidate()

    def _session_in_plan_ui(self) -> bool:
        kernel = getattr(self.session.extensions, "kernel", None)
        state = getattr(kernel, "session_mode_state", None) if kernel else None
        if state is None:
            return False
        return bool(
            getattr(state, "is_plan", lambda: False)()
            or getattr(state, "is_plan_ui", lambda: False)()
            or getattr(state, "awaiting_plan_approval", False)
        )

    def _close_ask_modal_only(self) -> None:
        """Dismiss dedicated ask float after answer."""
        if self._modal_kind != "ask" and not self._ask_active:
            return
        self._modal_kind = "agent"
        self._ask_other_editing = False
        self._detail_input.text = ""
        self._modal_open = False
        try:
            self.app.layout.focus(self._task_window)
        except Exception:  # noqa: BLE001
            try:
                self.app.layout.focus(self._input)
            except Exception:  # noqa: BLE001
                pass
        self.app.invalidate()

    def _current_ask_question(self) -> dict[str, Any] | None:
        if not self._ask_questions:
            return None
        if not 0 <= self._ask_q_index < len(self._ask_questions):
            return None
        return self._ask_questions[self._ask_q_index]

    def _ask_option_count(self) -> int:
        """Options + 1 for Other."""
        q = self._current_ask_question()
        if q is None:
            return 1
        opts = q.get("options") if isinstance(q.get("options"), list) else []
        return max(1, len(opts) + 1)

    def _ask_is_multi(self) -> bool:
        q = self._current_ask_question()
        if q is None:
            return False
        return bool(q.get("multi_select") or q.get("multiSelect"))

    def _resolve_ask(self, result: dict[str, Any]) -> None:
        if not self._ask_active:
            return
        self._ask_result = result
        self._ask_event.set()

    def _ask_move_option(self, delta: int) -> None:
        n = self._ask_option_count()
        self._ask_opt_index = (self._ask_opt_index + delta) % n
        self._ask_other_editing = False
        self.app.invalidate()

    def _ask_confirm_current(self) -> None:
        """Enter on current option (or Other free-text)."""
        q = self._current_ask_question()
        if q is None:
            self._resolve_ask({"outcome": "cancelled"})
            return
        qtext = str(q.get("question") or "")
        opts = q.get("options") if isinstance(q.get("options"), list) else []
        other_i = len(opts)
        multi = self._ask_is_multi()

        if self._ask_opt_index == other_i or self._ask_other_editing:
            # Free-text Other via main input (ask is a dedicated float, not detail).
            text = (self._input.text or "").strip()
            if not self._ask_other_editing:
                self._ask_other_editing = True
                self._input.text = ""
                try:
                    self.app.layout.focus(self._input)
                except Exception:  # noqa: BLE001
                    pass
                self._set_feedback("底部输入自定义答案后 Enter", "info")
                self.app.invalidate()
                return
            if not text:
                self._set_feedback("自定义答案不能为空", "warning")
                return
            labels = [text]
            self._ask_other_editing = False
            self._input.text = ""
        elif multi:
            # Multi: Enter finalizes picked set (Space toggles).
            if not self._ask_multi_picked:
                # Treat Enter as toggle+finalize single if nothing picked.
                self._ask_multi_picked.add(self._ask_opt_index)
            labels = []
            for i in sorted(self._ask_multi_picked):
                if 0 <= i < len(opts) and isinstance(opts[i], dict):
                    labels.append(str(opts[i].get("label") or f"选项{i+1}"))
            if not labels:
                self._set_feedback("请先空格勾选选项", "warning")
                return
        else:
            if 0 <= self._ask_opt_index < len(opts) and isinstance(
                opts[self._ask_opt_index], dict
            ):
                labels = [
                    str(opts[self._ask_opt_index].get("label") or "选项")
                ]
            else:
                labels = []
            if not labels:
                self._set_feedback("无效选项", "warning")
                return

        self._ask_answers[qtext] = labels
        self._ask_multi_picked = set()
        self._ask_opt_index = 0
        # Next question or finish.
        if self._ask_q_index + 1 < len(self._ask_questions):
            self._ask_q_index += 1
            self._ask_other_editing = False
            try:
                self.app.layout.focus(self._ask_window)
            except Exception:  # noqa: BLE001
                pass
            self.app.invalidate()
            return
        self._resolve_ask(
            {"outcome": "accepted", "answers": dict(self._ask_answers)}
        )

    def _ask_toggle_multi(self) -> None:
        if not self._ask_is_multi():
            return
        i = self._ask_opt_index
        opts = (self._current_ask_question() or {}).get("options") or []
        if i >= len(opts):
            # Other — switch to free-text instead of toggle.
            self._ask_confirm_current()
            return
        if i in self._ask_multi_picked:
            self._ask_multi_picked.discard(i)
        else:
            self._ask_multi_picked.add(i)
        self.app.invalidate()

    def _plan_mode_consent_fn(self) -> bool:
        """Block worker until user Enter/Esc on the turn-status consent strip."""
        if self._closing:
            return False
        task_id = self._active_task_id
        if task_id is None:
            sel = self._selected_task_view()
            task_id = sel.id if sel is not None else None
        self._plan_ui_task_id = task_id
        self._plan_consent_ok = False
        self._plan_consent_event.clear()
        self._plan_ui = "consent"
        if task_id:
            self.ledger.set_plan_state(task_id, "consent")
        self._call_in_ui_thread(self.app.invalidate)
        # Wait on worker thread; UI resolves via keys.
        signaled = self._plan_consent_event.wait(timeout=600)
        self._plan_ui = None
        if not signaled:
            self._plan_consent_ok = False
            self._call_in_ui_thread(
                lambda: self._set_feedback("Plan 同意超时 · 已拒绝", "warning")
            )
        ok = bool(self._plan_consent_ok)
        if task_id:
            self.ledger.set_plan_state(
                task_id, "planning" if ok else "none"
            )
        self._call_in_ui_thread(self.app.invalidate)
        return ok

    def _plan_mode_exit_fn(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Open task plan tab and wait for a / s / q."""
        payload = payload or {}
        if self._closing:
            return {"outcome": "abandoned"}
        task_id = self._active_task_id
        if task_id is None:
            sel = self._selected_task_view()
            task_id = sel.id if sel is not None else None
        plan_path = str(payload.get("plan_file_path") or "")
        self._plan_ui_task_id = task_id
        self._plan_exit_outcome = "approved"
        self._plan_exit_feedback = ""
        self._plan_exit_event.clear()
        self._plan_ui = "review"
        self._plan_exit_waiting = True
        kernel = getattr(self.session.extensions, "kernel", None)
        state = getattr(kernel, "session_mode_state", None) if kernel else None
        if state is not None:
            state.awaiting_plan_approval = True
            if kernel is not None and hasattr(kernel, "persist_plan_mode_state"):
                try:
                    kernel.persist_plan_mode_state()
                except Exception:  # noqa: BLE001
                    pass
        if task_id:
            self.ledger.set_plan_state(
                task_id, "awaiting_approval", plan_file=plan_path or None
            )
            # Open MAIN agent detail on 计划 tab.
            main_id = f"{task_id}:main"
            def _open() -> None:
                self._open_agent(task_id, main_id)
                self._detail_filter = "plan"
                self.app.invalidate()

            self._call_in_ui_thread(_open)
        else:
            self._call_in_ui_thread(self.app.invalidate)
        try:
            signaled = self._plan_exit_event.wait(timeout=600)
        finally:
            self._plan_exit_waiting = False
        self._plan_ui = None
        if state is not None:
            state.awaiting_plan_approval = False
            if kernel is not None and hasattr(kernel, "persist_plan_mode_state"):
                try:
                    kernel.persist_plan_mode_state()
                except Exception:  # noqa: BLE001
                    pass
        if not signaled:
            # Fail closed: stay in plan mode (Grok Cancelled), revise default.
            self._plan_exit_outcome = "revise"
            self._plan_exit_feedback = "计划确认超时 · 请修改后再次 exit_plan_mode"
            self._call_in_ui_thread(
                lambda: self._set_feedback("计划确认超时 · 保持 Plan", "warning")
            )
        outcome = self._plan_exit_outcome
        if task_id:
            if outcome == "approved":
                self.ledger.set_plan_state(task_id, "approved")
            elif outcome == "revise":
                self.ledger.set_plan_state(task_id, "planning")
            else:
                self.ledger.set_plan_state(task_id, "abandoned")
        self._call_in_ui_thread(self.app.invalidate)
        if outcome == "revise":
            # Grok Cancelled: exit_fn returns cancelled; tool keeps plan mode
            # active and feeds revise text to the model.
            fb = self._plan_exit_feedback or "请根据用户意见修改计划"
            return {"outcome": "cancelled", "feedback": fb}
        if outcome == "abandoned":
            return {"outcome": "abandoned"}
        return {"outcome": "approved"}

    def _resolve_plan_consent(self, ok: bool) -> None:
        self._plan_consent_ok = ok
        self._plan_ui = None
        self._plan_consent_event.set()
        self.app.invalidate()

    def _resolve_plan_exit(self, outcome: str) -> None:
        if outcome == "revise":
            # Capture optional detail input as revision notes.
            try:
                text = (self._detail_input.text or "").strip()
            except Exception:  # noqa: BLE001
                text = ""
            self._plan_exit_feedback = text or "请修改计划后再次 exit_plan_mode"
            if text:
                self._detail_input.text = ""
        self._plan_exit_outcome = outcome
        self._plan_ui = None
        if self._plan_exit_waiting:
            self._plan_exit_event.set()
        else:
            # Resume chrome (restart with awaiting_plan_approval, no parked tool).
            self._apply_plan_exit_resume(outcome)
        self.app.invalidate()

    def _apply_plan_exit_resume(self, outcome: str) -> None:
        """Handle a/s/q after process resume when no exit_fn is waiting."""
        kernel = getattr(self.session.extensions, "kernel", None)
        state = getattr(kernel, "session_mode_state", None) if kernel else None
        task_id = self._plan_ui_task_id
        if task_id is None and self._modal_ref is not None:
            task_id, _ = self._modal_ref
        if outcome == "approved":
            if kernel is not None and hasattr(kernel, "exit_plan_mode"):
                kernel.exit_plan_mode(approved=True)
            if task_id:
                self.ledger.set_plan_state(task_id, "approved")
            self._set_feedback("计划已批准 · 可开工实现", "success")
        elif outcome == "revise":
            if state is not None:
                state.awaiting_plan_approval = False
                if hasattr(state, "plan_phase") and state.plan_phase != "active":
                    if hasattr(kernel, "enter_plan_mode"):
                        kernel.enter_plan_mode(getattr(state, "plan_file", None))
                if kernel is not None and hasattr(kernel, "persist_plan_mode_state"):
                    kernel.persist_plan_mode_state()
            if task_id:
                self.ledger.set_plan_state(task_id, "planning")
            fb = self._plan_exit_feedback or "请修改计划"
            self._set_feedback(f"继续改计划 · {fb[:40]}", "info")
        else:
            if kernel is not None and hasattr(kernel, "exit_plan_mode"):
                kernel.exit_plan_mode(approved=False)
            if task_id:
                self.ledger.set_plan_state(task_id, "abandoned")
            self._set_feedback("已放弃计划", "warning")
        if state is not None:
            state.awaiting_plan_approval = False
            if kernel is not None and hasattr(kernel, "persist_plan_mode_state"):
                try:
                    kernel.persist_plan_mode_state()
                except Exception:  # noqa: BLE001
                    pass

    def _maybe_restore_plan_approval_chrome(self) -> None:
        """After restart: restore plan tab + a/s/q if awaiting_plan_approval."""
        kernel = getattr(self.session.extensions, "kernel", None)
        if kernel is None:
            return
        # Hydrate plan_mode.json if not already (build_session may have).
        if hasattr(kernel, "load_plan_mode_state"):
            try:
                if getattr(kernel, "session_mode_state", None) is None or not getattr(
                    getattr(kernel, "session_mode_state", None),
                    "awaiting_plan_approval",
                    False,
                ):
                    kernel.load_plan_mode_state()
            except Exception:  # noqa: BLE001
                pass
        state = getattr(kernel, "session_mode_state", None)
        if state is None or not getattr(state, "awaiting_plan_approval", False):
            return
        plan_path = str(getattr(state, "plan_file", "") or "")
        tasks = self.ledger.snapshots()
        task_id: str | None = None
        if tasks:
            task_id = tasks[-1].id
            self.ledger.set_plan_state(
                task_id, "awaiting_approval", plan_file=plan_path or None
            )
            main_id = f"{task_id}:main"
            self._plan_ui_task_id = task_id
            self._plan_ui = "review"
            self._plan_exit_waiting = False  # resume chrome, not blocked tool
            try:
                self._open_agent(task_id, main_id)
                self._detail_filter = "plan"
            except Exception:  # noqa: BLE001
                pass
            self._set_feedback(
                "恢复：计划待确认 · a 批准 · s 修改 · q 放弃",
                "warning",
            )
        else:
            self._plan_ui = "review"
            self._plan_exit_waiting = False
            self._set_feedback(
                "恢复：计划待确认（无任务卡）· a/s/q 仍可用",
                "warning",
            )
        self.app.invalidate()

    def _toggle_session_plan_mode(self) -> None:
        kernel = getattr(self.session.extensions, "kernel", None)
        if kernel is None:
            self._set_feedback("当前会话无 plan 模式", "warning")
            return
        state = getattr(kernel, "session_mode_state", None)
        turn_in_flight = bool(self._is_running())
        phase = str(getattr(state, "plan_phase", "") or "") if state else ""
        # Off when active / pending / exit_pending UI chrome is on.
        on = bool(
            state is not None
            and (
                getattr(state, "is_plan_ui", None)()
                if callable(getattr(state, "is_plan_ui", None))
                else (
                    getattr(state, "is_plan", lambda: False)()
                    or phase in {"pending", "active", "exit_pending"}
                )
            )
        )
        plan_path: str | None = None
        try:
            from codedoggy.tools.grok_build.plan_mode import (
                probe_or_create_empty_plan_file,
            )

            resolve = getattr(kernel, "_resolve_plan_file_path", None)
            if callable(resolve):
                plan_path = str(resolve(None))
            else:
                from codedoggy.orchestration.session_mode import plan_file_for_session

                cwd = Path(
                    getattr(kernel, "cwd", None)
                    or getattr(self.session, "cwd", Path.cwd())
                )
                sid = str(
                    getattr(kernel, "session_id", None)
                    or getattr(self.session, "id", "")
                    or "default"
                )
                plan_path = str(plan_file_for_session(cwd, sid).resolve())
            probe_or_create_empty_plan_file(Path(plan_path))
        except Exception:  # noqa: BLE001
            plan_path = None

        # Goal mode replaces Plan (Grok session modes are exclusive).
        if (
            not on
            and state is not None
            and getattr(state, "is_goal", lambda: False)()
        ):
            try:
                state.exit_goal(reason="switch_to_plan")
            except Exception:  # noqa: BLE001
                try:
                    state.exit_goal()
                except Exception:  # noqa: BLE001
                    pass
            if hasattr(kernel, "persist_plan_mode_state"):
                try:
                    kernel.persist_plan_mode_state()
                except Exception:  # noqa: BLE001
                    pass
            self._set_feedback("已退出 Goal · 切换 Plan", "info")

        if on:
            if hasattr(kernel, "user_exit_plan_mode"):
                kernel.user_exit_plan_mode(turn_in_flight=turn_in_flight)
            elif hasattr(kernel, "exit_plan_mode"):
                kernel.exit_plan_mode(approved=False)
            phase_after = str(getattr(state, "plan_phase", "") or "") if state else ""
            if phase_after == "exit_pending":
                self._set_feedback("将在本轮结束后退出 Plan", "info")
            else:
                self._set_feedback("已切换到 Auto", "info")
        else:
            # Idle Tab → Pending (activate on next prompt). Mid-turn Tab → Active
            # + interjection reminder (Grok mid-turn activate).
            if turn_in_flight and hasattr(kernel, "enter_plan_mode"):
                kernel.enter_plan_mode(plan_path)
                self._inject_mid_turn_plan_reminder()
                self._set_feedback("已切换到 Plan · 仅可改 plan 文件", "info")
            elif hasattr(kernel, "enter_plan_mode_pending"):
                kernel.enter_plan_mode_pending(plan_path)
                self._set_feedback("Plan 待启动 · 下一条消息进入计划模式", "info")
            elif hasattr(kernel, "enter_plan_mode"):
                kernel.enter_plan_mode(plan_path)
                self._set_feedback("已切换到 Plan · 仅可改 plan 文件", "info")
        self.app.invalidate()

    def _inject_mid_turn_plan_reminder(self) -> None:
        """Push a plan-mode activation note into the live turn (interjection)."""
        from codedoggy.orchestration.session_mode import PLAN_REMINDER_FULL

        text = (
            f"<system-reminder>\n{PLAN_REMINDER_FULL}\n</system-reminder>"
        )
        interject = getattr(self.session, "interject", None)
        if callable(interject):
            try:
                interject(text)
                return
            except Exception:  # noqa: BLE001
                pass
        # Fallback: enqueue as deferred full prompt if interject unavailable.
        enqueue = getattr(self.session, "enqueue_prompt", None)
        if callable(enqueue):
            try:
                enqueue(text, prompt_id="plan-mid-turn-activate")
            except Exception:  # noqa: BLE001
                pass

    def _render_plan_detail_body(self) -> StyleAndTextTuples:
        from codedoggy.tui.agent_detail import _render_text_block

        width = max(12, _terminal_width() - 8)
        path = ""
        if self._modal_ref is not None:
            task_id, _ = self._modal_ref
            for t in self.ledger.snapshots():
                if t.id == task_id:
                    path = t.plan_file
                    break
        if not path:
            kernel = getattr(self.session.extensions, "kernel", None)
            state = getattr(kernel, "session_mode_state", None) if kernel else None
            path = str(getattr(state, "plan_file", "") or "")
        if not path:
            try:
                kernel = getattr(self.session.extensions, "kernel", None)
                resolve = getattr(kernel, "_resolve_plan_file_path", None)
                if callable(resolve):
                    path = str(resolve(None))
            except Exception:  # noqa: BLE001
                path = ""
        if not path:
            from codedoggy.orchestration.session_mode import plan_file_for_session

            sid = str(getattr(self.session, "id", "") or "default")
            path = str(
                plan_file_for_session(
                    getattr(self.session, "cwd", Path.cwd()), sid
                ).resolve()
            )
        text = ""
        mtime = 0.0
        try:
            p = Path(path)
            if not p.is_absolute():
                p = Path(getattr(self.session, "cwd", Path.cwd())) / p
            if p.is_file():
                mtime = float(p.stat().st_mtime)
                text = p.read_text(encoding="utf-8", errors="replace")
                # Cap huge plan files for paint responsiveness.
                if len(text) > 80_000:
                    text = text[:80_000] + "\n\n…(计划过长，已截断显示)\n"
        except Exception:  # noqa: BLE001
            text = ""
        cache_key = (
            "plan",
            self._modal_ref,
            width,
            path,
            mtime,
            len(text),
            self._todo_badge_label(),
            self._task_awaiting_plan_approval(),
        )
        if (
            cache_key == self._detail_body_cache_key
            and self._detail_body_cache is not None
        ):
            return self._detail_body_cache
        frags: StyleAndTextTuples = []
        if self._task_awaiting_plan_approval():
            frags.append(
                (
                    "class:detail.thinking.header",
                    "  ◆ 计划待批 · a 批准 · s 修改 · q 放弃\n",
                )
            )
        if path:
            frags.append(
                ("class:detail.meta", f"  plan: {path}\n")
            )
        if not text.strip():
            frags.append(("class:detail.meta", "  （尚未写入计划内容）\n"))
        else:
            frags.extend(_render_text_block(text, width))
        frags = self._with_detail_scroll(self._ensure_fragments(frags))
        self._set_detail_line_count(frags)
        self._detail_body_cache_key = cache_key
        self._detail_body_cache = frags
        return frags

    def _plan_consent_mouse(
        self, ok: bool
    ) -> Callable[[MouseEvent], object]:
        return self._only_mouse_up(
            lambda _e: self._resolve_plan_consent(ok),
            scroll_target="none",
        )

    def _plan_action_mouse(
        self, outcome: str
    ) -> Callable[[MouseEvent], object]:
        return self._only_mouse_up(
            lambda _e: self._resolve_plan_exit(outcome),
            scroll_target="detail",
        )

    def _move_detail_cursor(self, delta: int) -> None:
        max_y = max(0, int(self._detail_line_count) - 1)
        self._detail_cursor_line = min(
            max(0, int(self._detail_cursor_line) + delta),
            max_y,
        )
        # Keep Window scroll aligned with the cursor anchor.
        try:
            win = self._detail_window
            info = getattr(win, "render_info", None)
            if info is not None:
                height = max(1, int(info.window_height))
                max_scroll = max(0, int(info.content_height) - height)
                y = int(self._detail_cursor_line)
                if y < int(win.vertical_scroll):
                    win.vertical_scroll = y
                elif y >= int(win.vertical_scroll) + height:
                    win.vertical_scroll = min(max_scroll, y - height + 1)
            else:
                # No render_info yet: still move the scroll hint with the cursor.
                win.vertical_scroll = max(
                    0, int(self._detail_cursor_line)
                )
        except Exception:  # noqa: BLE001
            pass
        self.app.invalidate()

    def _scroll_detail_to_line(self, line: int) -> None:
        """Jump detail viewport + cursor anchor to an absolute line."""
        max_y = max(0, int(self._detail_line_count) - 1)
        y = max(0, min(int(line), max_y))
        self._detail_cursor_line = y
        try:
            win = self._detail_window
            info = getattr(win, "render_info", None)
            if info is not None:
                height = max(1, int(info.window_height))
                max_scroll = max(0, int(info.content_height) - height)
                win.vertical_scroll = max(0, min(max_scroll, y))
            else:
                win.vertical_scroll = y
        except Exception:  # noqa: BLE001
            pass
        self.app.invalidate()

    def _scroll_detail_to_bottom(self) -> None:
        """Pin detail to the last lines (End key)."""
        max_y = max(0, int(self._detail_line_count) - 1)
        self._detail_cursor_line = max_y
        try:
            win = self._detail_window
            info = getattr(win, "render_info", None)
            if info is not None:
                height = max(1, int(info.window_height))
                max_scroll = max(0, int(info.content_height) - height)
                win.vertical_scroll = max_scroll
            else:
                win.vertical_scroll = max_y
        except Exception:  # noqa: BLE001
            pass
        self.app.invalidate()

    def _on_task_scrollbar(self, scroll: int) -> None:
        """Keep task cursor anchor on the scrollbar-driven viewport top."""
        self._follow_latest_task = False
        max_y = max(0, int(self._task_line_count) - 1)
        self._selected_line = max(0, min(max_y, int(scroll)))

    def _on_detail_scrollbar(self, scroll: int) -> None:
        """Keep detail cursor anchor on the scrollbar-driven viewport top."""
        max_y = max(0, int(self._detail_line_count) - 1)
        self._detail_cursor_line = max(0, min(max_y, int(scroll)))

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
            0, min(max_y, int(win.vertical_scroll))
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
        No pre-click / hover chrome.
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
                        if self._modal_open:
                            target = "detail"
                        elif self._todo_pane_open:
                            target = "todo"
                        else:
                            target = "tasks"
                    if target == "todo" and self._todo_pane_open:
                        # step is lines; map wheel to ~1 item per notch
                        self._scroll_todo_pane(-1 if step < 0 else 1)
                        return None
                    if target == "tasks" and not self._modal_open:
                        self._scroll_tasks(step)
                        return None
                    if target == "detail" and (
                        self._modal_open or self._ask_active
                    ):
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
        # Cursor follows the viewport top so ScrollOffsets cannot snap back.
        max_y = max(0, int(self._detail_line_count) - 1)
        self._detail_cursor_line = max(
            0, min(max_y, int(win.vertical_scroll))
        )
        self.app.invalidate()

    def _sync_detail_scroll_from_window(self) -> None:
        """Absorb scrollbar/wheel scroll into the detail cursor anchor.

        Without this, PT re-scrolls to ``get_cursor_position`` on every paint and
        the scrollbar thumb jumps back. Cursor may sit inside the viewport after
        ↑↓ keys — only pull it when it leaves the visible band.
        """
        if not self._modal_open:
            return
        if self._detail_scroll_syncing:
            return
        try:
            win = self._detail_window
            scroll = int(getattr(win, "vertical_scroll", 0) or 0)
        except Exception:  # noqa: BLE001
            return
        max_y = max(0, int(self._detail_line_count) - 1)
        cursor = int(self._detail_cursor_line)
        info = getattr(win, "render_info", None)
        height = int(info.window_height) if info is not None else 1
        height = max(1, height)
        # Outside viewport → follow scroll (drag / bare Window wheel).
        if cursor < scroll or cursor >= scroll + height:
            self._detail_scroll_syncing = True
            try:
                self._detail_cursor_line = max(0, min(max_y, scroll))
            finally:
                self._detail_scroll_syncing = False

    def _sync_task_scroll_from_window(self) -> None:
        """Same cursor/viewport coupling for the main task list scrollbar."""
        if self._modal_open or self._ask_active:
            return
        try:
            win = self._task_window
            scroll = int(getattr(win, "vertical_scroll", 0) or 0)
        except Exception:  # noqa: BLE001
            return
        max_y = max(0, int(self._task_line_count) - 1)
        cursor = int(self._selected_line)
        info = getattr(win, "render_info", None)
        height = int(info.window_height) if info is not None else 1
        height = max(1, height)
        if cursor < scroll or cursor >= scroll + height:
            self._selected_line = max(0, min(max_y, scroll))

    def _selected_task_view(self) -> TaskView | None:
        tasks = self.ledger.snapshots()
        if not tasks:
            return None
        if not self._task_selection_active or self._selected_task < 0:
            return None
        self._selected_task = max(0, min(int(self._selected_task), len(tasks) - 1))
        return tasks[self._selected_task]

    def _selected_agent_index(self, task: TaskView) -> int:
        if not task.agents:
            self._selected_agent_by_task[task.id] = 0
            return 0
        index = self._selected_agent_by_task.get(task.id, 0) % len(task.agents)
        self._selected_agent_by_task[task.id] = index
        return index

    def _tab_task_cycle(self) -> None:
        """Tab cycle: latest task → enter detail → exit detail.

        1. Outside / no selection → select + focus the newest task
        2. On a selected task card → open that task's detail
        3. Inside agent detail → close and return to the task list
        """
        if self._ask_active or (
            self._modal_open and self._modal_kind == "ask"
        ):
            return
        if self._modal_open and self._modal_kind == "auth":
            return
        # Inside task detail → leave (back to task list).
        if self._modal_open and self._modal_kind == "agent":
            saved_ref = self._modal_ref
            self._close_modal()
            # Keep the card that was open selected (do not jump elsewhere).
            if saved_ref is not None:
                tid = saved_ref[0]
                snaps = self.ledger.snapshots()
                for i, t in enumerate(snaps):
                    if t.id == tid:
                        self._selected_task = i
                        self._task_selection_active = True
                        self._follow_latest_task = i == len(snaps) - 1
                        self._pinned_task_for_line = None
                        break
            try:
                self.app.layout.focus(self._task_window)
            except Exception:  # noqa: BLE001
                pass
            self.app.invalidate()
            return
        # On a selected task (not typing in the prompt) → enter detail.
        # From the input box always re-land on the latest card first.
        input_focused = False
        try:
            input_focused = bool(self.app.layout.has_focus(self._input))
        except Exception:  # noqa: BLE001
            input_focused = False
        if (
            not input_focused
            and self._task_selection_active
            and self._selected_task >= 0
        ):
            self._open_selected_task()
            self.app.invalidate()
            return
        # Default: jump to the newest task (do not open yet).
        if self._focus_latest_task():
            try:
                self.app.layout.focus(self._task_window)
            except Exception:  # noqa: BLE001
                pass
        self.app.invalidate()

    def _focus_latest_task(self) -> bool:
        """Select the newest task and refresh refs. False when ledger is empty."""
        tasks = self.ledger.snapshots()
        if not tasks:
            self._task_refs = []
            return False
        self._selected_task = len(tasks) - 1
        self._task_selection_active = True
        self._follow_latest_task = True
        # Force re-pin of cursor line onto the latest card on next paint.
        self._pinned_task_for_line = None
        self._render_tasks()
        self.app.invalidate()
        return bool(self._task_refs)

    def _user_wants_latest_focus(self) -> bool:
        """True when the user is watching the bottom of the task list.

        Used so new/finished tasks re-select the latest card only if the user
        is already following or scrolled near the end — not when browsing older
        history above.
        """
        if self._follow_latest_task:
            return True
        if self._modal_open:
            return False
        win = self._task_window
        info = getattr(win, "render_info", None)
        if info is None:
            return False
        try:
            content_h = int(info.content_height)
            window_h = int(info.window_height)
            max_scroll = max(0, content_h - window_h)
            if max_scroll <= 0:
                # Everything fits: treat as "at bottom" only if selection is latest.
                tasks = self.ledger.snapshots()
                return bool(
                    tasks
                    and self._task_selection_active
                    and self._selected_task == len(tasks) - 1
                )
            # Within ~2 lines of the bottom counts as waiting on the latest.
            return int(win.vertical_scroll) >= max(0, max_scroll - 2)
        except Exception:  # noqa: BLE001
            return False

    def _maybe_focus_latest_after_task_event(self, task_id: str) -> None:
        """No-op: submit/turn events must not steal focus from the input.

        Kept as a stub so older call sites / tests stay import-safe.
        """
        return

    def _move_task(self, delta: int) -> None:
        tasks = self.ledger.snapshots()
        if not tasks:
            return
        if not self._task_selection_active or self._selected_task < 0:
            # First arrow from blank: pick latest (down) or first (up).
            self._selected_task = (
                len(tasks) - 1 if delta > 0 else 0
            )
            self._task_selection_active = True
        else:
            self._selected_task = (int(self._selected_task) + delta) % len(tasks)
        # Browsing older tasks pauses follow; landing on the last resumes it.
        self._follow_latest_task = self._selected_task == len(tasks) - 1
        self._pinned_task_for_line = None
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
        self._detail_filter = "message"
        self._detail_collapsed = set()
        self._detail_collapse_seeded_for = None
        self._detail_known_fold_keys = set()
        self._reset_detail_cursor_state()
        self._invalidate_detail_body_cache()
        self._detail_input.text = ""
        self._modal_open = True
        self.app.layout.focus(self._detail_window)
        self.app.invalidate()

    def _open_auth_wizard(self) -> None:
        self._modal_kind = "auth"
        self._modal_ref = None
        self._reset_detail_cursor_state()
        snap = session_surface.active_connection(self.session)
        self._auth_wizard.open(
            active_provider=session_surface.provider_id(self.session),
            active_model=session_surface.model_id(self.session),
            active_reasoning_effort=(
                snap.reasoning_effort if snap is not None else "high"
            ),
            active_reasoning_enabled=(
                snap.reasoning_enabled if snap is not None else True
            ),
        )
        self._detail_input.text = ""
        self._modal_open = True
        self.app.layout.focus(self._detail_window)
        self.app.invalidate()

    def _close_modal(self) -> None:
        was_auth = self._modal_kind == "auth"
        if self._modal_kind == "ask" and self._ask_active:
            # Closing × while questionnaire is open == cancel (do not hang worker).
            self._resolve_ask({"outcome": "cancelled"})
            return
        had_tasks = bool(self.ledger.snapshots())
        self._modal_open = False
        self._modal_kind = "agent"
        self._modal_ref = None
        self._reset_detail_cursor_state()
        self._invalidate_detail_body_cache()
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

    def _render_ask_dialog_title(self) -> StyleAndTextTuples:
        n = max(1, len(self._ask_questions))
        i = min(self._ask_q_index + 1, n)
        plan = " · 计划澄清" if self._session_in_plan_ui() else ""
        text = f"  {_DOG_EAR}  问卷  {i}/{n}{plan}  "
        return [("class:ask.header", text)]

    def _render_ask_dialog_hint(self) -> StyleAndTextTuples:
        if self._ask_other_editing:
            text = "  输入自定义答案 · Enter 提交 · Tab 退出"
        elif self._ask_is_multi():
            text = "  ↑↓ 移动 · Space 勾选 · Enter 完成 · Tab 退出"
        else:
            text = "  ↑↓ 选择 · Enter 确认 · Tab 退出"
        return [
            (
                "class:ask.hint",
                _truncate_display(text, max(12, _terminal_width() - 12)) + "\n",
            )
        ]

    def _render_ask_dialog(self) -> StyleAndTextTuples:
        """Dedicated questionnaire popup body (↑↓ selectable)."""
        return self._render_ask_body()

    def _render_ask_body(self) -> StyleAndTextTuples:
        """Questionnaire body — compact bordered float (not full-screen)."""
        width = max(18, _terminal_width() - 24)
        inner = max(10, width - 2)
        frags: StyleAndTextTuples = []
        q = self._current_ask_question()
        if q is None:
            frags.append(("class:ask.meta", "  （无问题）\n"))
            return self._ensure_fragments(frags)

        n = max(1, len(self._ask_questions))
        frags.append(("class:ask.meta", f"  问题 {self._ask_q_index + 1}/{n}\n"))
        qtext = str(q.get("question") or "").strip() or "（空问题）"
        for line in _wrap_display_lines(qtext, inner, max_lines=4):
            frags.append(("class:ask.question", f"  {line}\n"))
        frags.append(("class:ask.meta", "\n"))

        opts = q.get("options") if isinstance(q.get("options"), list) else []
        multi = self._ask_is_multi()
        for i, opt in enumerate(opts):
            if not isinstance(opt, dict):
                continue
            label = str(opt.get("label") or f"选项{i+1}")
            desc = str(opt.get("description") or "").strip()
            selected = i == self._ask_opt_index
            picked = i in self._ask_multi_picked
            if multi:
                mark = "☑" if picked else "☐"
            else:
                mark = "●" if selected else "○"
            prefix = "› " if selected else "  "
            style = "class:ask.option.selected" if selected else "class:ask.option"
            row = f"{prefix}{mark} {label}"
            frags.append((style, _truncate_display(row, width) + "\n"))
            if desc and selected:
                for dline in _wrap_display_lines(desc, max(8, inner - 4), max_lines=2):
                    frags.append(("class:ask.option.desc", f"    {dline}\n"))
            preview = opt.get("preview")
            if preview and selected:
                for pline in _wrap_display_lines(
                    str(preview), max(8, inner - 4), max_lines=3
                ):
                    frags.append(("class:ask.option.desc", f"    │ {pline}\n"))

        other_i = len(opts)
        other_sel = self._ask_opt_index == other_i or self._ask_other_editing
        ostyle = "class:ask.option.selected" if other_sel else "class:ask.option"
        omark = "●" if other_sel and not multi else ("○" if not multi else "☐")
        opfx = "› " if other_sel else "  "
        frags.append(
            (ostyle, _truncate_display(f"{opfx}{omark} Other（自己写）", width) + "\n")
        )
        if self._ask_other_editing:
            frags.append(
                ("class:ask.meta", "    在底部输入框写答案后 Enter\n")
            )
        return self._ensure_fragments(frags)

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

    def _task_card_mouse(
        self, task_index: int
    ) -> Callable[[MouseEvent], object]:
        """Whole-card click: select; double-click or Ctrl+left opens detail.

        No hover chrome, no auto-scroll-to-bottom on single click.
        """
        from prompt_toolkit.mouse_events import MouseButton, MouseModifier

        def handler(event: MouseEvent) -> object:
            btn = getattr(event, "button", None)
            if event.event_type is MouseEventType.MOUSE_DOWN:
                if btn in (None, MouseButton.LEFT):
                    self._task_mouse_down_index = task_index
                return None
            if event.event_type in {
                MouseEventType.SCROLL_UP,
                MouseEventType.SCROLL_DOWN,
            }:
                if not self._modal_open:
                    step = (
                        -3 if event.event_type is MouseEventType.SCROLL_UP else 3
                    )
                    self._scroll_tasks(step)
                    return None
                return NotImplemented
            if event.event_type is not MouseEventType.MOUSE_UP:
                return NotImplemented
            if btn not in (None, MouseButton.LEFT):
                self._task_mouse_down_index = None
                return NotImplemented
            down = self._task_mouse_down_index
            self._task_mouse_down_index = None
            # Must press and release on the same card (not drag in from void).
            if down != task_index:
                return None
            tasks = self.ledger.snapshots()
            if not 0 <= task_index < len(tasks):
                return None
            self._selected_task = task_index
            self._task_selection_active = True
            self._follow_latest_task = task_index == len(tasks) - 1
            self._pinned_task_for_line = None
            try:
                self.app.layout.focus(self._task_window)
            except Exception:  # noqa: BLE001
                pass
            mods = getattr(event, "modifiers", None) or ()
            now = time.monotonic()
            last = self._task_card_last_click
            is_double = (
                last is not None
                and last[0] == task_index
                and (now - last[1]) <= _DOUBLE_CLICK_S
            )
            if MouseModifier.CONTROL in mods or is_double:
                self._task_card_last_click = None
                self._open_selected_task()
            else:
                self._task_card_last_click = (task_index, now)
                self.app.invalidate()
            return None

        return handler

    def _clear_task_selection(self, *, focus_input: bool = True) -> None:
        """Drop selection/follow — never fall back to task index 0."""
        self._task_selection_active = False
        self._follow_latest_task = False
        self._selected_task = -1
        self._pinned_task_for_line = None
        self._task_mouse_down_index = None
        self._task_card_last_click = None
        if focus_input:
            try:
                self.app.layout.focus(self._input)
            except Exception:  # noqa: BLE001
                pass
        self._task_paint_cache = None
        self.app.invalidate()

    def _task_gap_mouse(self) -> Callable[[MouseEvent], object]:
        """1-line gap between cards: click clears selection."""

        def handler(event: MouseEvent) -> object:
            if event.event_type in {
                MouseEventType.SCROLL_UP,
                MouseEventType.SCROLL_DOWN,
            }:
                if not self._modal_open:
                    step = (
                        -3 if event.event_type is MouseEventType.SCROLL_UP else 3
                    )
                    self._scroll_tasks(step)
                    return None
                return NotImplemented
            if event.event_type is MouseEventType.MOUSE_DOWN:
                self._task_mouse_down_index = None
                return None
            if event.event_type is MouseEventType.MOUSE_UP:
                self._clear_task_selection(focus_input=True)
                return None
            return NotImplemented

        return handler

    def _task_void_mouse(self) -> Callable[[MouseEvent], object]:
        """Empty area below cards: click clears selection (no hover)."""

        def handler(event: MouseEvent) -> object:
            if event.event_type is MouseEventType.MOUSE_DOWN:
                self._task_mouse_down_index = None
                return None
            if event.event_type in {
                MouseEventType.SCROLL_UP,
                MouseEventType.SCROLL_DOWN,
            }:
                if not self._modal_open:
                    step = (
                        -3 if event.event_type is MouseEventType.SCROLL_UP else 3
                    )
                    self._scroll_tasks(step)
                    return None
                return NotImplemented
            if event.event_type is MouseEventType.MOUSE_UP:
                self._clear_task_selection(focus_input=True)
                return None
            return NotImplemented

        return handler

    def _task_blank_mouse(self) -> Callable[[MouseEvent], object]:
        """Alias for void (tests / callers)."""
        return self._task_void_mouse()

    @staticmethod
    def _make_prompt_history() -> FileHistory | InMemoryHistory:
        """Grok-style ↑/↓ prompt history; file-backed when home is writable."""
        try:
            path = Path.home() / ".codedoggy" / "prompt_history"
            path.parent.mkdir(parents=True, exist_ok=True)
            return FileHistory(str(path))
        except Exception:  # noqa: BLE001
            return InMemoryHistory()

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
            self._queue_or_apply_reload(
                provider=getattr(action, "provider", None),
                model=getattr(action, "model", None),
                reasoning_effort=getattr(action, "reasoning_effort", None),
                reasoning_enabled=getattr(action, "reasoning_enabled", None),
                message=message,
            )
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

    def _queue_or_apply_reload(
        self,
        *,
        provider: str | None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        reasoning_enabled: bool | None = None,
        message: str = "",
    ) -> None:
        """Apply connection now, or after the current turn if one is running.

        OAuth can finish while MAIN is still on bootstrap ollama; dropping the
        reload left tokens saved but sampler on 11434. Queue + flush fixes that.
        """
        if self._is_running():
            self._pending_reload = {
                "provider": provider,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "reasoning_enabled": reasoning_enabled,
                "message": message,
            }
            label = (provider or "provider").strip() or "provider"
            self._set_feedback(
                f"登录已保存，当前任务结束后自动切换到 {label}",
                "info",
            )
            self.app.invalidate()
            return
        self._pending_reload = None
        self._apply_reload_client(
            provider=provider,
            model=model,
            reasoning_effort=reasoning_effort,
            reasoning_enabled=reasoning_enabled,
            message=message,
        )

    def _flush_pending_reload(self) -> None:
        pending = self._pending_reload
        if not pending or self._is_running():
            return
        self._pending_reload = None
        self._apply_reload_client(
            provider=pending.get("provider"),
            model=pending.get("model"),
            reasoning_effort=pending.get("reasoning_effort"),
            reasoning_enabled=pending.get("reasoning_enabled"),
            message=str(pending.get("message") or ""),
        )

    def _apply_reload_client(
        self,
        *,
        provider: str | None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        reasoning_enabled: bool | None = None,
        message: str = "",
    ) -> None:
        try:
            snap = self._reload_model_client(
                provider,
                model=model,
                reasoning_effort=reasoning_effort,
                reasoning_enabled=reasoning_enabled,
            )
            if snap is not None:
                self._auth_wizard.active_provider = snap.provider
                self._auth_wizard.active_model = snap.model
                self._auth_wizard.active_reasoning_effort = snap.reasoning_effort
                self._auth_wizard.active_reasoning_enabled = snap.reasoning_enabled
                self._auth_wizard.pending_model = ""
                if self._auth_wizard.step is WizardStep.REASONING:
                    self._auth_wizard.step = WizardStep.PROVIDER
                    self._auth_wizard.provider = snap.provider
                    self._auth_wizard.cursor = 0
                if self._auth_wizard.step in {
                    WizardStep.PROVIDER,
                    WizardStep.MODEL,
                    WizardStep.REASONING,
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

    def _reload_model_client(
        self,
        provider: str | None = None,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        reasoning_enabled: bool | None = None,
    ) -> Any:
        """Apply provider/model/reasoning through ConnectionService only."""
        return session_surface.apply_connection(
            self.session,
            provider=provider,
            model=model,
            reasoning_effort=reasoning_effort,
            reasoning_enabled=reasoning_enabled,
            require_auth=True,
            source="panel",
        )

    def _wire_buffer_ctrl_click(self, area: TextArea) -> None:
        """Ctrl+click a path /「查看图片(…)」chip in the prompt → open file."""
        control = area.control
        original = control.mouse_handler

        def mouse_handler(mouse_event: MouseEvent) -> object:
            from prompt_toolkit.mouse_events import MouseButton, MouseModifier

            result = original(mouse_event)
            if mouse_event.event_type is not MouseEventType.MOUSE_UP:
                return result
            btn = getattr(mouse_event, "button", None)
            if btn not in (None, MouseButton.LEFT):
                return result
            mods = getattr(mouse_event, "modifiers", None) or ()
            if MouseModifier.CONTROL not in mods:
                return result
            buf = area.buffer
            path = path_under_cursor(buf.text, buf.cursor_position)
            if path is None and self._last_pasted_path:
                # Cursor may sit on「查看图片」label — open last paste.
                label = VIEW_IMAGE_LABEL
                pos = buf.cursor_position
                start = max(0, pos - len(label))
                if label in buf.text[start : pos + len(label)]:
                    path = self._last_pasted_path
            if path is None:
                # Chip form: 查看图片(relative/path.png)
                m = re.search(
                    rf"{re.escape(VIEW_IMAGE_LABEL)}\(([^)]+)\)",
                    buf.text,
                )
                if m and m.start() <= buf.cursor_position <= m.end():
                    path = m.group(1)
            if not path:
                return result
            cwd = getattr(self.session, "cwd", None)
            ok, message = open_local_path(path, cwd=cwd)
            self._set_feedback(message, "success" if ok else "warning")
            try:
                self.app.invalidate()
            except Exception:  # noqa: BLE001
                pass
            return None

        control.mouse_handler = mouse_handler  # type: ignore[method-assign]

    def _image_path_mouse(self, path: str) -> Callable[[MouseEvent], object]:
        """Ctrl+click-to-open for image / script paths in the agent detail pane."""

        def handler(event: MouseEvent) -> object:
            from prompt_toolkit.mouse_events import MouseModifier

            if event.event_type in {
                MouseEventType.SCROLL_UP,
                MouseEventType.SCROLL_DOWN,
            }:
                step = -3 if event.event_type is MouseEventType.SCROLL_UP else 3
                if self._modal_open or self._ask_active:
                    self._scroll_detail(step)
                    return None
                return NotImplemented
            if event.event_type is not MouseEventType.MOUSE_UP:
                return NotImplemented
            mods = getattr(event, "modifiers", None) or ()
            if MouseModifier.CONTROL not in mods:
                return NotImplemented
            cwd = getattr(self.session, "cwd", None)
            ok, message = open_local_path(path, cwd=cwd)
            self._set_feedback(message, "success" if ok else "warning")
            try:
                self.app.invalidate()
            except Exception:  # noqa: BLE001
                pass
            return None

        return handler


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
    from codedoggy.tui.agent_detail import strip_system_reminders

    parts: list[str] = []
    for message in messages:
        role = getattr(message, "role", None)
        if role is not Role.ASSISTANT and getattr(role, "value", role) != "assistant":
            continue
        content = strip_system_reminders(
            str(getattr(message, "content", "") or "").strip()
        )
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


def task_report_from_agent(text: str, *, max_chars: int | None = None) -> str:
    """Boss-list summary from MAIN wording — first paragraph, full text.

    Display wraps on the task card; do not hard-crop with ellipsis here.
    ``max_chars`` remains optional for callers that still want a soft cap.
    """
    clean = text.strip()
    if not clean:
        return "任务已结束。"
    soft = _friendly_failure_toast(clean)
    if soft != "任务未能完成" or _looks_like_transport_error(clean):
        # Prefer the human one-liner on the card; detail modal keeps raw if needed.
        if _looks_like_transport_error(clean) or clean.lower().startswith("sampler"):
            clean = soft
    paragraphs = [" ".join(part.split()) for part in re.split(r"\n\s*\n", clean)]
    report = next((part for part in paragraphs if part), clean)
    report = re.sub(r"^#{1,6}\s+", "", report)
    if max_chars is not None and len(report) > max_chars:
        return report[:max_chars].rstrip()
    return report


def _looks_like_transport_error(text: str) -> bool:
    low = (text or "").lower()
    return any(
        needle in low
        for needle in (
            "failed to reach",
            "winerror 10061",
            "积极拒绝",
            "connection refused",
            "timed out",
            "sampler error",
            "sampler failed",
        )
    )


def _friendly_failure_toast(text: str | None) -> str:
    """Short owner-facing failure line — no raw URL / WinError dump in the toast."""
    raw = (text or "").strip()
    if not raw:
        return "任务未能完成"
    low = raw.lower()
    if "10061" in low or "积极拒绝" in raw:
        return "连不上模型 · 检查网络或代理是否开启"
    if "failed to reach" in low or "connection refused" in low:
        return "连不上模型服务 · 请稍后重试或检查代理"
    if "timed out" in low or "timeout" in low:
        return "模型响应超时 · 可以再试一次"
    if "sampler" in low:
        return "模型调用失败"
    # Keep short; card should not reprint a novel.
    one = " ".join(raw.split())
    if len(one) > 42:
        return one[:41] + "…"
    return one or "任务未能完成"


def _turn_status(status: TurnStatus | Any) -> str:
    value = getattr(status, "value", status)
    if value == TurnStatus.COMPLETED.value:
        return "completed"
    if value == TurnStatus.CANCELLED.value:
        return "cancelled"
    if value == TurnStatus.MAX_TURNS_REACHED.value:
        return "max_turns"
    if value == TurnStatus.QUEUED.value:
        return "queued"
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
    if minutes < 60:
        return f"{minutes}m{remain:02d}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h{minutes:02d}m"


def _task_elapsed_seconds(
    task: TaskView, *, now: float | None = None
) -> float | None:
    """Wall-clock seconds for a task card; None if never started."""
    started = float(getattr(task, "started_at", 0.0) or 0.0)
    if started <= 0:
        return None
    ended = getattr(task, "ended_at", None)
    end = float(ended) if ended is not None else float(now if now is not None else time.time())
    return max(0.0, end - started)


def _task_elapsed_label(
    task: TaskView, *, now: float | None = None
) -> str:
    """Compact duration for homepage stage line (e.g. ``1m23s``)."""
    seconds = _task_elapsed_seconds(task, now=now)
    if seconds is None:
        return ""
    return _format_elapsed(seconds)


def _task_is_terminal(task: TaskView) -> bool:
    return task.phase in {"done", "failed", "cancelled"} or task.status in {
        "completed",
        "failed",
        "cancelled",
        "max_turns",
    }


def _task_list_summary(
    task: TaskView, *, interject: str | None = None
) -> str:
    """One human line for the task list — never tool-call live noise.

    Terminal tasks always prefer report / MAIN prose over plan draft chrome so
    a finished turn never keeps showing「正在起草计划…」.
    """
    if interject:
        return f"↩ 插入中 · {interject}"

    report = (task.report or "").strip()
    main_prose = ""
    for agent in task.agents:
        if agent.label.upper() == "MAIN" and (agent.output or "").strip():
            out = " ".join(agent.output.split())
            if out.startswith("→") or "· 调用中" in out:
                continue
            main_prose = out
            break

    if _task_is_terminal(task):
        if report:
            return " ".join(report.split())
        if main_prose:
            return main_prose
        return ""

    # Live plan chrome only while the task is still open.
    if task.plan_state == "awaiting_approval":
        return "计划待你确认 · Enter 打开"
    if task.plan_state == "consent":
        return "代理想进入 Plan · Enter 同意 / Esc 拒绝"
    if task.plan_state == "planning" and task.phase in {
        "planning",
        "plan_review",
        "dispatching",
    }:
        # Prefer live MAIN prose if the agent already spoke this turn.
        if main_prose:
            return main_prose
        if report:
            return " ".join(report.split())
        return "正在起草计划…"
    if report:
        return " ".join(report.split())
    if main_prose:
        return main_prose
    return ""


def _task_stage_text(task: TaskView, *, interject: str | None = None) -> str:
    # Terminal first — never stick on「Plan · 起草中」after the turn finished.
    if task.phase == "done" or task.status == "completed":
        return f"已完成 · {len(task.agents)} 个 Agent"
    if task.phase == "cancelled" or task.status == "cancelled":
        return STATUS_TEXT.get("cancelled", "已取消")
    if task.phase == "failed" or task.status in {"failed", "max_turns"}:
        return STATUS_TEXT.get(task.status, "失败")

    if interject:
        return "↩ 插入中"
    if task.plan_state == "consent":
        return "Plan · 等待同意"
    if task.phase == "planning" or (
        task.plan_state == "planning" and task.phase in {"planning", "plan_review"}
    ):
        return "Plan · 起草中"
    if task.phase == "plan_review" or task.plan_state == "awaiting_approval":
        return "Plan · 待批"
    if task.phase == "dispatching":
        return "MAIN 拆解中"
    if task.phase == "parallel":
        active = sum(
            agent.status in {"pending", "running"} for agent in task.agents
        )
        return f"{max(1, active)} 个 Agent 并行中"
    if task.phase == "reporting":
        return "MAIN 汇总中"
    return STATUS_TEXT.get(task.status, task.status)


def _compact_task_stage_text(
    task: TaskView, *, interject: str | None = None
) -> str:
    if task.phase == "done" or task.status == "completed":
        return f"完成·{len(task.agents)}"
    if task.phase == "cancelled" or task.status == "cancelled":
        return "取消"
    if task.phase == "failed" or task.status in {"failed", "max_turns"}:
        return "失败"
    if interject:
        return "插入中"
    if task.plan_state == "consent":
        return "待同意"
    if task.phase == "planning" or (
        task.plan_state == "planning" and task.phase in {"planning", "plan_review"}
    ):
        return "Plan"
    if task.phase == "plan_review" or task.plan_state == "awaiting_approval":
        return "待批"
    if task.phase == "dispatching":
        return "拆解中"
    if task.phase == "parallel":
        active = sum(
            agent.status in {"pending", "running"} for agent in task.agents
        )
        return f"{max(1, active)} 并行"
    if task.phase == "reporting":
        return "汇总中"
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
    ".": "#141414",
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


_MORE_HINT_WIDTH = 3
# 2Hz cycle (not 10Hz): ride Application.refresh_interval paints without
# thrashing the task-list fragment cache every tick.
_MORE_HINT_FRAMES = (
    "  >",
    " =>",
    "==>",
    "=> ",
    ">  ",
    "==>",
    " =>",
    "==>",
)


def _more_hint(*, now: float | None = None) -> str:
    """Animated more-marker (fixed width 3) for truncated task briefs.

    Paint-time only — never schedules invalidate. Prefer ``now=_paint_clock``
    so one frame uses one marker glyph.
    """
    clock = time.monotonic() if now is None else now
    return _MORE_HINT_FRAMES[int(clock * 2) % len(_MORE_HINT_FRAMES)]


def _strip_legacy_ellipsis(text: str) -> str:
    """Remove trailing … / ... left by older report storage — UI owns more-hint."""
    raw = text
    while True:
        if raw.endswith("..."):
            raw = raw[:-3].rstrip()
            continue
        if raw.endswith("…"):
            raw = raw[:-1].rstrip()
            continue
        break
    return raw


def _fill_display_width(text: str, width: int) -> tuple[str, str]:
    """Take a prefix of ``text`` that fits in ``width`` cells; return (prefix, rest)."""
    width = max(0, int(width))
    if width <= 0:
        return "", text
    out: list[str] = []
    used = 0
    for i, ch in enumerate(text):
        cw = get_cwidth(ch)
        if used + cw > width:
            return "".join(out), text[i:]
        out.append(ch)
        used += cw
    return "".join(out), ""


def _brief_two_lines(text: str, full_width: int) -> tuple[list[str], bool]:
    """Task brief: up to two full-width lines.

    Returns ``(lines, truncated)``. When ``truncated`` is True the second line
    is body-only (room reserved for the animated ``==>`` more-marker); the
    caller paints the marker with ``class:report.more``.
    """
    # Honor the caller's column budget (do not force min 8 — narrow cards overflow).
    full_width = max(1, int(full_width))
    raw = _strip_legacy_ellipsis(
        " ".join((text or "").replace("\r", "\n").split())
    )
    if not raw:
        return [""], False
    if get_cwidth(raw) <= full_width:
        return [raw], False
    line1, rest = _fill_display_width(raw, full_width)
    line1 = line1.rstrip()
    rest = rest.lstrip()
    if not rest:
        return [line1], False
    if get_cwidth(rest) <= full_width:
        return [line1, rest], False
    # Reserve marker only when at least one body cell remains.
    reserve = _MORE_HINT_WIDTH if full_width > _MORE_HINT_WIDTH else 0
    body_w = max(1, full_width - reserve)
    body, _leftover = _fill_display_width(rest, body_w)
    return [line1, body.rstrip()], True
