"""Prompt-toolkit reading stream with inline tools and plan approval."""

from __future__ import annotations

import hashlib
import re
import shutil
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from itertools import groupby
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.clipboard import ClipboardData
from prompt_toolkit.filters import Condition, has_selection
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.formatted_text.utils import split_lines
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
from prompt_toolkit.document import Document
from prompt_toolkit.widgets import Frame, TextArea

from codedoggy.session.types import SessionPhase, TurnStatus
from codedoggy.tui.clipboard_image import (
    coerce_image_path_text,
    get_system_clipboard_text,
    insert_image_chip,
    save_clipboard_image,
    set_system_clipboard_text,
)
from codedoggy.tui.agent_detail import (
    AgentDetailSnapshot,
    DetailRecord,
    snapshot_from_messages,
)
from codedoggy.tui.open_path import (
    VIEW_IMAGE_LABEL,
    extract_image_chip_paths,
    is_openable_file_path,
    open_local_path,
    path_under_cursor,
    resolve_openable_path,
    strip_image_chips,
)
from codedoggy.tui.plan_view import PlanMarkdownLexer
from codedoggy.tui.syntax import highlight_code_line
from codedoggy.tui.theme import build_style
from codedoggy.tui.activity import LiveActivityBoard
from codedoggy.tui.login_wizard import AuthWizard, WizardStep, run_browser_login
from codedoggy.tui.model import TaskLedger, TaskView
from codedoggy.tui import surface as session_surface
from codedoggy.attachments import AttachmentError, ImageAttachment
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

_STREAM_REFRESH_INTERVAL_S = 0.04


def _mouse_control_held(event: MouseEvent | None = None) -> bool:
    """True when Ctrl is held for a mouse event.

    prompt_toolkit's Win32 mouse path always passes an empty modifier set
    (``UNKNOWN_MODIFIER``), so ``MouseModifier.CONTROL in event.modifiers``
    is permanently false on classic Windows console input. Read the real
    keyboard state as a fallback.
    """
    if event is not None:
        mods = getattr(event, "modifiers", None) or ()
        try:
            from prompt_toolkit.mouse_events import MouseModifier

            if MouseModifier.CONTROL in mods:
                return True
        except Exception:  # noqa: BLE001
            pass
        for mod in mods:
            if getattr(mod, "value", mod) == "CONTROL":
                return True
    if sys.platform == "win32":
        try:
            import ctypes

            # VK_CONTROL = 0x11; high bit set ⇒ currently down.
            return bool(ctypes.windll.user32.GetAsyncKeyState(0x11) & 0x8000)
        except Exception:  # noqa: BLE001
            return False
    return False


def _append_stream_preview(
    chunks: list[str],
    state: dict[str, Any],
    piece: Any,
) -> tuple[str, bool]:
    """Append one model delta to the full live conversation text.

    ``preview`` is the whole turn for the homepage. ``draft`` is only the
    current, not-yet-archived assistant message for the live detail view.
    """
    chunk = str(piece or "")
    chunks.append(chunk)
    preview = f"{state.get('preview', '')}{chunk}"
    state["preview"] = preview
    state["draft"] = f"{state.get('draft', '')}{chunk}"
    now = time.monotonic()
    last_emit = float(state.get("last_emit", 0.0) or 0.0)
    should_emit = last_emit == 0.0 or now - last_emit >= _STREAM_REFRESH_INTERVAL_S
    if should_emit:
        state["last_emit"] = now
    return preview, should_emit


def _message_role_value(message: Any) -> str:
    role = (
        message.get("role")
        if isinstance(message, dict)
        else getattr(message, "role", None)
    )
    return str(getattr(role, "value", role) or "").strip().lower()


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
_MAIN_INPUT_PREFIX_COLS = 4  # "  › "
_MAIN_INPUT_RIGHT_COLS = 1  # one quiet breathing column
_MAIN_INPUT_SCROLLBAR_COLS = 1
_DOUBLE_CLICK_S = 0.45  # compact todo/fleet badge double-click window
# Windows Terminal steals Ctrl+V; we edge-detect the chord and paste images ourselves.
_WIN32_CTRL_V_DEBOUNCE_S = 0.35


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

        original_handlers: list[list[Callable[[MouseEvent], object]]] = []

        def capture_or_delegate(event: MouseEvent) -> object:
            """Capture an active drag; otherwise preserve the cell's real handler."""
            if self._dragging:
                et = event.event_type
                if et is MouseEventType.MOUSE_MOVE:
                    apply_thumb_drag(event.position.y)
                    return None
                if et is MouseEventType.MOUSE_UP:
                    apply_thumb_drag(event.position.y)
                    self._dragging = False
                    return None
            local_y = int(event.position.y) - ypos
            local_x = int(event.position.x) - capture_x_min
            if (
                0 <= local_y < len(original_handlers)
                and 0 <= local_x < len(original_handlers[local_y])
            ):
                return original_handlers[local_y][local_x](event)
            return NotImplemented

        # Install capture on the first paint, before a drag starts. Waiting for
        # MOUSE_DOWN to invalidate and repaint races with the first MOVE event:
        # as soon as the pointer leaves the one-cell rail, that MOVE would be
        # dispatched to the body and the drag would appear completely inert.
        # Delegation preserves links, selection and wheel handlers while idle.
        for screen_y in range(ypos, ypos + height):
            row = mouse_handlers.mouse_handlers[screen_y]
            original_handlers.append(
                [row[screen_x] for screen_x in range(capture_x_min, capture_x_max)]
            )
        mouse_handlers.set_mouse_handler_for_range(
            x_min=capture_x_min,
            x_max=capture_x_max,
            y_min=ypos,
            y_max=ypos + height,
            handler=capture_or_delegate,
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
    """Window with stable wheel routing and interactive scrollbar support."""

    def __init__(
        self,
        *args: Any,
        wheel_handler: Callable[[int], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._wheel_handler = wheel_handler

    def _install_wheel_handlers(
        self,
        mouse_handlers: Any,
        *,
        xpos: int,
        ypos: int,
        width: int,
        height: int,
    ) -> None:
        """Route wheel events once at window level; preserve all click handlers."""
        if self._wheel_handler is None or width <= 0 or height <= 0:
            return

        originals: list[list[Callable[[MouseEvent], object]]] = []
        for screen_y in range(ypos, ypos + height):
            row = mouse_handlers.mouse_handlers[screen_y]
            originals.append([row[x] for x in range(xpos, xpos + width)])

        def wheel_or_delegate(event: MouseEvent) -> object:
            if event.event_type is MouseEventType.SCROLL_UP:
                self._wheel_handler(-3)
                return None
            if event.event_type is MouseEventType.SCROLL_DOWN:
                self._wheel_handler(3)
                return None
            local_y = int(event.position.y) - ypos
            local_x = int(event.position.x) - xpos
            if (
                0 <= local_y < len(originals)
                and 0 <= local_x < len(originals[local_y])
            ):
                return originals[local_y][local_x](event)
            return NotImplemented

        mouse_handlers.set_mouse_handler_for_range(
            x_min=xpos,
            x_max=xpos + width,
            y_min=ypos,
            y_max=ypos + height,
            handler=wheel_or_delegate,
        )

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
        # Prefer the clamped rect Window actually painted into.
        painted = getattr(screen, "visible_windows_to_write_positions", {}).get(self)
        wp = painted if painted is not None else write_position
        xpos = int(wp.xpos)
        ypos = int(wp.ypos)
        width = int(wp.width)
        height = int(wp.height)
        self._install_wheel_handlers(
            mouse_handlers,
            xpos=xpos,
            ypos=ypos,
            width=width,
            height=height,
        )
        if total_right <= 0:
            return
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


# One visual truth: obsidian canvas and role-based reading colors.
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
        # Esc cancel is once-per-task + short grace: swallow key-repeat /
        # delayed Esc so the next turn is not cancelled after the user already
        # killed this one (GrokBuild: cancel must not bleed into the next prompt).
        self._cancelling_task_id: str | None = None
        self._cancel_grace_until: float = 0.0
        self._task_refs: list[str] = []
        # -1 = no intentional selection (never default to the first task).
        self._selected_task = -1
        # False until the user selects a task or follows the live tail.
        # Prevents "move/click blank → yellow first task" (selection must be intentional).
        self._task_selection_active = False
        self._selected_line = 0
        self._task_line_count = 1  # clamp cursor y — PT crashes if y >= line_count
        self._pinned_task_for_line: int | None = None  # re-pin only on task change
        # Active-task interject flash: task_id -> (until, preview)
        self._interject_flash: dict[str, tuple[float, str]] = {}
        self._modal_open = False
        self._modal_kind: str = "plan"  # plan | auth | ask
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
        # Transient projection of the one assistant message currently streaming.
        # It is cleared when that exact generation is archived into
        # ``_detail_messages``; it never becomes a second persisted transcript.
        self._detail_live_drafts: dict[
            tuple[str, str], tuple[int, str]
        ] = {}
        self._detail_cursor_line = 0
        self._detail_line_count = 1  # clamp detail cursor y — same class as task crash
        self._detail_scroll_syncing = False
        self._detail_scroll_handler: Callable[[MouseEvent], object] | None = None
        self._redraw_pending = False
        self._closing = False
        self._task_started_at: float | None = None
        self._quit_armed_until = 0.0
        self._feedback_text = ""
        self._feedback_kind = "info"
        # Plan approval host state. The page itself is shown only while the
        # canonical task state is ``awaiting_approval``.
        self._plan_ui_task_id: str | None = None
        self._plan_exit_event = threading.Event()
        self._plan_exit_outcome = "approved"
        self._plan_exit_feedback = ""
        # True while exit_plan_mode host fn is blocked on a/s/q.
        self._plan_exit_waiting = False
        # Grok-style todo plan badge + expandable list pane.
        self._todo_pane_open = False
        self._todo_scroll = 0  # first visible item index when list is long
        # Parallel fleet pane (independent of todo) — roster under turn status.
        self._fleet_pane_open = False
        self._fleet_scroll = 0  # first visible fleet row
        self._fleet_cursor = 0  # focused row within fleet entries
        self._pinned_agent_ref: tuple[str, str] | None = None  # (task_id, agent_id)
        self._fleet_badge_last_click: float | None = None
        # Worktree merge: first m arms confirm; second m within window lands.
        self._merge_confirm_ref: tuple[str, str] | None = None  # (task_id, agent_id)
        self._merge_confirm_until = 0.0
        self._merged_worktrees: set[str] = set()  # agent ids landed this session
        # Reading stream: DOWN+UP on the same task region selects it.
        self._task_mouse_down_index: int | None = None
        # FormattedTextControl mouse links consume the terminal's native drag
        # selection. Keep a native-feeling drag overlay: mouse-down only arms
        # it, movement starts it, mouse-up freezes it, Ctrl+C copies it.
        self._task_copy_pending: tuple[int, int] | None = None
        self._task_copy_dragging = False
        self._task_copy_anchor: tuple[int, int] | None = None
        self._task_copy_cursor: tuple[int, int] | None = None
        self._task_copy_text = ""
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
        self._pending_image_attachments: tuple[ImageAttachment, ...] = ()
        # One-shot startup brand (concept art). Dismissed forever on first task;
        # not "empty ledger" — finished tasks never bring the splash back.
        self._startup_brand = not bool(
            initial_prompt and str(initial_prompt).strip()
        )
        # before_render throttle + splash cache (ESC/modal close snappiness)
        self._last_sync_runtime_at = 0.0
        self._doggy_empty_cache: tuple[tuple[Any, ...], StyleAndTextTuples] | None = None
        # Full reading-stream cache: skip rebuilding unchanged transcript rows.
        self._task_paint_cache: tuple[Any, ...] | None = None
        # Conversation bodies start open. Collapsing only hides chat prose;
        # task title/status/agents remain visible as the task-oriented layer.
        self._collapsed_task_chats: set[str] = set()
        # Structural anchors inside the one continuous reading stream.
        self._task_anchor_lines: dict[str, int] = {}
        self._agent_anchor_lines: dict[tuple[str, str], int] = {}
        self._latest_task_tail_line = 0
        # Scroll-to-latest does not invent selection; it only follows the newest
        # visible conversation until the user scrolls upward.
        self._follow_latest_task = False
        # Live tool/activity lines from on_live_message (effect layer, not truth).
        self._activity = LiveActivityBoard()
        self._subagent_listener_bound = False
        self._session_listener_bound = False
        self._external_turn_views: dict[int, dict[str, Any]] = {}
        self._view_lock = threading.RLock()
        self._prompt_history = self._make_prompt_history()
        self._last_pasted_path: str | None = None
        self._win32_ctrl_v_down = False
        self._win32_ctrl_v_last_at = 0.0
        # A tool preview is a projection of one canonical transcript record.
        # Transient: Ctrl+move. Pinned: Ctrl+click. It is never a detail page.
        self._tool_hover_ref: tuple[str, str, str] | None = None
        self._tool_preview_ref: tuple[str, str, str] | None = None
        self._tool_preview_pinned = False
        self._tool_preview_buffer_key: tuple[Any, ...] | None = None

        self._task_control = FormattedTextControl(
            text=self._render_tasks,
            focusable=True,
            show_cursor=False,
            get_cursor_position=self._task_cursor_position,
        )
        self._wire_task_text_selection_control()
        self._task_window = ScrollableWindow(
            content=self._task_control,
            wrap_lines=False,  # line y == content row; wrap broke scroll/cursor map
            # This window owns wheel state. A hidden cursor must not pull a
            # manually selected viewport back during the next paint.
            wheel_handler=self._scroll_tasks,
            scroll_offsets=ScrollOffsets(top=0, bottom=0),
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
            # PT default is False: only the chrome (top/side/bottom) had our
            # focus mouse handlers, so clicks in the empty middle did nothing.
            focus_on_click=True,
            # A Window line prefix is repainted for every logical and wrapped
            # visual line. A TextArea ``prompt`` only covers the first line.
            get_line_prefix=self._render_prompt_line_prefix,
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
        # Plan tab body: read-only Buffer so the user can select/copy text.
        # NEVER dump full plan through FormattedText markdown (floods the TUI).
        self._plan_body_sync_key: tuple[Any, ...] | None = None
        self._plan_body_path: str = ""
        self._plan_chrome_control = FormattedTextControl(
            text=self._render_plan_chrome,
            focusable=False,
            show_cursor=False,
        )
        self._plan_body = TextArea(
            height=Dimension(weight=1, min=6),
            multiline=True,
            wrap_lines=True,
            scrollbar=True,
            read_only=True,
            focus_on_click=True,
            lexer=PlanMarkdownLexer(),
            style="class:plan.body",
        )
        self._detail_input = TextArea(
            height=self._detail_input_height,
            multiline=True,
            wrap_lines=True,
            scrollbar=True,
            dont_extend_height=True,
            focus_on_click=True,
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
                        "写下希望怎么改计划…",
                        style="class:detail.input.placeholder",
                    ),
                    Condition(
                        lambda: self._modal_kind == "plan"
                        and self._task_awaiting_plan_approval()
                        and (
                            not getattr(self, "_detail_input", None)
                            or not self._detail_input.text
                        )
                    ),
                ),
                ConditionalProcessor(
                    AfterInput(
                        "粘贴 Token / API Key…",
                        style="class:detail.input.placeholder",
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
                        style="class:detail.input.placeholder",
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
        self._tool_preview_body = TextArea(
            height=Dimension(weight=1, min=4),
            multiline=True,
            wrap_lines=True,
            scrollbar=True,
            read_only=True,
            focus_on_click=True,
            style="class:tool.preview.body",
        )
        self._wire_buffer_ctrl_click(self._input)
        self._wire_buffer_ctrl_click(self._detail_input)
        self._wire_buffer_outside_tool_preview(self._input)
        self._wire_buffer_outside_tool_preview(self._detail_input)

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
        fleet_pane = Window(
            FormattedTextControl(self._render_fleet_pane),
            height=self._fleet_pane_height,
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
            width=_MAIN_INPUT_RIGHT_COLS,
            # Match multi-line input height (no fixed height=1).
            style="class:root",
            dont_extend_width=True,
        )
        prompt_bottom = Window(
            FormattedTextControl(self._render_prompt_bottom),
            height=1,
            style="class:root",
        )
        shortcuts = ConditionalContainer(
            Window(
                FormattedTextControl(self._render_shortcuts),
                height=1,
                style="class:root",
            ),
            # A replacement page owns its own context-sensitive footer. Keeping the
            # global shortcut row underneath duplicates controls and creates two
            # competing visual bottoms.
            filter=Condition(lambda: not self._modal_open),
        )
        prompt_box = HSplit(
            [
                prompt_top,
                VSplit([self._input, prompt_right]),
                prompt_bottom,
            ],
            style="class:root",
        )
        main_page = ConditionalContainer(
            HSplit(
                [
                    self._task_window,
                    turn_status,
                    todo_pane,
                    fleet_pane,
                    Window(height=1, style="class:root"),
                    prompt_box,
                    shortcuts,
                ],
                style="class:root",
            ),
            filter=Condition(lambda: not self._detail_page_visible()),
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
        plan_panel = HSplit(
            [
                Window(
                    self._plan_chrome_control,
                    height=Dimension(min=2, max=8, preferred=4),
                    style="class:plan.chrome",
                ),
                self._plan_body,
            ],
            style="class:plan.surface",
        )
        modal_panel = HSplit(
            [
                modal_header,
                ConditionalContainer(
                    Window(
                        FormattedTextControl(self._render_modal_filters),
                        height=1,
                        style="class:agent-window",
                    ),
                    filter=Condition(lambda: self._modal_kind == "auth"),
                ),
                ConditionalContainer(
                    Window(height=1, char="─", style="class:separator"),
                    filter=Condition(lambda: self._modal_kind == "auth"),
                ),
                # Message/tool: FormattedText. Plan: selectable TextArea (no MD flood).
                ConditionalContainer(
                    self._detail_window,
                    filter=Condition(lambda: self._modal_kind == "auth"),
                ),
                ConditionalContainer(
                    plan_panel,
                    filter=Condition(self._plan_body_visible),
                ),
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
                    Window(height=1, style="class:agent-window"),
                    VSplit(
                        [
                            Window(
                                width=2,
                                style="class:agent-window",
                            ),
                            modal_panel,
                            Window(
                                width=2,
                                style="class:agent-window",
                            ),
                        ],
                        style="class:agent-window",
                    ),
                    Window(height=1, style="class:agent-window"),
                ],
                style="class:agent-window",
            ),
            # Detail/auth replace the main page below the persistent app header.
            # They are not overlays: hidden main chrome must never leak around
            # the page as disconnected rules or prompt-border fragments.
            filter=Condition(self._detail_page_visible),
        )
        body_inner = HSplit(
            [
                header,
                separator,
                main_page,
                modal_content,
            ],
            style="class:root",
        )
        # Outer breathing room — never paint chrome flush to the terminal edge.
        # Keep in sync with _EDGE_PAD_X / _EDGE_PAD_Y and _content_width().
        body = HSplit(
            [
                Window(height=_EDGE_PAD_Y, style="class:root"),
                VSplit(
                    [
                        Window(width=_EDGE_PAD_X, style="class:root"),
                        body_inner,
                        Window(width=_EDGE_PAD_X, style="class:root"),
                    ],
                    style="class:root",
                ),
                Window(height=_EDGE_PAD_Y, style="class:root"),
            ],
            style="class:root",
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
        tool_preview = ConditionalContainer(
            HSplit(
                [
                    Window(
                        FormattedTextControl(self._render_tool_preview_header),
                        height=2,
                        style="class:tool.preview.header",
                    ),
                    Window(
                        height=1,
                        char="─",
                        style="class:tool.preview.rule",
                    ),
                    self._tool_preview_body,
                    Window(
                        FormattedTextControl(self._render_tool_preview_footer),
                        height=1,
                        style="class:tool.preview.footer",
                    ),
                ],
                style="class:tool.preview",
            ),
            filter=Condition(self._tool_preview_visible),
        )
        root = FloatContainer(
            content=body,
            floats=[
                # Tool inspection stays over the reading stream. It is
                # transient on Ctrl+move and becomes selectable on Ctrl+click.
                Float(
                    top=3,
                    bottom=6,
                    left=6,
                    right=6,
                    content=tool_preview,
                    transparent=False,
                    z_index=30,
                ),
                # Questionnaire: compact bordered float (not full-screen).
                Float(
                    top=5,
                    bottom=9,
                    left=12,
                    right=12,
                    content=ask_dialog,
                    transparent=False,
                    z_index=20,
                ),
                Float(
                    bottom=8,
                    left=4,
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
            detail_key = (task_id, main_id)
            messages: list[Any] = []
            streamed: list[str] = []
            stream_state: dict[str, Any] = {}
            callback_state = {"active": True}
            prior_live = metadata.get("on_live_message")
            prior_delta = metadata.get("on_sample_delta")

            def on_live_message(message: Any) -> None:
                if not callback_state["active"] or self._closing:
                    return
                archived_generation: int | None = None
                if _message_role_value(message) == "assistant":
                    archived_generation = int(
                        stream_state.get("draft_generation", 0) or 0
                    )
                    stream_state["draft"] = ""
                    stream_state["draft_generation"] = archived_generation + 1
                # Track tools for status chips only — never paint tool noise on
                # the lightweight agent summary; tools remain canonical messages.
                self._activity.observe(task_id, main_id, message)
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
                    if archived_generation is not None:
                        self._clear_detail_live_draft(
                            detail_key, archived_generation
                        )
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
                # Card cover = message stream only (tools already under 工具 tab).
                if should_emit:
                    self.ledger.update_live_agent(
                        task_id,
                        main_id,
                        label="MAIN",
                        status="running",
                        output=preview,
                    )
                    self._set_detail_live_draft(
                        detail_key,
                        int(stream_state.get("draft_generation", 0) or 0),
                        str(stream_state.get("draft", "") or ""),
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
                self._detail_messages[detail_key] = messages
                self._activity.clear_task(task_id)
                self._subagent_baselines[task_id] = {
                    item.subagent_id for item in self._subagents()
                }
                self._dismiss_startup_brand()
                self._follow_latest_task = True
                self._pinned_task_for_line = None
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
            # Terminal state is already visible in the task header.
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

        # Tool and prose updates share the same canonical live message list.
        if message is not None:
            self._activity.observe(task_id, sub_id, message)
        elif live:
            self._activity.rebuild(task_id, sub_id, list(live))

        prose = ""
        if live:
            prose = agent_summary_text_from_messages(list(live))
        output = prose or subagent_text(snap)
        if _is_tool_activity_line(output):
            output = prose or ""
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
        ask_modal = Condition(
            lambda: bool(self._ask_active and self._modal_kind == "ask")
        )
        tasks_focused = Condition(
            lambda: not self._modal_open
            and not self._ask_active
            and get_app().layout.has_focus(self._task_window)
        )
        auth_list_focused = Condition(
            lambda: self._modal_open
            and self._modal_kind == "auth"
            and self._auth_wizard.step != WizardStep.PASTE
            and get_app().layout.has_focus(self._detail_window)
        )
        # Main prompt or detail interject (not auth token paste).
        prompt_paste = Condition(
            lambda: (
                get_app().layout.has_focus(self._input)
                or (
                    self._modal_open
                    and self._modal_kind == "plan"
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
            """Clipboard: image → chip, else text / image-file path."""
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

        # Tab = task cycle. Plan/Auto is agent-driven (enter/exit_plan_mode), not S-Tab.
        tab_tasks_ok = Condition(
            lambda: not (
                self._ask_active
                or (self._modal_open and self._modal_kind in {"auth", "ask"})
            )
        )

        @keys.add("tab", filter=tab_tasks_ok, eager=True)
        def _tab_to_tasks(_: Any) -> None:
            self._tab_task_cycle()

        # Ctrl+Space / Windows NUL (c-@): same as Tab cycle.
        @keys.add("c-space", filter=tab_tasks_ok, eager=True)
        @keys.add("c-@", filter=tab_tasks_ok, eager=True)
        def _ctrl_space_tasks(_: Any) -> None:
            self._tab_task_cycle()

        # Todo list open: ↑↓ scroll the checklist, not the transcript.
        todo_pane_nav = Condition(
            lambda: self._todo_pane_open
            and not self._fleet_pane_open
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

        # Fleet roster open: ↑↓ select agent · Enter open · p pin · P pinned open.
        fleet_pane_nav = Condition(
            lambda: self._fleet_pane_open
            and not self._modal_open
            and not get_app().layout.has_focus(self._input)
        )

        @keys.add("up", filter=fleet_pane_nav, eager=True)
        def _fleet_up(_: Any) -> None:
            self._move_fleet_cursor(-1)

        @keys.add("down", filter=fleet_pane_nav, eager=True)
        def _fleet_down(_: Any) -> None:
            self._move_fleet_cursor(1)

        @keys.add("pageup", filter=fleet_pane_nav, eager=True)
        def _fleet_page_up(_: Any) -> None:
            self._move_fleet_cursor(-self._FLEET_PANE_VISIBLE)

        @keys.add("pagedown", filter=fleet_pane_nav, eager=True)
        def _fleet_page_down(_: Any) -> None:
            self._move_fleet_cursor(self._FLEET_PANE_VISIBLE)

        @keys.add("enter", filter=fleet_pane_nav, eager=True)
        def _fleet_open(_: Any) -> None:
            self._open_fleet_cursor()

        @keys.add("p", filter=fleet_pane_nav, eager=True)
        def _fleet_pin(_: Any) -> None:
            self._pin_fleet_cursor()

        @keys.add("P", filter=fleet_pane_nav, eager=True)
        def _fleet_open_pinned(_: Any) -> None:
            self._open_pinned_agent()

        @keys.add("m", filter=fleet_pane_nav, eager=True)
        def _fleet_merge(_: Any) -> None:
            self._merge_fleet_cursor()

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

        @keys.add("enter", filter=tasks_focused)
        def _open_selected(_: Any) -> None:
            self._open_selected_task()

        task_text_selected = Condition(
            lambda: bool(self._task_copy_text) and not self._ask_active
        )

        @keys.add(
            "c-c",
            filter=~modal & ~has_selection & task_text_selected,
            eager=True,
        )
        def _copy_selected_task_text(_: Any) -> None:
            self._copy_task_text_to_clipboard(self._task_copy_text)

        @keys.add("up", filter=tasks_focused)
        def _scroll_transcript_up(_: Any) -> None:
            self._scroll_tasks(-1)

        @keys.add("down", filter=tasks_focused)
        def _scroll_transcript_down(_: Any) -> None:
            self._scroll_tasks(1)

        @keys.add("space", filter=tasks_focused)
        def _focus_prompt(event: Any) -> None:
            event.app.layout.focus(self._input)
            # Keep _task_selection_active so Tab returns to the same task
            # (chrome dims while input is focused, but selection is sticky).
            event.app.invalidate()

        @keys.add("pageup", filter=tasks_focused)
        def _tasks_page_up(_: Any) -> None:
            self._scroll_tasks(-max(4, _terminal_height() - 10))

        @keys.add("pagedown", filter=tasks_focused)
        def _tasks_page_down(_: Any) -> None:
            self._scroll_tasks(max(4, _terminal_height() - 10))

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

        @keys.add("escape", eager=True)
        def _escape(event: Any) -> None:
            if self._tool_preview_ref is not None:
                self._tool_hover_ref = None
                self._close_tool_preview(restore_focus=True)
                return
            # Esc = cancel running task only (not leave UI layers — use Tab).
            # eager=True: do not wait for ambiguous CSI sequences after Esc
            # when we only need a plain cancel (avoids "delayed" multi-cancel).
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

        # Approval keys exist only on the visible, canonical approval page.
        plan_review = Condition(
            lambda: self._plan_body_visible()
            and not self.app.layout.has_focus(self._detail_input)
        )
        plan_tab = Condition(self._plan_body_visible)

        @keys.add("a", filter=plan_review, eager=True)
        def _plan_approve(_: Any) -> None:
            self._resolve_plan_exit("approved")

        @keys.add("s", filter=plan_review, eager=True)
        def _plan_revise(_: Any) -> None:
            self._resolve_plan_exit("revise")

        @keys.add("q", filter=plan_review, eager=True)
        def _plan_quit(_: Any) -> None:
            self._resolve_plan_exit("abandoned")

        @keys.add("c-o", filter=plan_tab, eager=True)
        def _plan_open_os(_: Any) -> None:
            self._open_current_plan_in_os()
            self.app.invalidate()

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
        """Paste whatever the clipboard holds: image chip, else text / file path.

        Preference matches user expectation — last copied image → chip; last
        copied text → text. Windows Terminal may swallow Ctrl+V for images;
        ``_poll_win32_ctrl_v_image_paste`` covers that path.
        """
        buffer = event.current_buffer
        if buffer.selection_state is not None:
            buffer.cut_selection()
        cwd = getattr(self.session, "cwd", None) or Path.cwd()

        # Debounce against the Win32 Ctrl+V poll (same physical keypress).
        now = time.monotonic()
        if now - float(self._win32_ctrl_v_last_at) < _WIN32_CTRL_V_DEBOUNCE_S:
            return

        saved: Path | None = None
        try:
            saved = save_clipboard_image(cwd)
        except Exception:  # noqa: BLE001
            saved = None

        if saved is None:
            text_probe: str | None = None
            try:
                data = event.app.clipboard.get_data()
                raw = getattr(data, "text", None) if data is not None else None
                if isinstance(raw, str) and raw:
                    text_probe = raw
            except Exception:  # noqa: BLE001
                text_probe = None
            if not text_probe:
                try:
                    text_probe = get_system_clipboard_text()
                except Exception:  # noqa: BLE001
                    text_probe = None
            path_hit = coerce_image_path_text(text_probe, cwd=cwd)
            if path_hit is not None:
                saved = path_hit

        if saved is not None:
            self._win32_ctrl_v_last_at = now
            self._insert_image_chip(buffer, saved, cwd=cwd)
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
            self._set_feedback("剪贴板是空的", "warning")
        event.app.invalidate()

    def _insert_image_chip(
        self, buffer: Any, saved: Path | str, *, cwd: Path | str
    ) -> None:
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
            f"已粘贴{VIEW_IMAGE_LABEL}",
            "info",
        )

    def _image_attachments_for_prompt(
        self,
        prompt: str,
    ) -> tuple[ImageAttachment, ...]:
        """Resolve the image chips still present when the prompt is submitted."""
        cwd = getattr(self.session, "cwd", None) or Path.cwd()
        attachments: list[ImageAttachment] = []
        seen: set[str] = set()
        for raw_path in extract_image_chip_paths(prompt):
            resolved = resolve_openable_path(raw_path, cwd=cwd)
            if resolved is None:
                raise AttachmentError(f"file not found: {raw_path}")
            key = str(resolved).lower()
            if key in seen:
                continue
            seen.add(key)
            attachments.append(ImageAttachment.from_path(resolved))
        return tuple(attachments)

    def _accept_prompt(self, buffer: Any) -> bool:
        prompt = buffer.text.strip()
        if not prompt:
            buffer.text = ""
            return True
        try:
            image_attachments = self._image_attachments_for_prompt(prompt)
        except AttachmentError as exc:
            self._set_feedback(f"图片无法发送 · {exc}", "warning")
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
            # Dying / cancelled turn: do not interject into it (would look like
            # the next prompt was swallowed or cancelled with the old one).
            if tid and self._cancelling_task_id == tid:
                self._set_feedback("正在取消上一个任务 · 稍后再发", "warning")
                self._input.text = prompt
                self.app.invalidate()
                return True
            self.session.interject(
                strip_image_chips(prompt),
                prompt_id=tid,
                attachments=image_attachments,
            )
            if tid:
                self._note_interject(tid, prompt)
            # Interject flash lives in the active task header.
            self.app.invalidate()
            return True
        if not self._ensure_auth_ready():
            self._pending_prompt = prompt
            self._pending_image_attachments = image_attachments
            self._open_auth_wizard()
            # Keep text in the box so the user can edit; send resumes after login.
            self._input.text = prompt
            preview = _truncate_display(prompt.replace("\n", " "), 36)
            self._set_feedback(f"先登录 · 将发送：{preview}", "warning")
            return True
        self._start_task(prompt, image_attachments=image_attachments)
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
        if self._modal_kind == "plan":
            if buffer.text.strip():
                self._resolve_plan_exit("revise")
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
        # Stay in the interject box while the turn is live so the next note
        # does not fight stream redraws / forced detail-body focus.
        try:
            if self._detail_input_visible():
                self.app.layout.focus(self._detail_input)
            else:
                self.app.layout.focus(self._detail_window)
        except Exception:  # noqa: BLE001
            pass
        self.app.invalidate()
        return True

    def _note_interject(self, task_id: str, text: str) -> None:
        """Show the homepage 插入中 pulse until it expires."""
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

    def _start_task(
        self,
        prompt: str,
        *,
        image_attachments: tuple[ImageAttachment, ...] = (),
    ) -> None:
        self._dismiss_startup_brand()
        self._bind_subagent_listener()
        task = self.ledger.create(prompt)
        # Keep keyboard on the composer; the answer streams above it.
        self._active_task_id = task.id
        # New task is not mid-cancel; keep _cancel_grace_until so a lagged Esc
        # from the previous cancel cannot instantly kill this turn.
        self._cancelling_task_id = None
        self._detail_messages[(task.id, f"{task.id}:main")] = []
        self._activity.clear_task(task.id)
        self._task_started_at = time.monotonic()
        self._follow_latest_task = True
        self._pinned_task_for_line = None
        self._subagent_baselines[task.id] = {
            item.subagent_id for item in self._subagents()
        }
        self._clear_feedback()
        worker = threading.Thread(
            target=self._run_task,
            args=(
                task.id,
                strip_image_chips(prompt) if image_attachments else prompt,
                image_attachments,
            ),
            name=f"codedoggy-{task.id}",
            daemon=True,
        )
        self._worker = worker
        worker.start()
        # Do not yank focus away from a task or modal the user selected.
        try:
            layout = self.app.layout
            if self._modal_open:
                pass
            elif (
                layout.has_focus(self._task_window)
                and self._task_selection_active
                and self._selected_task >= 0
            ):
                pass
            else:
                layout.focus(self._input)
        except Exception:  # noqa: BLE001
            pass
        self.app.invalidate()

    def _dismiss_startup_brand(self) -> None:
        """Hide the launch splash for the rest of this process."""
        self._startup_brand = False

    def _showing_startup_brand(self) -> bool:
        return self._startup_brand and not self.ledger.snapshots()

    def _run_task(
        self,
        task_id: str,
        prompt: str,
        image_attachments: tuple[ImageAttachment, ...] = (),
    ) -> None:
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
            archived_generation: int | None = None
            if _message_role_value(message) == "assistant":
                archived_generation = int(
                    stream_state.get("draft_generation", 0) or 0
                )
                stream_state["draft"] = ""
                stream_state["draft_generation"] = archived_generation + 1
            # Tools stay on the activity board / detail 工具 tab — not the cover.
            self._activity.observe(task_id, main_agent_id, message)
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
                if archived_generation is not None:
                    self._clear_detail_live_draft(
                        detail_key, archived_generation
                    )
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
            # Cover description = assistant message stream only.
            if should_emit:
                self.ledger.update_live_agent(
                    task_id,
                    main_agent_id,
                    label="MAIN",
                    status="running",
                    output=preview,
                )
                self._set_detail_live_draft(
                    detail_key,
                    int(stream_state.get("draft_generation", 0) or 0),
                    str(stream_state.get("draft", "") or ""),
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
                attachments=image_attachments,
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
                # Task header already shows completion + duration.
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
                if self._cancelling_task_id == task_id:
                    # Keep grace_until; only clear the mid-cancel marker.
                    self._cancelling_task_id = None
                self._flush_pending_reload()
                # Never auto-focus task list after a turn ends — leave input alone.
                self._invalidate_safe()

            self._call_in_ui_thread(apply_finish)

    def _before_render(self) -> None:
        """prompt_toolkit before_render hook — keep off the hot path when idle."""
        # Windows mouse events do not fire when only the Ctrl key changes.
        # Poll the real key state so both orders work:
        # Ctrl→move and move→Ctrl. Pinned previews ignore key release.
        if sys.platform == "win32" and not self._tool_preview_pinned:
            control_held = _mouse_control_held()
            if control_held and self._tool_hover_ref is not None:
                if self._tool_preview_ref != self._tool_hover_ref:
                    self._show_tool_preview(
                        self._tool_hover_ref,
                        pinned=False,
                    )
            elif self._tool_preview_ref is not None:
                self._close_tool_preview(restore_focus=False)
        if self._tool_preview_ref is not None:
            try:
                self._sync_tool_preview_buffer()
            except Exception:  # noqa: BLE001
                self._close_tool_preview(restore_focus=False)
        # The approval page is a state surface, not a plan archive.
        if (
            self._modal_open
            and self._modal_kind == "plan"
            and not self._task_awaiting_plan_approval()
        ):
            self._leave_plan_approval()
        # Refresh selectable plan buffer when disk mtime changes (no FormattedText).
        if self._plan_body_visible():
            try:
                self._sync_plan_body_buffer(force=False)
            except Exception:  # noqa: BLE001
                pass
        # Fixed 30fps (GrokBuild animation.fps default). No dynamic downclock.
        try:
            self.app.refresh_interval = 1.0 / 30.0
        except Exception:  # noqa: BLE001
            pass
        # Goal mode is exclusive of plan approval chrome.
        self._clear_plan_approval_if_goal()
        # Interactive detail scrollbar only touches vertical_scroll;
        # re-anchor the detail cursor so get_cursor_position cannot snap back.
        self._sync_detail_scroll_from_window()
        # Live turn may hide the interject box when status flips terminal —
        # re-land focus before the next paint so stream redraws stay stable.
        self._land_focus_if_detail_input_gone()
        # WT steals Ctrl+V: when the chord is held and clipboard is an image,
        # paste the chip ourselves (text paste still comes from the terminal).
        self._poll_win32_ctrl_v_image_paste()
        self._sync_runtime()
        self._sync_task_follow_scroll()

    def _prompt_buffer_for_paste(self) -> Any | None:
        """Active prompt buffer (main or detail interject), or None."""
        try:
            layout = self.app.layout
            if layout.has_focus(self._input):
                return self._input.buffer
            if (
                self._modal_open
                and self._modal_kind == "plan"
                and layout.has_focus(self._detail_input)
            ):
                return self._detail_input.buffer
        except Exception:  # noqa: BLE001
            return None
        return None

    def _poll_win32_ctrl_v_image_paste(self) -> None:
        """Catch Ctrl+V when the host terminal swallows the key for text paste.

        Only acts when the clipboard holds an image — so plain text Ctrl+V is
        still handled by Windows Terminal / our ``c-v`` binding, never doubled.
        """
        if sys.platform != "win32":
            return
        buffer = self._prompt_buffer_for_paste()
        if buffer is None:
            self._win32_ctrl_v_down = False
            return
        try:
            import ctypes

            # VK_CONTROL=0x11, VK_V=0x56; high bit ⇒ currently down.
            ctrl = bool(ctypes.windll.user32.GetAsyncKeyState(0x11) & 0x8000)
            v_key = bool(ctypes.windll.user32.GetAsyncKeyState(0x56) & 0x8000)
        except Exception:  # noqa: BLE001
            return
        chord = ctrl and v_key
        was = bool(self._win32_ctrl_v_down)
        self._win32_ctrl_v_down = chord
        if not chord or was:
            return
        now = time.monotonic()
        if now - float(self._win32_ctrl_v_last_at) < _WIN32_CTRL_V_DEBOUNCE_S:
            return
        cwd = getattr(self.session, "cwd", None) or Path.cwd()
        try:
            saved = save_clipboard_image(cwd)
        except Exception:  # noqa: BLE001
            saved = None
        if saved is None:
            return
        self._win32_ctrl_v_last_at = now
        if buffer.selection_state is not None:
            buffer.cut_selection()
        self._insert_image_chip(buffer, saved, cwd=cwd)
        self._invalidate_safe()

    def _clear_plan_approval_if_goal(self) -> None:
        """Goal mode is exclusive: close and abandon a pending approval."""
        if not (
            self._plan_exit_waiting
            or (self._modal_open and self._modal_kind == "plan")
        ):
            return
        kernel = getattr(self.session.extensions, "kernel", None)
        state = getattr(kernel, "session_mode_state", None) if kernel else None
        if state is None or not getattr(state, "is_goal", lambda: False)():
            return
        self._plan_exit_outcome = "abandoned"
        self._plan_exit_waiting = False
        if self._modal_open and self._modal_kind == "plan":
            self._leave_plan_approval()
        try:
            self._plan_exit_event.set()
        except Exception:  # noqa: BLE001
            pass

    def _sync_task_plan_with_session(self) -> None:
        """Mirror session Plan state onto the active task boundary.

        - Session plan active → task plan_state at least ``planning``
        - Session awaiting approval → ``awaiting_approval``
        - Session left plan / task terminal → clear stale planning state
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
            if task.plan_state in {"planning", "awaiting_approval"}:
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
            if task.plan_state in {"none", ""}:
                self.ledger.set_plan_state(
                    tid, "planning", plan_file=plan_file or None
                )
            return
        # Session no longer in plan UI: clear stale draft chrome if we already
        # moved into real work (or never needed plan chrome).
        if task.plan_state == "planning" and task.phase not in {
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
        # Approval page open: throttle hard — full subagent + plan sync every
        # paint was freezing the UI when memory tools flooded the transcript.
        if self._modal_open and self._modal_kind == "plan":
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
                    # Rebuild tool activity for detail; keep cover on prose only.
                    self._activity.rebuild(task_id, snap.subagent_id, msgs)
                    if status in {"pending", "running"}:
                        prose = agent_summary_text_from_messages(msgs)
                        if prose:
                            self.ledger.apply_agent_status(
                                task_id,
                                snap.subagent_id,
                                label=label,
                                status=status,
                                output=prose,
                                description=description,
                            )

        # Keep task plan_state aligned with the canonical session mode.
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
                    end_status = "failed" if failed else "completed"
                    self.ledger.finish_task(task.id, end_status)
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
        """Cancel the active task once — ignore Esc key-repeat and delayed Esc.

        GrokBuild lesson: a cancelled turn must not bleed into the next prompt.
        Multiple Esc presses while the worker is still winding down used to
        re-enter cancel after the user already started the next task.
        """
        now = time.monotonic()
        # Grace after a recent cancel: drop buffered / repeated Esc.
        if now < float(self._cancel_grace_until or 0.0):
            return
        if not self._is_running():
            return
        task_id = self._active_task_id
        if not task_id:
            return
        # Already cancelling this exact task → swallow (repeat while winding down).
        if self._cancelling_task_id == task_id:
            return

        self._cancelling_task_id = task_id
        # ~0.8s covers terminal Esc lag + key-repeat after cancel feedback.
        self._cancel_grace_until = now + 0.8
        self.session.cancel()
        coordinator = self._subagent_coordinator()
        if coordinator is not None:
            for subagent_id, owner_task_id in list(self._subagent_task.items()):
                if owner_task_id != task_id:
                    continue
                try:
                    coordinator.cancel(subagent_id)
                except Exception:  # noqa: BLE001
                    pass
        self.ledger.set_task_status(task_id, "cancelled")
        self.ledger.set_task_phase(task_id, "cancelled")
        task = next(
            (item for item in self.ledger.snapshots() if item.id == task_id),
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
                task_id,
                str(aid),
                label=str(getattr(agent, "label", "") or "AGENT"),
                status="cancelled",
            )
        self._set_feedback("已取消当前任务", "info")
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
        width = max(1, _content_width())
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
            if event.event_type is MouseEventType.SCROLL_UP:
                self._scroll_tasks(-3)
                return None
            if event.event_type is MouseEventType.SCROLL_DOWN:
                self._scroll_tasks(3)
                return None
            if event.event_type is MouseEventType.MOUSE_MOVE:
                self._tool_hover_ref = None
                if not self._tool_preview_pinned:
                    self._close_tool_preview(restore_focus=False)
                return NotImplemented
            if event.event_type not in {
                MouseEventType.MOUSE_DOWN,
                MouseEventType.MOUSE_UP,
            }:
                return NotImplemented
            if self._modal_open or self._ask_active:
                return None
            self._close_tool_preview(restore_focus=False)
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

    def _render_prompt_line_prefix(
        self,
        line_number: int,
        wrap_count: int,
    ) -> StyleAndTextTuples:
        """Compact composer marker for every logical and wrapped input row."""
        first_visual_row = line_number == 0 and wrap_count == 0
        if first_visual_row:
            return self._with_input_focus_mouse(
                [("class:input", "  "), ("class:prompt", "› ")]
            )
        return self._with_input_focus_mouse([("class:input", "    ")])

    def _render_prompt_top(self) -> StyleAndTextTuples:
        width = max(16, _content_width())
        border = self._prompt_border_class()
        rule_width = max(1, width - 2)
        return self._with_input_focus_mouse(
            [(border, "  " + "━" * rule_width)]
        )

    def _render_prompt_right(self) -> StyleAndTextTuples:
        # One breathing cell per visual row. No right-hand frame.
        rows = 1
        try:
            rows = max(
                1,
                min(
                    _INPUT_MAX_LINES,
                    self._estimate_buffer_display_lines(
                        self._input.buffer.text,
                        available_cols=self._main_input_text_width(),
                    ),
                ),
            )
        except Exception:  # noqa: BLE001
            rows = 1
        fragments: StyleAndTextTuples = []
        for i in range(rows):
            fragments.append(("class:input", " "))
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
                self._input.buffer.text,
                available_cols=self._main_input_text_width(),
                max_lines=_INPUT_MAX_LINES,
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
        available_cols: int | None = None,
        max_lines: int | None = None,
    ) -> int:
        """Count soft-wrapped display rows for dynamic TextArea height."""
        cap = max_lines if max_lines is not None else _INPUT_MAX_LINES
        raw = text or ""
        if available_cols is None:
            # Detail input uses a different surface; retain its conservative
            # budget unless the caller supplies an exact viewport contract.
            avail = max(8, _content_width() - prefix_cols - 6)
        else:
            avail = max(8, int(available_cols))
        total = 0
        parts = raw.split("\n") if raw else [""]
        for part in parts:
            w = get_cwidth(part)
            if w <= 0:
                total += 1
            else:
                total += max(1, (w + avail - 1) // avail)
        return max(1, min(cap, total))

    @staticmethod
    def _main_input_text_width() -> int:
        """Exact text columns inside composer prefix, scrollbar and right rail."""
        return max(
            8,
            _content_width()
            - _MAIN_INPUT_PREFIX_COLS
            - _MAIN_INPUT_SCROLLBAR_COLS
            - _MAIN_INPUT_RIGHT_COLS,
        )

    def _render_prompt_bottom(self) -> StyleAndTextTuples:
        width = max(16, _content_width())
        caption_text = _truncate_display(
            session_surface.model_and_mode_text(self.session), width - 4
        )
        caption = f"  {caption_text}"
        fill = max(0, width - get_cwidth(caption))
        return self._with_input_focus_mouse(
            [
                ("class:prompt.caption", " " * fill + caption),
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
        elif self._modal_open and self._modal_kind == "plan":
            items = [
                ("a", "批准", "noop", False),
                ("s", "修改", "noop", False),
                ("q", "放弃", "noop", False),
                ("Tab", "返回正文", "tasks", False),
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
            if self._fleet_pane_open and not input_focused:
                items = [
                    ("↑↓", "选择", "noop", False),
                    ("Enter", "打开", "noop", False),
                    ("p", "钉住", "noop", False),
                    ("m", "合入", "noop", False),
                    ("Tab", "进入", "open", False),
                ]
                if self._merge_confirm_active():
                    items = [
                        ("m", "确认合入", "noop", False),
                        ("↑↓", "取消武装", "noop", False),
                    ]
                if self._is_running():
                    items.insert(0, ("Esc", "取消任务", "cancel", False))
                items.append(("Ctrl+Q", "退出", "quit", True))
            elif input_focused:
                items = [
                    ("Tab", "最新任务", "tasks", False),
                    ("^Enter", "换行", "noop", False),
                    ("Ctrl+L", "登录", "login", False),
                ]
                if self._is_running():
                    items.insert(0, ("Esc", "取消任务", "cancel", False))
                items.append(("Ctrl+Q", "退出", "quit", True))
            else:
                # Reading stream: no detail layer; Tab/Space return to composer.
                items = [
                    ("Space", "输入", "input", False),
                    ("Tab", "输入", "input", False),
                    ("Ctrl+移入", "工具预览", "noop", False),
                ]
                if self._is_running():
                    items.insert(0, ("Esc", "取消任务", "cancel", False))
                items.append(("Ctrl+L", "登录", "login", False))
                items.append(("Ctrl+Q", "退出", "quit", True))
        return self._fit_shortcuts(items, max(20, _content_width() - 4))

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
            elif action == "tasks":
                self._tab_task_cycle()
            elif action == "open":
                self._tab_task_cycle()
            elif action == "input":
                self.app.layout.focus(self._input)
            elif action == "prompt":
                self.app.layout.focus(self._input)
            self.app.invalidate()

        return self._only_mouse_up(_on_up, scroll_target="tasks")

    def _stop_mouse(self, event: MouseEvent) -> object:
        if event.event_type is not MouseEventType.MOUSE_UP:
            return NotImplemented
        self._cancel_current()
        return None

    def _render_header(self) -> StyleAndTextTuples:
        width = max(1, _content_width())
        left = "  DOGGY"
        right = session_surface.budget_text(self.session)
        badge = self._todo_badge_label()
        fleet = self._fleet_badge_label()
        if width < get_cwidth(left):
            shown = _truncate_display(left, width)
            dog_width = min(3, len(shown))
            return [
                ("class:header", shown[:2]),
                ("class:brand.dog", shown[2 : 2 + dog_width]),
                ("class:brand", shown[2 + dog_width :]),
            ]

        fragments: StyleAndTextTuples = [
            ("class:header", "  "),
            ("class:brand.dog", "DOG"),
            ("class:brand", "GY"),
        ]
        badge_handler = self._todo_badge_mouse() if badge else None
        badge_style = (
            "class:todo.badge.open" if self._todo_pane_open else "class:todo.badge"
        )
        fleet_handler = self._fleet_badge_mouse() if fleet else None
        fleet_style = (
            "class:fleet.badge.open" if self._fleet_pane_open else "class:fleet.badge"
        )
        # Layout: brand · [计划 n/m ✓] · [并行 n/m] · gap · budget
        used = get_cwidth(left)
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
        if fleet:
            fmid = f"  并行 {fleet}" if width >= 44 else f"  {fleet}"
            if used + get_cwidth(fmid) + 2 <= width:
                fragments.append((fleet_style, fmid, fleet_handler))
                used += get_cwidth(fmid)
            else:
                fmid = f"  {fleet}"
                if used + get_cwidth(fmid) + 2 <= width:
                    fragments.append((fleet_style, fmid, fleet_handler))
                    used += get_cwidth(fmid)
        if not right or used + get_cwidth(right) + 2 > width:
            return fragments
        gap = width - used - get_cwidth(right) - 1
        fragments.append(("class:meta", " " * max(1, gap) + right + " "))
        return fragments

    def _render_street_hud(self) -> StyleAndTextTuples:
        """Compact auth surface. Connection facts lead; decoration stays quiet."""
        width = 44
        snap = session_surface.hud_projection(self.session)
        frame = int(time.monotonic() * 4)
        pulse = frame % 2
        open_handler = self._hud_open_mouse

        def line(style: str, text: str) -> StyleAndTextTuples:
            shown = _truncate_display(text, width)
            padded = shown + " " * max(0, width - get_cwidth(shown))
            return [(style, padded, open_handler)]

        title_style = "class:hud.title"
        ok_style = "class:hud.ok"
        warn_style = "class:hud.warn"
        accent = "class:hud.accent"
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
        fragments.extend(line(title_style, "  登录与连接"))
        fragments.append((bg, "\n", open_handler))

        mid1 = _truncate_display(f"  {status_word} · {now_label}", width)
        fragments.append((status_style, mid1, open_handler))
        fragments.append(
            (
                accent if cur_ok else dim,
                " " * max(0, width - get_cwidth(mid1)),
                open_handler,
            )
        )
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
        mid2 = _truncate_display("  " + "  ".join(bits), width)
        fragments.extend(line(accent, mid2))
        fragments.append((bg, "\n", open_handler))

        action = "  Enter/单击打开登录向导 · Ctrl+L"
        fragments.extend(line(dim, action))
        fragments.append((bg, "\n", open_handler))

        fragments.extend(line(bg, ""))
        return fragments

    def _render_header_rule(self) -> StyleAndTextTuples:
        width = max(1, _content_width())
        return [("class:header.rule", "─" * width)]

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

    def _todo_counts(self) -> Any:
        from codedoggy.tools.grok_build.todo_logic import count_todos

        return count_todos(self._session_todo_state())

    def _todo_badge_label(self) -> str | None:
        counts = self._todo_counts()
        return counts.badge_text()

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
            # One bottom drawer at a time so ↑↓ ownership is unambiguous.
            self._fleet_pane_open = False
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
            now = time.monotonic()
            # Ctrl+left or double-click → open MAIN 计划 tab.
            if _mouse_control_held(event):
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
        """Open approval only when a revision is actually awaiting a decision."""
        self._focus_active_or_latest_task()
        task = self._selected_task_view()
        if task is None:
            self._set_feedback("暂无任务可打开", "warning")
            return
        if task.plan_state == "awaiting_approval":
            self._open_plan_approval(task.id)
            return
        self._toggle_todo_pane()

    def _todo_pane_mouse(self) -> Callable[[MouseEvent], object]:
        """Wheel scrolls the checklist; click no-ops (use badge to close)."""
        return self._only_mouse_up(lambda _e: None, scroll_target="todo")

    def _render_todo_pane(self) -> StyleAndTextTuples:
        """Expandable checklist as one quiet surface, without an ASCII box."""
        if not self._todo_pane_open:
            return []
        state = self._session_todo_state()
        width = max(12, _content_width())
        if state is None or getattr(state, "is_empty", lambda: True)():
            self._todo_pane_open = False
            return []

        wheel = self._todo_pane_mouse()
        counts = self._todo_counts()
        badge = counts.badge_text() or "0/0"
        title = f"  计划 {badge}"
        hint = "滚轮滚动 · 顶部计划可关闭"
        if get_cwidth(title) + get_cwidth(hint) + 4 <= width:
            pad = width - get_cwidth(title) - get_cwidth(hint) - 2
            top = title + " " * pad + hint + "  "
        else:
            top = _truncate_display(title, width)
            top += " " * max(0, width - get_cwidth(top))
        fragments: StyleAndTextTuples = [
            ("class:todo.pane.title", top + "\n", wheel),
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
            body = _truncate_display(
                f"    {icon} {item.content or _tid}", max(4, width - 2)
            )
            pad_r = max(0, width - get_cwidth(body))
            fragments.append((style, body + " " * pad_r + "\n", wheel))
        if max_scroll > 0:
            lo = self._todo_scroll + 1
            hi = self._todo_scroll + len(shown)
            more = _truncate_display(
                f"    {lo}-{hi}/{len(items)} · ↑↓/滚轮", max(4, width - 2)
            )
            pad_r = max(0, width - get_cwidth(more))
            fragments.append(
                ("class:todo.pane", more + " " * pad_r + "\n", wheel)
            )

        fragments.append(("class:todo.pane", " " * width, wheel))
        return fragments

    # ── Parallel fleet pane (roster under turn status) ────────────────

    _FLEET_PANE_VISIBLE = 6  # body rows (not counting chrome)
    _MERGE_CONFIRM_S = 4.0  # second `m` must land within this window

    def _fleet_child_entries(
        self,
    ) -> list[tuple[str, str, int, Any]]:
        """Global non-MAIN agents across *all* tasks.

        Order: live (running/pending/waiting) first, newest tasks first within
        each group — a cross-task fleet dashboard, not only the active task.
        """
        tasks = self.ledger.snapshots()
        if not tasks:
            return []

        live: list[tuple[str, str, int, Any]] = []
        idle: list[tuple[str, str, int, Any]] = []
        # Newest tasks first so recent parallel work surfaces on top.
        for task in reversed(tasks):
            for i, a in enumerate(task.agents):
                if str(a.id).endswith(":main") or str(a.label).upper() == "MAIN":
                    continue
                row = (task.id, task.title, i, a)
                if a.status in {"pending", "running", "waiting"}:
                    live.append(row)
                else:
                    idle.append(row)
        # Fallback: if MAIN filtering hid every child despite multi-agent tasks
        # (legacy MAIN not tagged), keep non-first agents.
        if not live and not idle:
            for task in reversed(tasks):
                if len(task.agents) <= 1:
                    continue
                for i, a in enumerate(task.agents):
                    if i == 0 and (
                        str(a.id).endswith(":main")
                        or str(a.label).upper() == "MAIN"
                    ):
                        continue
                    row = (task.id, task.title, i, a)
                    if a.status in {"pending", "running", "waiting"}:
                        live.append(row)
                    else:
                        idle.append(row)
        return live + idle

    def _fleet_badge_label(self) -> str | None:
        """Header chip: global live/total child agents, or None when empty."""
        entries = self._fleet_child_entries()
        if not entries:
            return None
        live = sum(
            1
            for _tid, _tt, _i, a in entries
            if a.status in {"pending", "running", "waiting"}
        )
        total = len(entries)
        if live:
            return f"{live}/{total}"
        return f"{total}"

    def _merge_confirm_active(self) -> bool:
        if self._merge_confirm_ref is None:
            return False
        if time.monotonic() > float(self._merge_confirm_until or 0):
            self._merge_confirm_ref = None
            return False
        return True

    def _clear_merge_confirm(self) -> None:
        self._merge_confirm_ref = None
        self._merge_confirm_until = 0.0

    def _fleet_pane_height(self) -> Dimension:
        if not self._fleet_pane_open:
            return Dimension.exact(0)
        entries = self._fleet_child_entries()
        if not entries:
            return Dimension.exact(0)
        n = len(entries)
        body = min(self._FLEET_PANE_VISIBLE, max(1, n))
        h = body + 2  # title + bottom
        if n > self._FLEET_PANE_VISIBLE:
            h += 1  # scroll hint
        return Dimension(min=h, max=h, preferred=h)

    def _toggle_fleet_pane(self) -> None:
        if self._fleet_badge_label() is None:
            self._fleet_pane_open = False
            self._set_feedback("暂无并行子 Agent", "warning")
            return
        self._fleet_pane_open = not self._fleet_pane_open
        if self._fleet_pane_open:
            self._fleet_scroll = 0
            # Clamp cursor into range.
            n = len(self._fleet_child_entries())
            self._fleet_cursor = max(0, min(self._fleet_cursor, max(0, n - 1)))
            # Opening fleet closes todo so only one bottom drawer owns keys.
            self._todo_pane_open = False
        else:
            self._clear_merge_confirm()
        self.app.invalidate()

    def _ensure_fleet_cursor_visible(self) -> None:
        n = len(self._fleet_child_entries())
        if n <= 0:
            self._fleet_cursor = 0
            self._fleet_scroll = 0
            return
        self._fleet_cursor = max(0, min(self._fleet_cursor, n - 1))
        vis = self._FLEET_PANE_VISIBLE
        if self._fleet_cursor < self._fleet_scroll:
            self._fleet_scroll = self._fleet_cursor
        elif self._fleet_cursor >= self._fleet_scroll + vis:
            self._fleet_scroll = self._fleet_cursor - vis + 1
        max_scroll = max(0, n - vis)
        self._fleet_scroll = max(0, min(max_scroll, self._fleet_scroll))

    def _move_fleet_cursor(self, delta: int) -> None:
        entries = self._fleet_child_entries()
        if not entries:
            return
        n = len(entries)
        prev = self._fleet_cursor
        self._fleet_cursor = max(0, min(n - 1, self._fleet_cursor + int(delta)))
        if self._fleet_cursor != prev:
            self._clear_merge_confirm()
        self._ensure_fleet_cursor_visible()
        self.app.invalidate()

    def _scroll_fleet_pane(self, delta: int) -> None:
        entries = self._fleet_child_entries()
        if not entries:
            return
        n = len(entries)
        max_scroll = max(0, n - self._FLEET_PANE_VISIBLE)
        self._fleet_scroll = max(
            0, min(max_scroll, self._fleet_scroll + int(delta))
        )
        # Keep cursor inside the visible window when scrolling with wheel.
        lo = self._fleet_scroll
        hi = self._fleet_scroll + self._FLEET_PANE_VISIBLE - 1
        if self._fleet_cursor < lo:
            self._fleet_cursor = lo
        elif self._fleet_cursor > hi:
            self._fleet_cursor = min(n - 1, hi)
        self.app.invalidate()

    def _pin_fleet_cursor(self) -> None:
        entries = self._fleet_child_entries()
        if not entries or not 0 <= self._fleet_cursor < len(entries):
            self._set_feedback("无可钉住的 Agent", "warning")
            return
        task_id, _title, _idx, agent = entries[self._fleet_cursor]
        self._toggle_pin_agent(task_id, agent.id, agent.label)

    def _toggle_pin_agent(
        self, task_id: str, agent_id: str, label: str | None = None
    ) -> None:
        ref = (str(task_id), str(agent_id))
        if self._pinned_agent_ref == ref:
            self._pinned_agent_ref = None
            self._set_feedback("已取消钉住", "info")
        else:
            self._pinned_agent_ref = ref
            name = (label or agent_id or "agent").strip() or "agent"
            self._set_feedback(f"已钉住 · {name}", "success")
        self.app.invalidate()

    def _open_fleet_cursor(self) -> None:
        entries = self._fleet_child_entries()
        if not entries or not 0 <= self._fleet_cursor < len(entries):
            return
        task_id, _title, _agent_index, agent = entries[self._fleet_cursor]
        snaps = self.ledger.snapshots()
        for i, t in enumerate(snaps):
            if t.id == task_id:
                self._selected_task = i
                self._task_selection_active = True
                self._follow_latest_task = i == len(snaps) - 1
                self._pinned_task_for_line = None
                break
        self._open_agent(task_id, agent.id)

    def _open_pinned_agent(self) -> None:
        ref = self._pinned_agent_ref
        if ref is None:
            self._set_feedback("尚未钉住 Agent（在并行面板按 p）", "warning")
            return
        task_id, agent_id = ref
        agent = self.ledger.get_agent(task_id, agent_id)
        if agent is None:
            self._pinned_agent_ref = None
            self._set_feedback("钉住的 Agent 已不存在", "warning")
            return
        snaps = self.ledger.snapshots()
        for i, t in enumerate(snaps):
            if t.id == task_id:
                self._selected_task = i
                self._task_selection_active = True
                self._follow_latest_task = i == len(snaps) - 1
                self._pinned_task_for_line = None
                break
        self._open_agent(task_id, agent_id)

    def _agent_worktree_merged(self, agent_id: str) -> bool:
        if str(agent_id) in self._merged_worktrees:
            return True
        info = self._agent_worktree_info(agent_id)
        return bool(info.get("merged"))

    def _agent_mergeable(self, agent_id: str, status: str) -> bool:
        """Completed worktree child that has not been landed yet."""
        st = (status or "").strip().lower()
        if st not in {"completed", "done"}:
            return False
        if self._agent_worktree_merged(agent_id):
            return False
        info = self._agent_worktree_info(agent_id)
        return bool(info.get("is_worktree"))

    def _merge_fleet_cursor(self) -> None:
        entries = self._fleet_child_entries()
        if not entries or not 0 <= self._fleet_cursor < len(entries):
            self._set_feedback("无可合入的 Agent", "warning")
            return
        task_id, _title, _idx, agent = entries[self._fleet_cursor]
        self._request_or_confirm_merge(
            task_id, agent.id, agent.label, status=agent.status
        )

    def _request_or_confirm_merge(
        self,
        task_id: str,
        agent_id: str,
        label: str | None = None,
        *,
        status: str = "",
    ) -> None:
        """Double-tap m / click 合入: arm, then land worktree into parent."""
        name = (label or agent_id or "agent").strip() or "agent"
        ref = (str(task_id), str(agent_id))
        now = time.monotonic()
        if (
            self._merge_confirm_ref == ref
            and now <= float(self._merge_confirm_until or 0)
        ):
            self._clear_merge_confirm()
            self._execute_worktree_merge(str(agent_id), name)
            return
        if not self._agent_mergeable(agent_id, status):
            if self._agent_worktree_merged(agent_id):
                self._set_feedback(f"已合入 · {name}", "info")
            elif (status or "").lower() not in {"completed", "done"}:
                self._set_feedback("仅已完成的 worktree 可合入", "warning")
            else:
                self._set_feedback("该 Agent 无 worktree 可合入", "warning")
            self._clear_merge_confirm()
            return
        self._merge_confirm_ref = ref
        self._merge_confirm_until = now + self._MERGE_CONFIRM_S
        path = str(self._agent_worktree_info(agent_id).get("short_path") or "wt")
        self._set_feedback(
            f"再按 m / 点合入 确认合入 · {name} · {path}",
            "warning",
        )
        self.app.invalidate()

    def _execute_worktree_merge(self, agent_id: str, label: str) -> None:
        """Call coordinator.merge_worktree (same as merge_subagent_worktree tool)."""
        coord = self._subagent_coordinator()
        if coord is None:
            self._set_feedback("无子 Agent 协调器，无法合入", "warning")
            return
        merge_fn = getattr(coord, "merge_worktree", None)
        if not callable(merge_fn):
            self._set_feedback("当前后端不支持 worktree 合入", "warning")
            return
        cwd = getattr(self.session, "cwd", None) or Path.cwd()
        try:
            result = merge_fn(
                str(agent_id),
                Path(cwd),
                strategy="merge",
                commit_message=f"merge worktree from {label}",
                cleanup_worktree=True,
                delete_branch=False,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_feedback(f"合入失败 · {exc}", "warning")
            return
        ok = bool(getattr(result, "ok", False))
        if ok:
            self._merged_worktrees.add(str(agent_id))
            commit = getattr(result, "commit", None) or ""
            tail = f" · {commit[:8]}" if commit else ""
            self._set_feedback(f"已合入 · {label}{tail}", "success")
        else:
            msg = str(getattr(result, "message", "") or "merge failed").strip()
            conflicts = list(getattr(result, "conflicts", None) or [])
            if conflicts:
                msg = f"{msg} · 冲突 {len(conflicts)}"
            self._set_feedback(f"合入失败 · {msg[:60]}", "warning")
        self.app.invalidate()

    def _fleet_badge_mouse(self) -> Callable[[MouseEvent], object]:
        def _on_up(event: MouseEvent) -> None:
            now = time.monotonic()
            # Ctrl+click → open pinned (or cursor) agent immediately.
            if _mouse_control_held(event):
                self._fleet_badge_last_click = None
                if self._pinned_agent_ref is not None:
                    self._open_pinned_agent()
                else:
                    self._fleet_pane_open = True
                    self._todo_pane_open = False
                    self._open_fleet_cursor()
                return
            last = self._fleet_badge_last_click
            if last is not None and (now - last) <= _DOUBLE_CLICK_S:
                self._fleet_badge_last_click = None
                # Double-click opens fleet and jumps to cursor agent.
                self._fleet_pane_open = True
                self._todo_pane_open = False
                self._open_fleet_cursor()
                return
            self._fleet_badge_last_click = now
            self._toggle_fleet_pane()

        return self._only_mouse_up(_on_up, scroll_target="fleet")

    def _fleet_pane_mouse(self) -> Callable[[MouseEvent], object]:
        """Wheel scrolls fleet; bare click on chrome no-ops."""
        return self._only_mouse_up(lambda _e: None, scroll_target="fleet")

    def _fleet_row_mouse(
        self, entry_index: int
    ) -> Callable[[MouseEvent], object]:
        """Click a fleet row: select; Ctrl+click pin; double-click open."""

        def _on_up(event: MouseEvent) -> None:
            entries = self._fleet_child_entries()
            if not 0 <= entry_index < len(entries):
                return
            self._fleet_cursor = entry_index
            self._ensure_fleet_cursor_visible()
            task_id, _title, _agent_index, agent = entries[entry_index]
            if _mouse_control_held(event):
                self._toggle_pin_agent(task_id, agent.id, agent.label)
                return
            # Single click focuses; open on second click within double-click window
            # is handled by selecting then Enter — here open immediately for
            # discoverability: one click jumps to that inline transcript.
            snaps = self.ledger.snapshots()
            for i, t in enumerate(snaps):
                if t.id == task_id:
                    self._selected_task = i
                    self._task_selection_active = True
                    self._follow_latest_task = i == len(snaps) - 1
                    self._pinned_task_for_line = None
                    break
            self._open_agent(task_id, agent.id)

        return self._only_mouse_up(_on_up, scroll_target="fleet")

    def _render_fleet_pane(self) -> StyleAndTextTuples:
        """Global parallel-agent roster on one flat, readable surface."""
        if not self._fleet_pane_open:
            return []
        entries = self._fleet_child_entries()
        width = max(12, _content_width())
        if not entries:
            self._fleet_pane_open = False
            return []

        self._ensure_fleet_cursor_visible()
        wheel = self._fleet_pane_mouse()
        live = sum(
            1
            for _t, _tt, _i, a in entries
            if a.status in {"pending", "running", "waiting"}
        )
        task_ids = {tid for tid, _tt, _i, _a in entries}
        multi = len(task_ids) > 1
        scope = "全局" if multi else "本任务"
        title = f"  并行 {scope} {live}/{len(entries)}"
        if self._merge_confirm_active():
            hint = "m 再按确认合入 · ↑↓ 取消"
        else:
            hint = "↑↓ 选择 · Enter 打开 · p 钉住 · m 合入"
        if get_cwidth(title) + get_cwidth(hint) + 4 <= width:
            pad = width - get_cwidth(title) - get_cwidth(hint) - 2
            top = title + " " * pad + hint + "  "
        else:
            top = _truncate_display(title, width)
            top += " " * max(0, width - get_cwidth(top))
        fragments: StyleAndTextTuples = [
            ("class:fleet.pane.title", top + "\n", wheel),
        ]

        shown = entries[
            self._fleet_scroll : self._fleet_scroll + self._FLEET_PANE_VISIBLE
        ]
        for offset, (task_id, task_title, _ai, agent) in enumerate(shown):
            entry_index = self._fleet_scroll + offset
            focus = entry_index == self._fleet_cursor
            mark = _agent_status_mark(agent.status)
            pin = (
                "★"
                if self._pinned_agent_ref == (task_id, agent.id)
                else " "
            )
            label = (agent.label or "AGENT").strip() or "AGENT"
            wt = self._agent_worktree_short(agent.id)
            merged = self._agent_worktree_merged(agent.id)
            live_line = ""
            if agent.status in {"pending", "running", "waiting"}:
                live_line = self._activity.line(task_id, agent.id)
            bullet = "›" if focus else " "
            mid = f" {bullet}{pin}{mark} {label}"
            if multi:
                # Short task title so cross-task rows stay identifiable.
                tshort = " ".join((task_title or "").split())[:10]
                if tshort:
                    mid += f" · {tshort}"
            if wt:
                mid += " · landed" if merged else f" · {wt}"
            if live_line:
                mid += f" · {live_line}"
            elif agent.status not in {"pending", "running", "waiting"}:
                tail = " ".join((agent.output or "").split())[:24]
                if tail:
                    mid += f" · {tail}"
            text = _truncate_display(f"  {mid}", max(4, width - 2))
            pad_r = max(0, width - get_cwidth(text))
            style = (
                "class:fleet.item.selected"
                if focus
                else (
                    "class:fleet.item.running"
                    if agent.status in {"pending", "running", "waiting"}
                    else "class:fleet.item"
                )
            )
            row_mouse = self._fleet_row_mouse(entry_index)
            fragments.append((style, text + " " * pad_r + "\n", row_mouse))

        max_scroll = max(0, len(entries) - self._FLEET_PANE_VISIBLE)
        if max_scroll > 0:
            lo = self._fleet_scroll + 1
            hi = self._fleet_scroll + len(shown)
            more = _truncate_display(
                f"    {lo}-{hi}/{len(entries)} · ↑↓/滚轮", max(4, width - 2)
            )
            pad_r = max(0, width - get_cwidth(more))
            fragments.append(
                ("class:fleet.pane", more + " " * pad_r + "\n", wheel)
            )

        # Bottom border: pin + worktree merge action (clickable when mergeable).
        foot_bits: list[str] = []
        merge_mouse = wheel
        if self._pinned_agent_ref is not None:
            pin_id = self._pinned_agent_ref[1]
            pin_label = pin_id
            for _tid, _tt, _i, a in entries:
                if a.id == pin_id:
                    pin_label = a.label or pin_id
                    break
            foot_bits.append(f"★{pin_label}")
        if 0 <= self._fleet_cursor < len(entries):
            cur_tid, _tt, _i, cur = entries[self._fleet_cursor]
            wt_info = self._agent_worktree_info(cur.id)
            if wt_info.get("is_worktree"):
                path = str(wt_info.get("short_path") or "wt")
                if self._agent_worktree_merged(cur.id):
                    foot_bits.append(f"已合入 {path}")
                elif self._agent_mergeable(cur.id, cur.status):
                    armed = (
                        self._merge_confirm_active()
                        and self._merge_confirm_ref == (cur_tid, cur.id)
                    )
                    foot_bits.append(
                        f"{'确认合入?' if armed else '合入[m]'} {path}"
                    )
                    merge_mouse = self._fleet_merge_mouse(cur_tid, cur.id, cur.label)
                else:
                    foot_bits.append(f"wt {path}")
        if foot_bits:
            footer = _truncate_display(
                "  " + " · ".join(foot_bits), max(4, width - 2)
            )
            pad_b = max(0, width - get_cwidth(footer))
            fragments.append(("class:fleet.pane.meta", footer, merge_mouse))
            fragments.append(("class:fleet.pane", " " * pad_b, wheel))
        else:
            fragments.append(("class:fleet.pane", " " * width, wheel))
        return fragments

    def _fleet_merge_mouse(
        self, task_id: str, agent_id: str, label: str | None
    ) -> Callable[[MouseEvent], object]:
        """Click footer 合入 chip — same double-confirm as key m."""

        def _on_up(_event: MouseEvent) -> None:
            agent = self.ledger.get_agent(task_id, agent_id)
            status = (agent.status if agent else "") or ""
            self._request_or_confirm_merge(
                task_id, agent_id, label or agent_id, status=status
            )

        return self._only_mouse_up(_on_up, scroll_target="fleet")

    def _agent_worktree_info(self, agent_id: str) -> dict[str, Any]:
        """Worktree path / isolation / merged for fleet footer and detail hint."""
        empty: dict[str, Any] = {
            "is_worktree": False,
            "path": "",
            "short_path": "",
            "isolation": "",
            "merged": False,
        }
        if str(agent_id) in self._merged_worktrees:
            empty["merged"] = True
            empty["is_worktree"] = True
            empty["short_path"] = "wt"
            return empty
        coord = self._subagent_coordinator()
        if coord is None:
            return empty
        lookup = getattr(coord, "lookup", None)
        if not callable(lookup):
            return empty
        try:
            snap = lookup(str(agent_id))
        except Exception:  # noqa: BLE001
            return empty
        if snap is None:
            return empty
        path = str(getattr(snap, "worktree_path", None) or "")
        meta = getattr(snap, "metadata", None) or {}
        isolation = str(meta.get("isolation") or "")
        merged = bool(meta.get("worktree_merged"))
        is_wt = bool(path) or isolation == "worktree" or merged
        short = ""
        if path:
            parts = path.replace("\\", "/").rstrip("/").split("/")
            short = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        elif is_wt:
            short = "wt"
        return {
            "is_worktree": is_wt,
            "path": path,
            "short_path": short,
            "isolation": isolation or ("worktree" if is_wt else ""),
            "merged": merged,
        }

    def _render_tasks(self) -> StyleAndTextTuples:
        """Render every task as one continuous, full-fidelity reading stream.

        Task is structure: a quiet divider and state. Chat is content: exact
        user text, assistant prose, thinking rails, and canonical tool anchors.
        There are no cards and no transcript/detail duplication.
        """
        tasks = self.ledger.snapshots()
        fragments: StyleAndTextTuples = []
        line = 0
        width = max(1, _content_width())

        # Plan/auth own the page. The questionnaire remains a float over the
        # live transcript, so only replacement pages suppress this stream.
        if self._detail_page_visible():
            if self._showing_startup_brand() or not tasks:
                self._task_refs = []
                empty: StyleAndTextTuples = [("", "\n")]
                # Underlay line count only — do not clamp free-scroll _selected_line.
                self._task_line_count = self._count_fragment_lines(empty)
                return empty
            # Keep task refs in sync for selection keys without repainting the
            # hidden transcript beneath the replacement page.
            self._task_refs = [task.id for task in tasks]
            if tasks and self._task_selection_active and self._selected_task >= 0:
                self._selected_task = max(
                    0, min(int(self._selected_task), len(tasks) - 1)
                )
            elif not self._task_selection_active:
                self._selected_task = -1
            empty = [("", "\n")]
            self._task_line_count = self._count_fragment_lines(empty)
            return empty

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
        self._task_anchor_lines = {}
        self._agent_anchor_lines = {}
        if self._task_selection_active:
            if self._selected_task < 0:
                self._task_selection_active = False
            else:
                self._selected_task = max(
                    0, min(int(self._selected_task), len(tasks) - 1)
                )
        else:
            self._selected_task = -1
        list_focused = self._task_list_has_focus()
        selected_line_start = 0
        latest_task_tail_line = 0

        for task_index, task in enumerate(tasks):
            has_sel = self._task_selection_active and self._selected_task >= 0
            selected = (
                list_focused
                and has_sel
                and task_index == self._selected_task
            )
            is_cursor_task = has_sel and task_index == self._selected_task
            task_start = line
            self._task_anchor_lines[task.id] = task_start
            if is_cursor_task:
                selected_line_start = task_start
            stream_mouse = self._task_section_mouse(task_index)
            face = "class:task.stream"

            def append_task_line(
                parts: StyleAndTextTuples,
                *,
                default_mouse: Callable[[MouseEvent], object] = stream_mouse,
                fill_style: str = face,
            ) -> None:
                """Paint a full-width stream row with reliable mouse mapping."""
                nonlocal line
                used = 0
                painted: list[
                    tuple[str, str, Callable[[MouseEvent], object]]
                ] = []
                for part in parts:
                    raw = part[0] or ""
                    style = f"{face} {raw}".strip() if raw else face
                    text = part[1]
                    used += get_cwidth(text)
                    if len(part) >= 3 and part[2] is not None:
                        painted.append((style, text, part[2]))
                    else:
                        painted.append((style, text, default_mouse))
                if used < width:
                    painted.append(
                        (
                            f"{face} {fill_style}".strip(),
                            " " * (width - used),
                            default_mouse,
                        )
                    )
                fragments.extend(self._paint_task_text_selection(painted, line))
                fragments.append(("", "\n", default_mouse))
                line += 1

            label_style = (
                "class:task.section.selected"
                if selected
                else "class:task.section.label"
            )
            flash = self._interject_preview(task.id)
            status = _compact_task_stage_text(task, interject=flash)
            elapsed_label = _task_elapsed_label(task)
            if elapsed_label:
                status = f"{status} · {elapsed_label}"
            status_style = (
                "class:task.interject"
                if flash
                else _task_status_style(task)
            )
            chat_collapsed = task.id in self._collapsed_task_chats
            fold_label = " 展开 " if chat_collapsed else " 收起 "
            fold_mouse = self._task_chat_toggle_mouse(task.id)
            task_label = f"  TASK {task_index + 1:02d}"
            right_parts: StyleAndTextTuples = [
                (status_style, status),
                ("class:task.section.meta", "  "),
                ("class:task.action", fold_label, fold_mouse),
                ("class:task.section.meta", " "),
            ]
            right_width = sum(get_cwidth(part[1]) for part in right_parts)
            gap = max(1, width - get_cwidth(task_label) - right_width)
            append_task_line(
                [
                    (label_style, task_label),
                    ("class:task.section.meta", " " * gap),
                    *right_parts,
                ]
            )

            if chat_collapsed:
                for title_line in _wrap_display_lines(
                    task.title, max(1, width - 6), max_lines=None
                ):
                    append_task_line(
                        [
                            ("class:task.section.meta", "    "),
                            ("class:task.title", title_line),
                        ]
                    )
            else:
                # Exact user prompt — title normalization never replaces chat.
                prompt_text = (getattr(task, "prompt", "") or task.title).strip()
                for row in _render_task_chat_rows(
                    prompt_text,
                    max(1, width - 4),
                    link_mouse=self._task_link_mouse,
                ):
                    append_task_line(
                        [("class:task.chat.gutter", "    "), *row]
                    )
                append_task_line([])

                for agent in task.agents:
                    self._agent_anchor_lines[(task.id, agent.id)] = line
                    is_main = (
                        str(agent.id).endswith(":main")
                        or str(agent.label).strip().upper() == "MAIN"
                    )
                    if not is_main:
                        agent_status = STATUS_TEXT.get(agent.status, agent.status)
                        agent_accent = (
                            "class:task.status.completed"
                            if agent.status in {"completed", "done"}
                            else (
                                "class:task.status.failed"
                                if agent.status in {"failed", "cancelled"}
                                else "class:task.status.running"
                            )
                        )
                        append_task_line(
                            [
                                ("class:task.chat.gutter", "    "),
                                (agent_accent, "━━ "),
                                ("class:task.actor.agent", agent.label or "AGENT"),
                                ("class:task.section.meta", f" · {agent_status}"),
                            ]
                        )

                    snapshot = self._snapshot_for_task_agent(task, agent)
                    records = () if snapshot is None else snapshot.records
                    if not records and agent.status in {
                        "pending",
                        "running",
                        "waiting",
                    }:
                        append_task_line(
                            [
                                ("class:task.chat.gutter", "    "),
                                ("class:task.status.running", "● "),
                                ("class:task.chat.muted", "正在思考…"),
                            ]
                        )
                    for record in records:
                        record_rows = self._task_record_rows(
                            task,
                            agent,
                            record,
                            max(1, width - 4),
                            stream_mouse,
                        )
                        for record_row, row_mouse, fill_style in record_rows:
                            append_task_line(
                                [
                                    ("class:task.chat.gutter", "    "),
                                    *record_row,
                                ],
                                default_mouse=row_mouse,
                                fill_style=fill_style,
                            )

                if task.plan_state == "awaiting_approval":
                    review_mouse = self._task_plan_review_mouse(task.id)
                    append_task_line([])
                    append_task_line(
                        [
                            ("class:task.chat.gutter", "    "),
                            ("class:task.plan.ready", "● PLAN READY"),
                            ("class:task.section.meta", "  "),
                            (
                                "class:task.plan.review",
                                "审阅唯一待批版本",
                                review_mouse,
                            ),
                        ],
                        default_mouse=review_mouse,
                    )

            if task_index == len(tasks) - 1:
                latest_task_tail_line = max(task_start, line - 1)

            fragments.extend(self._task_blank_line(width, self._task_gap_mouse()))
            line += 1

        void = self._task_void_mouse()
        try:
            win_h = int(getattr(self._task_window.render_info, "window_height", 0) or 0)
        except Exception:  # noqa: BLE001
            win_h = 0
        pad_lines = max(12, (win_h - line + 4) if win_h else 12)
        for _ in range(pad_lines):
            fragments.extend(self._task_blank_line(width, void))
            line += 1

        self._latest_task_tail_line = max(0, latest_task_tail_line)
        if self._follow_latest_task:
            self._selected_line = self._latest_task_tail_line
        elif self._task_selection_active and self._selected_task >= 0:
            pin_task = int(self._selected_task)
            if self._pinned_task_for_line != pin_task:
                self._selected_line = selected_line_start
                self._pinned_task_for_line = pin_task
        else:
            self._pinned_task_for_line = None
        self._set_task_line_count(fragments)
        store_key = self._task_paint_cache_key(tasks, width)
        self._task_paint_cache = (
            store_key,
            fragments,
            list(self._task_refs),
            int(self._selected_line),
            int(self._task_line_count),
        )
        return fragments

    def _snapshot_for_task_agent(
        self,
        task: TaskView,
        agent: Any,
    ) -> AgentDetailSnapshot | None:
        """Project one canonical message list into records for all UI surfaces."""
        key = (task.id, str(agent.id))
        messages = list(self._detail_messages.get(key, []))
        live_draft = self._detail_live_draft(key)
        if live_draft is not None and live_draft[1]:
            messages.append(Message(role=Role.ASSISTANT, content=live_draft[1]))
        if not messages:
            fallback = str(
                getattr(agent, "output", "")
                or getattr(agent, "description", "")
                or ""
            ).strip()
            if fallback:
                messages = [Message(role=Role.ASSISTANT, content=fallback)]
        if not messages:
            return None
        return snapshot_from_messages(
            messages,
            task_id=task.id,
            agent_id=str(agent.id),
            agent_label=str(agent.label or "AGENT"),
            task_title=task.title,
            initial_user_text=strip_image_chips(
                getattr(task, "prompt", "") or task.title
            ),
            status=str(agent.status or "running"),
        )

    def _task_record_rows(
        self,
        task: TaskView,
        agent: Any,
        record: DetailRecord,
        width: int,
        stream_mouse: Callable[[MouseEvent], object],
    ) -> list[
        tuple[
            StyleAndTextTuples,
            Callable[[MouseEvent], object],
            str,
        ]
    ]:
        """Render one transcript record; tool bodies stay in the hover float."""
        width = max(1, int(width))
        rows: list[
            tuple[
                StyleAndTextTuples,
                Callable[[MouseEvent], object],
                str,
            ]
        ] = []
        actor = str(record.actor or "").strip().upper()

        if actor == "TOOL":
            ref = (task.id, str(agent.id), str(record.id))
            mouse = self._task_tool_mouse(ref)
            status = str(record.status or "").lower()
            if status in {"pending", "running", "waiting"}:
                mark, mark_style = "●", "class:task.tool.running"
            elif status in {"failed", "error"}:
                mark, mark_style = "×", "class:task.tool.failed"
            else:
                mark, mark_style = "◇", "class:task.tool.done"
            prefix = f"{mark} "
            title_width = max(1, width - get_cwidth(prefix))
            title_lines = _wrap_display_lines(
                record.title or "Tool", title_width, max_lines=None
            )
            for index, title_line in enumerate(title_lines):
                if index == 0:
                    parts: StyleAndTextTuples = [
                        (mark_style, prefix, mouse),
                        ("class:task.tool.title", title_line, mouse),
                    ]
                else:
                    parts = [
                        ("class:task.tool.title", " " * get_cwidth(prefix), mouse),
                        ("class:task.tool.title", title_line, mouse),
                    ]
                rows.append((parts, mouse, "class:task.stream"))
            return rows

        if actor in {"THINK", "THINKING", "REASONING"}:
            rows.append(
                (
                    [("class:task.actor.thinking", "THINKING")],
                    stream_mouse,
                    "class:task.stream",
                )
            )
            body_width = max(1, width - 2)
            for block in record.blocks:
                for raw in (block.text or "").splitlines() or [""]:
                    for piece in _wrap_display_lines(
                        raw, body_width, max_lines=None
                    ):
                        rows.append(
                            (
                                [
                                    ("class:task.thinking.rail", "│ "),
                                    ("class:task.thinking.body", piece),
                                ],
                                stream_mouse,
                                "class:task.stream",
                            )
                        )
            rows.append(([], stream_mouse, "class:task.stream"))
            return rows

        # The surrounding task/agent section already owns speaker identity.
        # Repeating YOU/DOGGY before ordinary prose adds noise without meaning.
        for block in record.blocks:
            for md_row in _render_task_chat_rows(
                block.text,
                width,
                link_mouse=self._task_link_mouse,
            ):
                rows.append((md_row, stream_mouse, "class:task.stream"))
        rows.append(([], stream_mouse, "class:task.stream"))
        return rows

    def _agent_worktree_short(self, agent_id: str) -> str:
        """Compact worktree badge for roster / header."""
        coord = self._subagent_coordinator()
        if coord is None:
            return ""
        lookup = getattr(coord, "lookup", None)
        if not callable(lookup):
            return ""
        try:
            snap = lookup(str(agent_id))
        except Exception:  # noqa: BLE001
            return ""
        if snap is None:
            return ""
        path = getattr(snap, "worktree_path", None) or ""
        meta = getattr(snap, "metadata", None) or {}
        if path or str(meta.get("isolation") or "") == "worktree":
            return "wt"
        return ""

    def _task_paint_cache_key(self, tasks: list[TaskView], width: int) -> tuple[Any, ...]:
        """Stable identity for task-list paint; excludes free-scroll noise."""
        rows: list[tuple[Any, ...]] = []
        for task in tasks:
            agents: list[tuple[Any, ...]] = []
            for agent in task.agents:
                key = (task.id, str(agent.id))
                messages = list(self._detail_messages.get(key, []))
                draft = self._detail_live_draft(key)
                draft_text = draft[1] if draft is not None else ""
                agents.append(
                    (
                        agent.id,
                        agent.label,
                        agent.status,
                        agent.output,
                        agent.description,
                        _live_messages_signature(messages),
                        (
                            draft[0] if draft is not None else -1,
                            len(draft_text),
                            draft_text[:64],
                            draft_text[-32:] if draft_text else "",
                        ),
                    )
                )
            flash = self._interject_preview(task.id)
            # Bucket elapsed by whole seconds so running task headers refresh.
            el = _task_elapsed_seconds(task)
            el_bucket = int(el) if el is not None else -1
            rows.append(
                (
                    task.id,
                    task.title,
                    task.prompt,
                    task.phase,
                    task.status,
                    task.plan_state,
                    task.report,
                    task.reporter,
                    flash or "",
                    task.id in self._collapsed_task_chats,
                    el_bucket,
                    tuple(agents),
                )
            )
        return (
            width,
            int(self._selected_task),
            bool(self._task_selection_active),
            bool(self._follow_latest_task),
            self._task_copy_anchor,
            self._task_copy_cursor,
            self._pinned_task_for_line,
            int(self._selected_line),
            self._task_list_has_focus(),
            tuple(rows),
        )

    @staticmethod
    def _task_blank_line(
        width: int, mouse: Callable[[MouseEvent], object]
    ) -> StyleAndTextTuples:
        """One full-width blank row with a real mouse hit-target on every cell.

        Root cause (blank click → first task): prompt_toolkit ``Window`` mouse
        routing builds ``rowcol_to_yx`` only for *painted characters*. A bare
        ``\\n`` line has no cells. Clicks on those screen positions miss the map
        and fall through to ``Point(x=0, y=0)`` — the first task. Painting
        ``width`` spaces attaches the void/gap handler to the whole row.
        """
        w = max(1, int(width))
        return [("", " " * w + "\n", mouse)]

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
        old_n = max(1, int(self._detail_line_count))
        old_cursor = int(self._detail_cursor_line)
        n = self._count_fragment_lines(fragments)
        self._detail_line_count = n
        if preferred_cursor is not None:
            self._detail_cursor_line = int(preferred_cursor)
        elif old_n > 1 and old_cursor >= old_n - 2:
            # User was watching the live tail — stick to the new bottom so
            # stream appends do not strand the cursor one page above the end.
            self._detail_cursor_line = n - 1
        else:
            # User scrolled up to read history — keep their line stable while
            # the transcript grows below (do not auto-jump on every tool event).
            self._detail_cursor_line = old_cursor
        self._detail_cursor_line = max(0, min(int(self._detail_cursor_line), n - 1))

    def _render_modal_title(self) -> StyleAndTextTuples:
        width = max(12, _terminal_width() - 15)

        def title_row(
            title: str,
            status: str = "",
            *,
            status_style: str = "class:detail.meta",
        ) -> StyleAndTextTuples:
            left = f"  {title}"
            right = f"  {status}  " if status else ""
            if right and get_cwidth(left) + get_cwidth(right) + 1 <= width:
                gap = width - get_cwidth(left) - get_cwidth(right)
                return [
                    ("class:agent-window.header", left),
                    ("class:agent-window", " " * gap),
                    (status_style, right),
                ]
            return [
                (
                    "class:agent-window.header",
                    _truncate_display(left, width),
                )
            ]

        if self._modal_kind == "auth":
            right = "AUTH"
            if self._auth_wizard.busy:
                spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(time.monotonic() * 8) % 10]
                right = f"{spinner} 等待授权"
            return title_row(self._auth_wizard.title, right)
        if self._modal_kind == "ask":
            n = max(1, len(self._ask_questions))
            i = min(self._ask_q_index + 1, n)
            return title_row(f"计划提问 {i}/{n}", "ASK")
        if self._modal_kind == "plan":
            if not self._modal_ref:
                return title_row("计划审批", "等待决定")
            task_id, _agent_id = self._modal_ref
            task = next(
                (item for item in self.ledger.snapshots() if item.id == task_id),
                None,
            )
            goal = task.title if task is not None else "当前任务"
            return title_row(f"计划审批 · {goal}", self._plan_revision_label())
        return title_row("CodeDoggy")

    def _render_modal_filters(self) -> StyleAndTextTuples:
        text = f"  {self._auth_wizard.subtitle}"
        return [
            (
                "class:auth.note",
                _truncate_display(text, max(12, _terminal_width() - 12)),
            )
        ]

    def _set_detail_live_draft(
        self,
        key: tuple[str, str],
        generation: int,
        text: str,
    ) -> None:
        """Publish the current unarchived assistant message to the detail view."""
        with self._view_lock:
            if text:
                self._detail_live_drafts[key] = (int(generation), text)
            else:
                self._detail_live_drafts.pop(key, None)

    def _clear_detail_live_draft(
        self,
        key: tuple[str, str],
        generation: int,
    ) -> None:
        """Clear only the draft generation that just became a real message."""
        with self._view_lock:
            current = self._detail_live_drafts.get(key)
            if current is not None and current[0] == int(generation):
                self._detail_live_drafts.pop(key, None)

    def _detail_live_draft(
        self, key: tuple[str, str]
    ) -> tuple[int, str] | None:
        with self._view_lock:
            return self._detail_live_drafts.get(key)

    def _plan_body_visible(self) -> bool:
        """The canonical plan file is visible only on the approval surface."""
        return bool(
            self._modal_open
            and self._modal_kind == "plan"
            and self._task_awaiting_plan_approval()
        )

    def _render_modal_body(self) -> StyleAndTextTuples:
        if self._modal_kind == "auth":
            return self._render_auth_body()
        if self._modal_kind == "ask":
            return self._render_ask_body()
        # Plan text is owned by the separate selectable plan TextArea.
        return [("", "\n")]

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
            fragments.append(
                ("class:auth.item.active", f"  {spin} 等待浏览器授权完成…\n")
            )
            fragments.append(
                ("class:auth.hint", "  完成后回到结果页 · Tab 取消\n")
            )
        # First visual line of each menu item (for scroll-into-view).
        item_line_starts: list[int] = []
        for index, item in enumerate(wiz.items):
            # Next line index == total newlines so far when every fragment ends with \n.
            start_y = sum(str(it[1]).count("\n") for it in fragments)
            item_line_starts.append(start_y)
            selected = index == wiz.cursor
            marker = "  › " if selected else "    "
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
            label = _truncate_display(f"{marker}{item.label}", width)
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

    def _render_detail_prompt_prefix(self) -> StyleAndTextTuples:
        if self._modal_kind == "auth":
            text = "  › 凭证  "
            return [("class:detail.input.prompt", text)]
        if self._modal_kind == "ask":
            text = "  › Other  "
            return [("class:detail.input.prompt", text)]
        terminal_width = max(1, _terminal_width())
        budget = max(4, terminal_width - 20)
        if terminal_width < 40:
            text = "  › "
        else:
            text = "  › 修改意见  "
        text = _truncate_display(text, budget)
        return [("class:detail.input.prompt", text)]

    def _render_modal_hint(self) -> StyleAndTextTuples:
        if self._modal_kind == "auth":
            text = "  ↑↓ 选择 · Enter 确认 · Tab 返回 · Ctrl+L"
        elif self._modal_kind == "ask":
            if self._ask_other_editing:
                text = "  输入自定义答案 · Enter 提交 · Tab 退出"
            elif self._ask_is_multi():
                text = "  ↑↓ 移动 · Space 勾选 · Enter 完成 · Tab 退出"
            else:
                text = "  ↑↓ 选择 · Enter 确认 · Tab 退出"
        elif self._modal_kind == "plan":
            text = "  a 批准 · s 写修改意见 · q 放弃 · 滚轮浏览 · 拖选复制 · Tab 返回"
        else:
            text = "  Tab 返回"
        line = _truncate_display(text, max(1, _terminal_width() - 12))
        return [("class:agent-window.hint", line)]

    def _detail_page_visible(self) -> bool:
        """Only plan approval and auth can replace the reading stream."""
        return bool(
            self._modal_open and self._modal_kind in {"plan", "auth"}
        )

    def _detail_input_visible(self) -> bool:
        """Show interject/auth paste only when there is something to type for."""
        if not self._modal_open:
            return False
        if self._modal_kind == "auth":
            return self._auth_wizard.step == WizardStep.PASTE
        if self._modal_kind == "ask":
            return False  # dedicated ask float; Other uses main input
        if self._modal_kind != "plan":
            return False
        return self._task_awaiting_plan_approval()

    def _land_focus_if_detail_input_gone(self) -> None:
        """When the interject box disappears mid-turn, land on detail body.

        ConditionalContainer hide can leave focus on a detached control; stream
        redraws then fight the user. Explicitly move to the detail window.
        """
        if not self._modal_open or self._modal_kind != "plan":
            return
        if self._detail_input_visible():
            return
        try:
            if self.app.layout.has_focus(self._detail_input):
                self.app.layout.focus(self._detail_window)
        except Exception:  # noqa: BLE001
            pass

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
        # No enter-plan consent strip: model soft-enters plan mode; extra
        # "同意进入 Plan" would interrupt normal dialogue. Approval stays on exit.
        bag.pop("plan_mode_consent_fn", None)
        bag["plan_mode_exit_fn"] = self._plan_mode_exit_fn
        bag["todo_changed_fn"] = self._on_todo_changed
        # Override CLI stdin ask (which paints *outside* the TUI) with modal UI.
        bag["ask_user_fn"] = self._ask_user_fn

    def _on_todo_changed(self) -> None:
        """Called from todo_write worker thread after list mutates."""
        self._call_in_ui_thread(self.app.invalidate)

    def _ask_user_fn(self, questions: list[dict[str, Any]]) -> dict[str, Any]:
        """Host hook for ask_user_question — park worker; answer in a small float.

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
        """Show dedicated questionnaire float outside the approval/auth pages."""
        if self._modal_open and self._modal_kind in {"plan", "auth"}:
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
        self._modal_kind = "plan"
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
            def _open() -> None:
                self._open_plan_approval(task_id)

            self._call_in_ui_thread(_open)
        else:
            self._call_in_ui_thread(self.app.invalidate)
        try:
            signaled = self._plan_exit_event.wait(timeout=600)
        finally:
            self._plan_exit_waiting = False
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

    def _resolve_plan_exit(self, outcome: str) -> None:
        was_plan_page = self._modal_open and self._modal_kind == "plan"
        if outcome == "revise":
            # Revision is a real decision, not a vague button: collect the
            # requested change first, then Enter submits it.
            try:
                text = (self._detail_input.text or "").strip()
            except Exception:  # noqa: BLE001
                text = ""
            if not text and was_plan_page:
                self._set_feedback("写下具体修改意见，按 Enter 提交", "info")
                try:
                    self.app.layout.focus(self._detail_input)
                except Exception:  # noqa: BLE001
                    pass
                self.app.invalidate()
                return
            self._plan_exit_feedback = text or "请修改计划后再次 exit_plan_mode"
            if text:
                self._detail_input.text = ""
        self._plan_exit_outcome = outcome
        if self._plan_exit_waiting:
            self._plan_exit_event.set()
        else:
            # Resume chrome (restart with awaiting_plan_approval, no parked tool).
            self._apply_plan_exit_resume(outcome)
        if was_plan_page:
            self._leave_plan_approval()
        else:
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
            self._plan_ui_task_id = task_id
            self._plan_exit_waiting = False  # resume chrome, not blocked tool
            try:
                self._open_plan_approval(task_id)
            except Exception:  # noqa: BLE001
                pass
            self._set_feedback(
                "恢复：计划待确认 · 请在审批页决定",
                "warning",
            )
        else:
            self._plan_exit_waiting = False
            self._set_feedback(
                "恢复：有待确认计划，但当前没有可归属的任务",
                "warning",
            )
        self.app.invalidate()

    def _resolve_plan_file_path_for_ui(self) -> str:
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
        return path

    def _load_plan_file_for_ui(self) -> tuple[str, str, float]:
        """Return the exact approval file text and its disk identity."""
        path = self._resolve_plan_file_path_for_ui()
        text = ""
        mtime = 0.0
        try:
            p = Path(path)
            if not p.is_absolute():
                p = Path(getattr(self.session, "cwd", Path.cwd())) / p
            path = str(p)
            if p.is_file():
                mtime = float(p.stat().st_mtime)
                text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            text = ""
        return path, text, mtime

    def _plan_revision_label(self) -> str:
        """Content-derived revision id for the exact plan being approved."""
        text = ""
        try:
            text = self._plan_body.buffer.text or ""
        except Exception:  # noqa: BLE001
            pass
        if not text:
            _path, text, _mtime = self._load_plan_file_for_ui()
        if not text:
            return "REV EMPTY"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8].upper()
        return f"REV {digest}"

    def _render_plan_chrome(self) -> StyleAndTextTuples:
        """Approval controls around one exact, content-addressed plan revision."""
        if self._plan_body_visible():
            self._sync_plan_body_buffer(force=False)
        width = max(12, _terminal_width() - 10)
        path = self._plan_body_path or self._resolve_plan_file_path_for_ui()
        frags: StyleAndTextTuples = [
            ("class:plan.status.review", "  ● DECISION REQUIRED"),
            ("class:plan.meta", f"  ·  {self._plan_revision_label()}\n"),
            ("class:plan.chrome", "  "),
            (
                "class:plan.action.approve",
                " a  批准开工 ",
                self._plan_action_mouse("approved"),
            ),
            ("class:plan.chrome", "  "),
            (
                "class:plan.action.revise",
                " s  要求修改 ",
                self._plan_action_mouse("revise"),
            ),
            ("class:plan.chrome", "  "),
            (
                "class:plan.action.abandon",
                " q  放弃 ",
                self._plan_action_mouse("abandoned"),
            ),
            ("", "\n"),
        ]
        short_name = Path(path).name if path else "plan.md"
        frags.append(
            (
                "class:plan.meta",
                _truncate_display(
                    f"  {short_name} · 下方就是唯一待批原文，无摘要、日志或历史版本\n",
                    width,
                ),
            )
        )
        if not (self._plan_body.buffer.text or "").strip():
            frags.append(
                (
                    "class:plan.empty",
                    "  还没有草案 — 继续在对话里说明需求即可。\n",
                )
            )
        return self._ensure_fragments(frags)

    def _sync_plan_body_buffer(self, *, force: bool = False) -> None:
        """Load plan.md into the selectable TextArea when path/mtime changes.

        PLAN PAINT BUDGET: never route full plan through FormattedText markdown.
        """
        if not self._plan_body_visible() and not force:
            return
        path, text, mtime = self._load_plan_file_for_ui()
        self._plan_body_path = path
        key = (path, mtime, len(text))
        if (
            not force
            and key == self._plan_body_sync_key
            and self._plan_body.buffer.text == text
        ):
            return
        buf = self._plan_body.buffer
        # Keep selection/cursor when content is identical.
        if buf.text == text:
            self._plan_body_sync_key = key
            return
        cursor = min(int(buf.cursor_position or 0), len(text))
        try:
            buf.set_document(
                Document(text, cursor_position=cursor),
                bypass_readonly=True,
            )
        except TypeError:
            # Older prompt_toolkit: flip read_only briefly.
            was = buf.read_only
            try:
                buf.read_only = False  # type: ignore[assignment]
                buf.set_document(Document(text, cursor_position=cursor))
            finally:
                buf.read_only = was  # type: ignore[assignment]
        except Exception:  # noqa: BLE001
            try:
                buf.text = text
            except Exception:  # noqa: BLE001
                pass
        self._plan_body_sync_key = key

    def _open_current_plan_in_os(self) -> None:
        path = self._plan_body_path or self._resolve_plan_file_path_for_ui()
        if not path:
            self._set_feedback("没有 plan 文件", "warning")
            return
        cwd = getattr(self.session, "cwd", None)
        ok, message = open_local_path(path, cwd=cwd)
        self._set_feedback(message, "success" if ok else "warning")

    def _plan_action_mouse(
        self, outcome: str
    ) -> Callable[[MouseEvent], object]:
        return self._only_mouse_up(
            lambda _e: self._resolve_plan_exit(outcome),
            scroll_target="detail",
        )

    def _on_detail_scrollbar(self, scroll: int) -> None:
        """Keep detail cursor anchor on the scrollbar-driven viewport top."""
        max_y = max(0, int(self._detail_line_count) - 1)
        self._detail_cursor_line = max(0, min(max_y, int(scroll)))

    def _only_mouse_up(
        self,
        action: Callable[[MouseEvent], None],
        *,
        scroll_target: str = "auto",
    ) -> Callable[[MouseEvent], object]:
        """Fragment mouse handlers must not swallow wheel incorrectly.

        Returning None for non-UP events marks the event handled. Wheel over
        homepage task sections scroll the reading stream; modal/auth scrolls
        its own body. No pre-click chrome.
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
                        elif self._fleet_pane_open:
                            target = "fleet"
                        elif self._todo_pane_open:
                            target = "todo"
                        else:
                            target = "tasks"
                    if target == "fleet" and self._fleet_pane_open:
                        self._scroll_fleet_pane(-1 if step < 0 else 1)
                        return None
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

    def _scroll_tasks(self, delta_lines: int) -> None:
        """Scroll homepage conversation rows and pause/resume live following."""
        if self._modal_open:
            return
        win = self._task_window
        info = getattr(win, "render_info", None)
        # Window state is updated immediately by each wheel event; render_info
        # can lag one paint behind during a fast wheel gesture.
        current = int(getattr(win, "vertical_scroll", 0) or 0)
        if info is not None:
            height = max(1, int(info.window_height))
            content_max = max(0, int(info.content_height) - height)
            useful_max = max(0, int(self._latest_task_tail_line) - height + 1)
            max_scroll = min(content_max, useful_max)
        else:
            max_scroll = max(
                0,
                int(self._latest_task_tail_line),
                current + max(0, int(delta_lines)),
            )
        new_scroll = max(
            0, min(max_scroll, current + int(delta_lines))
        )
        win.vertical_scroll = new_scroll
        # Scrolling upward always means "let me read"; never let the next live
        # paint reclaim the viewport. Following resumes only after a deliberate
        # downward move reaches the real tail.
        at_latest = int(delta_lines) > 0 and new_scroll >= max_scroll
        self._follow_latest_task = at_latest
        if at_latest:
            self._selected_line = max(0, int(self._latest_task_tail_line))
        else:
            self._selected_line = new_scroll
        if self._task_selection_active and self._selected_task >= 0:
            # Manual scroll owns the cursor anchor until task selection changes.
            self._pinned_task_for_line = int(self._selected_task)
        self._task_paint_cache = None
        self.app.invalidate()

    def _sync_task_follow_scroll(self) -> None:
        """Keep the newest homepage conversation tail visible while it streams."""
        if self._modal_open or not self._follow_latest_task:
            return
        tail = max(0, int(self._latest_task_tail_line))
        self._selected_line = tail
        win = self._task_window
        info = getattr(win, "render_info", None)
        if info is None:
            return
        height = max(1, int(info.window_height))
        content_max = max(0, int(info.content_height) - height)
        target = min(content_max, max(0, tail - height + 1))
        if int(getattr(win, "vertical_scroll", 0) or 0) != target:
            win.vertical_scroll = target

    def _scroll_detail(self, delta_lines: int) -> None:
        """Scroll the auth/modal body without touching the transcript."""
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

    def _selected_task_view(self) -> TaskView | None:
        tasks = self.ledger.snapshots()
        if not tasks:
            return None
        if not self._task_selection_active or self._selected_task < 0:
            return None
        self._selected_task = max(0, min(int(self._selected_task), len(tasks) - 1))
        return tasks[self._selected_task]

    def _tab_task_cycle(self) -> None:
        """Tab switches only between reading stream and composer."""
        if self._ask_active or (
            self._modal_open and self._modal_kind == "ask"
        ):
            return
        if self._modal_open and self._modal_kind == "auth":
            return
        if self._modal_open and self._modal_kind == "plan":
            self._leave_plan_approval()
            return
        try:
            input_focused = bool(self.app.layout.has_focus(self._input))
            tasks_focused = bool(self.app.layout.has_focus(self._task_window))
        except Exception:  # noqa: BLE001
            input_focused = False
            tasks_focused = False
        if tasks_focused:
            try:
                self.app.layout.focus(self._input)
            except Exception:  # noqa: BLE001
                pass
            self.app.invalidate()
            return
        if not self._task_selection_active or self._selected_task < 0:
            self._focus_latest_task()
        if input_focused or self.ledger.snapshots():
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
        # Force re-pin of the cursor line onto the newest task on next paint.
        self._pinned_task_for_line = None
        self._render_tasks()
        self.app.invalidate()
        return bool(self._task_refs)

    def _open_selected_task(self) -> None:
        task = self._selected_task_view()
        if task is None:
            return
        if task.plan_state == "awaiting_approval":
            self._open_plan_approval(task.id)

    def _reset_detail_cursor_state(self) -> None:
        """Drop stale detail height before the next body paint measures it."""
        self._detail_cursor_line = 0
        self._detail_line_count = 1

    def _open_agent(self, task_id: str, agent_id: str) -> None:
        """Jump to an Agent's inline transcript; no generic detail page."""
        tasks = self.ledger.snapshots()
        task_index = next(
            (i for i, item in enumerate(tasks) if item.id == task_id),
            None,
        )
        if task_index is None:
            return
        task = tasks[task_index]
        if not any(str(item.id) == str(agent_id) for item in task.agents):
            return
        self._modal_open = False
        self._modal_ref = None
        self._selected_task = task_index
        self._task_selection_active = True
        self._follow_latest_task = False
        self._pinned_task_for_line = None
        self._task_paint_cache = None
        self._render_tasks()
        anchor = self._agent_anchor_lines.get(
            (task_id, str(agent_id)),
            self._task_anchor_lines.get(task_id, 0),
        )
        self._selected_line = max(0, int(anchor))
        try:
            self._task_window.vertical_scroll = max(0, int(anchor) - 2)
            self.app.layout.focus(self._task_window)
        except Exception:  # noqa: BLE001
            pass
        self.app.invalidate()

    def _open_plan_approval(self, task_id: str) -> None:
        """Open the sole non-home product page for one awaiting revision."""
        tasks = self.ledger.snapshots()
        task_index = next(
            (i for i, item in enumerate(tasks) if item.id == task_id),
            None,
        )
        if task_index is None:
            return
        task = tasks[task_index]
        if task.plan_state != "awaiting_approval":
            self._set_feedback("当前没有待审批计划", "warning")
            return
        main_id = f"{task.id}:main"
        if task.agents:
            main_id = str(task.agents[0].id)
            for agent in task.agents:
                if (
                    str(agent.id).endswith(":main")
                    or str(agent.label).strip().upper() == "MAIN"
                ):
                    main_id = str(agent.id)
                    break
        self._close_tool_preview(restore_focus=False)
        self._tool_hover_ref = None
        self._selected_task = task_index
        self._task_selection_active = True
        self._follow_latest_task = task_index == len(tasks) - 1
        self._plan_ui_task_id = task.id
        self._modal_kind = "plan"
        self._modal_ref = (task.id, main_id)
        self._modal_open = True
        self._plan_body_sync_key = None
        self._detail_input.text = ""
        self._sync_plan_body_buffer(force=True)
        try:
            self.app.layout.focus(self._plan_body)
        except Exception:  # noqa: BLE001
            pass
        self.app.invalidate()

    def _leave_plan_approval(self) -> None:
        """Return to the reading stream without changing approval state."""
        self._modal_open = False
        self._modal_kind = "plan"
        self._modal_ref = None
        self._detail_input.text = ""
        try:
            self.app.layout.focus(
                self._task_window if self.ledger.snapshots() else self._input
            )
        except Exception:  # noqa: BLE001
            pass
        self.app.invalidate()

    def _task_plan_review_mouse(
        self, task_id: str
    ) -> Callable[[MouseEvent], object]:
        """Homepage entry into the dedicated approval page."""

        def _on_up(_event: MouseEvent) -> None:
            self._open_plan_approval(task_id)

        return self._only_mouse_up(_on_up, scroll_target="tasks")

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
        if self._modal_kind == "plan":
            self._leave_plan_approval()
            return
        was_auth = self._modal_kind == "auth"
        if self._modal_kind == "ask" and self._ask_active:
            # Closing × while questionnaire is open == cancel (do not hang worker).
            self._resolve_ask({"outcome": "cancelled"})
            return
        had_tasks = bool(self.ledger.snapshots())
        self._modal_open = False
        self._modal_kind = "plan"
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
            image_attachments = self._pending_image_attachments
            self._pending_prompt = None
            self._pending_image_attachments = ()
            self._start_task(
                prompt,
                image_attachments=image_attachments,
            )

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

    def _task_chat_toggle_mouse(
        self, task_id: str
    ) -> Callable[[MouseEvent], object]:
        """Collapse/expand only the homepage conversation body."""

        def _toggle(_event: MouseEvent) -> None:
            if task_id in self._collapsed_task_chats:
                self._collapsed_task_chats.remove(task_id)
            else:
                self._collapsed_task_chats.add(task_id)
            self._task_paint_cache = None
            self._pinned_task_for_line = None
            self.app.invalidate()

        return self._only_mouse_up(_toggle, scroll_target="tasks")

    def _task_link_mouse(
        self,
        target: str,
    ) -> Callable[[MouseEvent], object]:
        """Open an underlined transcript path or HTTPS link on plain click."""
        from prompt_toolkit.mouse_events import MouseButton

        clean_target = (target or "").strip().strip("\"'")

        def handler(event: MouseEvent) -> object:
            button = getattr(event, "button", None)
            if event.event_type is MouseEventType.MOUSE_MOVE:
                self._tool_hover_ref = None
                if not self._tool_preview_pinned:
                    self._close_tool_preview(restore_focus=False)
                return None
            if event.event_type is MouseEventType.MOUSE_DOWN:
                return None
            if event.event_type is not MouseEventType.MOUSE_UP:
                return None
            if button not in (None, MouseButton.LEFT, MouseButton.UNKNOWN):
                return NotImplemented
            self._tool_hover_ref = None
            self._close_tool_preview(restore_focus=False)
            if re.match(r"^https?://", clean_target, re.IGNORECASE):
                try:
                    opened = bool(webbrowser.open(clean_target, new=2))
                except Exception:  # noqa: BLE001
                    opened = False
                message = (
                    "已在浏览器打开链接"
                    if opened
                    else "无法打开链接"
                )
                self._set_feedback(
                    message,
                    "success" if opened else "warning",
                )
            else:
                cwd = getattr(self.session, "cwd", None)
                opened, message = open_local_path(clean_target, cwd=cwd)
                self._set_feedback(
                    message,
                    "success" if opened else "warning",
                )
            self.app.invalidate()
            return None

        return handler

    def _task_tool_mouse(
        self,
        ref: tuple[str, str, str],
    ) -> Callable[[MouseEvent], object]:
        """Ctrl+move previews; Ctrl+click pins. No tool-navigation mode."""
        from prompt_toolkit.mouse_events import MouseButton

        def handler(event: MouseEvent) -> object:
            if event.event_type is MouseEventType.MOUSE_MOVE:
                self._tool_hover_ref = ref
                if _mouse_control_held(event):
                    if not self._tool_preview_pinned:
                        self._show_tool_preview(ref, pinned=False)
                    return None
                if not self._tool_preview_pinned:
                    self._close_tool_preview(restore_focus=False)
                return None
            button = getattr(event, "button", None)
            if event.event_type is MouseEventType.MOUSE_DOWN:
                self._tool_hover_ref = ref
                if button in (None, MouseButton.LEFT) and _mouse_control_held(event):
                    self._show_tool_preview(ref, pinned=False)
                    return None
                self._close_tool_preview(restore_focus=False)
                return None
            if event.event_type is MouseEventType.MOUSE_UP:
                self._tool_hover_ref = ref
                if button in (None, MouseButton.LEFT) and _mouse_control_held(event):
                    self._show_tool_preview(ref, pinned=True)
                    return None
                if not self._tool_preview_pinned:
                    self._close_tool_preview(restore_focus=False)
                return None
            return NotImplemented

        return handler

    def _tool_record_for_ref(
        self,
        ref: tuple[str, str, str] | None = None,
    ) -> tuple[TaskView, Any, DetailRecord] | None:
        target = ref or self._tool_preview_ref
        if target is None:
            return None
        task_id, agent_id, record_id = target
        task = next(
            (item for item in self.ledger.snapshots() if item.id == task_id),
            None,
        )
        if task is None:
            return None
        agent = next(
            (item for item in task.agents if str(item.id) == agent_id),
            None,
        )
        if agent is None:
            return None
        snapshot = self._snapshot_for_task_agent(task, agent)
        if snapshot is None:
            return None
        record = next(
            (
                item
                for item in snapshot.records
                if str(item.id) == record_id
                and str(item.actor).strip().upper() == "TOOL"
            ),
            None,
        )
        if record is None:
            return None
        return task, agent, record

    def _show_tool_preview(
        self,
        ref: tuple[str, str, str],
        *,
        pinned: bool,
    ) -> None:
        if self._modal_open or self._tool_record_for_ref(ref) is None:
            return
        if self._tool_preview_pinned and not pinned:
            return
        changed = self._tool_preview_ref != ref
        self._tool_preview_ref = ref
        self._tool_preview_pinned = bool(pinned)
        if changed:
            self._tool_preview_buffer_key = None
        self._sync_tool_preview_buffer()
        if pinned:
            try:
                self.app.layout.focus(self._tool_preview_body)
            except Exception:  # noqa: BLE001
                pass
        self.app.invalidate()

    def _close_tool_preview(self, *, restore_focus: bool) -> None:
        if self._tool_preview_ref is None:
            return
        was_focused = False
        try:
            was_focused = bool(self.app.layout.has_focus(self._tool_preview_body))
        except Exception:  # noqa: BLE001
            pass
        self._tool_preview_ref = None
        self._tool_preview_pinned = False
        self._tool_preview_buffer_key = None
        if restore_focus and was_focused:
            try:
                self.app.layout.focus(self._task_window)
            except Exception:  # noqa: BLE001
                pass
        self.app.invalidate()

    def _tool_preview_visible(self) -> bool:
        return bool(
            not self._modal_open
            and self._tool_preview_ref is not None
            and self._tool_record_for_ref() is not None
        )

    def _sync_tool_preview_buffer(self) -> None:
        resolved = self._tool_record_for_ref()
        if resolved is None:
            return
        _task, _agent, record = resolved
        parts: list[str] = []
        for block in record.blocks:
            label = (block.label or "").strip()
            if label:
                parts.append(label.upper())
            parts.append(block.text or "（空）")
        text = "\n\n".join(parts).strip() or "此工具没有正文输出。"
        signature = (
            self._tool_preview_ref,
            record.status,
            len(text),
            text[:96],
            text[-64:],
        )
        if signature == self._tool_preview_buffer_key:
            return
        buffer = self._tool_preview_body.buffer
        cursor = min(int(buffer.cursor_position or 0), len(text))
        try:
            buffer.set_document(
                Document(text, cursor_position=cursor),
                bypass_readonly=True,
            )
        except TypeError:
            was = buffer.read_only
            try:
                buffer.read_only = False  # type: ignore[assignment]
                buffer.set_document(Document(text, cursor_position=cursor))
            finally:
                buffer.read_only = was  # type: ignore[assignment]
        self._tool_preview_buffer_key = signature

    def _render_tool_preview_header(self) -> StyleAndTextTuples:
        resolved = self._tool_record_for_ref()
        if resolved is None:
            return [("class:tool.preview.meta", "  工具记录不可用\n")]
        task, agent, record = resolved
        width = max(12, _terminal_width() - 14)
        status = str(record.status or "").lower()
        if status in {"pending", "running", "waiting"}:
            status_text, status_style = "运行中", "class:tool.preview.running"
        elif status in {"failed", "error"}:
            status_text, status_style = "失败", "class:tool.preview.failed"
        else:
            status_text, status_style = "完成", "class:tool.preview.done"
        title = _truncate_display(f"  {record.title}", width)
        scope = _truncate_display(
            f"  {agent.label or 'AGENT'} · {task.title}", width
        )
        return [
            ("class:tool.preview.title", title),
            ("class:tool.preview.meta", "  "),
            (status_style, status_text),
            ("", "\n"),
            ("class:tool.preview.meta", scope),
        ]

    def _render_tool_preview_footer(self) -> StyleAndTextTuples:
        if self._tool_preview_pinned:
            text = "  已固定 · 滚轮浏览 · 拖选复制 · Esc 关闭"
        else:
            text = "  Ctrl+点击固定 · 松开 Ctrl 关闭"
        return [("class:tool.preview.footer", text)]

    def _wire_buffer_outside_tool_preview(self, area: TextArea) -> None:
        """A normal click in an input closes a pinned tool float."""
        control = area.control
        original = control.mouse_handler

        def handler(event: MouseEvent) -> object:
            if event.event_type is MouseEventType.MOUSE_DOWN:
                if (
                    self._task_copy_anchor is not None
                    or self._task_copy_pending is not None
                ):
                    self._clear_task_text_selection(invalidate=False)
                    self.app.invalidate()
            if event.event_type is MouseEventType.MOUSE_MOVE:
                self._tool_hover_ref = None
                if not self._tool_preview_pinned:
                    self._close_tool_preview(restore_focus=False)
            if (
                self._tool_preview_ref is not None
                and event.event_type is MouseEventType.MOUSE_DOWN
                and not _mouse_control_held(event)
            ):
                self._close_tool_preview(restore_focus=False)
            return original(event)

        control.mouse_handler = handler  # type: ignore[method-assign]

    def _wire_task_text_selection_control(self) -> None:
        """Let drag-copy win even over styled links and full-width padding."""
        from prompt_toolkit.mouse_events import MouseButton

        original = self._task_control.mouse_handler

        def handler(event: MouseEvent) -> object:
            button = getattr(event, "button", None)
            is_left = button in (
                None,
                MouseButton.LEFT,
                MouseButton.UNKNOWN,
            )
            if (
                event.event_type is MouseEventType.MOUSE_DOWN
                and is_left
                and not _mouse_control_held(event)
            ):
                self._begin_task_text_selection(event)
            elif (
                event.event_type is MouseEventType.MOUSE_MOVE
                and self._move_task_text_selection(event)
            ):
                return None
            elif (
                event.event_type is MouseEventType.MOUSE_UP
                and is_left
                and (
                    self._task_copy_dragging
                    or self._task_copy_pending is not None
                )
                and self._finish_task_text_selection(event)
            ):
                self._task_mouse_down_index = None
                return None
            return original(event)

        self._task_control.mouse_handler = handler  # type: ignore[method-assign]

    @staticmethod
    def _task_mouse_point(event: MouseEvent) -> tuple[int, int]:
        return (
            max(0, int(event.position.y)),
            max(0, int(event.position.x)),
        )

    def _begin_task_text_selection(self, event: MouseEvent) -> None:
        point = self._task_mouse_point(event)
        had_selection = self._task_copy_anchor is not None
        self._task_copy_pending = point
        self._task_copy_dragging = False
        self._task_copy_anchor = None
        self._task_copy_cursor = None
        self._task_copy_text = ""
        if had_selection:
            self._task_paint_cache = None
            self.app.invalidate()

    def _move_task_text_selection(self, event: MouseEvent) -> bool:
        point = self._task_mouse_point(event)
        if not self._task_copy_dragging:
            pending = self._task_copy_pending
            if pending is None:
                return False
            if point == pending:
                return True
            self._task_copy_anchor = pending
            self._task_copy_cursor = point
            self._task_copy_pending = None
            self._task_copy_dragging = True
            self._follow_latest_task = False
            self._task_paint_cache = None
            self.app.invalidate()
            return True
        if self._task_copy_anchor is None:
            return False
        if point != self._task_copy_cursor:
            self._task_copy_cursor = point
            self._task_paint_cache = None
            self.app.invalidate()
        return True

    def _finish_task_text_selection(self, event: MouseEvent) -> bool:
        """Finish a drag selection; return whether it consumed the click."""
        if not self._task_copy_dragging:
            self._task_copy_pending = None
            return False
        anchor = self._task_copy_anchor
        if anchor is None:
            self._task_copy_dragging = False
            self._task_copy_pending = None
            return False
        self._task_copy_cursor = self._task_mouse_point(event)
        if self._task_copy_cursor == anchor:
            self._clear_task_text_selection(invalidate=False)
            self._task_paint_cache = None
            return False

        self._task_copy_text = self._extract_task_text_selection()
        self._task_copy_dragging = False
        self._task_copy_pending = None
        self._task_paint_cache = None
        self.app.invalidate()
        return True

    def _clear_task_text_selection(self, *, invalidate: bool = True) -> None:
        self._task_copy_pending = None
        self._task_copy_dragging = False
        self._task_copy_anchor = None
        self._task_copy_cursor = None
        self._task_copy_text = ""
        self._task_paint_cache = None
        if invalidate:
            self.app.invalidate()

    def _task_text_selection_span(self, row: int) -> tuple[int, int] | None:
        anchor = self._task_copy_anchor
        cursor = self._task_copy_cursor
        if anchor is None or cursor is None or anchor == cursor:
            return None
        start, end = sorted((anchor, cursor))
        if row < start[0] or row > end[0]:
            return None
        start_col = start[1] if row == start[0] else 0
        end_col = end[1] + 1 if row == end[0] else 1 << 30
        return start_col, max(start_col, end_col)

    def _paint_task_text_selection(
        self,
        parts: list[tuple[str, str, Callable[[MouseEvent], object]]],
        row: int,
    ) -> list[tuple[str, str, Callable[[MouseEvent], object]]]:
        """Overlay selection without replacing each span's semantic color."""
        span = self._task_text_selection_span(row)
        if span is None:
            return parts
        start_col, end_col = span
        output: list[tuple[str, str, Callable[[MouseEvent], object]]] = []
        column = 0
        previous_selected = False

        def append(
            style: str,
            text: str,
            mouse: Callable[[MouseEvent], object],
        ) -> None:
            if (
                output
                and output[-1][0] == style
                and output[-1][2] is mouse
            ):
                previous = output[-1]
                output[-1] = (style, previous[1] + text, mouse)
            else:
                output.append((style, text, mouse))

        for style, text, mouse in parts:
            for char in text:
                cell_width = max(0, get_cwidth(char))
                selected = (
                    column < end_col
                    and column + cell_width > start_col
                )
                if cell_width == 0:
                    selected = previous_selected
                painted_style = (
                    f"{style} class:task.selection"
                    if selected
                    else style
                )
                append(painted_style, char, mouse)
                previous_selected = selected
                column += cell_width
        return output

    def _task_rendered_plain_lines(self) -> list[str]:
        fragments = getattr(self._task_control, "_fragments", None) or []
        return [
            "".join(part[1] for part in row)
            for row in split_lines(fragments)
        ]

    @staticmethod
    def _display_start_index(text: str, column: int) -> int:
        cell = 0
        for index, char in enumerate(text):
            width = max(0, get_cwidth(char))
            if width and cell + width > column:
                return index
            cell += width
        return len(text)

    @staticmethod
    def _display_end_index(text: str, column: int) -> int:
        cell = 0
        for index, char in enumerate(text):
            width = max(0, get_cwidth(char))
            if width and cell + width > column:
                return index + 1
            cell += width
        return len(text)

    def _extract_task_text_selection(self) -> str:
        anchor = self._task_copy_anchor
        cursor = self._task_copy_cursor
        lines = self._task_rendered_plain_lines()
        if anchor is None or cursor is None or not lines:
            return ""
        start, end = sorted((anchor, cursor))
        first_row = min(start[0], len(lines) - 1)
        last_row = min(end[0], len(lines) - 1)
        if first_row > last_row:
            return ""

        selected: list[str] = []
        for row in range(first_row, last_row + 1):
            text = lines[row]
            left = (
                self._display_start_index(text, start[1])
                if row == first_row
                else 0
            )
            right = (
                self._display_end_index(text, end[1])
                if row == last_row
                else len(text)
            )
            selected.append(text[left:right].rstrip())
        return "\n".join(selected).rstrip("\n")

    def _copy_task_text_to_clipboard(self, text: str) -> bool:
        if not text:
            return False
        try:
            self.app.clipboard.set_data(ClipboardData(text))
        except Exception:  # noqa: BLE001
            pass
        # Copy is intentionally silent: adding a feedback row changes layout
        # height and makes the reading viewport appear to jump.
        return set_system_clipboard_text(text)

    def _task_section_mouse(
        self, task_index: int
    ) -> Callable[[MouseEvent], object]:
        """Reading-stream click selects a task; it never opens a detail page."""
        from prompt_toolkit.mouse_events import MouseButton

        def handler(event: MouseEvent) -> object:
            btn = getattr(event, "button", None)
            if event.event_type is MouseEventType.MOUSE_MOVE:
                self._tool_hover_ref = None
                if not self._tool_preview_pinned:
                    self._close_tool_preview(restore_focus=False)
                return None
            if event.event_type is MouseEventType.MOUSE_DOWN:
                if btn in (None, MouseButton.LEFT):
                    self._task_mouse_down_index = task_index
                if not _mouse_control_held(event):
                    self._close_tool_preview(restore_focus=False)
                return None
            if event.event_type is not MouseEventType.MOUSE_UP:
                return NotImplemented
            if btn not in (None, MouseButton.LEFT):
                self._task_mouse_down_index = None
                return NotImplemented
            down = self._task_mouse_down_index
            self._task_mouse_down_index = None
            # Must press and release inside the same task section.
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
        self._clear_task_text_selection(invalidate=False)
        if focus_input:
            try:
                self.app.layout.focus(self._input)
            except Exception:  # noqa: BLE001
                pass
        self._task_paint_cache = None
        self.app.invalidate()

    def _task_gap_mouse(self) -> Callable[[MouseEvent], object]:
        """Gap between task sections: click clears selection."""

        def handler(event: MouseEvent) -> object:
            if event.event_type is MouseEventType.MOUSE_MOVE:
                self._tool_hover_ref = None
                if not self._tool_preview_pinned:
                    self._close_tool_preview(restore_focus=False)
                return None
            if event.event_type is MouseEventType.MOUSE_DOWN:
                self._task_mouse_down_index = None
                self._clear_task_text_selection(invalidate=False)
                self._close_tool_preview(restore_focus=False)
                return None
            if event.event_type is MouseEventType.MOUSE_UP:
                self._clear_task_selection(focus_input=True)
                return None
            return NotImplemented

        return handler

    def _task_void_mouse(self) -> Callable[[MouseEvent], object]:
        """Empty area below the stream: click clears selection."""

        def handler(event: MouseEvent) -> object:
            if event.event_type is MouseEventType.MOUSE_MOVE:
                self._tool_hover_ref = None
                if not self._tool_preview_pinned:
                    self._close_tool_preview(restore_focus=False)
                return None
            if event.event_type is MouseEventType.MOUSE_DOWN:
                self._task_mouse_down_index = None
                self._clear_task_text_selection(invalidate=False)
                self._close_tool_preview(restore_focus=False)
                return None
            if event.event_type is MouseEventType.MOUSE_UP:
                self._clear_task_selection(focus_input=True)
                return None
            return NotImplemented

        return handler


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
        """Focus on click + Ctrl+click path /「查看图片(…)」chip → open file.

        prompt_toolkit's BufferControl only focuses when ``focus_on_click`` is
        set; we also force-focus here so empty padding / middle cells always
        grab the caret (chrome-only hits used to be the only reliable path).
        """
        control = area.control
        original = control.mouse_handler

        def mouse_handler(mouse_event: MouseEvent) -> object:
            from prompt_toolkit.mouse_events import MouseButton

            # Plain click anywhere in the field → focus (even before original).
            if mouse_event.event_type in {
                MouseEventType.MOUSE_DOWN,
                MouseEventType.MOUSE_UP,
            }:
                btn = getattr(mouse_event, "button", None)
                if btn in (None, MouseButton.LEFT, MouseButton.UNKNOWN):
                    try:
                        if get_app().layout.current_control is not control:
                            get_app().layout.focus(area.window)
                    except Exception:  # noqa: BLE001
                        pass

            result = original(mouse_event)
            if mouse_event.event_type is not MouseEventType.MOUSE_UP:
                return result
            btn = getattr(mouse_event, "button", None)
            if btn not in (None, MouseButton.LEFT):
                return result
            if not _mouse_control_held(mouse_event):
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

def run_tui(session: Any, *, initial_prompt: str | None = None) -> None:
    CodeDoggyTUI(session, initial_prompt=initial_prompt).run()


def agent_summary_text_from_messages(messages: list[Any]) -> str:
    """Return assistant prose for the lightweight live-agent fallback."""
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


def _agent_status_mark(status: str) -> str:
    """One-glyph status for parallel roster rows."""
    st = (status or "").strip().lower()
    if st in {"pending", "running", "waiting"}:
        return "⋯"
    if st in {"completed", "done"}:
        return "✓"
    if st in {"failed", "error", "max_turns"}:
        return "×"
    if st in {"cancelled", "canceled"}:
        return "–"
    return "·"


def task_report_from_agent(text: str, *, max_chars: int | None = None) -> str:
    """Boss-list summary from MAIN wording — first paragraph, full text.

    Display wraps in the task section; do not hard-crop with ellipsis here.
    ``max_chars`` remains optional for callers that still want a soft cap.
    """
    clean = text.strip()
    if not clean:
        return "任务已结束。"
    soft = _friendly_failure_toast(clean)
    if soft != "任务未能完成" or _looks_like_transport_error(clean):
        # Prefer the human one-liner; canonical messages retain the full text.
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
    # Keep the terminal status short; the transcript already contains the prose.
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


# Outer layout air (must match body HSplit/VSplit pad windows).
# Renders that paint full-width lines MUST use _content_width(), not raw
# terminal size — otherwise right edges clip and look "truncated".
_EDGE_PAD_X = 2
_EDGE_PAD_Y = 1


def _terminal_width() -> int:
    try:
        return get_app().output.get_size().columns
    except Exception:  # noqa: BLE001
        return shutil.get_terminal_size(fallback=(100, 30)).columns


def _content_width() -> int:
    """Columns available inside the padded main chrome (tasks / prompt / panes)."""
    return max(1, _terminal_width() - 2 * _EDGE_PAD_X)


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
    """Wall-clock seconds for a task section; None if never started."""
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


def _is_tool_activity_line(text: str) -> bool:
    """True when transport text describes a tool rather than assistant prose."""
    t = (text or "").strip()
    if not t:
        return False
    if t.startswith(("→", "✓", "✗")):
        return True
    if "· 调用中" in t or "· 完成" in t or "· 失败" in t or "· 仍在 " in t:
        return True
    return False


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
    if task.phase == "planning" or (
        task.plan_state == "planning" and task.phase in {"planning", "plan_review"}
    ):
        return "起草"
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
    if task.plan_state == "awaiting_approval" or task.phase == "plan_review":
        return "class:task.status.running"
    return "class:task.status"


# Doggy neon couple splash — single source in doggy_brand.py
from codedoggy.tui.doggy_brand import (  # noqa: E402
    _DOGGY_ART_PALETTE,
    _DOGGY_COUPLE_ART,
    _DOGGY_COUPLE_FRAMES,
    _animate_doggy_couple,
    _compose_doggy_night,
    _half_block,
    _render_doggy_empty,
    _render_doggy_idle_panel,
)

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


_TASK_MD_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
_TASK_MD_FENCE_RE = re.compile(r"^\s*```([^`]*)$")
_TASK_MD_HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_TASK_MD_TABLE_RULE_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_TASK_MD_QUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_TASK_MD_ORDERED_RE = re.compile(r"^(\s*)(\d+[.)])\s+(.*)$")
_TASK_MD_BULLET_RE = re.compile(r"^(\s*)[-*+•]\s+(.*)$")
_TASK_MD_INLINE_RE = re.compile(
    r"`([^`\n]+)`"
    r"|\[([^\]\n]+)\]\(([^)\n]+)\)"
    r"|\*\*(.+?)\*\*"
    r"|__(.+?)__"
    r"|~~(.+?)~~"
    r"|(?<!\*)\*([^*\n]+)\*(?!\*)"
)


def _task_inline_open_target(text: str) -> str | None:
    """Return a path/URL target for clickable inline-code spans."""
    target = (text or "").strip().strip("\"'")
    if not target or any(char.isspace() for char in target):
        return None
    if re.match(r"^https?://", target, re.IGNORECASE):
        return target
    if is_openable_file_path(target) or "/" in target or "\\" in target:
        return target
    return None


def _task_inline_fragments(
    text: str,
    *,
    base_style: str = "class:task.chat.body",
    link_mouse: (
        Callable[[str], Callable[[MouseEvent], object]] | None
    ) = None,
) -> StyleAndTextTuples:
    """Render the small inline Markdown set used by homepage chat prose."""
    if not text:
        return [(base_style, "")]
    out: StyleAndTextTuples = []
    pos = 0
    heading_style = base_style.startswith("class:task.chat.h")
    for match in _TASK_MD_INLINE_RE.finditer(text):
        if match.start() > pos:
            out.append((base_style, text[pos : match.start()]))
        if match.group(1) is not None:
            code = match.group(1)
            target = _task_inline_open_target(code)
            if target is not None and link_mouse is not None:
                out.append(
                    (
                        "class:task.chat.path",
                        code,
                        link_mouse(target),
                    )
                )
            else:
                out.append(("class:task.chat.code", code))
        elif match.group(2) is not None:
            label = match.group(2)
            target = (match.group(3) or "").strip()
            if target and link_mouse is not None:
                out.append(
                    (
                        "class:task.chat.link",
                        label,
                        link_mouse(target),
                    )
                )
            else:
                out.append(("class:task.chat.link", label))
        elif match.group(4) is not None or match.group(5) is not None:
            strong = match.group(4) or match.group(5) or ""
            out.append(
                (
                    base_style if heading_style else "class:task.chat.strong",
                    strong,
                )
            )
        elif match.group(6) is not None:
            out.append(("class:task.chat.muted", match.group(6)))
        else:
            out.append(("class:task.chat.italic", match.group(7) or ""))
        pos = match.end()
    if pos < len(text):
        out.append((base_style, text[pos:]))
    return out or [(base_style, text)]


def _wrap_task_chat_fragments(
    fragments: StyleAndTextTuples,
    width: int,
) -> list[StyleAndTextTuples]:
    """Wrap styled spans by terminal cell width without losing their styles."""
    width = max(1, int(width))
    rows: list[StyleAndTextTuples] = []
    row: StyleAndTextTuples = []
    used = 0

    def append_piece(
        style: str,
        piece: str,
        mouse: Callable[[MouseEvent], object] | None,
    ) -> None:
        last_mouse = (
            row[-1][2]
            if row and len(row[-1]) >= 3
            else None
        )
        if row and row[-1][0] == style and last_mouse is mouse:
            if mouse is None:
                row[-1] = (style, row[-1][1] + piece)
            else:
                row[-1] = (style, row[-1][1] + piece, mouse)
        else:
            if mouse is None:
                row.append((style, piece))
            else:
                row.append((style, piece, mouse))

    for item in fragments:
        style = item[0] if item else ""
        text = item[1] if len(item) > 1 else ""
        mouse = item[2] if len(item) >= 3 else None
        for char in text:
            char_width = max(0, get_cwidth(char))
            if row and used + char_width > width:
                rows.append(row)
                row = []
                used = 0
            append_piece(style, char, mouse)
            used += char_width
    if row:
        rows.append(row)
    return rows or [[("class:task.chat.body", "")]]


def _pad_task_chat_rows(
    rows: list[StyleAndTextTuples],
    width: int,
    style: str,
) -> list[StyleAndTextTuples]:
    """Extend a semantic block background to the full reading column."""
    out: list[StyleAndTextTuples] = []
    for row in rows:
        used = sum(get_cwidth(item[1]) for item in row)
        padded = list(row)
        if used < width:
            padded.append((style, " " * (width - used)))
        out.append(padded)
    return out


def _render_task_chat_rows(
    text: str,
    width: int,
    *,
    link_mouse: (
        Callable[[str], Callable[[MouseEvent], object]] | None
    ) = None,
) -> list[StyleAndTextTuples]:
    """Homepage Markdown-lite with readable hierarchy and no raw markers."""
    width = max(1, int(width))
    rows: list[StyleAndTextTuples] = []
    in_fence = False
    raw_text = (text or "").replace("\r\n", "\n").replace("\r", "\n")

    for raw in raw_text.split("\n"):
        fence = _TASK_MD_FENCE_RE.match(raw)
        if fence:
            if in_fence:
                in_fence = False
            else:
                in_fence = True
                language = (fence.group(1) or "").strip()
                if language:
                    rows.extend(
                        _pad_task_chat_rows(
                            _wrap_task_chat_fragments(
                                [
                                    ("class:task.chat.code.rail", "│ "),
                                    ("class:task.chat.code.meta", language),
                                ],
                                width,
                            ),
                            width,
                            "class:task.chat.code.block",
                        )
                    )
            continue

        if in_fence:
            rows.extend(
                _pad_task_chat_rows(
                    _wrap_task_chat_fragments(
                        [
                            ("class:task.chat.code.rail", "│ "),
                            *highlight_code_line(
                                raw or " ",
                                style_prefix="task.chat.code",
                            ),
                        ],
                        width,
                    ),
                    width,
                    "class:task.chat.code.block",
                )
            )
            continue

        heading = _TASK_MD_HEADING_RE.match(raw)
        if heading:
            level = len(heading.group(1))
            style = {
                1: "class:task.chat.h1",
                2: "class:task.chat.h2",
            }.get(level, "class:task.chat.h3")
            rows.extend(
                _wrap_task_chat_fragments(
                    _task_inline_fragments(
                        heading.group(2),
                        base_style=style,
                        link_mouse=link_mouse,
                    ),
                    width,
                )
            )
            continue

        if _TASK_MD_HR_RE.match(raw):
            rows.append(
                [("class:task.chat.rule", "─" * width)]
            )
            continue

        if _TASK_MD_TABLE_RULE_RE.match(raw) and "|" in raw:
            rows.extend(
                _wrap_task_chat_fragments(
                    [("class:task.chat.rule", raw.strip())],
                    width,
                )
            )
            continue

        quote = _TASK_MD_QUOTE_RE.match(raw)
        if quote:
            spans: StyleAndTextTuples = [
                ("class:task.chat.code.rail", "│ "),
            ]
            spans.extend(
                _task_inline_fragments(
                    quote.group(1),
                    base_style="class:task.chat.quote",
                    link_mouse=link_mouse,
                )
            )
            rows.extend(_wrap_task_chat_fragments(spans, width))
            continue

        ordered = _TASK_MD_ORDERED_RE.match(raw)
        if ordered:
            indent, marker, body = ordered.groups()
            spans = [
                ("class:task.chat.muted", indent),
                ("class:task.chat.marker", f"{marker} "),
            ]
            spans.extend(
                _task_inline_fragments(body, link_mouse=link_mouse)
            )
            rows.extend(_wrap_task_chat_fragments(spans, width))
            continue

        bullet = _TASK_MD_BULLET_RE.match(raw)
        if bullet:
            indent, body = bullet.groups()
            spans = [
                ("class:task.chat.muted", indent),
                ("class:task.chat.marker", "• "),
            ]
            spans.extend(
                _task_inline_fragments(body, link_mouse=link_mouse)
            )
            rows.extend(_wrap_task_chat_fragments(spans, width))
            continue

        rows.extend(
            _wrap_task_chat_fragments(
                _task_inline_fragments(raw, link_mouse=link_mouse),
                width,
            )
        )

    return rows or [[("class:task.chat.body", "")]]


def _wrap_display_lines(
    text: str, width: int, *, max_lines: int | None = 40
) -> list[str]:
    """Wrap text to display-cell width.

    ``max_lines=None`` means no row cap in the reading stream. A positive
    int still hard-stops after that many display rows (finished 2-line abstract).
    """
    width = max(1, int(width))
    limit = None if max_lines is None else max(1, int(max_lines))
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return [""]
    lines: list[str] = []
    for para in raw.split("\n"):
        if not para:
            lines.append("")
            if limit is not None and len(lines) >= limit:
                break
            continue
        buf: list[str] = []
        used = 0
        for char in para:
            cw = get_cwidth(char)
            if buf and used + cw > width:
                lines.append("".join(buf))
                if limit is not None and len(lines) >= limit:
                    return lines
                buf = [char]
                used = cw
            else:
                buf.append(char)
                used += cw
        if buf:
            lines.append("".join(buf))
            if limit is not None and len(lines) >= limit:
                break
    return lines or [""]
