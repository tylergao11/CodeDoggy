"""Grok-shell app — Doggy Session wired to ported pager painters.

Presentation modules are source ports of xai-grok-pager.
This file only owns: session turn loop, live/subagent listeners, Doggy
exceptions (Ctrl+L login, image paste, doggy brand splash).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.filters import Condition
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
    Window,
)
from prompt_toolkit.layout.controls import UIControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.output.color_depth import ColorDepth

from codedoggy.attachments import AttachmentError, ImageAttachment
from codedoggy.session.types import SessionPhase
from codedoggy.tui import surface as session_surface
from codedoggy.tui.clipboard_image import (
    coerce_image_path_text,
    get_system_clipboard_text,
    insert_image_chip,
    save_clipboard_image,
    set_system_clipboard_text,
)
from codedoggy.tui.login_wizard import AuthWizard, WizardStep, run_browser_login
from codedoggy.tui.open_path import (
    VIEW_IMAGE_LABEL,
    extract_image_chip_paths,
    resolve_openable_path,
    strip_image_chips,
)
from codedoggy.tui_v2 import shortcuts, status_bar, turn_status
from codedoggy.tui_v2.brand import render_welcome
from codedoggy.tui_v2.context_bar import (
    context_bar_fragments,
    format_token_pair,
    fmt_tokens,
    usage_percentage,
    usage_style,
)
from codedoggy.tui_v2.prompt import PromptInfo, PromptStyle, create_prompt
from codedoggy.tui_v2.scrollback import (
    ScrollbackState,
    entry_at_line,
    reconcile_turn_finish,
    render_scrollback,
)
from codedoggy.tui_v2.theme import build_style

logger = logging.getLogger(__name__)

# Cap session-history seed so resume does not flood the viewport.
HISTORY_SEED_CAP = 80


class _SBControl(FormattedTextControl):
    def __init__(self, host: "GrokShellApp") -> None:
        self._host = host
        super().__init__(
            host._paint_scrollback,
            focusable=True,
            show_cursor=False,
            get_cursor_position=lambda: None,
        )

    def mouse_handler(self, event: MouseEvent) -> object:  # type: ignore[override]
        return self._host._sb_mouse(event)


class GrokShellApp:
    def __init__(self, session: Any, *, initial_prompt: str | None = None) -> None:
        self.session = session
        self._initial = initial_prompt
        self._closing = False
        self._scroll = ScrollbackState()
        self._worker: threading.Thread | None = None
        self._turn_started: float | None = None
        self._tick = 0
        self._feedback = ""
        self._feedback_kind = "info"
        self._focus = "prompt"
        self._auth_open = False
        self._auth = AuthWizard()
        self._pending_prompt: str | None = None
        self._pending_attach: tuple[ImageAttachment, ...] = ()
        self._sub_listener = False
        self._bg_listener = False
        self._esc_clear_until = 0.0
        self._cancel_grace = 0.0
        self._stream_buf: list[str] = []
        self._stream_state: dict[str, Any] = {}
        # Sticky last-known (used, total) so status does not flash empty mid-turn.
        self._usage_sticky: tuple[int | None, int | None] = (None, None)

        # Resume: project existing live transcript into scrollback (best-effort).
        self._seed_scrollback()

        hist_path = Path.home() / ".codedoggy" / "prompt_history"
        try:
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            history: Any = FileHistory(str(hist_path))
        except OSError:
            history = InMemoryHistory()

        self._prompt = create_prompt(
            style=PromptStyle(),
            info=PromptInfo(
                model_name=session_surface.model_and_mode_text(session) or ""
            ),
            history=history,
            accept_handler=self._on_accept,
            width_provider=lambda: self._term_size()[0],
        )
        self._sync_prompt_info()
        self._input = self._prompt.text_area
        self._kb = self._build_keys()
        self._sb = _SBControl(self)
        self._sb_win = Window(
            content=self._sb,
            wrap_lines=False,
            always_hide_cursor=True,
        )

        body = HSplit(
            [
                Window(FormattedTextControl(self._paint_status), height=1),
                self._sb_win,
                ConditionalContainer(
                    Window(FormattedTextControl(self._paint_turn), height=1),
                    filter=Condition(self._is_running),
                ),
                ConditionalContainer(
                    Window(FormattedTextControl(self._paint_feedback), height=1),
                    filter=Condition(lambda: bool(self._feedback)),
                ),
                self._prompt.container,
                Window(FormattedTextControl(self._paint_shortcuts), height=1),
            ]
        )
        self._modal = FormattedTextControl(self._paint_auth, focusable=True)
        root = FloatContainer(
            content=body,
            floats=[
                Float(
                    ConditionalContainer(
                        Window(
                            self._modal,
                            style="class:grok.bg_highlight",
                            width=Dimension(preferred=64, max=80),
                            height=Dimension(preferred=18, max=22),
                        ),
                        filter=Condition(lambda: self._auth_open),
                    ),
                    top=2,
                    left=4,
                )
            ],
        )
        self._layout = Layout(root, focused_element=self._input)
        self.app: Application[None] | None = None
        self._ensure_app()

    def _ensure_app(self) -> Application[None]:
        if self.app is not None:
            return self.app
        try:
            from prompt_toolkit.output.defaults import create_output

            output = create_output()
        except Exception:  # noqa: BLE001
            from prompt_toolkit.output import DummyOutput

            output = DummyOutput()
        self.app = Application(
            layout=self._layout,
            key_bindings=self._kb,
            style=build_style(),
            full_screen=True,
            mouse_support=True,
            color_depth=ColorDepth.TRUE_COLOR,
            output=output,
            refresh_interval=1.0 / 20.0,
            before_render=lambda _: self._on_before_render(),
        )
        self.app.ttimeoutlen = 0.05
        return self.app

    def run(self) -> None:
        app = self._ensure_app()
        self._bind_subagent()
        self._bind_bg_tasks()
        if self._initial and str(self._initial).strip():

            def kick() -> None:
                self._input.text = str(self._initial).strip()
                self._submit(self._input.text)

            app.pre_run_callables.append(kick)
        try:
            app.run()
        finally:
            self._closing = True
            self._unbind_subagent()
            self._unbind_bg_tasks()

    # ── paint ─────────────────────────────────────────────────────────────

    def _term_size(self) -> tuple[int, int]:
        try:
            if self.app is not None:
                s = self.app.output.get_size()
                return max(40, s.columns), max(12, s.rows)
        except Exception:  # noqa: BLE001
            pass
        return 80, 24

    def _on_before_render(self) -> None:
        self._tick += 1
        # Keep model caption + usage chips current (login, turns, mode switches).
        if self._tick % 4 == 0 or not self._is_running():
            self._sync_prompt_info()
        # Fallback sync for BackgroundTaskManager without add_listener yet.
        if self._tick % 20 == 0:
            self._poll_bg_tasks()

    def _seed_scrollback(self) -> None:
        """Project kernel/turn_runner live history into scrollback on open.

        Caps to the last :data:`HISTORY_SEED_CAP` messages. Never raises —
        empty/missing APIs leave an empty scrollback (welcome brand shows).
        """
        try:
            msgs = self._live_messages_snapshot()
            if not msgs:
                return
            if len(msgs) > HISTORY_SEED_CAP:
                msgs = msgs[-HISTORY_SEED_CAP:]
            self._scroll.seed_from_messages(msgs)
            # Resume: orphan tool_call rows have no matching result — settle
            # still-running tools so the viewport does not animate forever.
            for it in self._scroll.items:
                if it.kind == "tool" and it.status in {"running", "pending"}:
                    it.status = "completed"
            if self._scroll.items:
                self._scroll.follow_tail = True
                self._scroll.selected = len(self._scroll.items) - 1
        except Exception:  # noqa: BLE001
            logger.debug("scrollback seed failed", exc_info=True)

    def _live_messages_snapshot(self) -> list[Any]:
        """Best-effort live transcript from kernel or turn_runner."""
        ext = getattr(self.session, "extensions", None)
        if ext is None:
            return []
        kernel = getattr(ext, "kernel", None)
        if kernel is not None:
            lm = getattr(kernel, "live_messages", None)
            if callable(lm):
                try:
                    out = lm()
                    if out is not None:
                        return list(out)
                except Exception:  # noqa: BLE001
                    logger.debug("kernel.live_messages() failed", exc_info=True)
            elif lm is not None:
                try:
                    return list(lm)
                except Exception:  # noqa: BLE001
                    pass
        runner = getattr(ext, "turn_runner", None)
        if runner is not None:
            lm = getattr(runner, "live_messages", None)
            if lm is not None:
                try:
                    return list(lm)
                except Exception:  # noqa: BLE001
                    pass
        return []

    def _usage_pair(self) -> tuple[int | None, int | None]:
        """Sticky (used, total) tokens — same sources as ``surface.budget_text``.

        Reads ``extensions.context.budget`` (last_prompt_tokens / context_window),
        falls back to active connection window, keeps last good values mid-turn.
        """
        used_i: int | None = None
        total_i: int | None = None
        try:
            context = getattr(
                getattr(self.session, "extensions", None), "context", None
            )
            budget = getattr(context, "budget", None)
            used = getattr(budget, "last_prompt_tokens", None)
            total = getattr(budget, "context_window", None)
            if not total:
                snap = session_surface.active_connection(self.session)
                total = (
                    getattr(snap, "context_window", None) if snap is not None else None
                )
            if used is not None:
                used_i = int(used)
            if total:
                total_i = int(total)
        except Exception:  # noqa: BLE001
            pass

        prev_used, prev_total = self._usage_sticky
        if total_i is None:
            total_i = prev_total
        if used_i is None:
            used_i = prev_used
        if total_i:
            self._usage_sticky = (used_i, total_i)
            # Touch surface sticky so budget_text stays coherent if called elsewhere.
            try:
                session_surface.budget_text(self.session)
            except Exception:  # noqa: BLE001
                pass
        return used_i, total_i

    def _sync_prompt_info(self) -> None:
        """Refresh prompt bottom caption (model · mode · usage) from session."""
        try:
            info = self._prompt.info
            info.model_name = (
                session_surface.model_and_mode_text(self.session) or ""
            )
            used, total = self._usage_pair()
            if used is not None and total is not None and total > 0:
                pct = usage_percentage(used, total)
                # Always show compact pair; escalate style when near limit.
                info.usage_warning = format_token_pair(used, total)
                info.usage_warning_critical = pct >= 95.0
            elif total is not None and total > 0:
                info.usage_warning = f"… / {fmt_tokens(total)}"
                info.usage_warning_critical = False
            else:
                # Surface helper may still format a string (sticky / connection).
                budget = session_surface.budget_text(self.session)
                if budget:
                    info.usage_warning = budget
                    info.usage_warning_critical = False
                else:
                    info.usage_warning = None
                    info.usage_warning_critical = False
        except Exception:  # noqa: BLE001
            logger.debug("prompt info sync failed", exc_info=True)

    def _paint_status(self) -> StyleAndTextTuples:
        w, _ = self._term_size()
        model = session_surface.model_and_mode_text(self.session) or ""
        left = " doggy"
        usage_frags: StyleAndTextTuples = []
        usage_text = ""
        try:
            used, total = self._usage_pair()
            if used is not None and total is not None and total > 0:
                chip = context_bar_fragments(used, total)
                if chip:
                    # Leading space so pad math and visual gap stay clean.
                    usage_frags = [(chip[0][0], f" {chip[0][1]}")]
                    usage_text = usage_frags[0][1]
                else:
                    usage_text = f" {format_token_pair(used, total)}"
                    pct = usage_percentage(used, total)
                    usage_frags = [(usage_style(pct), usage_text)]
            elif total is not None and total > 0:
                usage_text = f" … / {fmt_tokens(total)}"
                usage_frags = [("class:grok.gray", usage_text)]
            else:
                budget = session_surface.budget_text(self.session)
                if budget:
                    usage_text = f" {budget}"
                    usage_frags = [("class:grok.gray", usage_text)]
        except Exception:  # noqa: BLE001
            pass

        right = model
        pad = max(1, w - len(left) - len(usage_text) - len(right) - 1)
        frags: StyleAndTextTuples = [
            ("class:grok.gray", left),
            ("class:grok.gray", " " * pad),
        ]
        frags.extend(usage_frags)
        frags.append(("class:grok.accent_model", f" {right}" if right else ""))
        return frags

    def _paint_scrollback(self) -> StyleAndTextTuples:
        w, h = self._term_size()
        reserved = 7
        if self._is_running():
            reserved += 1
        if self._feedback:
            reserved += 1
        body_h = max(4, h - reserved)
        welcome = None
        if not self._scroll.items:
            welcome = render_welcome(
                width=w,
                model_caption=session_surface.model_and_mode_text(self.session) or "",
            )
        return render_scrollback(
            self._scroll, width=w, height=body_h, welcome=welcome
        )

    def _paint_turn(self) -> StyleAndTextTuples:
        tools = sum(
            1
            for i in self._scroll.items
            if i.kind == "tool" and i.status in {"running", "pending"}
        )
        kids = self._running_subs()
        return turn_status.render(
            self._is_running(),
            self._turn_started,
            tools_running=tools,
            subagents_running=kids,
            tick=self._tick,
        )

    def _paint_feedback(self) -> StyleAndTextTuples:
        return [("class:grok.warning", f" {self._feedback}")]

    def _paint_shortcuts(self) -> StyleAndTextTuples:
        w, _ = self._term_size()
        focus = "auth" if self._auth_open else self._focus
        hints = shortcuts.hints_for(focus, self._is_running())
        return shortcuts.render(hints, width=w)

    def _paint_auth(self) -> StyleAndTextTuples:
        wiz = self._auth
        out: StyleAndTextTuples = [
            ("class:grok.text_primary", f" {wiz.title}\n"),
            ("class:grok.gray", f" {wiz.subtitle}\n"),
        ]
        if wiz.step == WizardStep.PASTE:
            out.append(("class:grok.gray", f" {wiz.paste_prompt}\n"))
            out.append(
                (
                    "class:grok.accent_user",
                    f" > {'•' * min(24, len(wiz.paste_buffer)) or '…'}\n",
                )
            )
        else:
            for i, item in enumerate(wiz.items):
                mark = "› " if i == wiz.cursor else "  "
                style = (
                    "class:grok.bg_hover"
                    if i == wiz.cursor
                    else "class:grok.text_secondary"
                )
                out.append((style, f"{mark}{item.label}\n"))
        out.append(("class:grok.gray", "\n Esc close · Tab back\n"))
        return out

    # ── keys ──────────────────────────────────────────────────────────────

    def _build_keys(self) -> KeyBindings:
        kb = KeyBindings()
        auth = Condition(lambda: self._auth_open)
        free = Condition(lambda: not self._auth_open)
        # Multiline TextArea: Enter inserts newline by default. Bind Enter to
        # validate_and_handle (accept_handler → _on_accept → _submit) when the
        # main prompt is focused and auth is closed — same pattern as legacy tui.
        prompt_focused = Condition(
            lambda: (not self._auth_open) and get_app().layout.has_focus(self._input)
        )

        @kb.add("enter", filter=prompt_focused, eager=True)
        def _submit_prompt(event: Any) -> None:
            self._input.buffer.validate_and_handle()

        # Hard newline: Ctrl+J (and Esc then Ctrl+J on Windows). Cap at prompt max.
        # Note: prompt_toolkit has no s-enter key — terminals rarely distinguish
        # Shift+Enter from Enter — so the shortcuts bar advertises Ctrl+J only.
        @kb.add("c-j", filter=prompt_focused, eager=True)
        @kb.add("escape", "c-j", filter=prompt_focused, eager=True)
        def _newline_prompt(event: Any) -> None:
            buffer = event.current_buffer
            max_lines = 12
            if buffer.text.count("\n") + 1 >= max_lines:
                return
            pos = buffer.cursor_position
            text = buffer.text
            buffer.text = text[:pos] + "\n" + text[pos:]
            buffer.cursor_position = pos + 1
            try:
                event.app.invalidate()
            except Exception:  # noqa: BLE001
                pass

        @kb.add("c-c")
        def _cc(event: Any) -> None:
            if self._auth_open:
                self._close_auth()
                return
            # Prefer copy of text selection when present
            from codedoggy.tui_v2.scrollback import copy_text_selection

            sel_text = copy_text_selection(self._scroll)
            if sel_text:
                if set_system_clipboard_text(sel_text):
                    self._set_feedback("copied selection")
                    self._scroll.text_sel = None
                    self._invalidate()
                return
            if self._is_running():
                if self._input.text.strip():
                    self._input.text = ""
                    return
                self._cancel()
                return
            if self._input.text.strip():
                self._input.text = ""
                return
            event.app.exit()

        @kb.add("c-l", filter=free)
        def _cl(_: Any) -> None:
            self._open_auth()

        @kb.add("c-v", filter=free)
        def _cv(_: Any) -> None:
            self._paste_image()

        @kb.add("c-y", filter=free)
        def _cy(_: Any) -> None:
            from codedoggy.tui_v2.scrollback import copy_text_selection

            sel_text = copy_text_selection(self._scroll)
            if sel_text and set_system_clipboard_text(sel_text):
                self._set_feedback("copied selection")
                return
            it = self._scroll.selected_item()
            if it is None:
                return
            text = it.text or it.tool_result or it.tool_name
            if text and set_system_clipboard_text(text):
                self._set_feedback("copied")

        @kb.add("escape", filter=auth, eager=True)
        def _ea(_: Any) -> None:
            self._close_auth()

        @kb.add("escape", filter=free, eager=True)
        def _esc(_: Any) -> None:
            if self._is_running():
                self._cancel()
                return
            now = time.monotonic()
            if self._input.text.strip():
                if now < self._esc_clear_until:
                    self._input.text = ""
                    self._esc_clear_until = 0.0
                else:
                    self._esc_clear_until = now + 0.8
                    self._set_feedback("Esc again to clear")
                return

        @kb.add("enter", filter=auth, eager=True)
        def _ae(_: Any) -> None:
            if self._auth.step == WizardStep.PASTE and self._auth.paste_buffer.strip():
                action = self._auth.submit_paste_text(self._auth.paste_buffer)
            else:
                action = self._auth.activate()
            self._wizard_action(action)

        @kb.add("up", filter=auth, eager=True)
        def _au(_: Any) -> None:
            self._auth.move(-1)
            self._invalidate()

        @kb.add("down", filter=auth, eager=True)
        def _ad(_: Any) -> None:
            self._auth.move(1)
            self._invalidate()

        @kb.add("tab", filter=auth, eager=True)
        def _at(_: Any) -> None:
            self._wizard_action(self._auth.go_back())

        @kb.add("<any>", filter=auth, eager=True)
        def _atype(event: Any) -> None:
            if self._auth.step != WizardStep.PASTE:
                return
            data = event.data or ""
            if data in {"\x7f", "\x08"}:
                self._auth.paste_buffer = self._auth.paste_buffer[:-1]
            elif data.isprintable():
                self._auth.paste_buffer += data
            self._invalidate()

        @kb.add("tab", filter=free)
        def _tab(_: Any) -> None:
            if self._focus == "prompt":
                self._focus = "scrollback"
                try:
                    if self.app:
                        self.app.layout.focus(self._sb_win)
                except Exception:  # noqa: BLE001
                    pass
            else:
                self._focus = "prompt"
                try:
                    if self.app:
                        self.app.layout.focus(self._input)
                except Exception:  # noqa: BLE001
                    pass
            self._invalidate()

        @kb.add("up", filter=free)
        def _up(event: Any) -> None:
            if self._focus == "scrollback":
                self._scroll.select_delta(-1)
                self._scroll.follow_tail = False
                self._invalidate()
            else:
                event.current_buffer.auto_up(count=event.arg)

        @kb.add("down", filter=free)
        def _down(event: Any) -> None:
            if self._focus == "scrollback":
                self._scroll.select_delta(1)
                self._invalidate()
            else:
                event.current_buffer.auto_down(count=event.arg)

        @kb.add("left", filter=free)
        def _left(event: Any) -> None:
            if self._focus == "scrollback":
                # Prefer re-folding an expanded verb group, else collapse / reverse fold.
                if self._scroll.collapse_group_at_selection():
                    self._invalidate()
                    return
                it = self._scroll.selected_item()
                if it:
                    self._collapse_item_fold(it)
                    self._invalidate()
            else:
                event.current_buffer.cursor_left()

        @kb.add("right", filter=free)
        def _right(event: Any) -> None:
            if self._focus == "scrollback":
                # Expand folded verb group, else expand once toward truncated/expanded.
                if self._scroll.expand_group_at_selection():
                    self._invalidate()
                    return
                it = self._scroll.selected_item()
                if it:
                    self._expand_item_fold(it)
                    self._invalidate()
            else:
                event.current_buffer.cursor_right()

        @kb.add("pageup", filter=free)
        def _pu(_: Any) -> None:
            self._scroll.select_delta(-10)
            self._focus = "scrollback"
            self._invalidate()

        @kb.add("pagedown", filter=free)
        def _pd(_: Any) -> None:
            self._scroll.select_delta(10)
            self._focus = "scrollback"
            self._invalidate()

        return kb

    def _on_accept(self, buf: Any) -> bool:
        text = (buf.text or "").strip()
        if not text:
            return True
        self._submit(text)
        return True

    # ── prompt / turn ─────────────────────────────────────────────────────

    def _submit(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            return
        try:
            attachments = self._attachments(prompt)
        except AttachmentError as exc:
            self._set_feedback(f"image: {exc}")
            return
        try:
            hist = getattr(self._input.buffer, "history", None)
            if hist is not None:
                hist.append_string(prompt)
        except Exception:  # noqa: BLE001
            pass
        self._input.text = ""
        self._feedback = ""

        if self._is_running():
            clean = strip_image_chips(prompt)
            try:
                self.session.interject(clean, attachments=attachments)
            except Exception as exc:  # noqa: BLE001
                self._set_feedback(str(exc))
                self._input.text = prompt
                return
            self._scroll.items.append(
                __import__("codedoggy.tui_v2.project", fromlist=["ScrollItem"]).ScrollItem(
                    kind="user",
                    id=self._scroll.new_id("user"),
                    text=(clean or prompt).strip() or prompt,
                    meta={"interject": True, "optimistic_user": True},
                )
            )
            self._invalidate()
            return

        if not session_surface.ready_to_sample(self.session):
            self._pending_prompt = prompt
            self._pending_attach = attachments
            self._open_auth()
            self._input.text = prompt
            self._set_feedback("login required")
            return

        self._start_turn(prompt, attachments=attachments)

    def _emit_session_event(self, event: str, text: str = "") -> None:
        """Append a session_event row when the tail is not already the same event.

        Idempotent against consecutive duplicates; different events may stack.
        """
        from codedoggy.tui_v2.project import ScrollItem

        ev = (event or "").strip()
        if not ev:
            return
        items = self._scroll.items
        if (
            items
            and items[-1].kind == "session_event"
            and items[-1].meta.get("event") == ev
        ):
            return
        self._scroll.items.append(
            ScrollItem(
                kind="session_event",
                id=self._scroll.new_id("session_event"),
                text=text or "",
                meta={"event": ev},
                status="done",
            )
        )
        if self._scroll.follow_tail and self._scroll.items:
            self._scroll.selected = len(self._scroll.items) - 1

    def _append_turn_session_event(
        self,
        *,
        status: Any,
        error: str | None,
        started: float | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a Grok-like session_event row after reconcile (finish/fail only).

        Primary marker is idempotent (skips when the scroll tail is already a
        session_event). Supplementary markers (compact / context / reauth) use
        ``_emit_session_event`` and only fire when clearly warranted.
        """
        from codedoggy.tui_v2.project import ScrollItem

        st = str(getattr(status, "value", status) or "").lower()
        elapsed = ""
        if started is not None:
            try:
                elapsed = turn_status.format_turn_timer(
                    max(0.0, time.monotonic() - float(started))
                )
            except Exception:  # noqa: BLE001
                elapsed = ""

        event: str | None = None
        text = ""
        if st == "completed":
            event = "worked"
            text = elapsed
        elif st == "cancelled":
            event = "turn_cancelled"
        elif st in {"max_turns_reached", "max_turns"}:
            event = "max_turns"
        elif st in {"failed", "error"}:
            event = "turn_failed"
            raw = (error or "").strip()
            if raw:
                # First line, keep the marker compact.
                raw = raw.splitlines()[0].strip()
                if len(raw) > 160:
                    raw = raw[:157] + "…"
            text = raw

        items = self._scroll.items
        already_session = bool(items and items[-1].kind == "session_event")
        if event and not already_session:
            self._scroll.items.append(
                ScrollItem(
                    kind="session_event",
                    id=self._scroll.new_id("session_event"),
                    text=text,
                    meta={"event": event},
                    status="done",
                )
            )
            if self._scroll.follow_tail and self._scroll.items:
                self._scroll.selected = len(self._scroll.items) - 1

        # ── supplementary markers (careful — no flood) ────────────────────
        meta = metadata if isinstance(metadata, dict) else {}
        # compact_done only when loop/result metadata records a compaction.
        compact_n = meta.get("context_compactions")
        if compact_n is None:
            compact_n = meta.get("compactions")
        try:
            compact_hits = int(compact_n or 0) > 0
        except (TypeError, ValueError):
            compact_hits = bool(compact_n)
        if not compact_hits and meta.get("context_last"):
            compact_hits = True
        if compact_hits:
            self._emit_session_event("compact_done")

        err_l = (error or "").lower()
        if st in {"failed", "error"} and err_l:
            # context_too_large: require "context" + (large|overflow) co-signal
            if "context" in err_l and (
                "large" in err_l or "overflow" in err_l
            ):
                self._emit_session_event("context_too_large")
            # reauth: auth-ish failure while sampler is not ready
            authish = any(
                tok in err_l
                for tok in (
                    "auth",
                    "unauthorized",
                    "401",
                    "login",
                    "credential",
                    "token expired",
                    "not authenticated",
                )
            )
            if authish and not session_surface.ready_to_sample(self.session):
                self._emit_session_event("reauth")

    def _start_turn(
        self, prompt: str, *, attachments: tuple[ImageAttachment, ...] = ()
    ) -> None:
        from codedoggy.tui_v2.project import ScrollItem

        # Display text must match what the runner archives (strip chips) so
        # on_live / reconcile can dedupe the optimistic user row.
        model_text = strip_image_chips(prompt)
        display_user = (model_text or prompt).strip() or prompt
        self._scroll.follow_tail = True
        self._scroll.items.append(
            ScrollItem(
                kind="user",
                id=self._scroll.new_id("user"),
                text=display_user,
                meta={"optimistic_user": True},
            )
        )
        # Index of this turn's user row — finish only reconciles from here on.
        turn_scroll_start = max(0, len(self._scroll.items) - 1)
        self._scroll.tool_open.clear()
        self._stream_buf = []
        self._stream_state = {
            "draft": "",
            "pending": 0,
            "draft_generation": 0,
        }
        self._turn_started = time.monotonic()
        self._invalidate()

        def worker() -> None:
            runner = getattr(self.session.extensions, "turn_runner", None)
            old = getattr(runner, "on_live_message", None) if runner else None
            active = {"on": True}
            # Messages already in the live transcript before this turn body runs.
            live_start = len(self._live_messages_snapshot())

            def on_live(msg: Any) -> None:
                if not active["on"] or self._closing:
                    return

                # Multi-sample: when an assistant message is archived, end the
                # current draft generation so the next sample does not prefix
                # previous sample text (and late UI deltas for this gen drop).
                from codedoggy.tui_v2.scrollback import _msg_role

                if _msg_role(msg) == "assistant":
                    archived_generation = int(
                        self._stream_state.get("draft_generation", 0) or 0
                    )
                    self._stream_buf = []
                    self._stream_state["draft"] = ""
                    self._stream_state["pending"] = 0
                    self._stream_state["draft_generation"] = (
                        archived_generation + 1
                    )

                def apply() -> None:
                    if not active["on"] or self._closing:
                        return
                    # Final assistant (or tool) supersedes streaming draft.
                    self._scroll.clear_draft()
                    # UI owns the optimistic user row; runner also archives USER
                    # at loop start — skip duplicates (and framed interjects).
                    if self._should_skip_live_message(msg, turn_scroll_start):
                        return
                    self._scroll.append_message(msg)
                    self._request_redraw()

                self._ui(apply)

            def on_delta(piece: str) -> bool:
                if not active["on"] or self._closing:
                    return False
                self._stream_buf.append(piece)
                draft = self._stream_state.get("draft", "") + piece
                self._stream_state["draft"] = draft
                pend = int(self._stream_state.get("pending", 0)) + len(piece)
                self._stream_state["pending"] = pend
                gen = int(self._stream_state.get("draft_generation", 0) or 0)
                if pend >= 8 or piece.endswith(("\n", ".", "。")):
                    self._stream_state["pending"] = 0

                    def apply_d() -> None:
                        if not active["on"] or self._closing:
                            return
                        # Stale after multi-sample assistant archive (gen bump).
                        if (
                            int(
                                self._stream_state.get("draft_generation", 0)
                                or 0
                            )
                            != gen
                        ):
                            return
                        self._scroll.set_draft(draft)
                        self._request_redraw()

                    self._ui(apply_d)
                return active["on"] and not self._closing

            if runner is not None:
                runner.on_live_message = on_live
            result: Any = None
            try:
                result = self.session.handle_prompt(
                    model_text,
                    metadata={
                        "on_live_message": on_live,
                        "stream_sample": True,
                        "on_sample_delta": on_delta,
                    },
                    attachments=attachments,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("turn failed")
                err = f"{type(exc).__name__}: {exc}"

                def fail() -> None:
                    started = self._turn_started
                    self._turn_started = None
                    streamed = "".join(self._stream_buf).strip() or None
                    draft = str(self._stream_state.get("draft") or "").strip() or None
                    live = self._live_messages_since(live_start)
                    reconcile_turn_finish(
                        self._scroll,
                        result_status="error",
                        result_error=err,
                        final_text=streamed or draft,
                        live_messages=live,
                        turn_scroll_start=turn_scroll_start,
                    )
                    self._append_turn_session_event(
                        status="error",
                        error=err,
                        started=started,
                        metadata=None,
                    )
                    self._sync_prompt_info()
                    self._set_feedback("turn failed")
                    self._request_redraw()

                self._ui(fail)
                return
            finally:
                active["on"] = False
                if runner is not None:
                    runner.on_live_message = old

            def finish() -> None:
                started = self._turn_started
                self._turn_started = None
                # Prefer result.final_text; fall back to streamed draft buffer.
                final = str(getattr(result, "final_text", None) or "").strip()
                if not final:
                    final = "".join(self._stream_buf).strip()
                if not final:
                    final = str(self._stream_state.get("draft") or "").strip()
                live = self._live_messages_since(live_start)
                st = getattr(result, "status", None)
                err = str(getattr(result, "error", "") or "") or None
                meta = getattr(result, "metadata", None)
                reconcile_turn_finish(
                    self._scroll,
                    result_status=st,
                    result_error=err,
                    final_text=final or None,
                    live_messages=live,
                    turn_scroll_start=turn_scroll_start,
                )
                self._append_turn_session_event(
                    status=st,
                    error=err,
                    started=started,
                    metadata=meta if isinstance(meta, dict) else None,
                )
                # Cheap caption / model refresh after a finished turn.
                self._sync_prompt_info()
                self._request_redraw()

            self._ui(finish)

        self._worker = threading.Thread(target=worker, name="doggy-turn", daemon=True)
        self._worker.start()

    def _should_skip_live_message(self, msg: Any, turn_scroll_start: int) -> bool:
        """True when live projection would duplicate UI-owned rows.

        Runner archives the USER prompt at loop start; we already painted an
        optimistic user row. Interjection USER is framed by
        ``format_interjection`` — skip those too when a plain interject row
        already exists.
        """
        from codedoggy.tui_v2.scrollback import (
            _message_represented_in_scroll,
            _msg_content,
            _msg_role,
        )

        role = _msg_role(msg)
        if role != "user":
            # Still dedupe assistant/tool when already represented (finish
            # path is the main safety net; live path is best-effort).
            try:
                return _message_represented_in_scroll(
                    self._scroll, msg, since=turn_scroll_start
                )
            except Exception:  # noqa: BLE001
                return False

        content = _msg_content(msg).strip()
        if not content:
            return True
        # Exact match against optimistic / prior user rows this turn.
        if _message_represented_in_scroll(
            self._scroll, msg, since=turn_scroll_start
        ):
            return True
        # Framed interjection: "The user sent a message while you were working"
        low = content.lower()
        if "while you were working" in low or "<user_query>" in low:
            since = max(0, turn_scroll_start)
            for it in self._scroll.items[since:]:
                if it.kind == "user" and (
                    it.meta.get("interject") or it.meta.get("optimistic_user")
                ):
                    return True
            # Also skip if any plain user body is embedded in the frame.
            for it in self._scroll.items[since:]:
                if it.kind != "user":
                    continue
                body = (it.text or "").strip()
                if body and body in content:
                    return True
        return False

    def _live_messages_since(self, start: int) -> list[Any]:
        """Slice of live transcript messages produced for the current turn.

        ``start`` is the length of the live list captured before
        ``handle_prompt``. When the runner replaces the list with a shorter
        turn-local snapshot (``start > n``), fall back to the full snapshot
        so finish can still project missing rows. When ``start == n`` there
        is nothing new — return ``[]`` so seeded history is not re-projected.
        """
        live = self._live_messages_snapshot()
        if not live:
            return []
        n = len(live)
        if 0 <= start < n:
            return list(live[start:])
        if start > n:
            # List replaced with shorter turn-local snapshot
            return list(live)
        # start == n: nothing new
        return []

    def _cancel(self) -> None:
        now = time.monotonic()
        if now < self._cancel_grace or not self._is_running():
            return
        self._cancel_grace = now + 0.8
        try:
            self.session.cancel()
        except Exception:  # noqa: BLE001
            pass
        coord = self._coord()
        if coord is not None:
            try:
                for snap in coord.list_for_parent(str(self.session.id)):
                    if getattr(snap, "status", "") in {"pending", "running"}:
                        coord.cancel(snap.subagent_id)
            except Exception:  # noqa: BLE001
                pass
        self._set_feedback("cancelled")

    # ── auth / paste ──────────────────────────────────────────────────────

    def _open_auth(self) -> None:
        self._auth_open = True
        self._auth.open(
            active_provider=session_surface.provider_id(self.session),
            active_model=session_surface.model_id(self.session),
        )
        self._invalidate()

    def _close_auth(self) -> None:
        self._auth_open = False
        self._focus = "prompt"
        if self._pending_prompt and session_surface.ready_to_sample(self.session):
            p, a = self._pending_prompt, self._pending_attach
            self._pending_prompt = None
            self._pending_attach = ()
            self._input.text = ""
            self._start_turn(p, attachments=a)
        self._invalidate()

    def _wizard_action(self, action: Any) -> None:
        kind = getattr(action, "kind", "none")
        msg = getattr(action, "message", "") or ""
        if msg:
            self._set_feedback(msg)
        if kind == "close":
            self._close_auth()
        elif kind == "start_login":
            provider = getattr(action, "provider", None)
            if provider:
                self._browser_login(str(provider))
        elif kind == "reload_client":
            self._reload(action)
        self._invalidate()

    def _browser_login(self, provider: str) -> None:
        def job() -> None:
            try:
                status = run_browser_login(provider)
            except Exception as exc:  # noqa: BLE001
                from codedoggy.model.auth.base import AUTH_OAUTH, AuthStatus

                status = AuthStatus(
                    provider=provider,
                    kind=AUTH_OAUTH,
                    logged_in=False,
                    detail=str(exc),
                )
            self._ui(lambda: self._wizard_action(self._auth.on_login_finished(status)))

        threading.Thread(target=job, daemon=True).start()

    def _reload(self, action: Any) -> None:
        from codedoggy.model.connection import connection_of

        svc = connection_of(self.session)
        if svc is None:
            return
        try:
            svc.apply(
                provider=getattr(action, "provider", None),
                model=getattr(action, "model", None),
                reasoning_effort=getattr(action, "reasoning_effort", None),
                reasoning_enabled=getattr(action, "reasoning_enabled", None),
            )
            self._set_feedback("connection applied")
        except Exception as exc:  # noqa: BLE001
            self._set_feedback(str(exc))

    def _paste_image(self) -> None:
        cwd = getattr(self.session, "cwd", None) or Path.cwd()
        saved = save_clipboard_image(cwd)
        if saved is None:
            text = get_system_clipboard_text()
            if text:
                path = coerce_image_path_text(text, cwd=cwd)
                if path is not None:
                    saved = path
                else:
                    self._input.buffer.insert_text(text)
                    return
            else:
                return
        token = insert_image_chip(saved, cwd=cwd)
        buf = self._input.buffer
        pos = buf.cursor_position
        lead = " " if buf.text[:pos] and not buf.text[pos - 1].isspace() else ""
        insert = f"{lead}{token} "
        buf.text = buf.text[:pos] + insert + buf.text[pos:]
        buf.cursor_position = pos + len(insert)
        self._set_feedback(f"pasted {VIEW_IMAGE_LABEL}")

    def _attachments(self, prompt: str) -> tuple[ImageAttachment, ...]:
        cwd = getattr(self.session, "cwd", None) or Path.cwd()
        out: list[ImageAttachment] = []
        seen: set[str] = set()
        for raw in extract_image_chip_paths(prompt):
            resolved = resolve_openable_path(raw, cwd=cwd)
            if resolved is None:
                raise AttachmentError(f"missing {raw}")
            key = str(resolved).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(ImageAttachment.from_path(resolved))
        return tuple(out)

    # ── subagent ──────────────────────────────────────────────────────────

    def _coord(self) -> Any | None:
        k = getattr(getattr(self.session, "extensions", None), "kernel", None)
        return getattr(k, "subagent_coordinator", None)

    def _bind_subagent(self) -> None:
        if self._sub_listener:
            return
        c = self._coord()
        if c is None or not hasattr(c, "add_listener"):
            return
        c.add_listener(self._on_sub)
        self._sub_listener = True

    def _unbind_subagent(self) -> None:
        if not self._sub_listener:
            return
        c = self._coord()
        if c is not None and hasattr(c, "remove_listener"):
            try:
                c.remove_listener(self._on_sub)
            except Exception:  # noqa: BLE001
                pass
        self._sub_listener = False

    def _on_sub(self, snap: Any, message: Any = None) -> None:
        if self._closing:
            return

        def apply() -> None:
            from codedoggy.tui_v2.project import ScrollItem

            sid = str(getattr(snap, "subagent_id", "") or "")
            st = str(getattr(snap, "status", "") or "running")
            label = str(
                getattr(snap, "description", "")
                or getattr(snap, "subagent_type", "")
                or "subagent"
            )
            body = str(getattr(snap, "output", "") or getattr(snap, "error", "") or "")
            if st in {"pending", "running"} and not body:
                body = "…"
            # update existing system-like item by meta id
            activity = body if st in {"pending", "running"} else ""
            for it in self._scroll.items:
                if it.meta.get("subagent_id") == sid:
                    it.text = label
                    it.status = st
                    it.kind = "subagent"
                    if activity:
                        it.meta["activity"] = activity
                    err = str(getattr(snap, "error", "") or "")
                    if err:
                        it.meta["error"] = err
                    el = getattr(snap, "elapsed_ms", None)
                    if el is not None:
                        try:
                            it.elapsed_ms = int(el)
                        except (TypeError, ValueError):
                            pass
                    self._request_redraw()
                    return
            meta: dict[str, Any] = {
                "subagent_id": sid,
                "background": bool(getattr(snap, "is_background", False)),
            }
            if activity:
                meta["activity"] = activity
            err = str(getattr(snap, "error", "") or "")
            if err:
                meta["error"] = err
            self._scroll.items.append(
                ScrollItem(
                    kind="subagent",
                    id=self._scroll.new_id("sub"),
                    text=label,
                    status=st,
                    collapsed=True,
                    meta=meta,
                )
            )
            self._request_redraw()

        self._ui(apply)

    def _running_subs(self) -> int:
        c = self._coord()
        if c is None:
            return 0
        try:
            return sum(
                1
                for s in c.list_for_parent(str(self.session.id))
                if getattr(s, "status", "") in {"pending", "running"}
            )
        except Exception:  # noqa: BLE001
            return 0

    # ── background tasks (kind=bg_task) ───────────────────────────────────

    def _task_manager(self) -> Any | None:
        k = getattr(getattr(self.session, "extensions", None), "kernel", None)
        return getattr(k, "task_manager", None)

    def _bind_bg_tasks(self) -> None:
        if getattr(self, "_bg_listener", False):
            return
        tm = self._task_manager()
        if tm is None or not hasattr(tm, "add_listener"):
            return
        try:
            tm.add_listener(self._on_bg_task)
            self._bg_listener = True
        except Exception:  # noqa: BLE001
            logger.debug("bg task bind failed", exc_info=True)

    def _unbind_bg_tasks(self) -> None:
        if not getattr(self, "_bg_listener", False):
            return
        tm = self._task_manager()
        if tm is not None and hasattr(tm, "remove_listener"):
            try:
                tm.remove_listener(self._on_bg_task)
            except Exception:  # noqa: BLE001
                pass
        self._bg_listener = False

    def _on_bg_task(self, event: str, snap: Any) -> None:
        """Listener: BackgroundTaskManager → scrollback bg_task row (UI thread)."""
        if self._closing:
            return

        def apply() -> None:
            try:
                self._apply_bg_task_snap(snap)
            except Exception:  # noqa: BLE001
                logger.debug("bg task UI apply failed", exc_info=True)

        self._ui(apply)

    def _bg_task_output_from_snap(self, snap: Any) -> str:
        """Best-effort task output: in-memory snap.output, else short file head."""
        out = str(getattr(snap, "output", None) or "")
        if out.strip():
            return out
        path = str(getattr(snap, "output_file", None) or "").strip()
        if not path:
            return out
        try:
            p = Path(path)
            if not p.is_file():
                return out
            # Cap read so a huge log does not stall the UI thread (~64 KiB).
            data = p.read_bytes()[: 64 * 1024]
            return data.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            logger.debug("bg task output_file read failed", exc_info=True)
            return out

    def _refresh_bg_task_output(self, it: Any) -> None:
        """Pull latest task output into the scroll item when expanding."""
        tid = (it.meta or {}).get("task_id") if getattr(it, "meta", None) else None
        tm = self._task_manager()
        if not tm or not tid:
            return
        snap = tm.get(tid) if hasattr(tm, "get") else None
        if snap is None:
            return
        out = self._bg_task_output_from_snap(snap)
        if out:
            it.tool_result = out
            it.meta["output"] = out

    def _apply_bg_task_snap(self, snap: Any) -> None:
        """Upsert a scrollback row for one TaskSnapshot. Never raises."""
        try:
            from codedoggy.tui_v2.project import ScrollItem

            tid = str(getattr(snap, "task_id", "") or "")
            if not tid:
                return
            description = (
                str(getattr(snap, "description", "") or "")
                or str(getattr(snap, "display_command", "") or "")
                or str(getattr(snap, "command", "") or "")
                or "background task"
            )
            status_fn = getattr(snap, "status_label", None)
            if callable(status_fn):
                status = str(status_fn() or "running")
            else:
                status = "running" if not getattr(snap, "completed", False) else "completed"
            completed = bool(getattr(snap, "completed", False)) or status in {
                "completed",
                "failed",
                "cancelled",
                "canceled",
                "killed",
            }
            elapsed_ms: int | None = None
            if completed:
                dur_fn = getattr(snap, "duration_secs", None)
                if callable(dur_fn):
                    try:
                        elapsed_ms = int(float(dur_fn()) * 1000)
                    except (TypeError, ValueError):
                        elapsed_ms = None

            exit_code = getattr(snap, "exit_code", None)
            signal = getattr(snap, "signal", None)
            output = self._bg_task_output_from_snap(snap)

            for it in self._scroll.items:
                if it.kind == "bg_task" and it.meta.get("task_id") == tid:
                    it.text = description
                    it.status = status
                    if elapsed_ms is not None:
                        it.elapsed_ms = elapsed_ms
                    it.meta["task_id"] = tid
                    it.meta["exit_code"] = exit_code
                    it.meta["signal"] = signal
                    if output:
                        it.tool_result = output
                        it.meta["output"] = output
                    self._request_redraw()
                    return

            self._scroll.items.append(
                ScrollItem(
                    kind="bg_task",
                    id=self._scroll.new_id("bg"),
                    text=description,
                    status=status,
                    elapsed_ms=elapsed_ms,
                    tool_result=output,
                    collapsed=True,
                    meta={
                        "task_id": tid,
                        "exit_code": exit_code,
                        "signal": signal,
                        **({"output": output} if output else {}),
                    },
                )
            )
            self._request_redraw()
        except Exception:  # noqa: BLE001
            logger.debug("apply bg task snap failed", exc_info=True)

    def _poll_bg_tasks(self) -> None:
        """Periodic list_tasks sync when manager has no live listeners (or to catch up)."""
        if self._closing:
            return
        try:
            tm = self._task_manager()
            if tm is None or not hasattr(tm, "list_tasks"):
                return
            snaps = tm.list_tasks()
            if not snaps:
                return
            for snap in snaps:
                self._apply_bg_task_snap(snap)
        except Exception:  # noqa: BLE001
            logger.debug("bg task poll failed", exc_info=True)

    # ── util ──────────────────────────────────────────────────────────────

    def _is_running(self) -> bool:
        if self._worker is not None and self._worker.is_alive():
            return True
        if getattr(self.session, "phase", None) is SessionPhase.TURN_RUNNING:
            return True
        return self._running_subs() > 0

    def _expand_item_fold(self, it: Any) -> None:
        """Expand selected item one step toward truncated / fully expanded.

        When ``ScrollItem`` exposes ``cycle_fold`` (fold cycle), tools and
        bg_task go collapsed → truncated → expanded; otherwise just
        ``collapsed=False``. Expanding a bg_task refreshes live output.
        """
        if hasattr(it, "cycle_fold"):
            # expand once toward truncated/expanded
            if it.collapsed:
                it.collapsed = False
                if it.kind in {"tool", "bg_task"}:
                    it.truncated = True
            elif getattr(it, "truncated", False):
                it.truncated = False
            else:
                it.collapsed = False
        else:
            it.collapsed = False

        if it.kind == "bg_task" and not it.collapsed:
            self._refresh_bg_task_output(it)

    def _collapse_item_fold(self, it: Any) -> None:
        """Collapse selected item one step (reverse fold cycle) or fully.

        When fold cycle is available: expanded → truncated (tools/bg_task) →
        collapsed. Else: collapsed=True and truncated=False if present.
        """
        if hasattr(it, "cycle_fold"):
            if not it.collapsed and not getattr(it, "truncated", False):
                # fully expanded → truncated for tools/bg_task, else collapsed
                if it.kind in {"tool", "bg_task"} and hasattr(it, "truncated"):
                    it.truncated = True
                else:
                    it.collapsed = True
            elif getattr(it, "truncated", False):
                it.collapsed = True
                it.truncated = False
            else:
                it.collapsed = True
                if hasattr(it, "truncated"):
                    it.truncated = False
        else:
            it.collapsed = True
            if hasattr(it, "truncated"):
                it.truncated = False

    def _sb_mouse(self, event: MouseEvent) -> object:
        from codedoggy.tui_v2.text_selection import TextSel, col_at_x

        et = event.event_type
        if et == MouseEventType.SCROLL_UP:
            # Line scroll using last painted viewport_h (set by render_scrollback).
            # Do not re-estimate body height here — mismatch jumps at top/bottom.
            self._scroll.scroll_by_lines(-3)
            self._scroll.text_sel = None
            self._focus = "scrollback"
            self._invalidate()
            return None
        if et == MouseEventType.SCROLL_DOWN:
            self._scroll.scroll_by_lines(3)
            self._scroll.text_sel = None
            self._focus = "scrollback"
            self._invalidate()
            return None

        row = int(getattr(event.position, "y", 0) or 0)
        col = int(getattr(event.position, "x", 0) or 0)
        # Clamp to viewport row for col mapping
        vr = self._scroll.viewport_rows
        if 0 <= row < len(vr):
            col = col_at_x(vr[row], col)

        if et == MouseEventType.MOUSE_DOWN:
            self._scroll.text_sel = TextSel(
                anchor_row=row,
                anchor_col=col,
                head_row=row,
                head_col=col,
                active=True,
            )
            self._focus = "scrollback"
            self._invalidate()
            return None

        if et == MouseEventType.MOUSE_MOVE:
            sel = self._scroll.text_sel
            if sel is not None and sel.active:
                sel.head_row = row
                sel.head_col = col
                self._invalidate()
                return None
            return NotImplemented

        if et == MouseEventType.MOUSE_UP:
            sel = self._scroll.text_sel
            if sel is not None and sel.active:
                sel.head_row = row
                sel.head_col = col
                sel.active = False
                if sel.is_empty():
                    self._scroll.text_sel = None
                    # Block selection / double-click expand
                    idx = entry_at_line(self._scroll, row)
                    if idx is not None:
                        now = time.monotonic()
                        if (
                            idx == self._scroll._last_click_owner
                            and (now - self._scroll._last_click_ms) < 0.4
                        ):
                            self._scroll.selected = idx
                            self._scroll.expand_group_at_selection()
                            # also expand once toward truncated/expanded
                            it = self._scroll.selected_item()
                            if it is not None:
                                self._expand_item_fold(it)
                            self._scroll._last_click_ms = 0.0
                        else:
                            self._scroll.selected = idx
                            self._scroll.follow_tail = (
                                idx >= len(self._scroll.items) - 1
                            )
                            self._scroll._last_click_owner = idx
                            self._scroll._last_click_ms = now
                        self._focus = "scrollback"
                self._invalidate()
                return None
            idx = entry_at_line(self._scroll, row)
            if idx is not None:
                self._scroll.selected = idx
                self._scroll.follow_tail = idx >= len(self._scroll.items) - 1
                self._focus = "scrollback"
                self._invalidate()
            return None

        return NotImplemented

    def _set_feedback(self, text: str) -> None:
        self._feedback = (text or "").strip()
        self._invalidate()

    def _invalidate(self) -> None:
        try:
            if self.app is not None:
                self.app.invalidate()
        except Exception:  # noqa: BLE001
            pass

    def _ui(self, cb: Any) -> None:
        if self._closing:
            return
        loop = getattr(self.app, "loop", None) if self.app else None
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(cb)
        else:
            cb()

    def _request_redraw(self) -> None:
        self._ui(self._invalidate)


def run_tui(session: Any, *, initial_prompt: str | None = None) -> None:
    GrokShellApp(session, initial_prompt=initial_prompt).run()
