"""Regression: FormattedTextControl cursor y must never exceed line_count."""

from __future__ import annotations

from prompt_toolkit.formatted_text.utils import split_lines
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput

from codedoggy.tui.app import CodeDoggyTUI
from codedoggy.tui.login_wizard import WizardStep


class _Session:
    cwd = "."
    id = "s"
    phase = None

    class _Ext:
        kernel = None
        connection = None
        context = None

    extensions = _Ext()

    def interject(self, *a, **k):  # noqa: ANN001
        return None

    def cancel(self) -> None:
        return None


def _pt_line_count(fragments) -> int:  # noqa: ANN001
    return len(list(split_lines(fragments)))


def _assert_cursor_safe(tui: CodeDoggyTUI, fragments, *, which: str) -> None:
    pt_n = _pt_line_count(fragments)
    shadow = (
        tui._task_line_count if which == "task" else tui._detail_line_count
    )
    assert shadow == pt_n, f"{which} shadow={shadow} pt={pt_n}"
    pos = (
        tui._task_cursor_position()
        if which == "task"
        else tui._detail_cursor_position()
    )
    assert 0 <= pos.y < pt_n
    # Real FormattedTextControl path used by Window scroll.
    control = FormattedTextControl(
        lambda: fragments,
        focusable=True,
        show_cursor=False,
        get_cursor_position=(
            tui._task_cursor_position
            if which == "task"
            else tui._detail_cursor_position
        ),
    )
    content = control.create_content(80, None)
    assert 0 <= content.cursor_position.y < content.line_count
    content.get_line(content.cursor_position.y)  # must not IndexError


def test_count_fragment_lines_matches_split_lines() -> None:
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
    cases = [
        [],
        [("", "")],
        [("", "\n")],
        [("", "a\n")],
        [("", "a\nb")],
        [("", "a"), ("", "\n")],
        [("", "a\n"), ("", "b\n")],
        [("", "line\n", lambda e: None)],
    ]
    for fr in cases:
        assert tui._count_fragment_lines(fr) == _pt_line_count(fr)


def test_task_cursor_clamped_when_selected_line_huge() -> None:
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        tui._selected_line = 999_999
        tui._follow_latest_task = True
        for i in range(5):
            tui.ledger.create(f"task-{i}-" + ("x" * 120))
        fr = tui._render_tasks()
        _assert_cursor_safe(tui, fr, which="task")


def test_detail_cursor_clamped_on_empty_and_body() -> None:
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        tui._modal_open = True
        tui._modal_kind = "agent"
        tui._modal_ref = ("missing", "missing:main")
        tui._detail_cursor_line = 500
        fr = tui._render_modal_body()
        _assert_cursor_safe(tui, fr, which="detail")

        task = tui.ledger.create("hello world")
        main_id = f"{task.id}:main"
        tui._modal_ref = (task.id, main_id)
        tui._detail_messages[(task.id, main_id)] = []
        tui._detail_cursor_line = 999
        fr = tui._render_modal_body()
        _assert_cursor_safe(tui, fr, which="detail")


def test_auth_body_uses_real_line_count_not_item_index() -> None:
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        tui._modal_open = True
        tui._modal_kind = "auth"
        tui._auth_wizard.open(active_provider="grok", active_model="grok-4.5")
        tui._auth_wizard.cursor = 0
        fr = tui._render_auth_body()
        assert fr
        _assert_cursor_safe(tui, fr, which="detail")
        assert tui._detail_line_count == tui._count_fragment_lines(fr)


def test_auth_body_empty_items_never_returns_empty_fragments() -> None:
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        tui._modal_open = True
        tui._modal_kind = "auth"
        tui._auth_wizard.open(active_provider="grok", active_model="grok-4.5")
        tui._auth_wizard.items = []
        tui._auth_wizard.body_note = ""
        # Force non-WAITING so chrome does not fill the body alone.
        tui._auth_wizard.step = WizardStep.HOME
        fr = tui._render_auth_body()
        assert fr
        _assert_cursor_safe(tui, fr, which="detail")


def test_modal_underlay_empty_content_clamps_task_cursor() -> None:
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        tui._startup_brand = False
        tui._follow_latest_task = False
        for i in range(4):
            tui.ledger.create(f"t{i}")
        tui._render_tasks()
        tui._selected_line = 12
        saved = tui._selected_line
        tui._modal_open = True
        fr = tui._render_tasks()
        pos = tui._task_cursor_position()
        # Underlay is empty while modal is open — only line count / clamp matter.
        assert tui._task_line_count == _pt_line_count(fr)
        # Cursor position is clamped for paint; free-scroll y is preserved.
        assert 0 <= pos.y < tui._task_line_count
        assert tui._selected_line == saved
        tui._modal_open = False
        tui._render_tasks()
        assert tui._selected_line == saved


def test_modal_title_never_empty() -> None:
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        tui._modal_open = True
        tui._modal_kind = "agent"
        tui._modal_ref = None
        assert tui._render_modal_title()
        tui._modal_ref = ("gone", "gone:main")
        assert tui._render_modal_title()


def test_scroll_mouse_routing_modal_vs_tasks() -> None:
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        for i in range(3):
            tui.ledger.create(f"t{i}")
        tui._render_tasks()
        tui._follow_latest_task = True
        before = tui._selected_line

        class _Pos:
            x = 0
            y = 0

        def make_event(etype: MouseEventType) -> MouseEvent:
            return MouseEvent(position=_Pos(), event_type=etype, button=None, modifiers=None)

        # Task handler moves card focus when modal closed (not line-scroll).
        # Wheel is damped — fire enough same-direction ticks to commit one move.
        from codedoggy.tui.app import _WHEEL_TASK_NOTCHES

        handler = tui._only_mouse_up(lambda _e: None, scroll_target="tasks")
        tui._modal_open = False
        tui._task_selection_active = True
        tui._selected_task = 0
        tui._follow_latest_task = True
        tui._wheel_task_accum = 0
        tui._wheel_task_last_move_at = 0.0
        for _ in range(_WHEEL_TASK_NOTCHES):
            assert handler(make_event(MouseEventType.SCROLL_DOWN)) is None
        assert tui._selected_task == 1
        assert tui._follow_latest_task is False

        # Same handler must not scroll tasks while modal is open.
        tui._follow_latest_task = True
        tui._selected_line = before
        tui._modal_open = True
        assert handler(make_event(MouseEventType.SCROLL_DOWN)) is NotImplemented

        # Detail target only works with modal open.
        detail = tui._only_mouse_up(lambda _e: None, scroll_target="detail")
        tui._detail_line_count = 20
        tui._detail_cursor_line = 0
        assert detail(make_event(MouseEventType.SCROLL_DOWN)) is None
        assert tui._detail_cursor_line > 0

        none_h = tui._only_mouse_up(lambda _e: None, scroll_target="none")
        assert none_h(make_event(MouseEventType.SCROLL_UP)) is NotImplemented


def test_task_card_wheel_moves_focus() -> None:
    from codedoggy.tui.app import _WHEEL_TASK_NOTCHES

    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        for i in range(4):
            tui.ledger.create(f"t{i}")
        tui._task_selection_active = True
        tui._selected_task = 1
        tui._follow_latest_task = False
        tui._wheel_task_accum = 0
        tui._wheel_task_last_move_at = 0.0

        class _Pos:
            x = 0
            y = 0

        handler = tui._task_card_mouse(1)
        up = MouseEvent(
            position=_Pos(),
            event_type=MouseEventType.SCROLL_UP,
            button=None,
            modifiers=None,
        )
        down = MouseEvent(
            position=_Pos(),
            event_type=MouseEventType.SCROLL_DOWN,
            button=None,
            modifiers=None,
        )
        # Single tick must not leap; need accumulated notches.
        assert handler(up) is None
        assert tui._selected_task == 1
        for _ in range(_WHEEL_TASK_NOTCHES - 1):
            assert handler(up) is None
        assert tui._selected_task == 0
        tui._wheel_task_last_move_at = 0.0
        for _ in range(_WHEEL_TASK_NOTCHES):
            assert handler(down) is None
        assert tui._selected_task == 1


def test_wheel_task_focus_resets_on_direction_flip() -> None:
    from codedoggy.tui.app import _WHEEL_TASK_NOTCHES

    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        for i in range(3):
            tui.ledger.create(f"t{i}")
        tui._task_selection_active = True
        tui._selected_task = 1
        tui._wheel_task_accum = 0
        tui._wheel_task_last_move_at = 0.0
        # Two down, then up should wipe the accumulator (no move yet).
        tui._wheel_task_focus(1)
        tui._wheel_task_focus(1)
        assert tui._selected_task == 1
        assert abs(tui._wheel_task_accum) == 2
        tui._wheel_task_focus(-1)
        assert tui._wheel_task_accum == -1
        assert tui._selected_task == 1
        for _ in range(_WHEEL_TASK_NOTCHES - 1):
            tui._wheel_task_focus(-1)
        assert tui._selected_task == 0


def test_auth_set_cursor_rejects_disabled() -> None:
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        tui._auth_wizard.open(active_provider="grok", active_model="grok-4.5")
        items = tui._auth_wizard.items
        assert items
        disabled_idx = next(
            (i for i, it in enumerate(items) if not it.enabled), None
        )
        if disabled_idx is None:
            items[0].enabled = False
            disabled_idx = 0
            tui._auth_wizard.cursor = min(1, len(items) - 1)
        prev = tui._auth_wizard.cursor
        assert tui._auth_wizard.set_cursor(disabled_idx) is False
        assert tui._auth_wizard.cursor == prev


def test_task_scroll_does_not_repin_every_paint() -> None:
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        for i in range(4):
            tui.ledger.create(f"task long title {i} " + ("body " * 30))
        tui._follow_latest_task = False
        tui._selected_task = 1
        tui._pinned_task_for_line = None
        tui._render_tasks()
        first = tui._selected_line
        tui._selected_line = min(first + 3, tui._task_line_count - 1)
        moved = tui._selected_line
        tui._render_tasks()
        # Same task: free-scroll line must survive re-render.
        assert tui._selected_line == moved


def test_ensure_fragments_never_empty() -> None:
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
    assert tui._ensure_fragments([])
    assert tui._ensure_fragments(None)
    assert tui._count_fragment_lines(tui._ensure_fragments([])) >= 1


def test_wrap_display_lines_caps_and_fits() -> None:
    from codedoggy.tui.app import _wrap_display_lines
    from prompt_toolkit.utils import get_cwidth

    assert _wrap_display_lines("hello world", 20, max_lines=2) == ["hello world"]
    text = "abcdefghij" * 2
    lines = _wrap_display_lines(text, 10, max_lines=2)
    assert len(lines) == 2
    assert "".join(lines) == text
    long = "abcdefghij" * 3
    capped = _wrap_display_lines(long, 10, max_lines=2)
    assert len(capped) == 2
    assert all(get_cwidth(line) <= 10 for line in capped)


def test_task_paint_cache_skips_rebuild_when_idle() -> None:
    """Idle task list must not full-walk cards on every refresh tick."""
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        tui._startup_brand = False
        tui.ledger.create("short done")
        fr1 = tui._render_tasks()
        fr2 = tui._render_tasks()
        assert fr1 is fr2
        # Content change busts the cache.
        tui.ledger.create("another")
        fr3 = tui._render_tasks()
        assert fr3 is not fr1


def test_focus_latest_task_from_prompt_ignores_prior_selection() -> None:
    """Tab / focus-latest from input lands on the newest task, not last browse."""
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        for i in range(4):
            tui.ledger.create(f"task-{i}")
        tui._selected_task = 0
        tui._follow_latest_task = False
        tui._pinned_task_for_line = 0
        assert tui._focus_latest_task() is True
        assert tui._selected_task == 3
        assert tui._follow_latest_task is True
        assert tui._task_refs[-1] == tui.ledger.snapshots()[-1].id


def test_blank_click_clears_selection_not_first_task() -> None:
    """Void click clears selection; paint must never invent index 0."""
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        for i in range(3):
            tui.ledger.create(f"t{i}")
        assert tui._selected_task == -1
        assert tui._task_selection_active is False
        tui._render_tasks()
        assert tui._selected_task == -1

        tui._selected_task = 2
        tui._task_selection_active = True
        tui._follow_latest_task = True
        tui._render_tasks()

        class _Pos:
            x = 1
            y = 1

        void = tui._task_void_mouse()
        void(
            MouseEvent(
                position=_Pos(),
                event_type=MouseEventType.MOUSE_UP,
                button=MouseButton.LEFT,
                modifiers=None,
            )
        )
        assert tui._selected_task == -1
        assert tui._task_selection_active is False
        assert tui._follow_latest_task is False

        fr = tui._render_tasks()
        plain = "".join(p[1] for p in fr)
        assert tui._selected_task == -1
        assert "›" not in plain


def test_blank_lines_are_full_width_hit_targets(monkeypatch: object) -> None:
    """Bare ``\\n`` rows remap to (0,0)=first card in prompt_toolkit Window.

    Every gap/void row must paint full-width spaces with a clear-selection handler.
    """
    monkeypatch.setattr(
        "codedoggy.tui.app._terminal_width",
        lambda: 40,
    )
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        for i in range(2):
            tui.ledger.create(f"t{i}")
        tui._selected_task = 1
        tui._task_selection_active = True
        fr = tui._render_tasks()

        blank_rows = [
            f
            for f in fr
            if len(f) >= 3
            and f[2] is not None
            and f[1].endswith("\n")
            and f[1][:-1] != ""
            and f[1][:-1].strip() == ""
        ]
        assert blank_rows, "expected gap/void rows with space hit-targets"
        for _style, text, _handler in blank_rows:
            body = text[:-1]
            assert len(body) >= 40, f"short blank row remaps to card0: {text!r}"

        # Clicking a void/gap handler must clear, not select card 0.
        from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

        class _Pos:
            x = 20
            y = 0

        blank_rows[-1][2](
            MouseEvent(
                position=_Pos(),
                event_type=MouseEventType.MOUSE_UP,
                button=MouseButton.LEFT,
                modifiers=None,
            )
        )
        assert tui._selected_task == -1
        assert tui._task_selection_active is False


def test_interject_shows_on_homepage_card() -> None:
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        t = tui.ledger.create("running")
        tui.ledger.update_agent(
            t.id, f"{t.id}:main", label="MAIN", status="running", output="…"
        )
        tui._active_task_id = t.id
        tui._note_interject(t.id, "补充：请用 JWT")
        preview = tui._interject_preview(t.id)
        assert preview is not None
        assert "JWT" in preview
        from codedoggy.tui.app import _task_list_summary, _task_stage_text

        snap = tui.ledger.snapshots()[0]
        assert "插入中" in _task_stage_text(snap, interject=preview)
        assert "插入中" in _task_list_summary(snap, interject=preview)


def test_tab_cycle_latest_enter_exit() -> None:
    """Tab: latest → enter detail → exit back to that card."""
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        for i in range(3):
            t = tui.ledger.create(f"t{i}")
            tui.ledger.update_agent(
                t.id,
                f"{t.id}:main",
                label="MAIN",
                status="completed",
                output="ok",
            )

        # 1) From input → latest task selected, no modal.
        tui.app.layout.focus(tui._input)
        tui._task_selection_active = False
        tui._selected_task = -1
        tui._tab_task_cycle()
        assert tui._selected_task == 2
        assert tui._task_selection_active is True
        assert tui._modal_open is False

        # 2) On selected card → open detail.
        tui.app.layout.focus(tui._task_window)
        tui._tab_task_cycle()
        assert tui._modal_open is True
        assert tui._modal_kind == "agent"
        assert tui._modal_ref is not None
        assert tui._modal_ref[0] == tui.ledger.snapshots()[2].id

        # 3) Inside detail → exit, keep that card selected.
        tui._tab_task_cycle()
        assert tui._modal_open is False
        assert tui._selected_task == 2
        assert tui._task_selection_active is True


def test_user_wants_latest_focus_respects_browse_above() -> None:
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        for i in range(3):
            tui.ledger.create(f"t{i}")
        tui._follow_latest_task = True
        assert tui._user_wants_latest_focus() is True
        tui._follow_latest_task = False
        tui._selected_task = 0
        # No render_info → not at bottom unless following.
        assert tui._user_wants_latest_focus() is False


def test_task_card_plain_click_selects_double_and_ctrl_open() -> None:
    """Plain left: select. Double-click or Ctrl+left: open detail."""
    import time

    from prompt_toolkit.mouse_events import (
        MouseButton,
        MouseEvent,
        MouseEventType,
        MouseModifier,
    )

    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        t = tui.ledger.create("open-me")
        tui.ledger.update_agent(
            t.id, f"{t.id}:main", label="MAIN", status="completed", output="ok"
        )
        tui._render_tasks()
        handler = tui._task_card_mouse(0)

        class _Pos:
            x = 1
            y = 1

        def fire(
            etype: MouseEventType, *, mods: frozenset | None = None
        ) -> object:
            return handler(
                MouseEvent(
                    position=_Pos(),
                    event_type=etype,
                    button=MouseButton.LEFT,
                    modifiers=mods,
                )
            )

        # Plain click: select, never open modal.
        fire(MouseEventType.MOUSE_DOWN)
        fire(MouseEventType.MOUSE_UP)
        assert tui._modal_open is False
        assert tui._selected_task == 0

        # Slow second click still selects only (outside double-click window).
        tui._task_card_last_click = (0, time.monotonic() - 1.0)
        fire(MouseEventType.MOUSE_DOWN)
        fire(MouseEventType.MOUSE_UP)
        assert tui._modal_open is False
        assert tui._selected_task == 0

        # Double-click opens detail.
        tui._task_card_last_click = (0, time.monotonic())
        fire(MouseEventType.MOUSE_DOWN)
        fire(MouseEventType.MOUSE_UP)
        assert tui._modal_open is True

        tui._close_modal()
        assert tui._modal_open is False

        # Ctrl+left opens detail.
        fire(MouseEventType.MOUSE_DOWN)
        fire(MouseEventType.MOUSE_UP, mods=frozenset({MouseModifier.CONTROL}))
        assert tui._modal_open is True


def test_detail_path_link_opens_without_control_modifier(
    tmp_path: object, monkeypatch: object
) -> None:
    """Dedicated detail links must open on plain click (Win32 reports no Ctrl)."""
    from pathlib import Path

    from prompt_toolkit.mouse_events import MouseButton, MouseModifier

    from codedoggy.tui.app import _mouse_control_held
    import codedoggy.tui.open_path as op

    root = Path(str(tmp_path))
    img = root / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    opened: list[str] = []
    monkeypatch.setattr(op.sys, "platform", "win32")
    monkeypatch.setattr(op.os, "startfile", lambda p: opened.append(p), raising=False)

    # Helper: enum CONTROL counts; empty frozenset does not (without real key).
    assert _mouse_control_held(
        MouseEvent(
            position=type("P", (), {"x": 0, "y": 0})(),
            event_type=MouseEventType.MOUSE_UP,
            button=MouseButton.LEFT,
            modifiers=frozenset({MouseModifier.CONTROL}),
        )
    )
    # Empty mods: only true if physical Ctrl is down — do not assert False on
    # a live Ctrl-held developer machine; just exercise the call.
    _mouse_control_held(
        MouseEvent(
            position=type("P", (), {"x": 0, "y": 0})(),
            event_type=MouseEventType.MOUSE_UP,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
    )

    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        tui.session.cwd = root  # type: ignore[attr-defined]
        handler = tui._image_path_mouse(str(img))
        result = handler(
            MouseEvent(
                position=type("P", (), {"x": 0, "y": 0})(),
                event_type=MouseEventType.MOUSE_UP,
                button=MouseButton.LEFT,
                modifiers=frozenset(),  # Win32 reality
            )
        )
        assert result is None
        assert opened and str(opened[0]).endswith("shot.png")


def test_task_card_frame_carries_mouse_handler() -> None:
    """Whole card (frame included) is a click target."""
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        tui.ledger.create("card")
        fr = tui._render_tasks()
        # Top border fragment should have a mouse handler (3-tuple).
        tops = [f for f in fr if len(f) >= 2 and "╭" in f[1]]
        assert tops
        assert len(tops[0]) >= 3 and tops[0][2] is not None


def test_detail_scroll_helpers_move_cursor_and_window() -> None:
    """↑↓ / End / absolute jump must move both cursor anchor and vertical_scroll."""
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        t = tui.ledger.create("scroll-me")
        tui.ledger.update_agent(
            t.id, f"{t.id}:main", label="MAIN", status="completed", output="body"
        )
        tui._open_agent(t.id, f"{t.id}:main")
        tui._detail_line_count = 80
        tui._detail_cursor_line = 0
        tui._detail_window.vertical_scroll = 0

        tui._move_detail_cursor(10)
        assert tui._detail_cursor_line == 10

        tui._scroll_detail(5)
        assert tui._detail_window.vertical_scroll >= 5
        assert tui._detail_cursor_line == tui._detail_window.vertical_scroll

        tui._scroll_detail_to_bottom()
        assert tui._detail_cursor_line == 79
        assert tui._detail_window.vertical_scroll == 79  # no render_info → max_y

        tui._scroll_detail_to_line(3)
        assert tui._detail_cursor_line == 3
        assert tui._detail_window.vertical_scroll == 3


def test_interactive_scrollbar_margin_emits_handlers() -> None:
    """Margin paints glyphs; mouse wiring is via install_mouse_handlers."""
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

    from codedoggy.tui.app import InteractiveScrollbarMargin

    class _Win:
        vertical_scroll = 5

    class _Info:
        content_height = 40
        window_height = 10
        vertical_scroll = 5
        window = _Win()

    margin = InteractiveScrollbarMargin()
    fr = margin.create_margin(_Info(), 1, 10)
    assert any(f[1] in {"▴", "▾", " "} for f in fr)

    handlers = MouseHandlers()
    info = _Info()
    win = info.window

    def _set(w: object, max_scroll: int, value: int) -> None:
        setattr(w, "vertical_scroll", max(0, min(max_scroll, int(value))))

    margin._set_scroll = _set  # type: ignore[method-assign]
    margin.install_mouse_handlers(
        handlers,
        window_render_info=info,
        bar_xpos=20,
        ypos=0,
        height=10,
        capture_x_min=0,
        capture_x_max=21,
    )
    # Row 0 = ▴; thumb near top; row 7 is track below → jump scroll.
    before = win.vertical_scroll
    handlers.mouse_handlers[7][20](
        MouseEvent(
            position=Point(x=20, y=7),
            event_type=MouseEventType.MOUSE_DOWN,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
    )
    assert win.vertical_scroll != before

    # Thumb at track index 1 → screen y=2 → starts drag.
    win.vertical_scroll = 5
    info.vertical_scroll = 5
    margin._dragging = False
    margin.install_mouse_handlers(
        handlers,
        window_render_info=info,
        bar_xpos=20,
        ypos=0,
        height=10,
        capture_x_min=0,
        capture_x_max=21,
    )
    handlers.mouse_handlers[2][20](
        MouseEvent(
            position=Point(x=20, y=2),
            event_type=MouseEventType.MOUSE_DOWN,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
    )
    assert margin._dragging


def test_scrollbar_drag_capture_follows_move_off_rail() -> None:
    """While dragging, MOVE over content (not the 1-cell rail) still scrolls."""
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

    from codedoggy.tui.app import InteractiveScrollbarMargin

    class _Win:
        vertical_scroll = 10

    class _Info:
        content_height = 100
        window_height = 10
        vertical_scroll = 10
        window = _Win()

    margin = InteractiveScrollbarMargin()
    info = _Info()
    win = info.window

    def _set(w: object, max_scroll: int, value: int) -> None:
        setattr(w, "vertical_scroll", max(0, min(max_scroll, int(value))))

    margin._set_scroll = _set  # type: ignore[method-assign]
    handlers = MouseHandlers()
    margin.install_mouse_handlers(
        handlers,
        window_render_info=info,
        bar_xpos=20,
        ypos=0,
        height=10,
        capture_x_min=0,
        capture_x_max=21,
    )
    # Thumb at track index 1 → screen y=2.
    handlers.mouse_handlers[2][20](
        MouseEvent(
            position=Point(x=20, y=2),
            event_type=MouseEventType.MOUSE_DOWN,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
    )
    assert margin._dragging
    margin.install_mouse_handlers(
        handlers,
        window_render_info=info,
        bar_xpos=20,
        ypos=0,
        height=10,
        capture_x_min=0,
        capture_x_max=21,
    )
    start = win.vertical_scroll
    handlers.mouse_handlers[7][5](
        MouseEvent(
            position=Point(x=5, y=7),
            event_type=MouseEventType.MOUSE_MOVE,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
    )
    assert win.vertical_scroll != start


def test_task_cards_never_show_agent_rows() -> None:
    """Selecting / following a card must not inject ↳ MAIN lines (height jiggle)."""
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        t = tui.ledger.create("done-task")
        tui.ledger.update_agent(
            t.id, f"{t.id}:main", label="MAIN", status="completed", output="ok"
        )
        tui.ledger.finish_task(t.id, "completed")
        tui._follow_latest_task = True
        tui._selected_task = 0
        tui._task_selection_active = True
        text = "".join(p[1] for p in tui._render_tasks())
        assert "↳" not in text
