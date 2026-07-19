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
        for i in range(4):
            tui.ledger.create(f"t{i}")
        tui._selected_line = 80
        tui._modal_open = True
        fr = tui._render_tasks()
        pos = tui._task_cursor_position()
        assert tui._agent_refs == []
        assert tui._task_line_count == _pt_line_count(fr)
        assert 0 <= pos.y < tui._task_line_count


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

        # Task handler scrolls tasks when modal closed.
        handler = tui._only_mouse_up(lambda _e: None, scroll_target="tasks")
        tui._modal_open = False
        assert handler(make_event(MouseEventType.SCROLL_DOWN)) is None
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
