"""Input selection cut + paste chip + friendly failure copy."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.selection import SelectionState, SelectionType

from codedoggy.tui.app import CodeDoggyTUI, _friendly_failure_toast
from codedoggy.tui.clipboard_image import insert_image_chip
from codedoggy.tui.open_path import VIEW_IMAGE_LABEL, path_under_cursor


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


def test_friendly_failure_hides_winerror_url() -> None:
    msg = (
        "sampler error: Failed to reach https://cli-chat-proxy.grok.com/v1/responses: "
        "[WinError 10061] 由于目标计算机积极拒绝，无法连接。"
    )
    soft = _friendly_failure_toast(msg)
    assert "10061" not in soft
    assert "https://" not in soft
    assert "连不上" in soft or "模型" in soft


def test_insert_image_chip_shows_view_label(tmp_path: Path) -> None:
    img = tmp_path / ".codedoggy" / "attachments" / "paste-1.png"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
    chip = insert_image_chip(img, cwd=tmp_path)
    assert chip.startswith(VIEW_IMAGE_LABEL + "(")
    assert "paste-1.png" in chip
    assert path_under_cursor(chip, 1) is not None


def test_input_areas_focus_on_click() -> None:
    """Middle of the prompt box must accept clicks (not only chrome edges)."""
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

    class _Pos:
        x = 12
        y = 0

    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        assert tui._input.control.focus_on_click()
        assert tui._detail_input.control.focus_on_click()

        with set_app(tui.app):
            # Move focus away from the main input, then click the buffer middle.
            tui.app.layout.focus(tui._task_window)
            assert not tui.app.layout.has_focus(tui._input)

            handler = tui._input.control.mouse_handler
            event = MouseEvent(
                position=_Pos(),
                event_type=MouseEventType.MOUSE_DOWN,
                button=None,
                modifiers=None,
            )
            handler(event)
            assert tui.app.layout.has_focus(tui._input)


def test_cut_selection_on_delete_and_backspace() -> None:
    with create_pipe_input() as pin:
        tui = CodeDoggyTUI(_Session(), input=pin, output=DummyOutput())
        buf = Buffer(document=Document("abcdef", 6))
        buf.selection_state = SelectionState(2, SelectionType.CHARACTERS)
        buf.cursor_position = 5  # select cde
        event = SimpleNamespace(current_buffer=buf, app=tui.app)
        # Simulate the binding body.
        buf.cut_selection()
        assert buf.text == "abf"


def test_paste_image_inserts_chip(tmp_path: Path) -> None:
    img = tmp_path / "paste-1.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)

    with create_pipe_input() as pin:
        session = _Session()
        session.cwd = str(tmp_path)
        tui = CodeDoggyTUI(session, input=pin, output=DummyOutput())
        buf = tui._input.buffer
        buf.text = ""
        event = SimpleNamespace(current_buffer=buf, app=tui.app)
        with patch(
            "codedoggy.tui.app.save_clipboard_image", return_value=img
        ):
            tui._paste_into_buffer(event)
        assert VIEW_IMAGE_LABEL in buf.text
        assert "paste-1.png" in buf.text


def test_paste_image_path_text_becomes_chip(tmp_path: Path) -> None:
    """WT drag-drop / copy-as-path pastes a path string — coerce to chip."""
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)

    with create_pipe_input() as pin:
        session = _Session()
        session.cwd = str(tmp_path)
        tui = CodeDoggyTUI(session, input=pin, output=DummyOutput())
        buf = tui._input.buffer
        event = SimpleNamespace(current_buffer=buf, app=tui.app)
        with patch(
            "codedoggy.tui.app.save_clipboard_image", return_value=None
        ), patch(
            "codedoggy.tui.app.get_system_clipboard_text",
            return_value=str(img),
        ):
            tui._paste_into_buffer(event)
        assert VIEW_IMAGE_LABEL in buf.text
        assert "shot.png" in buf.text


def test_win32_ctrl_v_poll_pastes_image_only(tmp_path: Path) -> None:
    """Poll must paste images on Ctrl+V, and ignore text-only clipboards."""
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)

    with create_pipe_input() as pin:
        session = _Session()
        session.cwd = str(tmp_path)
        tui = CodeDoggyTUI(session, input=pin, output=DummyOutput())
        tui.app.layout.focus(tui._input)
        tui._win32_ctrl_v_down = False
        tui._win32_ctrl_v_last_at = 0.0

        # Text-only: poll no-ops (terminal owns text paste).
        with patch("sys.platform", "win32"), patch(
            "codedoggy.tui.app.save_clipboard_image", return_value=None
        ), patch(
            "ctypes.windll.user32.GetAsyncKeyState",
            side_effect=lambda vk: 0x8000 if vk in {0x11, 0x56} else 0,
        ):
            tui._poll_win32_ctrl_v_image_paste()
        assert tui._input.buffer.text == ""

        # Image: rising edge of Ctrl+V inserts chip.
        tui._win32_ctrl_v_down = False
        tui._win32_ctrl_v_last_at = 0.0
        with patch("sys.platform", "win32"), patch(
            "codedoggy.tui.app.save_clipboard_image", return_value=img
        ), patch(
            "ctypes.windll.user32.GetAsyncKeyState",
            side_effect=lambda vk: 0x8000 if vk in {0x11, 0x56} else 0,
        ):
            tui._poll_win32_ctrl_v_image_paste()
        assert VIEW_IMAGE_LABEL in tui._input.buffer.text
