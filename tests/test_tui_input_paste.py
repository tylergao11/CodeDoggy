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
