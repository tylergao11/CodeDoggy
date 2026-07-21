"""Clipboard image → path helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from codedoggy.tui.clipboard_image import (
    coerce_image_path_text,
    get_system_clipboard_text,
    insert_path_token,
    save_clipboard_image,
)


def test_save_clipboard_image_none_when_no_image(tmp_path: Path) -> None:
    with patch(
        "codedoggy.tui.clipboard_image._save_windows", return_value=None
    ), patch(
        "codedoggy.tui.clipboard_image._save_macos", return_value=None
    ), patch(
        "codedoggy.tui.clipboard_image._save_linux", return_value=None
    ):
        assert save_clipboard_image(tmp_path) is None


def test_save_clipboard_image_writes_under_attachments(tmp_path: Path) -> None:
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

    def _fake_save(dest: Path, **_kwargs: object) -> Path | None:
        dest.write_bytes(payload)
        return dest

    with patch("codedoggy.tui.clipboard_image.sys.platform", "win32"), patch(
        "codedoggy.tui.clipboard_image._save_windows", side_effect=_fake_save
    ):
        path = save_clipboard_image(tmp_path)
    assert path is not None
    assert path.is_file()
    assert ".codedoggy" in path.parts
    assert "attachments" in path.parts
    assert path.name.startswith("paste-")
    token = insert_path_token(path)
    assert path.name in token or "paste-1" in token


def test_insert_path_token_spaces(tmp_path: Path) -> None:
    p = tmp_path / "my file.png"
    p.write_bytes(b"x")
    tok = insert_path_token(p)
    assert tok.startswith('"') and tok.endswith('"')


def test_get_system_clipboard_text_uses_os_hook(monkeypatch) -> None:
    with patch(
        "codedoggy.tui.clipboard_image._windows_clipboard_text",
        return_value="hello clip",
    ), patch("codedoggy.tui.clipboard_image.sys.platform", "win32"):
        assert get_system_clipboard_text() == "hello clip"


def test_unique_paste_names(tmp_path: Path) -> None:
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

    def _fake_save(dest: Path, **_kwargs: object) -> Path | None:
        dest.write_bytes(payload)
        return dest

    with patch("codedoggy.tui.clipboard_image.sys.platform", "win32"), patch(
        "codedoggy.tui.clipboard_image._save_windows", side_effect=_fake_save
    ):
        a = save_clipboard_image(tmp_path)
        b = save_clipboard_image(tmp_path)
    assert a is not None and b is not None
    assert a != b


def test_coerce_image_path_text(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
    assert coerce_image_path_text(str(img)) == img.resolve()
    assert coerce_image_path_text(f'"{img}"') == img.resolve()
    assert coerce_image_path_text("not-a-file.png", cwd=tmp_path) is None
    assert coerce_image_path_text("readme.txt\n", cwd=tmp_path) is None


def test_save_windows_prefers_png_format_bytes(tmp_path: Path) -> None:
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    out_dir = tmp_path / ".codedoggy" / "attachments"
    out_dir.mkdir(parents=True)
    dest = out_dir / "paste-1.png"

    with patch(
        "codedoggy.tui.clipboard_image._windows_clipboard_format_bytes",
        side_effect=lambda name: payload if name in {"PNG", "image/png"} else None,
    ), patch(
        "codedoggy.tui.clipboard_image._windows_hdrop_paths", return_value=[]
    ), patch(
        "codedoggy.tui.clipboard_image._save_windows_dib", return_value=None
    ), patch(
        "codedoggy.tui.clipboard_image._save_windows_powershell_png",
        return_value=None,
    ):
        from codedoggy.tui.clipboard_image import _save_windows

        path = _save_windows(dest, out_dir=out_dir, prefix="paste")
    assert path is not None
    assert path.read_bytes().startswith(b"\x89PNG")
