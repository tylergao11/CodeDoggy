"""Behavioral detail tests for core file tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.tools.builtins.read_file import (
    extract_file_content_lines,
    resolve_read_start_line,
)
from codedoggy.tools.runtime import ToolCallContext, ToolError
from codedoggy.tools import ToolRegistryBuilder


@pytest.fixture
def tools(tmp_path: Path):
    return ToolRegistryBuilder.new().finalize(), ToolCallContext(cwd=tmp_path), tmp_path


def test_read_line_numbers_every_tenth() -> None:
    # 12 lines: prefixes on 1 and 10
    content = "\n".join(f"L{i}" for i in range(1, 13)) + "\n"
    out = extract_file_content_lines(content, offset=1, limit=12)
    assert out.startswith("1→L1\n")
    assert "\n2→" not in out  # line 2 has no prefix
    assert "L2" in out
    assert "10→L10" in out


def test_read_offset_default_and_zero() -> None:
    assert resolve_read_start_line("a\nb\n", None) == 1
    assert resolve_read_start_line("a\nb\n", 0) == 1
    assert resolve_read_start_line("a\nb\n", 2) == 2


def test_read_offset_negative() -> None:
    # offset=-1 starts at last content line (with or without trailing newline)
    assert resolve_read_start_line("a\nb\nc", -1) == 3
    assert resolve_read_start_line("a\nb\nc\n", -1) == 3
    assert resolve_read_start_line("a\nb\nc\n", -3) == 1
    assert resolve_read_start_line("a\nb\nc", -3) == 1
    assert resolve_read_start_line("a\nb\nc\n", -999) == 1
    out = extract_file_content_lines("a\nb\nc", offset=-1, limit=1)
    assert "c" in out
    assert out.startswith("3→")


def test_read_no_phantom_line_after_trailing_newline() -> None:
    """Trailing \\n does not invent an extra empty numbered line."""
    out = extract_file_content_lines("hello\nworld\n", offset=1, limit=10)
    assert out == "1→hello\nworld"
    assert "3→" not in out
    # Real blank line (file ends with \\n\\n) still appears.
    out2 = extract_file_content_lines("hello\n\n", offset=1, limit=10)
    assert out2 == "1→hello\n"


def test_read_empty_file(tools) -> None:
    set_, ctx, tmp = tools
    (tmp / "e.txt").write_text("", encoding="utf-8")
    out = set_.call("read_file", {"target_file": "e.txt"}, ctx)
    assert out == ""


def test_read_binary_rejected(tools) -> None:
    set_, ctx, tmp = tools
    (tmp / "x.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    with pytest.raises(ToolError) as ei:
        set_.call("read_file", {"target_file": "x.png"}, ctx)
    assert ei.value.code == "binary_file"


def test_search_replace_create_file(tools) -> None:
    set_, ctx, tmp = tools
    msg = set_.call(
        "search_replace",
        {"file_path": "new.txt", "old_string": "", "new_string": "hello"},
        ctx,
    )
    assert "created successfully" in msg
    assert (tmp / "new.txt").read_text(encoding="utf-8") == "hello"


def test_search_replace_same_string_rejected(tools) -> None:
    set_, ctx, tmp = tools
    (tmp / "a.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ToolError, match="same"):
        set_.call(
            "search_replace",
            {"file_path": "a.txt", "old_string": "x", "new_string": "x"},
            ctx,
        )


def test_search_replace_ambiguous(tools) -> None:
    set_, ctx, tmp = tools
    (tmp / "a.txt").write_text("x x", encoding="utf-8")
    with pytest.raises(ToolError) as ei:
        set_.call(
            "search_replace",
            {"file_path": "a.txt", "old_string": "x", "new_string": "y"},
            ctx,
        )
    assert ei.value.code == "edit_ambiguous"


def test_search_replace_all(tools) -> None:
    set_, ctx, tmp = tools
    (tmp / "a.txt").write_text("x x", encoding="utf-8")
    msg = set_.call(
        "search_replace",
        {
            "file_path": "a.txt",
            "old_string": "x",
            "new_string": "y",
            "replace_all": True,
        },
        ctx,
    )
    assert "All occurrences" in msg
    assert (tmp / "a.txt").read_text(encoding="utf-8") == "y y"


def test_search_replace_success_message(tools) -> None:
    set_, ctx, tmp = tools
    (tmp / "a.txt").write_text("hello", encoding="utf-8")
    msg = set_.call(
        "search_replace",
        {"file_path": "a.txt", "old_string": "hello", "new_string": "hi"},
        ctx,
    )
    assert "updated successfully" in msg


def test_list_dir_format(tools) -> None:
    set_, ctx, tmp = tools
    (tmp / "f.txt").write_text("1", encoding="utf-8")
    (tmp / "sub").mkdir()
    (tmp / "sub" / "g.txt").write_text("2", encoding="utf-8")
    (tmp / ".hidden").write_text("h", encoding="utf-8")
    out = set_.call("list_dir", {"target_directory": "."}, ctx)
    assert out.startswith(f"- {tmp}/") or out.startswith("- ")
    assert "- f.txt" in out
    assert "- sub/" in out
    assert ".hidden" not in out


def test_list_dir_file_error(tools) -> None:
    set_, ctx, tmp = tools
    (tmp / "f.txt").write_text("1", encoding="utf-8")
    with pytest.raises(ToolError, match="is a file"):
        set_.call("list_dir", {"target_directory": "f.txt"}, ctx)


def test_search_replace_crlf_preserves_endings(tools) -> None:
    """LF old_string matches CRLF file; write keeps \\r\\n."""
    set_, ctx, tmp = tools
    path = tmp / "crlf.txt"
    path.write_bytes(b"hello\r\nworld\r\n")
    msg = set_.call(
        "search_replace",
        {
            "file_path": "crlf.txt",
            "old_string": "hello\nworld\n",
            "new_string": "goodbye\nearth\n",
        },
        ctx,
    )
    assert "updated successfully" in msg
    assert path.read_bytes() == b"goodbye\r\nearth\r\n"


def test_search_replace_crlf_single_line(tools) -> None:
    set_, ctx, tmp = tools
    path = tmp / "crlf2.txt"
    path.write_bytes(b"aaa\r\nbbb\r\nccc\r\n")
    set_.call(
        "search_replace",
        {"file_path": "crlf2.txt", "old_string": "bbb", "new_string": "BBB"},
        ctx,
    )
    assert path.read_bytes() == b"aaa\r\nBBB\r\nccc\r\n"


def test_search_replace_lf_file_stays_lf(tools) -> None:
    set_, ctx, tmp = tools
    path = tmp / "lf.txt"
    path.write_bytes(b"hello\nworld\n")
    set_.call(
        "search_replace",
        {"file_path": "lf.txt", "old_string": "hello", "new_string": "goodbye"},
        ctx,
    )
    assert path.read_bytes() == b"goodbye\nworld\n"


def test_list_dir_description_honest(tools) -> None:
    set_, _ctx, _tmp = tools
    defs = {d.name: d for d in set_.tool_definitions()}
    desc = defs["list_dir"].description or ""
    assert "gitignore" in desc.lower() or "Does not apply" in desc
    assert "depth" in desc.lower() or "3" in desc
