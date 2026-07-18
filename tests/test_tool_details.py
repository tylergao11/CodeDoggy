"""Behavioral detail tests for core file tools (Grok-aligned)."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.tools.builtins.read_file import (
    extract_file_content_lines,
    resolve_read_start_line,
)
from codedoggy.tools.grok_build.list_dir import (
    DEFAULT_MAX_OUTPUT_CHARS,
    TOP_K_EXTENSIONS,
    budget_expand,
    build_tree,
    root_truncation_notice,
)
from codedoggy.tools.grok_build.read_file_extract import (
    extract_file_content_lines as extract_full,
)
from codedoggy.tools.runtime import ToolCallContext, ToolError
from codedoggy.tools import ToolRegistryBuilder


@pytest.fixture
def tools(tmp_path: Path):
    return ToolRegistryBuilder.new().finalize(), ToolCallContext(cwd=tmp_path), tmp_path


def test_read_line_numbers_every_tenth() -> None:
    # Grok: extract_decade_line_numbered_in_addition_to_first
    file_content = "".join(f"L{i}\n" for i in range(1, 13))
    out = extract_file_content_lines(file_content, offset=None, limit=None)
    assert out == "1→L1\nL2\nL3\nL4\nL5\nL6\nL7\nL8\nL9\n10→L10\nL11\nL12\n"


def test_read_offset_default_and_zero() -> None:
    assert resolve_read_start_line("a\nb\n", None) == 1
    assert resolve_read_start_line("a\nb\n", 0) == 1
    assert resolve_read_start_line("a\nb\n", 2) == 2


def test_read_offset_negative_harness() -> None:
    # Grok resolve_read_start_line_* tests
    assert resolve_read_start_line("a\nb\nc\n", -3) == 2
    assert resolve_read_start_line("a\nb\nc", -3) == 2
    assert resolve_read_start_line("a\nb\nc\n", -999) == 1
    assert resolve_read_start_line("a\nb\nc\n", 0) == 1


def test_extract_basic_crlf() -> None:
    # Grok test_extract_file_content_lines_basic
    extracted = extract_full("1\n2\r\n3\n", None, None, 4)
    assert extracted.content == "1→1\n2\n3\n"
    assert extracted.raw_output == "1\n2\r\n3\n"


def test_extract_with_offset() -> None:
    extracted = extract_full("1\n2\n3\r\n4\r", 3, None, 4)
    assert extracted.content == "3→3\n4\r"
    assert extracted.raw_output == "3\r\n4\r"


def test_extract_offset_and_limit() -> None:
    extracted = extract_full("1\n2\n3\r\n4\r", 2, 2, 4)
    assert extracted.content == "2→2\n3"
    assert extracted.raw_output == "2\n3\n"


def test_extract_negative_offset() -> None:
    file_content = "line1\nline2\nline3\nline4\nline5\n"
    total_lines = file_content.count("\n") + 1
    extracted = extract_full(file_content, -2, 2, total_lines)
    assert extracted.content == "5→line5\n"


def test_extract_trailing_empty_line() -> None:
    # Grok: trailing \n yields an extra blank (unnumbered unless decade)
    assert extract_file_content_lines("a\nb\nc\n", None, None) == "1→a\nb\nc\n"
    assert extract_file_content_lines("hello\nworld\n", 1, 10) == "1→hello\nworld\n"


def test_read_empty_file(tools) -> None:
    set_, ctx, tmp = tools
    (tmp / "e.txt").write_text("", encoding="utf-8")
    out = set_.call("read_file", {"target_file": "e.txt"}, ctx)
    assert out == ""


def test_read_image_returns_metadata(tools) -> None:
    set_, ctx, tmp = tools
    (tmp / "x.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    out = set_.call("read_file", {"target_file": "x.png"}, ctx)
    assert "[image]" in out
    assert "size_bytes" in out


def test_read_binary_rejected(tools) -> None:
    set_, ctx, tmp = tools
    (tmp / "x.bin").write_bytes(b"\x00\x01\x02\x03" + b"\xff" * 100)
    with pytest.raises(ToolError) as ei:
        set_.call("read_file", {"target_file": "x.bin"}, ctx)
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
    set_.call(
        "search_replace",
        {
            "file_path": "a.txt",
            "old_string": "x",
            "new_string": "y",
            "replace_all": True,
        },
        ctx,
    )
    assert (tmp / "a.txt").read_text(encoding="utf-8") == "y y"


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
    assert "g.txt" in out  # BFS expands small subdirs


def test_list_dir_file_error(tools) -> None:
    set_, ctx, tmp = tools
    (tmp / "f.txt").write_text("1", encoding="utf-8")
    with pytest.raises(ToolError, match="is a file"):
        set_.call("list_dir", {"target_directory": "f.txt"}, ctx)


def test_list_dir_deep_expanded(tmp_path: Path) -> None:
    # Grok small_deep_dirs_are_expanded
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "deep.rs").write_text("x", encoding="utf-8")
    tree, trunc = build_tree(tmp_path)
    body = budget_expand(
        tree, DEFAULT_MAX_OUTPUT_CHARS, TOP_K_EXTENSIONS, trunc, root_truncation_notice()
    )
    assert "deep.rs" in body
    assert "- a/" in body
    assert "- b/" in body
    assert "- c/" in body


def test_list_dir_large_summarized(tmp_path: Path) -> None:
    # Grok large_directory_is_summarized_by_budget
    subdir = tmp_path / "big"
    subdir.mkdir()
    for i in range(50):
        (subdir / f"file{i}.rs").write_text("x", encoding="utf-8")
    tree, trunc = build_tree(tmp_path)
    body = budget_expand(tree, 200, TOP_K_EXTENSIONS, trunc, root_truncation_notice())
    assert "- big/" in body
    assert "[50 files in subtree: 50 *.rs]" in body
    assert "file0.rs" not in body


def test_list_dir_hidden_excluded(tmp_path: Path) -> None:
    (tmp_path / ".hidden").write_text("x", encoding="utf-8")
    (tmp_path / "visible.rs").write_text("x", encoding="utf-8")
    tree, trunc = build_tree(tmp_path)
    body = budget_expand(
        tree, DEFAULT_MAX_OUTPUT_CHARS, TOP_K_EXTENSIONS, trunc, root_truncation_notice()
    )
    assert "visible.rs" in body
    assert ".hidden" not in body


def test_list_dir_cutoff_notice(tmp_path: Path) -> None:
    (tmp_path / "file.rs").write_text("x", encoding="utf-8")
    tree, _ = build_tree(tmp_path)
    body = budget_expand(
        tree, DEFAULT_MAX_OUTPUT_CHARS, TOP_K_EXTENSIONS, True, root_truncation_notice()
    )
    assert "more than 100000 items" in body


def test_search_replace_crlf_preserves_endings(tools) -> None:
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
    assert "gitignore" in desc.lower()
    assert "large" in desc.lower() or "summar" in desc.lower()
