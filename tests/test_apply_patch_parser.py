"""Source-level tests for codex apply_patch (Grok parser.rs / apply.rs cases)."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.tools.codex.apply_patch.apply_logic import derive_new_contents
from codedoggy.tools.codex.apply_patch.parser import (
    AddFile,
    DeleteFile,
    ParseError,
    ParseMode,
    UpdateFile,
    parse_patch,
    parse_patch_text,
)
from codedoggy.tools.codex.apply_patch.seek_sequence import seek_sequence
from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.runtime import ToolCallContext, ToolError


def test_parse_patch_bad_first_line() -> None:
    with pytest.raises(ParseError, match="Begin Patch"):
        parse_patch_text("bad", ParseMode.Strict)


def test_parse_patch_bad_last_line() -> None:
    with pytest.raises(ParseError, match="End Patch"):
        parse_patch_text("*** Begin Patch\nbad", ParseMode.Strict)


def test_parse_patch_empty_update_hunk() -> None:
    with pytest.raises(ParseError, match="empty"):
        parse_patch(
            "*** Begin Patch\n"
            "*** Update File: test.py\n"
            "*** End Patch"
        )


def test_parse_patch_empty_hunks() -> None:
    p = parse_patch("*** Begin Patch\n*** End Patch")
    assert p.hunks == []


def test_parse_patch_all_hunk_types() -> None:
    p = parse_patch(
        "*** Begin Patch\n"
        "*** Add File: path/add.py\n"
        "+abc\n"
        "+def\n"
        "*** Delete File: path/delete.py\n"
        "*** Update File: path/update.py\n"
        "*** Move to: path/update2.py\n"
        "@@ def f():\n"
        "-    pass\n"
        "+    return 123\n"
        "*** End Patch"
    )
    assert len(p.hunks) == 3
    assert isinstance(p.hunks[0], AddFile)
    assert p.hunks[0].contents == "abc\ndef\n"
    assert isinstance(p.hunks[1], DeleteFile)
    assert isinstance(p.hunks[2], UpdateFile)
    assert p.hunks[2].move_path == "path/update2.py"
    assert p.hunks[2].chunks[0].change_context == "def f():"
    assert p.hunks[2].chunks[0].old_lines == ["    pass"]
    assert p.hunks[2].chunks[0].new_lines == ["    return 123"]


def test_seek_sequence_exact_and_trim() -> None:
    lines = ["  foo  ", "bar"]
    assert seek_sequence(lines, ["foo"], 0, False) == 0  # trim pass
    assert seek_sequence(lines, ["bar"], 0, False) == 1


def test_derive_new_contents_update() -> None:
    from codedoggy.tools.codex.apply_patch.parser import UpdateFileChunk

    original = "foo\nbar\n"
    chunks = [
        UpdateFileChunk(
            change_context=None,
            old_lines=["bar"],
            new_lines=["baz"],
        )
    ]
    out = derive_new_contents(original, "t.txt", chunks)
    assert out == "foo\nbaz\n"


def test_tool_apply_all_hunk_types(tmp_path: Path) -> None:
    (tmp_path / "path").mkdir()
    (tmp_path / "path" / "delete.py").write_text("gone\n", encoding="utf-8")
    (tmp_path / "path" / "update.py").write_text(
        "def f():\n    pass\n", encoding="utf-8"
    )
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    patch = (
        "*** Begin Patch\n"
        "*** Add File: path/add.py\n"
        "+abc\n"
        "+def\n"
        "*** Delete File: path/delete.py\n"
        "*** Update File: path/update.py\n"
        "*** Move to: path/update2.py\n"
        "@@ def f():\n"
        "-    pass\n"
        "+    return 123\n"
        "*** End Patch"
    )
    out = tools.call("apply_patch", {"patch": patch}, ctx)
    assert "Success" in out
    assert (tmp_path / "path" / "add.py").read_text(encoding="utf-8") == "abc\ndef\n"
    assert not (tmp_path / "path" / "delete.py").exists()
    assert "return 123" in (tmp_path / "path" / "update2.py").read_text(encoding="utf-8")
