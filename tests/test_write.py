"""OpenCode write tool — Grok source-aligned behavior.

Mirrors crates/codegen/xai-grok-tools/src/implementations/opencode/write/mod.rs tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.builtins.write import WriteTool
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import ToolCallContext, ToolError


@pytest.fixture
def tools(tmp_path: Path):
    return ToolRegistryBuilder.new().finalize(), ToolCallContext(cwd=tmp_path), tmp_path


def test_tool_metadata() -> None:
    tool = WriteTool()
    assert tool.id() == "write"
    assert tool.kind() is ToolKind.Write
    assert tool.tool_namespace() is ToolNamespace.Doggy
    desc = tool.description(None).description
    assert "Create or overwrite a file" in desc
    assert "read_file" in desc
    assert "Parent directories are created" in desc


def test_write_new_file_creates_with_correct_content(tools) -> None:
    set_, ctx, tmp = tools
    out = set_.call(
        "write",
        {"file_path": "new.txt", "content": "hello\nworld\n"},
        ctx,
    )
    assert "created" in out
    assert "successfully" not in out or "created" in out
    # Grok: "The file {} has been created."
    assert out.startswith("The file ") and out.endswith(" has been created.")
    assert (tmp / "new.txt").read_text(encoding="utf-8") == "hello\nworld\n"


def test_overwrite_existing_file(tools) -> None:
    set_, ctx, tmp = tools
    (tmp / "existing.txt").write_text("old content\n", encoding="utf-8")
    out = set_.call(
        "write",
        {"file_path": "existing.txt", "content": "new content\n"},
        ctx,
    )
    # Grok: "Wrote file successfully to {}."
    assert out.startswith("Wrote file successfully to ")
    assert out.endswith(".")
    assert (tmp / "existing.txt").read_text(encoding="utf-8") == "new content\n"


def test_creates_parent_directories(tools) -> None:
    set_, ctx, tmp = tools
    out = set_.call(
        "write",
        {"file_path": "a/b/c/file.txt", "content": "nested\n"},
        ctx,
    )
    assert "created" in out
    assert (tmp / "a" / "b" / "c" / "file.txt").read_text(encoding="utf-8") == "nested\n"


def test_empty_content_write(tools) -> None:
    set_, ctx, tmp = tools
    out = set_.call("write", {"file_path": "empty.txt", "content": ""}, ctx)
    assert "created" in out
    assert (tmp / "empty.txt").exists()
    assert (tmp / "empty.txt").read_text(encoding="utf-8") == ""


def test_overwrite_preserves_path_in_output(tools) -> None:
    set_, ctx, tmp = tools
    (tmp / "output_check.txt").write_text("old\n", encoding="utf-8")
    resolved = str((tmp / "output_check.txt").resolve())
    out = set_.call(
        "write",
        {"file_path": "output_check.txt", "content": "new\n"},
        ctx,
    )
    assert resolved in out
    assert out == f"Wrote file successfully to {resolved}."


def test_relative_path_resolution(tools) -> None:
    set_, ctx, tmp = tools
    out = set_.call(
        "write",
        {"file_path": "subdir/relative.txt", "content": "resolved\n"},
        ctx,
    )
    expected = (tmp / "subdir" / "relative.txt").resolve()
    assert expected.read_text(encoding="utf-8") == "resolved\n"
    assert str(expected) in out


def test_absolute_path_write(tools) -> None:
    set_, ctx, tmp = tools
    target = (tmp / "abs.txt").resolve()
    out = set_.call(
        "write",
        {"file_path": str(target), "content": "via-abs\n"},
        ctx,
    )
    assert target.read_text(encoding="utf-8") == "via-abs\n"
    assert "created" in out
    assert str(target) in out


def test_mutation_create_and_overwrite(tools) -> None:
    set_, ctx, tmp = tools
    set_.call("write", {"file_path": "m.txt", "content": "v1\n"}, ctx)
    mut = ctx.extra.get("mutation")
    assert mut is not None
    assert mut.is_create is True
    assert mut.before is None
    assert mut.after == "v1\n"
    assert mut.tool_name == "write"

    set_.call("write", {"file_path": "m.txt", "content": "v2\n"}, ctx)
    mut2 = ctx.extra.get("mutation")
    assert mut2 is not None
    assert mut2.is_create is False
    assert mut2.before == "v1\n"
    assert mut2.after == "v2\n"


def test_missing_args(tools) -> None:
    set_, ctx, _tmp = tools
    with pytest.raises(ToolError) as ei:
        set_.call("write", {"content": "x"}, ctx)
    assert ei.value.code == "invalid_arguments"

    with pytest.raises(ToolError) as ei2:
        set_.call("write", {"file_path": "a.txt"}, ctx)
    assert ei2.value.code == "invalid_arguments"


def test_filename_too_long(tools) -> None:
    set_, ctx, _tmp = tools
    long_name = "a" * 300 + ".txt"
    with pytest.raises(ToolError) as ei:
        set_.call("write", {"file_path": long_name, "content": "x"}, ctx)
    assert ei.value.code == "filename_too_long"
