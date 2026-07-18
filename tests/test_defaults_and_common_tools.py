"""Defaults + grep / terminal cmd."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.builtins.grep import resolve_effective_head_limit, OutputMode
from codedoggy.tools.defaults import (
    BASH_DEFAULT_MAX_TIMEOUT_MS,
    BASH_DEFAULT_TIMEOUT_MS,
    DEFAULT_TOOL_OUTPUT_BYTES,
    DEFAULT_TOOL_OUTPUT_CHARS,
    GREP_CONTENT_LINE_DEFAULT,
    GREP_CONTENT_LINE_LIMIT,
    MAX_LINES_READ_DEFAULT,
)
from codedoggy.tools.runtime import ToolCallContext, ToolError


def test_hardcoded_defaults() -> None:
    assert DEFAULT_TOOL_OUTPUT_BYTES == 40_000
    assert DEFAULT_TOOL_OUTPUT_CHARS == 20_000
    assert MAX_LINES_READ_DEFAULT == 1_000
    assert BASH_DEFAULT_TIMEOUT_MS == 120_000
    assert BASH_DEFAULT_MAX_TIMEOUT_MS == 300_000
    assert GREP_CONTENT_LINE_DEFAULT == 200
    assert GREP_CONTENT_LINE_LIMIT == 2_000


def test_grep_head_limit_defaults() -> None:
    assert resolve_effective_head_limit(None, OutputMode.Content) == 200
    assert resolve_effective_head_limit(5000, OutputMode.Content) == 2_000
    assert resolve_effective_head_limit(None, OutputMode.FilesWithMatches) == 500


def test_grep_finds_content(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("x = 2\n", encoding="utf-8")
    set_ = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    out = set_.call("grep", {"pattern": "hello", "path": str(tmp_path)}, ctx)
    assert "hello" in out
    assert "workspace_result" in out


def test_run_terminal_cmd_echo(tmp_path: Path) -> None:
    set_ = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    out = set_.call(
        "run_terminal_cmd",
        {
            "command": 'python -c "print(12345)"',
            "description": "print a marker for the test",
        },
        ctx,
    )
    assert out.startswith("exit: 0")
    assert "12345" in out


def test_run_terminal_cmd_rejects_trailing_ampersand(tmp_path: Path) -> None:
    set_ = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    with pytest.raises(ToolError, match="must not end with"):
        set_.call(
            "run_terminal_cmd",
            {
                "command": 'python -c "print(1)" &',
                "description": "should reject background op",
            },
            ctx,
        )


def test_run_terminal_cmd_requires_description(tmp_path: Path) -> None:
    set_ = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    with pytest.raises(ToolError, match="description"):
        set_.call("run_terminal_cmd", {"command": 'python -c "print(1)"'}, ctx)


def test_run_terminal_cmd_timeout_zero_is_default(tmp_path: Path) -> None:
    from codedoggy.tools.builtins.run_terminal_cmd import resolve_fg_timeout_ms
    from codedoggy.tools.defaults import BASH_DEFAULT_TIMEOUT_MS

    assert resolve_fg_timeout_ms(0) == BASH_DEFAULT_TIMEOUT_MS
    assert resolve_fg_timeout_ms(None) == BASH_DEFAULT_TIMEOUT_MS


def test_run_terminal_cmd_timeout(tmp_path: Path) -> None:
    set_ = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    out = set_.call(
        "run_terminal_cmd",
        {
            "command": 'python -c "import time; time.sleep(5)"',
            "timeout": 200,
            "description": "force a short timeout",
        },
        ctx,
    )
    assert "exit: killed (timeout)" in out


def test_grep_workspace_wrapper(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    set_ = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    out = set_.call("grep", {"pattern": "hello", "path": str(tmp_path)}, ctx)
    assert "<workspace_result" in out
    assert "Found " in out and "matching lines" in out
    assert "</workspace_result>" in out


def test_builtins_registered() -> None:
    b = ToolRegistryBuilder.new()
    for qid in (
        "Doggy:read_file",
        "Doggy:search_replace",
        "Doggy:list_dir",
        "Doggy:grep",
        "Doggy:run_terminal_cmd",
        "Doggy:memory",
        "Doggy:session_search",
    ):
        assert b.has_tool_id(qid), qid


def test_grep_rejects_context_without_rg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without rg, context flags must fail closed — not silently ignored."""
    import codedoggy.tools.builtins.grep as grep_mod

    monkeypatch.setattr(grep_mod.shutil, "which", lambda _name: None)
    (tmp_path / "a.py").write_text("hello\n", encoding="utf-8")
    set_ = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    with pytest.raises(ToolError) as ei:
        set_.call(
            "grep",
            {"pattern": "hello", "path": str(tmp_path), "-A": 2},
            ctx,
        )
    assert ei.value.code == "unsupported_without_rg"


def test_grep_rg_exit2_is_error_not_no_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rg exit 2 must not look like a clean miss."""
    import codedoggy.tools.builtins.grep as grep_mod
    from types import SimpleNamespace

    def fake_run(*_a, **_k):
        return SimpleNamespace(
            returncode=2,
            stdout=b"",
            stderr=b"regex parse error: unclosed group",
        )

    monkeypatch.setattr(grep_mod.shutil, "which", lambda name: "rg" if "rg" in name else None)
    monkeypatch.setattr(grep_mod.subprocess, "run", fake_run)
    set_ = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    with pytest.raises(ToolError) as ei:
        set_.call("grep", {"pattern": "(oops", "path": str(tmp_path)}, ctx)
    assert ei.value.code == "rg_error"
    assert "No matches found" not in str(ei.value)


def test_grep_invalid_context_arg(tmp_path: Path) -> None:
    set_ = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    with pytest.raises(ToolError) as ei:
        set_.call("grep", {"pattern": "x", "path": str(tmp_path), "-A": "nope"}, ctx)
    assert ei.value.code == "invalid_arguments"


def test_grep_python_fallback_basic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import codedoggy.tools.builtins.grep as grep_mod

    monkeypatch.setattr(grep_mod.shutil, "which", lambda _name: None)
    (tmp_path / "a.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    set_ = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    out = set_.call("grep", {"pattern": "hello", "path": str(tmp_path)}, ctx)
    assert "hello" in out
    assert "workspace_result" in out


def test_run_terminal_cmd_description_mentions_tree_kill(tmp_path: Path) -> None:
    set_ = ToolRegistryBuilder.new().finalize()
    defs = {d.name: d for d in set_.tool_definitions()}
    desc = defs["run_terminal_cmd"].description or ""
    # Must not claim unimplemented Job Object; must state actual kill mechanism.
    assert "Job Object" not in desc
    assert "taskkill" in desc or "process group" in desc or "SIGTERM" in desc
