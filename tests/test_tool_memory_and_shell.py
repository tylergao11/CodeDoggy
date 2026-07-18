"""Tool-layer only: memory_search/get + shell state + auto-bg."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.memory.store import MemoryStore
from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.runtime import ToolCallContext, ToolError
from codedoggy.tools.task_manager import BackgroundTaskManager
from codedoggy.tools.util.shell_state import ShellState, ensure_shell_state
from codedoggy.tools.builtins.memory_get import format_with_line_numbers


def _tools():
    return ToolRegistryBuilder.new().finalize()


class _FakeMemoryBackend:
    """Host-injected MemoryBackend stand-in (Grok pattern)."""

    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits

    def search(self, query: str, max_results=None, min_score=None):  # noqa: ANN001
        q = query.lower()
        out = [h for h in self._hits if q.split()[0] in h.get("snippet", "").lower()]
        if max_results is not None:
            out = out[: int(max_results)]
        return out


def test_memory_search_requires_backend(tmp_path: Path) -> None:
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    out = tools.call("memory_search", {"query": "anything"}, ctx)
    assert "not enabled" in out.lower()


def test_memory_search_and_get(tmp_path: Path) -> None:
    mem_dir = tmp_path / "mem"
    store = MemoryStore(memory_dir=mem_dir)
    store.load_from_disk()
    store.add("memory", "prefer ripgrep over shell grep for code search")
    store.add("user", "user likes concise answers")

    backend = _FakeMemoryBackend(
        [
            {
                "score": 0.9,
                "source": "workspace",
                "path": str(mem_dir / "MEMORY.md"),
                "start_line": 1,
                "end_line": 3,
                "snippet": "prefer ripgrep over shell grep for code search",
            }
        ]
    )

    tools = _tools()
    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={"memory_store": store, "memory_backend": backend},
    )
    out = tools.call("memory_search", {"query": "ripgrep code search"}, ctx)
    assert "Found " in out
    assert "### Result" in out
    assert "ripgrep" in out.lower()

    got = tools.call("memory_get", {"target": "memory"}, ctx)
    assert "**File:**" in got
    assert "→" in got
    assert "ripgrep" in got

    # Grok: from is 0-based; lines = max count
    got2 = tools.call(
        "memory_get",
        {"path": str(mem_dir / "USER.md"), "from": 0, "lines": 50},
        ctx,
    )
    assert "concise" in got2
    assert "**Lines:**" in got2


def test_memory_get_requires_store(tmp_path: Path) -> None:
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    # Grok: soft text when memory disabled (not ToolError)
    out = tools.call("memory_get", {"target": "memory"}, ctx)
    assert "not enabled" in out.lower()


def test_format_with_line_numbers_trailing() -> None:
    # Grok get_tool.rs: split on \n keeps trailing blank line numbered
    body = format_with_line_numbers("a\n", 1)
    assert body == "1→a\n2→"


def test_format_with_line_numbers_basic() -> None:
    out = format_with_line_numbers("alpha\nbeta\ngamma", 1)
    assert out == "1→alpha\n2→beta\n3→gamma"


def test_format_with_line_numbers_offset() -> None:
    # from=4 (0-based) → first_line_num=5
    out = format_with_line_numbers("line five\nline six", 5)
    assert out.startswith("5→line five")
    assert out.endswith("6→line six")


def test_format_with_line_numbers_empty() -> None:
    assert format_with_line_numbers("", 1) == ""


def test_format_with_line_numbers_single() -> None:
    assert format_with_line_numbers("only line", 1) == "1→only line"


def test_format_with_line_numbers_no_trailing_no_blank() -> None:
    assert format_with_line_numbers("alpha", 1) == "1→alpha"


def test_format_with_line_numbers_double_trailing() -> None:
    assert format_with_line_numbers("a\n\n", 1) == "1→a\n2→\n3→"


def test_memory_get_full_file_header_and_trailing_newline(tmp_path: Path) -> None:
    """Grok: lines=None returns raw content; total_lines uses lines().count()."""
    mem_dir = tmp_path / "mem"
    store = MemoryStore(memory_dir=mem_dir)
    store.load_from_disk()
    path = mem_dir / "MEMORY.md"
    path.write_text("lineA\nlineB\n", encoding="utf-8")

    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path, extra={"memory_store": store})
    out = tools.call("memory_get", {"path": str(path)}, ctx)

    assert "**File:**" in out
    assert "from: start" in out
    assert "limit: all" in out
    # Rust content.lines().count() on "lineA\nlineB\n" == 2
    assert "**Lines:** 2 (from: start, limit: all)" in out
    # format_with_line_numbers still emits trailing blank numbered line
    assert "1→lineA" in out
    assert "2→lineB" in out
    assert "3→" in out


def test_shell_state_persists_cwd(tmp_path: Path) -> None:
    tools = _tools()
    sub = tmp_path / "sub"
    sub.mkdir()
    extra: dict = {}
    ctx = ToolCallContext(cwd=tmp_path, extra=extra)
    import sys

    if sys.platform == "win32":
        cmd = f"Set-Location -LiteralPath '{sub}'"
    else:
        cmd = f"cd '{sub}'"
    tools.call(
        "run_terminal_cmd",
        {"command": cmd, "description": "change directory"},
        ctx,
    )
    st = ensure_shell_state(extra, tmp_path)
    assert isinstance(st, ShellState)
    assert st.cwd is not None
    if sys.platform == "win32":
        out = tools.call(
            "run_terminal_cmd",
            {"command": "(Get-Location).Path", "description": "print cwd"},
            ctx,
        )
    else:
        out = tools.call(
            "run_terminal_cmd",
            {"command": "pwd", "description": "print cwd"},
            ctx,
        )
    assert "exit: 0" in out or "exit:" in out


def test_auto_background_on_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import codedoggy.tools.builtins.run_terminal_cmd as rtc
    import codedoggy.tools.defaults as defaults
    import codedoggy.tools.grok_build.bash_params as bash_params

    monkeypatch.setattr(defaults, "BASH_AUTO_BACKGROUND_ON_TIMEOUT", True)
    monkeypatch.setattr(rtc, "BASH_AUTO_BACKGROUND_ON_TIMEOUT", True)
    monkeypatch.setattr(defaults, "BASH_DEFAULT_FOREGROUND_BLOCK_BUDGET_MS", 300)
    monkeypatch.setattr(bash_params, "DEFAULT_FOREGROUND_BLOCK_BUDGET_MS", 300)

    tools = _tools()
    tm = BackgroundTaskManager(work_dir=tmp_path / "tasks")
    ctx = ToolCallContext(cwd=tmp_path, extra={"task_manager": tm})
    out = tools.call(
        "run_terminal_cmd",
        {
            "command": 'python -c "import time; time.sleep(5); print(99)"',
            "description": "sleep then print for auto-bg",
        },
        ctx,
    )
    assert "<task-id>" in out or "background" in out.lower()
    if "<task-id>" in out:
        tid = out.split("<task-id>")[1].split("</task-id>")[0]
        tools.call("kill_task", {"task_id": tid}, ctx)
