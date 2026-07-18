"""Attack-style P1: memory provider tools must be model-visible and callable.

Regression for: external providers load after ToolRegistryBuilder.finalize(),
so notes_append etc. never appear in the toolset (half-wired).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codedoggy.memory.manager import MemoryManager
from codedoggy.memory.plugins.notes import NotesMemoryProvider
from codedoggy.memory.tool_injection import (
    MemoryProviderDispatchTool,
    inject_memory_provider_tools,
)
from codedoggy.tools.registry import FinalizedToolset, ToolRegistryBuilder
from codedoggy.tools.runtime import ToolCallContext


def test_p1_finalize_without_inject_hides_notes_append() -> None:
    """Would-fail-before-fix attack: product toolset lacks provider tools."""
    toolset = ToolRegistryBuilder.new().finalize()
    names = set(toolset.client_names())
    assert "notes_append" not in names
    # Manager knows the tool, but toolset does not — the half-wired failure mode
    mm = MemoryManager()
    assert mm.add_provider(NotesMemoryProvider()) is True
    assert mm.has_tool("notes_append")
    assert "notes_append" not in set(toolset.client_names())


def test_p1_inject_makes_notes_append_visible_and_callable(tmp_path: Path) -> None:
    notes_path = tmp_path / "notes.md"
    mm = MemoryManager()
    provider = NotesMemoryProvider(path=notes_path)
    provider.initialize(session_id="s1")
    assert mm.add_provider(provider) is True

    toolset = ToolRegistryBuilder.new().finalize()
    n = inject_memory_provider_tools(toolset, mm)
    assert n >= 1
    assert "notes_append" in toolset.client_names()
    defs = {d.name: d for d in toolset.tool_definitions()}
    assert "notes_append" in defs
    props = (defs["notes_append"].parameters or {}).get("properties") or {}
    assert "content" in props

    ctx = ToolCallContext(
        cwd=tmp_path, extra={"memory_manager": mm}
    )
    out = toolset.call(
        "notes_append", {"content": "DECIDE: use httponly cookies"}, ctx
    )
    data = json.loads(out)
    assert data.get("success") is True
    assert "httponly" in notes_path.read_text(encoding="utf-8")


def test_p1_inject_idempotent_no_duplicate() -> None:
    mm = MemoryManager()
    mm.add_provider(NotesMemoryProvider())
    toolset = ToolRegistryBuilder.new().finalize()
    assert inject_memory_provider_tools(toolset, mm) >= 1
    assert inject_memory_provider_tools(toolset, mm) == 0
    # single primary listing
    listed = [d.name for d in toolset.tool_definitions() if d.name == "notes_append"]
    assert listed == ["notes_append"]


def test_p1_bootstrap_wires_notes_when_env_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: CODEDOGGY_MEMORY_PROVIDER=notes → toolset + call path."""
    from codedoggy.bootstrap import build_session
    from codedoggy.model import CompletionResult, ModelConfig

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CODEDOGGY_HOME", str(home))
    monkeypatch.setenv("CODEDOGGY_MEMORY_PROVIDER", "notes")

    class _Client:
        def __init__(self) -> None:
            self.config = ModelConfig(
                provider="fake", model="t", base_url="http://x", api_key="x"
            )

        def complete(self, messages, **kwargs):
            return CompletionResult(content="ok", model="t")

    s = build_session(
        tmp_path,
        main_client=_Client(),
        audit_client=_Client(),
        enable_audit=False,
        enable_graph=False,
        enable_policy=False,
        memory_dir=home / "memories",
        session_db=tmp_path / "sess.db",
    )
    try:
        tools = s.extensions.tools
        assert tools is not None
        assert "notes_append" in tools.client_names()
        mm = s.extensions.memory_manager
        assert mm is not None
        assert mm.has_tool("notes_append")

        ctx = ToolCallContext(
            cwd=tmp_path,
            session_id=str(s.id),
            extra=dict(getattr(s.extensions.kernel, "tool_extra", None) or {
                "memory_manager": mm,
            }),
        )
        out = tools.call("notes_append", {"content": "P1: provider tools live"}, ctx)
        data = json.loads(out)
        assert data.get("success") is True
        notes_file = home / "memories" / "notes.md"
        assert notes_file.is_file()
        assert "provider tools live" in notes_file.read_text(encoding="utf-8")
    finally:
        s.close()


def test_p1_dispatch_tool_routes_through_handle_tool_call(tmp_path: Path) -> None:
    """No second protocol — only MemoryManager.handle_tool_call."""
    mm = MemoryManager()
    provider = NotesMemoryProvider(path=tmp_path / "n.md")
    provider.initialize()
    mm.add_provider(provider)
    tool = MemoryProviderDispatchTool(
        "notes_append",
        description="append",
        parameters={
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
        memory_manager=mm,
    )
    ctx = ToolCallContext(cwd=tmp_path)
    out = tool.run(ctx, {"content": "via-dispatch"})
    assert json.loads(out).get("success") is True
    assert "via-dispatch" in (tmp_path / "n.md").read_text(encoding="utf-8")


def test_p1_unknown_provider_tool_not_found_on_empty_manager(tmp_path: Path) -> None:
    toolset = ToolRegistryBuilder.new().finalize()
    mm = MemoryManager()  # no external tools
    assert inject_memory_provider_tools(toolset, mm) == 0
    assert "notes_append" not in toolset.client_names()
    with pytest.raises(Exception) as ei:
        toolset.call("notes_append", {"content": "x"}, ToolCallContext(cwd=tmp_path))
    assert "not found" in str(ei.value).lower() or getattr(
        ei.value, "code", ""
    ) == "not_found"
