"""Tool registry and dispatch tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.tools import (
    ToolCallContext,
    ToolConfig,
    ToolError,
    ToolRegistryBuilder,
    ToolServerConfig,
    register_tool_pack,
)
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.registry import FinalizeError, _reset_packs_for_tests
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolDescription,
    ToolId,
)


@pytest.fixture(autouse=True)
def _clean_packs() -> None:
    _reset_packs_for_tests()
    yield
    _reset_packs_for_tests()


def test_new_registers_file_tools() -> None:
    b = ToolRegistryBuilder.new()
    assert b.has_tool_id("Doggy:read_file")
    assert b.has_tool_id("Doggy:list_dir")
    assert b.has_tool_id("Doggy:search_replace")
    assert b.has_tool_id("Doggy:grep")
    assert b.has_tool_id("Doggy:run_terminal_cmd")
    assert b.has_tool_id("Doggy:memory")
    kinds = b.known_tool_kinds()
    assert kinds["Doggy:read_file"] is ToolKind.Read


def test_finalize_tool_definitions_and_call(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
    set_ = ToolRegistryBuilder.new().finalize()
    names = set_.client_names()
    assert "read_file" in names
    assert "list_dir" in names

    defs = set_.tool_definitions()
    assert any(d.name == "read_file" and d.parameters.get("properties") for d in defs)

    ctx = ToolCallContext(cwd=tmp_path)
    out = set_.call("read_file", {"target_file": "a.txt"}, ctx)
    # Line prefix on first visible line and every 10th line number only.
    assert out.startswith("1→hello")
    assert "world" in out
    assert "2→" not in out


def test_search_replace(tmp_path: Path) -> None:
    p = tmp_path / "b.txt"
    p.write_text("foo bar foo", encoding="utf-8")
    set_ = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    set_.call(
        "search_replace",
        {"file_path": "b.txt", "old_string": "bar", "new_string": "baz"},
        ctx,
    )
    assert p.read_text(encoding="utf-8") == "foo baz foo"


def test_config_enable_subset() -> None:
    b = ToolRegistryBuilder.new()
    cfg = ToolServerConfig(
        tools=[ToolConfig(id="Doggy:read_file", kind=ToolKind.Read)]
    )
    set_ = b.finalize(cfg)
    assert set_.client_names() == ["read_file"]


def test_name_override() -> None:
    b = ToolRegistryBuilder.new()
    cfg = ToolServerConfig(
        tools=[ToolConfig(id="Doggy:read_file").with_name("Read")]
    )
    set_ = b.finalize(cfg)
    assert set_.client_names() == ["Read"]


def test_duplicate_client_name_rejected() -> None:
    b = ToolRegistryBuilder.new()
    cfg = ToolServerConfig(
        tools=[
            ToolConfig(id="Doggy:read_file", name_override="x"),
            ToolConfig(id="Doggy:list_dir", name_override="x"),
        ]
    )
    with pytest.raises(FinalizeError, match="duplicate"):
        b.finalize(cfg)


def test_unknown_tool_call() -> None:
    set_ = ToolRegistryBuilder.new().finalize()
    with pytest.raises(ToolError) as ei:
        set_.call("nope", {}, ToolCallContext(cwd=Path.cwd()))
    assert ei.value.code == "not_found"


def test_register_tool_pack() -> None:
    class Extra(Tool):
        def id(self) -> ToolId:
            return ToolId("extra")

        def tool_namespace(self) -> ToolNamespace:
            return ToolNamespace.Doggy

        def kind(self) -> ToolKind:
            return ToolKind.Other

        def description(self, ctx: ListToolsContext | None = None) -> ToolDescription:
            return ToolDescription("extra", "extra tool")

        def parameters_schema(self) -> dict:
            return {"type": "object", "properties": {}}

        def run(self, ctx: ToolCallContext, args: dict) -> str:
            return "ok"

    def pack(b: ToolRegistryBuilder) -> None:
        b.register(Extra())

    _reset_packs_for_tests()
    register_tool_pack(pack)
    b = ToolRegistryBuilder.new()
    assert b.has_tool_id("Doggy:extra")
