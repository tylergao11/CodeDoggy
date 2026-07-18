"""Attack-style P1: use_tool must prepare (schema/policy) before host mcp_dispatch.

Before fix: bad args and path escapes went straight to host; Shadow saw no MCP mutations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.policy import WorkspacePolicy
from codedoggy.tools.runtime import ToolCallContext, ToolError


def _tools():
    return ToolRegistryBuilder.new().finalize()


def test_attack_missing_catalog_still_dispatches(tmp_path: Path) -> None:
    """No catalog / no schema → still call host dispatch (glue, not full MCP)."""
    tools = _tools()
    calls: list[tuple[str, dict]] = []

    def dispatch(name: str, args: dict) -> str:
        calls.append((name, dict(args)))
        return f"dispatched:{name}"

    ctx = ToolCallContext(cwd=tmp_path, extra={"mcp_dispatch": dispatch})
    out = tools.call(
        "use_tool",
        {"tool_name": "github__create_issue", "tool_input": {"title": "x"}},
        ctx,
    )
    assert "dispatched:github__create_issue" in out
    assert calls == [("github__create_issue", {"title": "x"})]


def test_attack_invalid_schema_args_rejected_when_catalog_has_schema(
    tmp_path: Path,
) -> None:
    """Catalog with schema: type-wrong / missing required must NOT reach dispatch."""
    tools = _tools()
    calls: list[tuple[str, dict]] = []

    def dispatch(name: str, args: dict) -> str:
        calls.append((name, dict(args)))
        return "should-not-run"

    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "priority": {"type": "integer"},
        },
        "required": ["title", "priority"],
    }
    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={
            "mcp_dispatch": dispatch,
            "mcp_tools": [
                {
                    "name": "linear__save_issue",
                    "description": "save",
                    "parameters": schema,
                }
            ],
        },
    )

    # Wrong type for priority
    with pytest.raises(ToolError) as ei:
        tools.call(
            "use_tool",
            {
                "tool_name": "linear__save_issue",
                "tool_input": {"title": "t", "priority": "high"},
            },
            ctx,
        )
    assert ei.value.code == "invalid_arguments"
    assert "priority" in ei.value.message or "schema" in ei.value.message.lower()
    assert calls == []

    # Missing required
    with pytest.raises(ToolError) as ei2:
        tools.call(
            "use_tool",
            {"tool_name": "linear__save_issue", "tool_input": {"title": "only"}},
            ctx,
        )
    assert ei2.value.code == "invalid_arguments"
    assert calls == []


def test_attack_path_escape_in_tool_input_denied_when_policy_present(
    tmp_path: Path,
) -> None:
    """Path-like keys outside workspace must be denied before dispatch."""
    tools = _tools()
    calls: list[tuple[str, dict]] = []

    def dispatch(name: str, args: dict) -> str:
        calls.append((name, dict(args)))
        return "escaped"

    policy = WorkspacePolicy(cwd=tmp_path)
    outside = str((tmp_path / ".." / "secrets" / "id_rsa").resolve())
    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={
            "mcp_dispatch": dispatch,
            "policy": policy,
            # no catalog — policy alone must still block path escape
        },
    )
    with pytest.raises(ToolError) as ei:
        tools.call(
            "use_tool",
            {
                "tool_name": "fs__write_file",
                "tool_input": {"path": outside, "content": "pwn"},
            },
            ctx,
        )
    assert ei.value.code in {"path_escape", "policy_denied"}
    assert "escape" in ei.value.message.lower() or "denied" in ei.value.message.lower()
    assert calls == []


def test_attack_deny_write_protected_path_in_tool_input(tmp_path: Path) -> None:
    tools = _tools()
    calls: list = []

    def dispatch(name: str, args: dict) -> str:
        calls.append(name)
        return "nope"

    policy = WorkspacePolicy(cwd=tmp_path)
    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={"mcp_dispatch": dispatch, "policy": policy},
    )
    with pytest.raises(ToolError) as ei:
        tools.call(
            "use_tool",
            {
                "tool_name": "fs__write_file",
                "tool_input": {"file_path": ".env", "content": "SECRET=1"},
            },
            ctx,
        )
    assert ei.value.code in {"deny_path", "policy_denied"}
    assert calls == []


def test_valid_schema_and_in_workspace_path_dispatches(tmp_path: Path) -> None:
    tools = _tools()
    calls: list[tuple[str, dict]] = []

    def dispatch(name: str, args: dict) -> str:
        calls.append((name, dict(args)))
        return "ok"

    target = tmp_path / "notes.txt"
    target.write_text("hi", encoding="utf-8")
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }
    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={
            "mcp_dispatch": dispatch,
            "policy": WorkspacePolicy(cwd=tmp_path),
            "mcp_tools": [
                {"name": "fs__write_file", "parameters": schema},
            ],
        },
    )
    out = tools.call(
        "use_tool",
        {
            "tool_name": "fs__write_file",
            "tool_input": {"path": "notes.txt", "content": "bye"},
        },
        ctx,
    )
    assert out == "ok"
    assert calls and calls[0][0] == "fs__write_file"


def test_host_structured_mutations_recorded(tmp_path: Path) -> None:
    """Host can report mutations via structured return → ctx.set_mutation."""
    tools = _tools()

    def dispatch(name: str, args: dict) -> dict:
        return {
            "text": "wrote file",
            "mutations": [
                {
                    "path": "out.txt",
                    "before": None,
                    "after": "data",
                    "is_create": True,
                }
            ],
        }

    ctx = ToolCallContext(cwd=tmp_path, extra={"mcp_dispatch": dispatch})
    out = tools.call(
        "use_tool",
        {"tool_name": "fs__write_file", "tool_input": {"path": "out.txt"}},
        ctx,
    )
    assert out == "wrote file"
    mut = (ctx.extra or {}).get("mutation")
    assert mut is not None
    assert mut.path == "out.txt"
    assert mut.is_create is True
    assert mut.after == "data"


def test_mcp_tool_index_schema_validation(tmp_path: Path) -> None:
    tools = _tools()
    calls: list = []

    def dispatch(name: str, args: dict) -> str:
        calls.append(name)
        return "x"

    class Index:
        def get_schema(self, name: str):
            if name == "svc__echo":
                return {
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                    "required": ["msg"],
                }
            return None

    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={"mcp_dispatch": dispatch, "mcp_tool_index": Index()},
    )
    with pytest.raises(ToolError) as ei:
        tools.call(
            "use_tool",
            {"tool_name": "svc__echo", "tool_input": {"msg": 123}},
            ctx,
        )
    assert ei.value.code == "invalid_arguments"
    assert calls == []
