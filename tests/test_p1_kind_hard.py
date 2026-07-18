"""Attack-style regressions: config kind must not downgrade mutating tools.

P1 residual — registration kind is authoritative for Write/Edit/Delete/Move/Execute
and hard-named write/execute tools (write, memory, scheduler_*, shell).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from codedoggy.orchestration.agent_def import filter_toolset
from codedoggy.orchestration.types import CapabilityMode
from codedoggy.tools import ToolConfig, ToolRegistryBuilder, ToolServerConfig
from codedoggy.tools.gate import effective_kind, enforce_policy
from codedoggy.tools.kinds import (
    HARD_EXECUTE_TOOL_NAMES,
    HARD_WRITE_TOOL_NAMES,
    ToolKind,
    resolve_authoritative_kind,
)
from codedoggy.tools.policy import PolicyDecision, WorkspacePolicy
from codedoggy.tools.runtime import ToolCallContext, ToolError


def test_resolve_authoritative_kind_blocks_search_downgrade() -> None:
    assert (
        resolve_authoritative_kind(
            short_id="write",
            registered_kind=ToolKind.Write,
            config_kind=ToolKind.Search,
        )
        is ToolKind.Write
    )
    assert (
        resolve_authoritative_kind(
            short_id="memory",
            registered_kind=ToolKind.Edit,
            config_kind=ToolKind.Search,
        )
        is ToolKind.Edit
    )
    assert (
        resolve_authoritative_kind(
            short_id="run_terminal_cmd",
            registered_kind=ToolKind.Execute,
            config_kind=ToolKind.Search,
        )
        is ToolKind.Execute
    )
    # Hard name even when registered kind is Other (scheduler)
    assert (
        resolve_authoritative_kind(
            short_id="scheduler_create",
            registered_kind=ToolKind.Other,
            config_kind=ToolKind.Search,
        )
        is ToolKind.Other
    )
    # Non-mutating tools may take config kind
    assert (
        resolve_authoritative_kind(
            short_id="read_file",
            registered_kind=ToolKind.Read,
            config_kind=ToolKind.Search,
        )
        is ToolKind.Search
    )


def test_effective_kind_write_names_ignore_config_search() -> None:
    for name in ("write", "memory", "search_replace", "scheduler_create", "scheduler_delete"):
        assert name in HARD_WRITE_TOOL_NAMES
        ek = effective_kind(
            tool_name=name,
            registered_kind=ToolKind.Write if name != "memory" else ToolKind.Edit,
            config_kind=ToolKind.Search,
        )
        assert ek is not ToolKind.Search, f"{name} must not be masked as Search"
        assert ek in {ToolKind.Write, ToolKind.Edit}


def test_effective_kind_execute_names_ignore_config_search() -> None:
    for name in HARD_EXECUTE_TOOL_NAMES:
        ek = effective_kind(
            tool_name=name,
            registered_kind=ToolKind.Execute,
            config_kind=ToolKind.Search,
        )
        assert ek is ToolKind.Execute, f"{name} must stay Execute"


def test_finalize_write_config_search_still_write_kind() -> None:
    """ToolConfig(kind=Search) on write builtin must not downgrade finalized kind."""
    b = ToolRegistryBuilder.new()
    cfg = ToolServerConfig(
        tools=[
            ToolConfig(id="Doggy:write", kind=ToolKind.Search),
            ToolConfig(id="Doggy:search_replace", kind=ToolKind.Search),
            ToolConfig(id="Doggy:read_file", kind=ToolKind.Read),
        ]
    )
    tools = b.finalize(cfg)
    assert tools.kind_of("write") is ToolKind.Write
    assert tools.kind_of("search_replace") is ToolKind.Edit
    assert tools.kind_of("read_file") is ToolKind.Read


def test_finalize_memory_config_search_remains_edit() -> None:
    b = ToolRegistryBuilder.new()
    cfg = ToolServerConfig(
        tools=[ToolConfig(id="Doggy:memory", kind=ToolKind.Search)]
    )
    tools = b.finalize(cfg)
    ft = tools.by_client_name.get("memory")
    assert ft is not None
    assert ft.kind is ToolKind.Edit


def test_filter_toolset_read_only_excludes_memory() -> None:
    parent = ToolRegistryBuilder.new().finalize()
    child = filter_toolset(parent, capability=CapabilityMode.READ_ONLY)
    names = set(child.client_names())
    assert "read_file" in names
    assert "memory" not in names
    assert "search_replace" not in names
    assert "write" not in names


def test_filter_toolset_read_only_excludes_memory_even_after_search_override() -> None:
    """Attack: config tries to mark memory as Search so it slips into READ_ONLY."""
    b = ToolRegistryBuilder.new()
    cfg = ToolServerConfig(
        tools=[
            ToolConfig(id="Doggy:memory", kind=ToolKind.Search),
            ToolConfig(id="Doggy:read_file", kind=ToolKind.Read),
            ToolConfig(id="Doggy:write", kind=ToolKind.Search),
        ]
    )
    parent = b.finalize(cfg)
    assert parent.kind_of("memory") is ToolKind.Edit
    child = filter_toolset(parent, capability=CapabilityMode.READ_ONLY)
    names = set(child.client_names())
    assert "read_file" in names
    assert "memory" not in names
    assert "write" not in names


def test_write_with_search_config_still_hits_write_policy(tmp_path: Path) -> None:
    """Downgraded config kind must still enforce check_write (not check_read)."""
    b = ToolRegistryBuilder.new()
    cfg = ToolServerConfig(
        tools=[ToolConfig(id="Doggy:write", kind=ToolKind.Search)]
    )
    tools = b.finalize(cfg)
    assert tools.kind_of("write") is ToolKind.Write

    writes: list[str] = []
    reads: list[str] = []

    class SpyPolicy:
        def check_write(self, path: str) -> PolicyDecision:
            writes.append(path)
            return PolicyDecision(allowed=False, reason=f"write denied for {path}", code="policy_denied")

        def check_read(self, path: str) -> PolicyDecision:
            reads.append(path)
            return PolicyDecision(allowed=True)

        def check_shell(self, cmd: str) -> PolicyDecision:
            return PolicyDecision(allowed=True)

    ctx = ToolCallContext(cwd=tmp_path, extra={"policy": SpyPolicy()})
    with pytest.raises(ToolError) as ei:
        tools.call(
            "write",
            {"file_path": "secret.txt", "content": "pwned"},
            ctx,
        )
    assert ei.value.code == "policy_denied"
    assert writes == ["secret.txt"]
    assert reads == [], "write must not take the read policy path"


def test_effective_kind_direct_policy_path_for_masked_write(tmp_path: Path) -> None:
    """Even if caller passes kind=Search, registered Write still hits write policy."""
    class SpyPolicy:
        def __init__(self) -> None:
            self.writes: list[str] = []
            self.reads: list[str] = []

        def check_write(self, path: str) -> Any:
            self.writes.append(path)
            return PolicyDecision(allowed=False, reason="nope", code="policy_denied")

        def check_read(self, path: str) -> Any:
            self.reads.append(path)
            return PolicyDecision(allowed=True)

    pol = SpyPolicy()
    ctx = ToolCallContext(cwd=tmp_path, extra={"policy": pol})
    with pytest.raises(ToolError):
        enforce_policy(
            tool_name="write",
            kind=ToolKind.Search,  # attacker-supplied config kind
            args={"file_path": "x.txt"},
            ctx=ctx,
            registered_kind=ToolKind.Write,
        )
    assert pol.writes == ["x.txt"]
    assert pol.reads == []


def test_workspace_policy_write_still_works_after_finalize_attack(tmp_path: Path) -> None:
    """End-to-end: WorkspacePolicy + Search-kind config on write still denies .env."""
    b = ToolRegistryBuilder.new()
    cfg = ToolServerConfig(
        tools=[ToolConfig(id="Doggy:write", kind=ToolKind.Search)]
    )
    tools = b.finalize(cfg)
    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    policy = WorkspacePolicy(cwd=tmp_path)
    ctx = ToolCallContext(cwd=tmp_path, extra={"policy": policy})
    with pytest.raises(ToolError) as ei:
        tools.call(
            "write",
            {"file_path": ".env", "content": "pwned"},
            ctx,
        )
    assert ei.value.code in {"policy_denied", "deny_path"}
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "SECRET=1\n"
