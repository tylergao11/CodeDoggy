"""Subagent policy: isolation defaults, depth, discovery, cancel drain."""

from __future__ import annotations

from pathlib import Path

from codedoggy.orchestration.agent_def import (
    refresh_custom_agents,
    resolve_agent_definition,
)
from codedoggy.orchestration.subagent_policy import (
    default_isolation_for,
    drain_prompt_queue_after_cancel,
    effective_max_subagent_depth,
    load_discovered_agents,
)
from codedoggy.orchestration.types import IsolationMode
from codedoggy.tools.builtins.merge_subagent_worktree import MergeSubagentWorktreeTool
from codedoggy.tools.builtins.spawn_subagent import TaskTool
from codedoggy.tools.runtime import ToolCallContext, ToolError
from codedoggy.orchestration.subagent import SubagentCoordinator, SubagentRequest, SubagentSnapshot


def test_effective_max_subagent_depth_env(monkeypatch: object) -> None:
    monkeypatch.delenv("CODEDOGGY_MAX_SUBAGENT_DEPTH", raising=False)
    assert effective_max_subagent_depth() == 1
    monkeypatch.setenv("CODEDOGGY_MAX_SUBAGENT_DEPTH", "3")
    assert effective_max_subagent_depth() == 3
    monkeypatch.setenv("CODEDOGGY_MAX_SUBAGENT_DEPTH", "99")
    assert effective_max_subagent_depth() == 5  # hard ceiling


def test_default_isolation_auto(monkeypatch: object) -> None:
    monkeypatch.setenv("CODEDOGGY_SUBAGENT_ISOLATION", "auto")
    assert default_isolation_for("explore") is IsolationMode.NONE
    assert default_isolation_for("general-purpose") is IsolationMode.WORKTREE
    monkeypatch.setenv("CODEDOGGY_SUBAGENT_ISOLATION", "worktree")
    assert default_isolation_for("explore") is IsolationMode.WORKTREE
    monkeypatch.setenv("CODEDOGGY_SUBAGENT_ISOLATION", "none")
    assert default_isolation_for("general-purpose") is IsolationMode.NONE


def test_drain_after_cancel_default_off(monkeypatch: object) -> None:
    monkeypatch.delenv("CODEDOGGY_DRAIN_AFTER_CANCEL", raising=False)
    assert drain_prompt_queue_after_cancel() is False
    monkeypatch.setenv("CODEDOGGY_DRAIN_AFTER_CANCEL", "1")
    assert drain_prompt_queue_after_cancel() is True


def test_discover_custom_agent_md(tmp_path: Path, monkeypatch: object) -> None:
    agents = tmp_path / ".codedoggy" / "agents"
    agents.mkdir(parents=True)
    (agents / "reviewer.md").write_text(
        "---\n"
        "name: reviewer\n"
        "description: PR review helper\n"
        "capability_mode: read-only\n"
        "isolation: none\n"
        "---\n"
        "Role: review only. Do not edit files.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEDOGGY_AGENTS_PATHS", "")
    found = load_discovered_agents(tmp_path)
    assert "reviewer" in found
    defn = found["reviewer"]
    assert defn.capability_mode.value == "read-only"
    assert "review" in defn.system_prompt_body.lower()

    n = refresh_custom_agents(tmp_path, force=True)
    assert n >= 1
    resolved = resolve_agent_definition("reviewer", cwd=tmp_path)
    assert resolved is not None
    assert resolved.name == "reviewer"
    # Builtin still wins
    assert resolve_agent_definition("explore", cwd=tmp_path).name == "explore"


def test_spawn_uses_isolation_default(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("CODEDOGGY_SUBAGENT_ISOLATION", "worktree")
    coord = SubagentCoordinator()
    seen: list[SubagentRequest] = []

    def run_fn(req: SubagentRequest, cancel) -> SubagentSnapshot:
        seen.append(req)
        return SubagentSnapshot(
            subagent_id=req.id,
            subagent_type=req.subagent_type,
            status="completed",
            output="ok",
            duration_ms=1,
        )

    TaskTool().run(
        ToolCallContext(
            cwd=tmp_path,
            session_id="s1",
            extra={
                "subagent_coordinator": coord,
                "subagent_run_fn": run_fn,
                "task_model_validator": lambda _s: None,
            },
        ),
        {
            "prompt": "p",
            "description": "d",
            "subagent_type": "explore",
            "run_in_background": False,
        },
    )
    assert len(seen) == 1
    assert seen[0].isolation is IsolationMode.WORKTREE
    coord.shutdown(wait=True)


def test_merge_subagent_worktree_missing_id(tmp_path: Path) -> None:
    coord = SubagentCoordinator()
    with __import__("pytest").raises(ToolError):
        MergeSubagentWorktreeTool().run(
            ToolCallContext(
                cwd=tmp_path,
                session_id="s",
                extra={"subagent_coordinator": coord},
            ),
            {},
        )
    coord.shutdown(wait=True)


def test_merge_unknown_subagent(tmp_path: Path) -> None:
    coord = SubagentCoordinator()
    with __import__("pytest").raises(ToolError) as ei:
        MergeSubagentWorktreeTool().run(
            ToolCallContext(
                cwd=tmp_path,
                session_id="s",
                extra={"subagent_coordinator": coord},
            ),
            {"subagent_id": "sub_missing"},
        )
    assert "Unknown" in ei.value.message or "merge failed" in ei.value.message.lower()
    coord.shutdown(wait=True)
