"""Attack-style regressions for graph reliability P1.

1. Incremental reindex must extract-before-swap (failed extract keeps prior defs).
2. code_nav reindex mutates cache — must fail closed under allow_writes=False.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.graph import FileEvent, IndexBuilder, IndexManager, Navigator
from codedoggy.graph.handle import CodebaseGraph
from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.policy import WorkspacePolicy
from codedoggy.tools.runtime import ToolCallContext, ToolError


def _sample_repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        """\
class AuthService:
    def login(self, user):
        return token_for(user)

def token_for(user):
    return f\"tok-{user}\"
""",
        encoding="utf-8",
    )
    return tmp_path


def test_p1_reindex_extract_failure_keeps_prior_defs(tmp_path: Path) -> None:
    """If extract raises mid-reindex, prior definitions for that path must remain.

    Swap-then-extract would wipe AuthService/token_for and leave a hole.
    """
    root = _sample_repo(tmp_path)
    index = IndexBuilder().build(root)
    mgr = IndexManager(root, index=index)
    nav = Navigator(mgr.index, root=root)
    assert nav.goto_definition_by_name("AuthService").locations
    assert nav.goto_definition_by_name("token_for").locations
    prior_defs = list(mgr.index.definitions.get("AuthService", []))
    assert prior_defs

    mod = root / "pkg" / "mod.py"
    # File still valid on disk; extract is forced to fail
    mod.write_text(
        "class AuthService:\n    pass\n\ndef token_for(user):\n    return user\n",
        encoding="utf-8",
    )

    def _boom(*_a, **_k):
        raise RuntimeError("simulated extract failure")

    mgr.registry.extract = _boom  # type: ignore[method-assign]
    mgr.send_event(FileEvent.modified(mod))

    nav2 = Navigator(mgr.index, root=root)
    kept = nav2.goto_definition_by_name("AuthService").locations
    assert kept, "extract failure must not wipe prior AuthService defs"
    assert list(mgr.index.definitions.get("AuthService", [])) == prior_defs
    assert nav2.goto_definition_by_name("token_for").locations


def test_p1_reindex_denied_when_allow_writes_false(tmp_path: Path) -> None:
    """code_nav reindex writes .goto_index.json — deny when allow_writes=False."""
    root = _sample_repo(tmp_path)
    graph = CodebaseGraph(root, use_cache=True)
    graph.reindex()  # seed index + cache outside the policy gate

    tools = ToolRegistryBuilder.new().finalize()
    policy = WorkspacePolicy(cwd=root, allow_writes=False)
    ctx = ToolCallContext(cwd=root, extra={"graph": graph, "policy": policy})

    with pytest.raises(ToolError) as ei:
        tools.call("code_nav", {"action": "reindex"}, ctx)
    assert ei.value.code in {"write_disabled", "policy_denied"}
    assert "write" in (ei.value.message or "").lower() or "denied" in (
        ei.value.message or ""
    ).lower()

    # Read-only actions still allowed under the same policy
    stats = tools.call("code_nav", {"action": "stats"}, ctx)
    assert "definitions" in stats
    out = tools.call(
        "code_nav",
        {"action": "definition", "symbol": "AuthService"},
        ctx,
    )
    assert "AuthService" in out


def test_p1_reindex_allowed_when_writes_enabled(tmp_path: Path) -> None:
    """Positive control: reindex succeeds under allow_writes=True."""
    root = _sample_repo(tmp_path)
    graph = CodebaseGraph(root, use_cache=True)
    tools = ToolRegistryBuilder.new().finalize()
    policy = WorkspacePolicy(cwd=root, allow_writes=True)
    ctx = ToolCallContext(cwd=root, extra={"graph": graph, "policy": policy})
    out = tools.call("code_nav", {"action": "reindex"}, ctx)
    assert '"ok": true' in out.lower() or '"ok": true' in out or '"ok":true' in out.replace(
        " ", ""
    )
