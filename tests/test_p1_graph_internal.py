"""Attack-style P1: CodebaseGraph.reindex / persist must honor write policy.

Tool path (code_nav action=reindex) is gated in code_nav + gate; residual gap is
direct graph.reindex() / watcher-driven persist writing .goto_index.json without
consulting policy. See handle.CodebaseGraph.set_policy / allow_write.
"""

from __future__ import annotations

from pathlib import Path

from codedoggy.graph.cache import CACHE_FILE_NAME, get_cache_path
from codedoggy.graph.handle import CodebaseGraph
from codedoggy.graph.index_manager import FileEvent
from codedoggy.tools.policy import WorkspacePolicy


def _sample_repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        "class AuthService:\n    def login(self):\n        return 1\n",
        encoding="utf-8",
    )
    return tmp_path


def test_reindex_allow_writes_false_does_not_write_cache(tmp_path: Path) -> None:
    """graph.reindex with policy allow_writes=False rebuilds memory but not disk."""
    root = _sample_repo(tmp_path)
    cache = get_cache_path(root)
    assert not cache.is_file()

    graph = CodebaseGraph(root, use_cache=True)
    graph.set_policy(WorkspacePolicy(cwd=root, allow_writes=False))
    stats = graph.reindex()
    assert stats["definitions"] >= 1
    # In-memory index usable
    assert graph.navigator.goto_definition_by_name("AuthService").locations
    assert not cache.is_file(), f"{CACHE_FILE_NAME} must not be written under allow_writes=False"

    # Explicit kwarg also forces skip even if policy were permissive
    graph2 = CodebaseGraph(root, use_cache=True)
    graph2.reindex(allow_write=False)
    assert not get_cache_path(root).is_file()


def test_reindex_allow_writes_true_writes_cache(tmp_path: Path) -> None:
    """Positive: reindex with allow_writes=True persists .goto_index.json when use_cache."""
    root = _sample_repo(tmp_path)
    cache = get_cache_path(root)
    graph = CodebaseGraph(root, use_cache=True)
    graph.set_policy(WorkspacePolicy(cwd=root, allow_writes=True))
    graph.reindex()
    assert cache.is_file(), f"expected {CACHE_FILE_NAME} after reindex with writes allowed"
    assert cache.stat().st_size > 0


def test_reindex_allow_write_true_overrides_deny_policy(tmp_path: Path) -> None:
    """Explicit allow_write=True wins over attached deny policy (caller already gated)."""
    root = _sample_repo(tmp_path)
    graph = CodebaseGraph(root, use_cache=True)
    graph.set_policy(WorkspacePolicy(cwd=root, allow_writes=False))
    graph.reindex(allow_write=True)
    assert get_cache_path(root).is_file()


def test_persist_if_dirty_skips_when_writes_denied(tmp_path: Path) -> None:
    """Watcher/session path: dirty in-memory updates must not flush cache under deny."""
    root = _sample_repo(tmp_path)
    graph = CodebaseGraph(root, use_cache=True)
    # Seed cache while writes allowed
    graph.reindex()
    cache = get_cache_path(root)
    assert cache.is_file()
    mtime_before = cache.stat().st_mtime_ns
    size_before = cache.stat().st_size

    graph.set_policy(WorkspacePolicy(cwd=root, allow_writes=False))
    # Incremental change (watcher spirit) marks dirty
    mod = root / "pkg" / "mod.py"
    mod.write_text(
        "class AuthService:\n    def login(self):\n        return 2\n\ndef helper():\n    pass\n",
        encoding="utf-8",
    )
    graph.send_event(FileEvent.modified(mod))
    assert graph._dirty is True  # noqa: SLF001 — attack probe

    graph.persist_if_dirty()
    assert graph._dirty is True  # noqa: SLF001 — still dirty; skip does not clear
    # Disk unchanged
    assert cache.stat().st_mtime_ns == mtime_before
    assert cache.stat().st_size == size_before
    # Memory updated
    assert graph.navigator.goto_definition_by_name("helper").locations


def test_policy_constructor_and_no_policy_default_writes(tmp_path: Path) -> None:
    """policy= ctor works; no policy keeps prior write-on-reindex behavior."""
    root = _sample_repo(tmp_path)
    g_deny = CodebaseGraph(
        root, use_cache=True, policy=WorkspacePolicy(cwd=root, allow_writes=False)
    )
    g_deny.reindex()
    assert not get_cache_path(root).is_file()

    g_open = CodebaseGraph(root, use_cache=True)
    g_open.reindex()
    assert get_cache_path(root).is_file()
