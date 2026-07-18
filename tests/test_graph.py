"""Codebase graph — API parity with xai-codebase-graph Navigator / IndexBuilder."""

from __future__ import annotations

from pathlib import Path

from codedoggy.graph import (
    IndexBuilder,
    Navigator,
    get_cache_path,
    load_index,
    save_index,
)
from codedoggy.tools import ToolCallContext, ToolRegistryBuilder


def _sample_repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        """\
class AuthService:
    def login(self, user):
        return token_for(user)

def token_for(user):
    return f\"tok-{user}\"

def main():
    svc = AuthService()
    return svc.login(\"a\")
""",
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "other.py").write_text(
        """\
from mod import AuthService, token_for as tf

def use():
    return tf(\"b\")
""",
        encoding="utf-8",
    )
    return tmp_path


def test_index_builder_and_goto_definition(tmp_path: Path) -> None:
    root = _sample_repo(tmp_path)
    index = IndexBuilder().build(root)
    stats = index.stats()
    assert stats.files >= 2
    assert stats.definitions >= 3

    nav = Navigator(index, root=root)
    by_name = nav.goto_definition_by_name("AuthService")
    assert by_name.symbol == "AuthService"
    assert by_name.locations
    assert any("mod.py" in loc.path for loc in by_name.locations)

    refs = nav.goto_references_by_name("token_for", include_definition=True)
    assert refs.locations
    assert any(loc.line > 0 for loc in refs.locations)


def test_index_builder_parallel_matches_serial(tmp_path: Path) -> None:
    root = _sample_repo(tmp_path)
    for i in range(20):
        (root / "pkg" / f"gen_{i}.py").write_text(
            f"def f_{i}():\n    return g_{i}()\n\ndef g_{i}():\n    return {i}\n",
            encoding="utf-8",
        )
    serial = IndexBuilder().with_threads(1).with_chunk_size(4).build(root)
    parallel = (
        IndexBuilder().with_threads(4).with_chunk_size(4).with_build_batch_size(8).build(root)
    )
    assert serial.stats().files == parallel.stats().files
    assert serial.stats().definitions == parallel.stats().definitions
    assert set(serial.definitions.keys()) == set(parallel.definitions.keys())
    assert set(serial.definitions.get("AuthService", [])) == set(
        parallel.definitions.get("AuthService", [])
    )


def test_goto_definition_at_position(tmp_path: Path) -> None:
    root = _sample_repo(tmp_path)
    index = IndexBuilder().build(root)
    nav = Navigator(index, root=root)
    result = nav.goto_definition("pkg/mod.py", 1, 7)
    assert result.symbol == "AuthService"
    assert result.locations


def test_cache_roundtrip(tmp_path: Path) -> None:
    root = _sample_repo(tmp_path)
    index = IndexBuilder().build(root)
    cache = get_cache_path(root)
    save_index(cache, index)
    loaded = load_index(cache)
    assert loaded.stats().definitions == index.stats().definitions
    nav = Navigator(loaded, root=root)
    assert nav.goto_definition_by_name("login").locations


def test_code_nav_tool(tmp_path: Path) -> None:
    from codedoggy.graph.handle import CodebaseGraph

    root = _sample_repo(tmp_path)
    graph = CodebaseGraph(root, use_cache=False)
    graph.reindex()
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=root, extra={"graph": graph})
    out = tools.call(
        "code_nav",
        {"action": "definition", "symbol": "AuthService"},
        ctx,
    )
    assert "AuthService" in out
    assert "mod.py" in out
    stats = tools.call("code_nav", {"action": "stats"}, ctx)
    assert "definitions" in stats


def test_index_manager_incremental(tmp_path: Path) -> None:
    from codedoggy.graph import FileEvent, IndexBuilder, IndexManager, Navigator

    root = _sample_repo(tmp_path)
    index = IndexBuilder().with_threads(2).build(root)
    mgr = IndexManager(root, index=index)
    nav = Navigator(mgr.index, root=root)
    assert nav.goto_definition_by_name("AuthService").locations

    mod = root / "pkg" / "mod.py"
    mod.write_text(
        "class AuthServiceV2:\n    pass\n\ndef token_for(user):\n    return user\n",
        encoding="utf-8",
    )
    mgr.send_event(FileEvent.modified(mod))
    nav2 = Navigator(mgr.index, root=root)
    defs = nav2.goto_definition_by_name("AuthServiceV2").locations
    assert defs
    assert any("mod.py" in d.path for d in defs)

    mgr.send_event(FileEvent.removed(mod))
    nav3 = Navigator(mgr.index, root=root)
    assert not nav3.goto_definition_by_name("AuthServiceV2").locations


def test_js_extract_and_index(tmp_path: Path) -> None:
    (tmp_path / "app.js").write_text(
        "class Foo {}\nfunction bar() { return Foo(); }\nconst baz = () => bar();\n",
        encoding="utf-8",
    )
    index = IndexBuilder().build(tmp_path)
    assert "Foo" in index.definitions
    assert "bar" in index.definitions
    nav = Navigator(index, root=tmp_path)
    assert nav.goto_definition_by_name("bar").locations


def test_build_session_has_graph(tmp_path: Path) -> None:
    from codedoggy.bootstrap import build_session
    from codedoggy.model import CompletionResult
    from tests.test_bootstrap import ScriptClient

    main = ScriptClient([CompletionResult(content="ok", model="m")], name="m")
    s = build_session(
        tmp_path,
        main_client=main,
        audit_client=main,
        enable_audit=False,
        enable_memory=False,
        enable_session_store=False,
        enable_graph=True,
    )
    try:
        assert s.extensions.graph is not None
    finally:
        s.close()


def test_query_version_set_and_cache_invalidation(tmp_path: Path) -> None:
    from codedoggy.graph import (
        IndexBuilder,
        LanguageRegistry,
        QueryVersion,
        get_cache_path,
        load_index,
        save_index,
    )
    from codedoggy.graph.handle import CodebaseGraph

    root = _sample_repo(tmp_path)
    reg = LanguageRegistry()
    h1 = reg.compute_query_hash()
    h2 = LanguageRegistry().compute_query_hash()
    assert h1 == h2
    assert h1 != 0

    index = IndexBuilder(registry=reg).build(root)
    assert not index.query_version.is_legacy
    assert index.query_version.hash == h1
    assert not index.needs_query_rebuild(h1)
    assert index.needs_query_rebuild(h1 ^ 0xDEAD)
    assert QueryVersion.legacy().needs_rebuild(h1)

    cache = get_cache_path(root)
    save_index(cache, index)
    loaded = load_index(cache)
    assert loaded.query_version == index.query_version
    assert not loaded.needs_query_rebuild(h1)

    loaded.query_version = QueryVersion.version(1)
    save_index(cache, loaded)
    g = CodebaseGraph(root, use_cache=True)
    g.ensure_indexed()
    assert g.stats().definitions >= 3
    assert not g.ensure_indexed().needs_query_rebuild(h1)


def test_event_debouncer_coalesces(tmp_path: Path) -> None:
    import time

    from codedoggy.graph import EventDebouncer, FileEvent

    received: list[list[FileEvent]] = []
    deb = EventDebouncer(lambda batch: received.append(batch), debounce_secs=0.15)
    p = tmp_path / "a.py"
    p.write_text("x=1\n", encoding="utf-8")
    deb.push(FileEvent.modified(p))
    deb.push(FileEvent.modified(p))
    deb.push(FileEvent.modified(p))
    time.sleep(0.35)
    assert len(received) == 1
    assert len(received[0]) == 1
    deb.close()


def test_workspace_watcher_picks_up_new_file(tmp_path: Path) -> None:
    import time

    from codedoggy.graph import IndexBuilder, IndexManager, Navigator, WorkspaceWatcher

    root = _sample_repo(tmp_path)
    index = IndexBuilder().build(root)
    mgr = IndexManager(root, index=index)
    w = WorkspaceWatcher(root, mgr, debounce_secs=0.2)
    w.start()
    try:
        assert w.is_running()
        new = root / "pkg" / "fresh.py"
        new.write_text("def brand_new():\n    return 1\n", encoding="utf-8")
        deadline = time.time() + 4.0
        found = False
        while time.time() < deadline:
            if Navigator(mgr.index, root=root).goto_definition_by_name(
                "brand_new"
            ).locations:
                found = True
                break
            time.sleep(0.1)
        assert found, "watchdog should reindex new file via FileEvent"
    finally:
        w.stop()


def test_rust_go_index(tmp_path: Path) -> None:
    from codedoggy.graph import IndexBuilder, Navigator

    (tmp_path / "lib.rs").write_text(
        """\
pub struct AuthService {}
impl AuthService {
    pub fn login(&self) {}
}
pub fn token_for(user: &str) -> String {
    format!(\"tok-{user}\")
}
fn call_it() {
    token_for(\"a\");
}
use std::io as sio;
""",
        encoding="utf-8",
    )
    (tmp_path / "main.go").write_text(
        """\
package main

type AuthService struct{}

func (a *AuthService) Login() {}

func TokenFor(user string) string {
    return user
}

func main() {
    TokenFor(\"x\")
}
""",
        encoding="utf-8",
    )
    index = IndexBuilder().build(tmp_path)
    assert "AuthService" in index.definitions
    nav = Navigator(index, root=tmp_path)
    assert nav.goto_definition_by_name("AuthService").locations
    assert nav.goto_definition_by_name("token_for").locations
    assert nav.goto_definition_by_name("TokenFor").locations
    assert "sio" in index.aliases


def test_python_tree_sitter_extract() -> None:
    from codedoggy.graph.languages import LanguageRegistry

    reg = LanguageRegistry()
    src = "class Z:\n    def m(self):\n        helper()\n\ndef helper():\n    pass\n"
    ex = reg.extract("x.py", src)
    names = {d.name for d in ex.definitions}
    assert "Z" in names
    assert "helper" in names
    assert any(r.name == "helper" for r in ex.references)


def test_removed_then_created_debounce_indexes_file(tmp_path: Path) -> None:
    """Atomic rewrite: Removed → Created coalesces to Modified; index keeps symbols."""
    from codedoggy.graph import EventDebouncer, FileEvent, IndexBuilder, IndexManager, Navigator

    root = _sample_repo(tmp_path)
    index = IndexBuilder().build(root)
    mgr = IndexManager(root, index=index)
    received: list[list[FileEvent]] = []

    def on_flush(batch: list[FileEvent]) -> None:
        received.append(batch)
        mgr.send_events(batch)

    deb = EventDebouncer(on_flush, debounce_secs=0.1)
    target = root / "pkg" / "mod.py"
    # Simulate editor atomic save: delete then recreate
    deb.push(FileEvent.removed(target))
    # File still on disk for reindex after coalesced Modified/Created
    deb.push(FileEvent.created(target))
    deb.flush_now()

    assert len(received) == 1
    assert len(received[0]) == 1
    assert received[0][0].kind.value in ("Created", "Modified")
    nav = Navigator(mgr.index, root=root)
    assert nav.goto_definition_by_name("AuthService").locations
    assert nav.goto_definition_by_name("token_for").locations


def test_reindex_updates_watcher_manager_pointer(tmp_path: Path) -> None:
    from codedoggy.graph.handle import CodebaseGraph

    root = _sample_repo(tmp_path)
    g = CodebaseGraph(root, use_cache=False)
    g.reindex()
    g.start_watch(debounce_secs=0.2)
    try:
        assert g.watcher is not None
        old_mgr = g.manager
        assert g.watcher.manager is old_mgr
        g.reindex()
        new_mgr = g.manager
        assert new_mgr is not None
        assert new_mgr is not old_mgr
        assert g.watcher.manager is new_mgr
    finally:
        g.stop_watch()


def test_cache_format_version_mismatch_rebuilds(tmp_path: Path) -> None:
    import json

    from codedoggy.graph import CACHE_FORMAT_VERSION, CacheFormatError, get_cache_path, load_index
    from codedoggy.graph.handle import CodebaseGraph

    root = _sample_repo(tmp_path)
    g = CodebaseGraph(root, use_cache=True)
    g.reindex()
    cache = get_cache_path(root)
    assert cache.is_file()

    raw = json.loads(cache.read_text(encoding="utf-8"))
    raw["format_version"] = CACHE_FORMAT_VERSION - 1
    # Poison definitions so a successful load of stale format would be visible
    raw["definitions"] = {"OnlyInStaleCache": [["pkg/mod.py", 1]]}
    cache.write_text(json.dumps(raw), encoding="utf-8")

    try:
        load_index(cache)
        raise AssertionError("expected CacheFormatError")
    except CacheFormatError:
        pass

    g2 = CodebaseGraph(root, use_cache=True)
    idx = g2.ensure_indexed()
    assert "OnlyInStaleCache" not in idx.definitions
    assert "AuthService" in idx.definitions
    # Fresh cache written with current format
    reloaded = load_index(cache)
    assert reloaded.stats().definitions == idx.stats().definitions


def test_send_event_marks_dirty_for_persist(tmp_path: Path) -> None:
    from codedoggy.graph import FileEvent
    from codedoggy.graph.handle import CodebaseGraph

    root = _sample_repo(tmp_path)
    g = CodebaseGraph(root, use_cache=True)
    g.reindex()
    assert g._dirty is False  # noqa: SLF001
    mod = root / "pkg" / "mod.py"
    mod.write_text("def after_edit():\n    return 1\n", encoding="utf-8")
    g.send_event(FileEvent.modified(mod))
    assert g._dirty is True  # noqa: SLF001
    g.persist_if_dirty()
    assert g._dirty is False  # noqa: SLF001


def test_respect_gitignore_walk_skips_ignored(tmp_path: Path) -> None:
    """When git is unavailable, walk path still applies .gitignore."""
    from codedoggy.graph.builder import IndexBuilder

    (tmp_path / "keep.py").write_text("def keep_me():\n    pass\n", encoding="utf-8")
    (tmp_path / "skip_me.py").write_text("def skip_me():\n    pass\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("skip_me.py\n", encoding="utf-8")

    b = IndexBuilder(respect_gitignore=True)
    # Force walk path (skip git) by monkeypatching
    b._collect_files_git = lambda root: None  # type: ignore[method-assign]
    files = b._collect_files(tmp_path)
    names = {p.name for p in files}
    assert "keep.py" in names
    assert "skip_me.py" not in names

    b2 = IndexBuilder(respect_gitignore=False)
    b2._collect_files_git = lambda root: None  # type: ignore[method-assign]
    files2 = b2._collect_files(tmp_path)
    names2 = {p.name for p in files2}
    assert "skip_me.py" in names2
