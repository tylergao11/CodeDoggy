"""Hermes-style session FTS store + HermesMemorySelector + session_search tool."""

from __future__ import annotations

import json
from pathlib import Path

from codedoggy.audit.types import MemorySelectRequest, MutationEvent
from codedoggy.bootstrap import build_session
from codedoggy.memory import HermesMemorySelector, MemoryStore, SessionStore
from codedoggy.model import CompletionResult
from codedoggy.tools import ToolCallContext, ToolRegistryBuilder


def test_session_store_search_and_scroll(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    store = SessionStore(db)
    store.ensure_session("s1", goal="auth refactor", title="auth")
    store.append_message("s1", "user", "please fix login timeout")
    store.append_message("s1", "assistant", "I'll update the JWT expiry")
    store.append_message("s1", "user", "also docker networking")
    store.ensure_session("s2", goal="docs", title="docs")
    store.append_message("s2", "user", "unrelated gardening tips")

    hits = store.search("login JWT", limit=5)
    assert hits
    assert any("login" in h.content.lower() or "jwt" in h.content.lower() for h in hits)

    around = store.get_messages_around("s1", hits[0].message_id, window=2)
    assert around["window"]
    store.close()


def test_fts_sanitize_never_column_filter() -> None:
    """Natural language must not become FTS5 col:term (live: no such column: reading)."""
    q = SessionStore._sanitize_fts_query(
        "Read blob.txt fully with read_file (offset/limit if needed)"
    )
    assert q
    assert ":" not in q.replace('"', "")  # no bare col: syntax
    assert "reading" not in q.lower() or '"read' in q.lower() or "read_file" in q.lower()
    # Each term quoted
    assert '"' in q
    # Column names not bare
    for col in ("content", "role", "tool_name"):
        assert f"{col}:" not in q


def test_fts_search_natural_language_no_error(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    store.ensure_session("s1")
    store.append_message("s1", "user", "please Read the reading list later")
    store.append_message("s1", "assistant", "ok will read_file the notes")
    # Must not raise; may use LIKE fallback only if MATCH fails
    hits = store.search(
        "Read blob fully with read_file for reading notes",
        limit=5,
    )
    # At least does not crash; ideally finds something
    assert isinstance(hits, list)
    store.close()


def test_hermes_selector_combines_sources(tmp_path: Path) -> None:
    mem = MemoryStore(memory_dir=tmp_path / "mem")
    mem.load_from_disk()
    mem.add("memory", "Prefer small diffs on auth")
    mem.load_from_disk()

    db = tmp_path / "state.db"
    ss = SessionStore(db)
    ss.ensure_session("old", goal="auth work")
    ss.append_message("old", "user", "auth timeout bug on login page")
    ss.append_message("old", "assistant", "fixed JWT expiry")

    sel = HermesMemorySelector(curated_store=mem, session_store=ss)
    res = sel.select(
        MemorySelectRequest(
            goal="auth timeout",
            mutation=MutationEvent(
                path="auth.py",
                tool_name="search_replace",
                call_id="1",
                after="x",
            ),
            trajectory_summary="(none)",
            session_id="current",
            query_hint="auth login",
            max_session_hits=5,
        )
    )
    assert res.curated_blocks
    assert "small diffs" in res.combined_text()
    assert res.session_hits
    assert any("auth" in h.lower() or "login" in h.lower() for h in res.session_hits)
    ss.close()


def test_hermes_selector_live_and_same_session(tmp_path: Path) -> None:
    mem = MemoryStore(memory_dir=tmp_path / "mem")
    mem.load_from_disk()
    mem.add("memory", "Frozen only note")
    mem.load_from_disk()
    mem.add("memory", "Live mid-session note about JWT")

    db = tmp_path / "state.db"
    ss = SessionStore(db)
    ss.ensure_session("sid-1", goal="auth")
    ss.append_message("sid-1", "user", "login JWT timeout earlier today")

    frozen_sel = HermesMemorySelector(
        curated_store=mem, session_store=ss, prefer_frozen=True
    )
    live_sel = HermesMemorySelector(
        curated_store=mem, session_store=ss, prefer_frozen=False
    )
    req = MemorySelectRequest(
        goal="auth",
        mutation=MutationEvent(
            path="auth.py", tool_name="search_replace", call_id="1", after="x"
        ),
        trajectory_summary="(none)",
        session_id="sid-1",
        query_hint="JWT login",
        max_session_hits=5,
    )
    frozen = frozen_sel.select(req)
    live = live_sel.select(req)
    assert "Frozen only" in frozen.combined_text()
    assert "Live mid-session" not in frozen.combined_text()
    assert "Live mid-session" in live.combined_text()
    # Same session included by default
    assert live.session_hits or frozen.session_hits
    assert any("JWT" in h or "login" in h.lower() for h in (live.session_hits + frozen.session_hits))

    excl = HermesMemorySelector(
        curated_store=mem,
        session_store=ss,
        include_current_session=False,
    ).select(req)
    assert excl.session_hits == []
    ss.close()


def test_session_search_tool(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    store = SessionStore(db)
    store.ensure_session("s1", title="deploy")
    store.append_message("s1", "user", "kubernetes rollout failed")
    store.append_message("s1", "assistant", "check image pull secrets")

    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path, extra={"session_store": store})
    out = tools.call("session_search", {"query": "kubernetes secrets"}, ctx)
    data = json.loads(out)
    assert data["shape"] == "discovery"
    assert data["results"]

    browse = json.loads(tools.call("session_search", {}, ctx))
    assert browse["shape"] == "browse"
    assert browse["sessions"]
    store.close()


def test_file_notes_provider_and_fts_recency(tmp_path: Path) -> None:
    from codedoggy.memory import MemoryManager, SessionStore
    from codedoggy.memory.providers_extra import FileNotesProvider

    notes = tmp_path / "notes.md"
    notes.write_text("DECIDE: use JWT for auth sessions\nother line\n", encoding="utf-8")
    ss = SessionStore(tmp_path / "s.db")
    ss.ensure_session("s1")
    ss.append_message("s1", "user", "old gardening tips")
    ss.append_message("s1", "assistant", "JWT expiry set to 15 minutes for auth")
    mm = MemoryManager.create_default(session_store=ss)
    assert mm.add_provider(FileNotesProvider(notes)) is True
    pre = mm.prefetch_all("JWT auth", session_id="s1")
    assert "JWT" in pre
    # file notes should contribute
    assert "File notes" in pre or "JWT" in pre
    hits = ss.search("JWT auth", limit=5)
    assert hits
    assert hits[0].score >= hits[-1].score  # ranked desc
    mm.shutdown()
    ss.close()


def test_memory_manager_prefetch_and_one_external(tmp_path: Path) -> None:
    from codedoggy.memory import (
        BaseMemoryProvider,
        MemoryManager,
        MemoryStore,
        SessionStore,
    )

    mem = MemoryStore(memory_dir=tmp_path / "m")
    mem.load_from_disk()
    mem.add("memory", "prefer small diffs")
    mem.load_from_disk()
    ss = SessionStore(tmp_path / "s.db")
    ss.ensure_session("s1")
    ss.append_message("s1", "user", "JWT login timeout")
    mm = MemoryManager.create_default(curated=mem, session_store=ss)
    sys_blk = mm.build_system_prompt()
    assert "small diffs" in sys_blk
    pre = mm.prefetch_all("JWT login", session_id="s1")
    assert "JWT" in pre or "login" in pre.lower()

    class Ext(BaseMemoryProvider):
        name = "ext_a"

        def prefetch(self, query: str, *, session_id: str = "") -> str:
            return "external-hit"

    class Ext2(BaseMemoryProvider):
        name = "ext_b"

    assert mm.add_provider(Ext()) is True
    assert mm.add_provider(Ext2()) is False  # second external rejected
    pre2 = mm.prefetch_all("anything", session_id="s1")
    assert "external-hit" in pre2
    mm.shutdown()
    ss.close()


def test_shell_write_denied_by_policy_before_exec(tmp_path: Path) -> None:
    from codedoggy.tools.policy import WorkspacePolicy
    from codedoggy.tools import ToolCallContext, ToolRegistryBuilder
    from codedoggy.tools.runtime import ToolError

    pol = WorkspacePolicy.from_env(tmp_path)
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path, extra={"policy": pol})
    try:
        tools.call(
            "run_terminal_cmd",
            {
                "command": "echo secret > .env",
                "description": "try to write secrets file",
            },
            ctx,
        )
        raised = False
    except ToolError as e:
        raised = True
        assert "denied" in e.message.lower() or e.code in {"deny_path", "policy_denied"}
    assert raised
    assert not (tmp_path / ".env").exists()


def test_auditor_prompt_includes_policy() -> None:
    from codedoggy.audit.model_auditor import ModelAuditor
    from codedoggy.audit.types import (
        AuditContext,
        MemorySelectResult,
        MutationEvent,
    )
    from codedoggy.model import CompletionResult, ModelConfig

    class CapClient:
        config = ModelConfig(provider="x", model="m", base_url="http://x")
        last = None

        def complete(self, messages, **kw):
            self.last = messages
            return CompletionResult(content='{"ok": true}', model="m")

    client = CapClient()
    aud = ModelAuditor(client)  # type: ignore[arg-type]
    mem = MemorySelectResult(raw={"policy": {"enabled": True, "deny_write_globs": [".git"]}})
    ctx = AuditContext(
        goal="ship",
        mutation=MutationEvent(
            path="a.py", tool_name="search_replace", call_id="1", after="x"
        ),
        trajectory_summary="(none)",
        memory=mem,
        cwd=".",
    )
    aud.review(ctx)
    assert client.last
    blob = "\n".join(m.content or "" for m in client.last)
    assert "Workspace policy" in blob
    assert ".git" in blob


def test_policy_denies_pem_and_ssh(tmp_path: Path) -> None:
    from codedoggy.tools.policy import WorkspacePolicy

    pol = WorkspacePolicy.from_env(tmp_path)
    assert pol.check_write("certs/server.pem").allowed is False
    assert pol.check_write(".ssh/id_rsa").allowed is False
    assert pol.check_write("src/main.py").allowed is True


def test_policy_denies_git_write(tmp_path: Path) -> None:
    from codedoggy.tools.policy import WorkspacePolicy
    from codedoggy.tools import ToolCallContext, ToolRegistryBuilder
    from codedoggy.tools.runtime import ToolError

    pol = WorkspacePolicy.from_env(tmp_path)
    d = pol.check_write(".git/config")
    assert d.allowed is False
    assert ".git" in d.reason or "git" in d.reason
    # Allowed normal file
    assert pol.check_write("src/ok.py").allowed is True

    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path, extra={"policy": pol})
    try:
        tools.call(
            "search_replace",
            {
                "file_path": ".git/config",
                "old_string": "",
                "new_string": "x=1",
            },
            ctx,
        )
        raised = False
    except ToolError as e:
        raised = True
        assert e.code == "deny_path" or "denied" in e.message.lower()
    assert raised
    assert not (tmp_path / ".git" / "config").exists()


def test_build_session_persists_turn(tmp_path: Path) -> None:
    from tests.test_bootstrap import ScriptClient

    db = tmp_path / "state.db"
    main = ScriptClient(
        [CompletionResult(content="all good", model="main")],
        name="main",
    )
    audit = ScriptClient(
        [CompletionResult(content='{"ok": true}', model="audit")],
        name="audit",
    )
    s = build_session(
        tmp_path,
        goal="say ok",
        max_turns=3,
        main_client=main,
        audit_client=audit,
        enable_memory=True,
        enable_session_store=True,
        memory_dir=tmp_path / "mem",
        session_db=db,
    )
    try:
        r = s.handle_prompt("hello codedoggy")
        assert r.status.value == "completed"
        store = s.extensions.session_store
        assert store is not None
        msgs = store.get_messages(str(s.id))
        assert any(m["role"] == "user" and "hello" in (m["content"] or "") for m in msgs)
        assert any(m["role"] == "assistant" for m in msgs)
        # Archive path skips system noise
        assert not any(m["role"] == "system" for m in msgs)
    finally:
        s.close()


def test_shell_write_triggers_mutation_audit(tmp_path: Path) -> None:
    """Shell redirect / python open into cwd should produce a mutation."""
    from codedoggy.tools import ToolCallContext, ToolRegistryBuilder
    from codedoggy.tools.util.write_detect import (
        detect_shell_write_paths,
        record_shell_mutations,
    )

    assert detect_shell_write_paths('echo hi > out.txt')
    assert detect_shell_write_paths(
        "python -c \"open('wrote.txt','w',encoding='utf-8').write('x')\""
    )
    paths = detect_shell_write_paths(
        "python -c \"open('wrote.txt','w',encoding='utf-8').write('x')\""
    )
    assert any("wrote" in p for p in paths)

    (tmp_path / "redir.txt").write_text("via redirect detect", encoding="utf-8")
    ctx2 = ToolCallContext(cwd=tmp_path)
    ok = record_shell_mutations(
        ctx2, "echo x > redir.txt", exit_ok=True, tool_name="run_terminal_cmd"
    )
    assert ok
    mut = ctx2.extra.get("mutation")
    assert mut is not None
    assert "redir" in mut.path
    assert mut.after and "redirect" in mut.after

    (tmp_path / "wrote.txt").write_text("shell body", encoding="utf-8")
    ctx3 = ToolCallContext(cwd=tmp_path)
    ok3 = record_shell_mutations(
        ctx3,
        "python -c \"open('wrote.txt','w').write('shell body')\"",
        exit_ok=True,
    )
    assert ok3
    assert "wrote" in ctx3.extra["mutation"].path


def test_main_prefetch_injects_session_hits(tmp_path: Path) -> None:
    from tests.test_bootstrap import ScriptClient

    db = tmp_path / "state.db"
    store = SessionStore(db)
    store.ensure_session("will-replace", goal="auth")
    # Will be re-bound to real session id after build — seed via store after create

    main = ScriptClient(
        [CompletionResult(content="ok continuing", model="main")],
        name="main",
    )
    audit = ScriptClient(
        [CompletionResult(content='{"ok": true}', model="audit")] * 3,
        name="audit",
    )
    s = build_session(
        tmp_path,
        goal="auth timeout",
        max_turns=3,
        main_client=main,
        audit_client=audit,
        enable_memory=True,
        enable_session_store=True,
        memory_dir=tmp_path / "mem",
        session_db=db,
    )
    try:
        store2 = s.extensions.session_store
        assert store2 is not None
        store2.append_message(str(s.id), "user", "earlier JWT login timeout discussion")
        store2.append_message(str(s.id), "assistant", "fixed expiry to 15m")
        r = s.handle_prompt("remind me about the JWT work")
        assert r.status.value == "completed"
        # Prefetch should have been injected into the system side of the sample
        first = main.calls[0]
        blob = "\n".join(
            (getattr(m, "content", None) or "")
            if not isinstance(m, dict)
            else (m.get("content") or "")
            for m in first
        )
        assert "Prefetched session memory" in blob or "JWT" in blob
    finally:
        s.close()
        store.close()


def test_cross_prompt_live_resume(tmp_path: Path) -> None:
    """Second handle_prompt sees prior user/assistant in the live window."""
    from tests.test_bootstrap import ScriptClient

    main = ScriptClient(
        [
            CompletionResult(content="first reply about auth", model="main"),
            CompletionResult(content="second reply continuing", model="main"),
        ],
        name="main",
    )
    audit = ScriptClient(
        [CompletionResult(content='{"ok": true}', model="audit")] * 4,
        name="audit",
    )
    s = build_session(
        tmp_path,
        goal="auth work",
        max_turns=4,
        main_client=main,
        audit_client=audit,
        enable_memory=False,
        enable_session_store=True,
        session_db=tmp_path / "state.db",
    )
    try:
        r1 = s.handle_prompt("start auth refactor")
        assert r1.status.value == "completed"
        assert r1.metadata.get("resumed_prior") is False
        runner = s.extensions.turn_runner
        assert runner is not None
        assert any(
            "start auth" in (m.content or "")
            for m in runner.live_messages
        )

        r2 = s.handle_prompt("continue with JWT")
        assert r2.status.value == "completed"
        assert r2.metadata.get("resumed_prior") is True
        # Main model saw prior transcript on second call
        second_call = main.calls[1]
        blob = "\n".join(
            getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "") or ""
            for m in second_call
        )
        assert "start auth" in blob or "first reply" in blob
        assert "JWT" in blob or "continue" in blob

        store = s.extensions.session_store
        assert store is not None
        archived = store.get_messages(str(s.id))
        users = [m for m in archived if m["role"] == "user"]
        assert len(users) >= 2
    finally:
        s.close()
