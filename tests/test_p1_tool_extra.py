"""Attack-style P1: full mid-turn path for memory_manager + provider tools.

Would-fail-before-fix failure modes:
  1. kernel.tool_extra missing memory_manager / session_store → provider tools
     and session_search cannot resolve stores mid-turn.
  2. Hermes initialize_all never run → providers lack session_id at first turn.
  3. Runner does not pass kernel.tool_extra into the loop.
  4. Custom tools= toolset finalized before provider inject → notes_append hidden.

Regression targets: bootstrap + RuntimeKernel.refresh_tool_extra + AgentTurnRunner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codedoggy.bootstrap import build_session
from codedoggy.model import CompletionResult, ModelConfig
from codedoggy.tools.registry import ToolRegistryBuilder
from codedoggy.tools.runtime import ToolCallContext
from codedoggy.turn.types import Message, Role


class _ScriptClient:
    """Deterministic chat client with optional tool-call script."""

    def __init__(
        self,
        script: list[CompletionResult] | None = None,
        *,
        name: str = "script",
    ) -> None:
        self.script = list(script or [])
        self.n = 0
        self.name = name
        self.config = ModelConfig(
            provider="fake", model=name, base_url="http://fake", api_key="x"
        )
        self.calls: list = []

    def complete(self, messages, **kwargs):
        self.calls.append(messages)
        if self.n >= len(self.script):
            return CompletionResult(content="(exhausted)", model=self.name)
        out = self.script[self.n]
        self.n += 1
        return out


def _notes_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tools=None,
    main_script: list[CompletionResult] | None = None,
):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CODEDOGGY_HOME", str(home))
    monkeypatch.setenv("CODEDOGGY_MEMORY_PROVIDER", "notes")
    main = _ScriptClient(
        main_script
        or [CompletionResult(content="ok", model="main")],
        name="main",
    )
    audit = _ScriptClient(
        [CompletionResult(content='{"ok": true}', model="audit")],
        name="audit",
    )
    s = build_session(
        tmp_path,
        main_client=main,
        audit_client=audit,
        enable_audit=False,
        enable_graph=False,
        enable_policy=True,
        memory_dir=home / "memories",
        session_db=tmp_path / "sess.db",
        tools=tools,
    )
    return s, main, home


def test_p1_kernel_tool_extra_carries_memory_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attack: after build_session, tool_extra must expose mm/store/policy."""
    s, _main, _home = _notes_session(tmp_path, monkeypatch)
    try:
        kernel = s.extensions.kernel
        assert kernel is not None
        extra = kernel.tool_extra
        assert extra.get("memory_manager") is s.extensions.memory_manager
        assert extra.get("memory_manager") is not None
        assert extra.get("session_store") is s.extensions.session_store
        assert extra.get("memory_store") is s.extensions.memory
        assert extra.get("policy") is s.extensions.policy
        # Hermes bind_session ran with real session id
        mm = s.extensions.memory_manager
        assert getattr(mm, "_session_id", None) == str(s.id)
        notes = next(
            (p for p in mm.providers if getattr(p, "name", "") == "notes"), None
        )
        assert notes is not None
        assert getattr(notes, "_session_id", None) == str(s.id)
    finally:
        s.close()


def test_p1_notes_append_via_kernel_tool_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attack: tools.call with tool_extra from kernel must write notes."""
    s, _main, home = _notes_session(tmp_path, monkeypatch)
    try:
        tools = s.extensions.tools
        assert tools is not None
        assert "notes_append" in tools.client_names()

        kernel = s.extensions.kernel
        kernel.refresh_tool_extra()
        ctx = ToolCallContext(
            cwd=tmp_path,
            session_id=str(s.id),
            extra=dict(kernel.tool_extra),
        )
        assert ctx.extra.get("memory_manager") is not None
        out = tools.call(
            "notes_append",
            {"content": "P1-tool-extra: httponly cookies"},
            ctx,
        )
        data = json.loads(out)
        assert data.get("success") is True
        notes_file = home / "memories" / "notes.md"
        assert notes_file.is_file()
        assert "httponly cookies" in notes_file.read_text(encoding="utf-8")
    finally:
        s.close()


def test_p1_handle_prompt_mid_turn_notes_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attack: live handle_prompt path must deliver memory_manager to tools."""
    tool_call = {
        "id": "c1",
        "type": "function",
        "function": {
            "name": "notes_append",
            "arguments": json.dumps(
                {"content": "live-turn: provider tools see tool_extra"}
            ),
        },
    }
    script = [
        CompletionResult(content="writing", model="main", tool_calls=[tool_call]),
        CompletionResult(content="done", model="main"),
    ]
    s, _main, home = _notes_session(tmp_path, monkeypatch, main_script=script)
    try:
        r = s.handle_prompt("remember the cookie decision")
        assert r.status.value == "completed"
        assert "notes_append" in (r.tools_called or [])
        notes_file = home / "memories" / "notes.md"
        body = notes_file.read_text(encoding="utf-8")
        assert "live-turn: provider tools see tool_extra" in body
    finally:
        s.close()


def test_p1_session_search_via_tool_extra_cwd_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attack: session_search resolves session_store from tool_extra + cwd scope."""
    # No notes provider required for this attack — default session store is enough.
    monkeypatch.delenv("CODEDOGGY_MEMORY_PROVIDER", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CODEDOGGY_HOME", str(home))

    s = build_session(
        tmp_path,
        main_client=_ScriptClient(
            [CompletionResult(content="ok", model="main")], name="main"
        ),
        audit_client=_ScriptClient(
            [CompletionResult(content="ok", model="audit")], name="audit"
        ),
        enable_audit=False,
        enable_graph=False,
        enable_policy=False,
        memory_dir=home / "memories",
        session_db=tmp_path / "sess.db",
    )
    try:
        kernel = s.extensions.kernel
        assert kernel is not None
        kernel.refresh_tool_extra()
        store = kernel.tool_extra.get("session_store")
        assert store is not None

        other = tmp_path / "other_proj"
        other.mkdir()
        # Same cwd as session → visible
        store.ensure_session("same-cwd", cwd=str(tmp_path.resolve()), title="in-scope")
        store.append_message(
            "same-cwd", "user", "alpha-zeta-unique-cookie decision in scope"
        )
        # Foreign cwd → out of scope
        store.ensure_session("foreign", cwd=str(other.resolve()), title="out")
        store.append_message(
            "foreign", "user", "alpha-zeta-unique-cookie decision foreign"
        )

        tools = s.extensions.tools
        ctx = ToolCallContext(
            cwd=tmp_path,
            session_id=str(s.id),
            extra=dict(kernel.tool_extra),
        )
        out = tools.call(
            "session_search",
            {"query": "alpha-zeta-unique-cookie", "limit": 10},
            ctx,
        )
        data = json.loads(out)
        assert data.get("shape") == "discovery"
        assert data.get("cwd") == str(tmp_path.resolve())
        ids = {r.get("session_id") for r in data.get("results") or []}
        assert "same-cwd" in ids
        assert "foreign" not in ids
    finally:
        s.close()


def test_p1_handle_prompt_session_search_mid_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attack: live turn session_search must use tool_extra.session_store."""
    monkeypatch.delenv("CODEDOGGY_MEMORY_PROVIDER", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CODEDOGGY_HOME", str(home))

    tool_call = {
        "id": "c1",
        "type": "function",
        "function": {
            "name": "session_search",
            "arguments": json.dumps(
                {"query": "omega-unique-search-token", "limit": 5}
            ),
        },
    }
    main = _ScriptClient(
        [
            CompletionResult(content="search", model="main", tool_calls=[tool_call]),
            CompletionResult(content="found it", model="main"),
        ],
        name="main",
    )
    s = build_session(
        tmp_path,
        main_client=main,
        audit_client=_ScriptClient(
            [CompletionResult(content="ok", model="audit")], name="audit"
        ),
        enable_audit=False,
        enable_graph=False,
        enable_policy=False,
        memory_dir=home / "memories",
        session_db=tmp_path / "sess.db",
    )
    try:
        store = s.extensions.session_store
        assert store is not None
        store.ensure_session("prior", cwd=str(tmp_path.resolve()), title="prior")
        store.append_message(
            "prior", "user", "we set omega-unique-search-token for sessions"
        )

        r = s.handle_prompt("find the token")
        assert r.status.value == "completed"
        assert "session_search" in (r.tools_called or [])
        # Tool observation in live history must be discovery with a hit
        live = s.extensions.turn_runner.live_messages
        tool_msgs = [
            m
            for m in live
            if (m.role is Role.TOOL)
            or (getattr(m.role, "value", None) == "tool")
        ]
        assert tool_msgs
        payload = json.loads(tool_msgs[0].content or "{}")
        assert payload.get("shape") == "discovery"
        assert any(
            "omega-unique-search-token" in (hit.get("snippet") or "")
            or hit.get("session_id") == "prior"
            for hit in payload.get("results") or []
        )
    finally:
        s.close()


def test_p1_custom_tools_still_get_provider_inject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attack: custom tools= finalized without notes still injects when mm present."""
    bare = ToolRegistryBuilder.new().finalize()
    assert "notes_append" not in bare.client_names()
    s, _main, home = _notes_session(tmp_path, monkeypatch, tools=bare)
    try:
        tools = s.extensions.tools
        assert tools is bare  # same object
        assert "notes_append" in tools.client_names()
        kernel = s.extensions.kernel
        kernel.refresh_tool_extra()
        ctx = ToolCallContext(
            cwd=tmp_path,
            session_id=str(s.id),
            extra=dict(kernel.tool_extra),
        )
        out = tools.call(
            "notes_append", {"content": "custom-toolset-inject"}, ctx
        )
        assert json.loads(out).get("success") is True
        body = (home / "memories" / "notes.md").read_text(encoding="utf-8")
        assert "custom-toolset-inject" in body
    finally:
        s.close()


def test_p1_refresh_tool_extra_preserves_mcp_hooks(tmp_path: Path) -> None:
    """Attack: host MCP hooks must survive kernel.refresh_tool_extra."""
    from codedoggy.session.kernel import RuntimeKernel

    def _dispatch(name: str, inp: dict) -> str:
        return json.dumps({"ok": True, "name": name})

    k = RuntimeKernel(cwd=tmp_path, session_id="s-mcp")
    k.tool_extra["mcp_dispatch"] = _dispatch
    k.tool_extra["mcp_tools"] = [{"name": "demo__ping", "parameters": {}}]
    k.refresh_tool_extra()
    assert k.tool_extra.get("mcp_dispatch") is _dispatch
    assert k.tool_extra.get("mcp_tools") == [{"name": "demo__ping", "parameters": {}}]
    # Managed keys still present
    assert k.tool_extra.get("kernel") is k
    assert k.tool_extra.get("task_manager") is not None


def test_p1_runner_merges_kernel_tool_extra_into_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attack: AgentTurnRunner must pass memory_manager through tool_extra."""
    captured: dict = {}

    def _spy_loop(**kwargs):
        captured["tool_extra"] = dict(kwargs.get("tool_extra") or {})
        # Minimal completed loop result shape
        from codedoggy.turn.types import LoopResult

        user = Message(role=Role.USER, content=kwargs.get("user_text") or "")
        return LoopResult(
            messages=[user, Message(role=Role.ASSISTANT, content="ok")],
            final_text="ok",
            tools_called=[],
            rounds=1,
            completed=True,
            error=None,
            cancelled=False,
            aborted=False,
            max_turns_reached=False,
            metadata={},
        )

    monkeypatch.setattr("codedoggy.turn.runner.run_agent_loop", _spy_loop)

    s, _main, _home = _notes_session(tmp_path, monkeypatch)
    try:
        r = s.handle_prompt("hi")
        assert r.status.value == "completed"
        te = captured.get("tool_extra") or {}
        assert te.get("memory_manager") is s.extensions.memory_manager
        assert te.get("session_store") is s.extensions.session_store
        assert te.get("memory_store") is s.extensions.memory
        assert te.get("policy") is s.extensions.policy
    finally:
        s.close()
