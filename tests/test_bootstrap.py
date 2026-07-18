"""Dual model-brain session bootstrap."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.bootstrap import build_session
from codedoggy.model import ModelConfig, ModelProfiles, CompletionResult, ChatMessage
from codedoggy.model.profiles import model_profiles_from_env
from codedoggy.turn.types import Role


class ScriptClient:
    """Deterministic chat client for bootstrap tests."""

    def __init__(self, script: list[CompletionResult], *, name: str = "script") -> None:
        self.script = list(script)
        self.n = 0
        self.name = name
        self.config = ModelConfig(
            provider="fake", model=name, base_url="http://fake", api_key="x"
        )
        self.calls: list[list] = []

    def complete(self, messages, **kwargs):
        self.calls.append(messages)
        if self.n >= len(self.script):
            return CompletionResult(content="(exhausted)", model=self.name)
        out = self.script[self.n]
        self.n += 1
        return out


def test_profiles_audit_falls_back_to_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEDOGGY_PROVIDER", "ollama")
    monkeypatch.setenv("CODEDOGGY_MODEL", "qwen3:8b")
    monkeypatch.setenv("CODEDOGGY_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.delenv("CODEDOGGY_AUDIT_MODEL", raising=False)
    monkeypatch.delenv("CODEDOGGY_AUX_MODEL", raising=False)
    prof = model_profiles_from_env()
    assert prof.main.model == "qwen3:8b"
    assert prof.audit.model == "qwen3:8b"


def test_profiles_audit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEDOGGY_MODEL", "main-model")
    monkeypatch.setenv("CODEDOGGY_AUDIT_MODEL", "audit-model")
    monkeypatch.setenv("CODEDOGGY_PROVIDER", "ollama")
    monkeypatch.setenv("CODEDOGGY_BASE_URL", "http://127.0.0.1:11434/v1")
    prof = model_profiles_from_env()
    assert prof.main.model == "main-model"
    assert prof.audit.model == "audit-model"


def test_build_session_wires_main_and_audit(tmp_path: Path) -> None:
    # Main: no tools, final answer only
    main = ScriptClient(
        [CompletionResult(content="hello world", model="main")],
        name="main",
    )
    # Audit unused if no mutation
    audit = ScriptClient(
        [CompletionResult(content='{"ok": true}', model="audit")],
        name="audit",
    )
    s = build_session(
        tmp_path,
        goal="say hi",
        max_turns=3,
        main_client=main,
        audit_client=audit,
        enable_memory=False,
        enable_audit=True,  # legacy path only
    )
    try:
        assert s.goal == "say hi"
        assert s.extensions.turn_runner is not None
        assert s.extensions.audit is not None
        r = s.handle_prompt("hi")
        assert r.status.value == "completed"
        assert r.final_text == "hello world"
        assert main.n == 1
    finally:
        s.close()


def test_build_session_default_no_audit(tmp_path: Path) -> None:
    """Product path: Shadow/audit is off by default."""
    main = ScriptClient(
        [CompletionResult(content="ok", model="main")],
        name="main",
    )
    s = build_session(tmp_path, main_client=main, enable_memory=False)
    try:
        assert s.extensions.audit is None
        prompt = s.extensions.turn_runner.system_prompt or ""
        assert "Shadow" not in prompt
        assert "shadow P0" not in prompt
        assert "parallel" in prompt.lower() or "parallel_tasks" in prompt
        assert "MAIN" in prompt
        # Agency: tendency in prompt, not harness auto-orchestrate
        assert "does **not**" in prompt or "does not" in prompt.lower()
        assert "auto" in prompt.lower()
    finally:
        s.close()


def test_build_session_important_deferred_to_turn_end(tmp_path: Path) -> None:
    """important is not a mid-turn red card — flushed at turn end."""
    import json

    tool_call = {
        "id": "c1",
        "type": "function",
        "function": {
            "name": "search_replace",
            "arguments": json.dumps(
                {
                    "file_path": "note.txt",
                    "old_string": "",
                    "new_string": "hello",
                }
            ),
        },
    }
    main = ScriptClient(
        [
            CompletionResult(content="writing", model="main", tool_calls=[tool_call]),
            CompletionResult(content="done", model="main"),
        ],
        name="main",
    )
    audit = ScriptClient(
        [
            CompletionResult(
                content=json.dumps(
                    {
                        "ok": False,
                        "findings": [
                            {"severity": "important", "message": "off goal rethink"}
                        ],
                    }
                ),
                model="audit",
            )
        ],
        name="audit",
    )
    s = build_session(
        tmp_path,
        goal="only edit README",
        max_turns=5,
        main_client=main,
        audit_client=audit,
        enable_memory=False,
        enable_audit=True,  # legacy audit package tests only
    )
    try:
        r = s.handle_prompt("create note")
        assert r.status.value == "completed"
        assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "hello"
        assert len(s.extensions.audit.trajectory) == 1
        assert audit.n == 1
        # Mid-turn tool observation must NOT carry non-P0 finding text
        second_batch = main.calls[1]
        from codedoggy.model.types import ChatMessage as CM

        tool_texts = []
        for m in second_batch:
            role = m.role if isinstance(m, CM) else (m.get("role") if isinstance(m, dict) else None)
            if str(role) == "tool":
                content = m.content if isinstance(m, CM) else m.get("content")
                tool_texts.append(content or "")
        joined = "\n".join(tool_texts)
        assert "off goal rethink" not in joined
        assert "shadow P0" not in joined
        # End-of-turn summary for host / final_text
        assert "shadow_deferred" in r.metadata
        assert "off goal rethink" in r.metadata["shadow_deferred"]
        assert "off goal rethink" in (r.final_text or "")
    finally:
        s.close()


def test_p0_critical_is_immediate_red_card(tmp_path: Path) -> None:
    import json

    tool_call = {
        "id": "c1",
        "type": "function",
        "function": {
            "name": "search_replace",
            "arguments": json.dumps(
                {
                    "file_path": "secrets.env",
                    "old_string": "",
                    "new_string": "KEY=1",
                }
            ),
        },
    }
    main = ScriptClient(
        [
            CompletionResult(content="writing", model="main", tool_calls=[tool_call]),
            CompletionResult(content="done", model="main"),
        ],
        name="main",
    )
    audit = ScriptClient(
        [
            CompletionResult(
                content=json.dumps(
                    {
                        "ok": False,
                        "findings": [
                            {
                                "severity": "critical",
                                "message": "do not write secrets",
                                "path": "secrets.env",
                            },
                            {
                                "severity": "suggestion",
                                "message": "prefer .env.example",
                            },
                        ],
                    }
                ),
                model="audit",
            )
        ],
        name="audit",
    )
    s = build_session(
        tmp_path,
        goal="add docs only",
        max_turns=5,
        main_client=main,
        audit_client=audit,
        enable_memory=False,
        enable_audit=True,  # legacy audit package tests only
    )
    try:
        r = s.handle_prompt("write secrets")
        # P0 aborts the turn (writes paused / remaining tools cancelled)
        assert r.status.value in {"completed", "error"}
        # P0 text on tool observation and/or error path
        blob = (r.final_text or "") + str(r.error or "") + str(r.metadata or "")
        # First sample's tool result should have been archived into next call if any
        all_texts: list[str] = [blob]
        for batch in main.calls:
            for m in batch:
                c = m.content if isinstance(m, ChatMessage) else m.get("content")
                all_texts.append(str(c or ""))
        joined = "\n".join(all_texts)
        assert "P0" in joined or "do not write secrets" in joined or "critical" in joined.lower()
    finally:
        s.close()
