"""Session bootstrap + model profiles."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.bootstrap import build_session
from codedoggy.model import ModelConfig, CompletionResult
from codedoggy.model.profiles import model_profiles_from_env


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


def test_profiles_aux_falls_back_to_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEDOGGY_PROVIDER", "ollama")
    monkeypatch.setenv("CODEDOGGY_MODEL", "qwen3:8b")
    monkeypatch.setenv("CODEDOGGY_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.delenv("CODEDOGGY_AUX_MODEL", raising=False)
    prof = model_profiles_from_env()
    assert prof.main.model == "qwen3:8b"
    assert prof.aux.model == "qwen3:8b"


def test_profiles_aux_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEDOGGY_MODEL", "main-model")
    monkeypatch.setenv("CODEDOGGY_AUX_MODEL", "aux-model")
    monkeypatch.setenv("CODEDOGGY_PROVIDER", "ollama")
    monkeypatch.setenv("CODEDOGGY_BASE_URL", "http://127.0.0.1:11434/v1")
    prof = model_profiles_from_env()
    assert prof.main.model == "main-model"
    assert prof.aux.model == "aux-model"


def test_build_session_wires_main(tmp_path: Path) -> None:
    main = ScriptClient(
        [CompletionResult(content="hello world", model="main")],
        name="main",
    )
    s = build_session(
        tmp_path,
        goal="say hi",
        max_turns=3,
        main_client=main,
        enable_memory=False,
    )
    try:
        assert s.goal == "say hi"
        assert s.extensions.turn_runner is not None
        assert not hasattr(s.extensions, "audit") or getattr(s.extensions, "audit", None) is None
        r = s.handle_prompt("hi")
        assert r.status.value == "completed"
        assert r.final_text == "hello world"
        assert main.n == 1
    finally:
        s.close()


def test_build_session_product_prompt_has_parallel_bias(tmp_path: Path) -> None:
    main = ScriptClient(
        [CompletionResult(content="ok", model="main")],
        name="main",
    )
    s = build_session(tmp_path, main_client=main, enable_memory=False)
    try:
        prompt = s.extensions.turn_runner.system_prompt or ""
        assert "shadow" not in prompt.lower()
        assert "parallel" in prompt.lower() or "parallel_tasks" in prompt
        assert "MAIN" in prompt
        assert "does **not**" in prompt or "does not" in prompt.lower()
        assert "auto" in prompt.lower()
    finally:
        s.close()
