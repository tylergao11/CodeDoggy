"""Unified ActiveConnection / ConnectionService truth."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from codedoggy.model.connection import (
    ActiveConnection,
    ConnectionService,
    connection_from_config,
    connection_of,
)
from codedoggy.model.types import ModelConfig
from codedoggy.session.extensions import SessionExtensions
from codedoggy.tui import surface as session_surface


def _cfg(**kwargs: object) -> ModelConfig:
    base = dict(
        provider="ollama",
        model="qwen3:8b",
        base_url="http://127.0.0.1:11434/v1",
        temperature=0.2,
        context_window=32768,
        timeout_s=120.0,
    )
    base.update(kwargs)
    return ModelConfig(**base)  # type: ignore[arg-type]


def test_connection_from_config_snapshot_fields() -> None:
    snap = connection_from_config(_cfg(), source="bootstrap")
    assert isinstance(snap, ActiveConnection)
    assert snap.provider == "ollama"
    assert snap.model == "qwen3:8b"
    assert snap.ready_to_sample is True
    assert snap.label == "ollama/qwen3:8b"
    assert snap.generation == 0
    # Missing extra → product default high
    assert snap.reasoning_enabled is True
    assert snap.reasoning_effort == "high"
    assert snap.reasoning_label == "推理:high"


def test_connection_reads_reasoning_from_config_extra() -> None:
    snap = connection_from_config(
        _cfg(extra={"reasoning": {"enabled": True, "effort": "medium"}}),
    )
    assert snap.reasoning_effort == "medium"
    assert snap.reasoning_label == "推理:medium"
    off = connection_from_config(
        _cfg(extra={"reasoning": {"enabled": False}}),
    )
    assert off.reasoning_enabled is False
    assert off.reasoning_label == "推理:off"


def test_bootstrap_service_snapshot_stable() -> None:
    client = SimpleNamespace(config=_cfg())
    runner = SimpleNamespace(sampler=SimpleNamespace(client=client, stream=False), system_prompt="Model: x\nProvider: y\n")
    svc = ConnectionService.bootstrap(_cfg(), client=client, runner=runner)
    a = svc.snapshot()
    b = svc.snapshot()
    assert a == b
    assert a.model == "qwen3:8b"
    assert svc.client() is client


def test_apply_increments_generation_and_swaps_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEDOGGY_PROVIDER", "ollama")
    monkeypatch.setenv("CODEDOGGY_MODEL", "qwen3:8b")
    monkeypatch.setenv("CODEDOGGY_BASE_URL", "http://127.0.0.1:11434/v1")

    created: list[ModelConfig] = []

    class _FakeClient:
        def __init__(self, config: ModelConfig) -> None:
            self.config = config
            created.append(config)

    def _fake_create(config: ModelConfig, *, require_auth: bool | None = None) -> _FakeClient:
        return _FakeClient(config)

    monkeypatch.setattr(
        "codedoggy.model.connection.create_client",
        _fake_create,
    )

    old_client = _FakeClient(_cfg(model="old"))
    runner = SimpleNamespace(
        sampler=SimpleNamespace(client=old_client, stream=True, on_delta=None),
        system_prompt="Model: old\nProvider: ollama\n",
    )
    svc = ConnectionService.bootstrap(_cfg(model="old"), client=old_client, runner=runner)
    assert svc.snapshot().generation == 0

    snap = svc.apply(provider="ollama", model="new-model", require_auth=False, source="panel")
    assert snap.generation == 1
    assert snap.model == "new-model"
    assert snap.provider == "ollama"
    assert snap.source == "panel"
    assert created[-1].model == "new-model"
    assert runner.sampler.client.config.model == "new-model"
    assert "Model: new-model" in runner.system_prompt
    assert "Provider: ollama" in runner.system_prompt


def test_connection_of_reads_extensions() -> None:
    svc = ConnectionService.bootstrap(_cfg())
    session = SimpleNamespace(extensions=SessionExtensions(connection=svc))
    assert connection_of(session) is svc
    assert session_surface.provider_id(session) == "ollama"
    assert session_surface.model_id(session) == "qwen3:8b"
    caption = session_surface.model_and_mode_text(session)
    assert "qwen3:8b" in caption
    assert "推理:high" in caption
    assert "auto" in caption


def test_hud_projection_uses_connection_not_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEDOGGY_PROVIDER", "grok")
    monkeypatch.setenv("CODEDOGGY_MODEL", "grok-3")
    svc = ConnectionService.bootstrap(
        _cfg(provider="ollama", model="local-model"),
    )
    session = SimpleNamespace(extensions=SessionExtensions(connection=svc))
    hud = session_surface.hud_projection(session)
    assert hud["provider"] == "ollama"
    assert hud["model"] == "local-model"
    assert hud["label"] == "ollama/local-model"


def test_suggested_models_include_default() -> None:
    from codedoggy.model.catalog import suggested_models
    from codedoggy.model.profile_registry import get_profile

    assert get_profile("grok").default_model == "grok-4.5"
    assert get_profile("claude").default_model == "claude-opus-4-5"
    assert get_profile("codex").default_model == "gpt-5.6-sol"
    assert get_profile("deepseek").default_model == "deepseek-reasoner"
    assert get_profile("openai").default_model == "gpt-5.6-sol"

    grok = suggested_models("grok")
    assert grok[0] == "grok-4.5"
    assert "grok-3" in grok
    ollama = suggested_models("ollama")
    assert "qwen3:8b" in ollama


def test_reasoning_defaults_to_high(monkeypatch: pytest.MonkeyPatch) -> None:
    from codedoggy.model.registry import model_config_from_env

    monkeypatch.delenv("CODEDOGGY_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("CODEDOGGY_REASONING_ENABLED", raising=False)
    monkeypatch.setenv("CODEDOGGY_PROVIDER", "ollama")
    cfg = model_config_from_env()
    assert cfg.extra.get("reasoning", {}).get("effort") == "high"
    assert cfg.extra.get("reasoning", {}).get("enabled") is True


def test_apply_sets_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEDOGGY_PROVIDER", "ollama")
    monkeypatch.setenv("CODEDOGGY_MODEL", "qwen3:8b")
    monkeypatch.setenv("CODEDOGGY_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.delenv("CODEDOGGY_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("CODEDOGGY_REASONING_ENABLED", raising=False)

    class _Client:
        def __init__(self, config: ModelConfig) -> None:
            self.config = config

    def _fake_create(cfg: ModelConfig, **_k: object) -> _Client:
        return _Client(cfg)

    monkeypatch.setattr(
        "codedoggy.model.connection.create_client",
        _fake_create,
    )
    svc = ConnectionService.bootstrap(_cfg())
    snap = svc.apply(
        provider="ollama",
        model="qwen3:8b",
        reasoning_effort="medium",
        reasoning_enabled=True,
        require_auth=False,
    )
    assert snap.reasoning_effort == "medium"
    assert snap.reasoning_label == "推理:medium"
    assert snap.reasoning_enabled is True
    import os

    assert os.environ.get("CODEDOGGY_REASONING_EFFORT") == "medium"
    # refresh_auth must not drop reasoning fields
    refreshed = svc.refresh_auth()
    assert refreshed.reasoning_effort == "medium"
    assert refreshed.reasoning_enabled is True


def test_apply_failure_does_not_publish_reasoning_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEDOGGY_PROVIDER", "ollama")
    monkeypatch.setenv("CODEDOGGY_MODEL", "qwen3:8b")
    monkeypatch.setenv("CODEDOGGY_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("CODEDOGGY_REASONING_EFFORT", "high")
    monkeypatch.setenv("CODEDOGGY_REASONING_ENABLED", "1")

    def _boom(cfg: ModelConfig, **_k: object) -> None:
        raise RuntimeError("no client")

    monkeypatch.setattr("codedoggy.model.connection.create_client", _boom)
    svc = ConnectionService.bootstrap(
        _cfg(extra={"reasoning": {"enabled": True, "effort": "high"}})
    )
    import os

    with pytest.raises(RuntimeError, match="no client"):
        svc.apply(
            provider="ollama",
            model="qwen3:8b",
            reasoning_effort="low",
            reasoning_enabled=True,
            require_auth=False,
        )
    # Failed apply must not leave the new effort in process env.
    assert os.environ.get("CODEDOGGY_REASONING_EFFORT") == "high"
    assert svc.snapshot().reasoning_effort == "high"


def test_apply_model_keeps_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEDOGGY_PROVIDER", "ollama")
    monkeypatch.setenv("CODEDOGGY_MODEL", "qwen3:8b")
    monkeypatch.setenv("CODEDOGGY_BASE_URL", "http://127.0.0.1:11434/v1")

    class _FakeClient:
        def __init__(self, config: ModelConfig) -> None:
            self.config = config

    monkeypatch.setattr(
        "codedoggy.model.connection.create_client",
        lambda config, require_auth=None: _FakeClient(config),
    )
    client = _FakeClient(_cfg())
    runner = SimpleNamespace(
        sampler=SimpleNamespace(client=client, stream=False, on_delta=None),
        system_prompt="Model: qwen3:8b\nProvider: ollama\n",
    )
    svc = ConnectionService.bootstrap(_cfg(), client=client, runner=runner)
    snap = svc.apply(model="llama3.2", require_auth=False, source="panel")
    assert snap.provider == "ollama"
    assert snap.model == "llama3.2"
    assert runner.sampler.client.config.model == "llama3.2"
    assert session_surface.model_id(
        SimpleNamespace(extensions=SessionExtensions(connection=svc))
    ) == "llama3.2"


def test_build_session_attaches_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEDOGGY_PROVIDER", "ollama")
    monkeypatch.setenv("CODEDOGGY_MODEL", "qwen3:8b")
    monkeypatch.setenv("CODEDOGGY_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("CODEDOGGY_MCP", "0")
    from codedoggy.bootstrap import build_session

    session = build_session(
        tmp_path,
        enable_memory=False,
        enable_session_store=False,
        enable_graph=False,
        enable_mcp=False,
    )
    try:
        svc = connection_of(session)
        assert svc is not None
        snap = svc.snapshot()
        assert snap.provider == "ollama"
        client = getattr(
            getattr(session.extensions.turn_runner, "sampler", None), "client", None
        )
        assert client is not None
        assert client.config.model == snap.model
        assert session_surface.model_id(session) == snap.model
    finally:
        session.close()
