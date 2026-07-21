"""Startup provider preference — auth success must not leave MAIN on ollama."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from codedoggy.model.preferred_provider import (
    load_preferred_provider,
    resolve_startup_provider,
    save_preferred_provider,
)
from codedoggy.model.registry import model_config_from_env


def test_save_load_preferred_provider(tmp_path: Path) -> None:
    save_preferred_provider("grok", home=tmp_path)
    assert load_preferred_provider(home=tmp_path) == "grok"


def test_resolve_startup_prefers_remembered_when_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_preferred_provider("claude", home=tmp_path)
    monkeypatch.setattr(
        "codedoggy.model.preferred_provider.preferred_provider_path",
        lambda home=None: tmp_path / "active_provider",
    )
    monkeypatch.setattr(
        "codedoggy.model.preferred_provider.provider_usable",
        lambda name: name == "claude",
    )
    assert resolve_startup_provider() == "claude"


def test_resolve_startup_scans_imperial_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "codedoggy.model.preferred_provider.preferred_provider_path",
        lambda home=None: tmp_path / "missing",
    )
    monkeypatch.setattr(
        "codedoggy.model.preferred_provider.provider_usable",
        lambda name: name == "grok",
    )
    assert resolve_startup_provider() == "grok"


def test_model_config_from_env_prefers_logged_in_grok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CODEDOGGY_PROVIDER", raising=False)
    monkeypatch.delenv("CODEDOGGY_MODEL", raising=False)
    monkeypatch.delenv("CODEDOGGY_BASE_URL", raising=False)
    monkeypatch.setenv("CODEDOGGY_HOME", str(tmp_path))
    monkeypatch.setattr(
        "codedoggy.model.preferred_provider.provider_usable",
        lambda name: name == "grok",
    )
    cfg = model_config_from_env()
    assert cfg.provider == "grok"
    assert "11434" not in (cfg.base_url or "")


def test_model_config_from_env_strips_stale_ollama_base_for_imperial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEDOGGY_PROVIDER", "grok")
    monkeypatch.delenv("CODEDOGGY_MODEL", raising=False)
    monkeypatch.setenv("CODEDOGGY_BASE_URL", "http://127.0.0.1:11434/v1")
    cfg = model_config_from_env()
    assert cfg.provider == "grok"
    assert "11434" not in (cfg.base_url or "")


def test_apply_saves_preferred_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codedoggy.model.connection import ConnectionService
    from codedoggy.model.types import ModelConfig

    monkeypatch.setenv("CODEDOGGY_HOME", str(tmp_path))
    monkeypatch.setenv("CODEDOGGY_PROVIDER", "ollama")
    monkeypatch.setenv("CODEDOGGY_MODEL", "qwen3:8b")
    monkeypatch.setenv("CODEDOGGY_BASE_URL", "http://127.0.0.1:11434/v1")

    class _FakeClient:
        def __init__(self, config: ModelConfig) -> None:
            self.config = config

    def _create(cfg: ModelConfig, **kwargs: object) -> _FakeClient:
        return _FakeClient(cfg)

    monkeypatch.setattr(
        "codedoggy.model.connection.create_client",
        _create,
    )
    monkeypatch.setattr(
        "codedoggy.model.connection.auth_status",
        lambda _p: SimpleNamespace(logged_in=True, kind="oauth", source="file", detail="ok"),
    )
    monkeypatch.setattr(
        "codedoggy.model.connection.auth_kind_for_provider",
        lambda _p: "oauth",
    )

    client = _FakeClient(
        ModelConfig(
            provider="ollama",
            model="qwen3:8b",
            base_url="http://127.0.0.1:11434/v1",
        )
    )
    runner = SimpleNamespace(
        sampler=SimpleNamespace(client=client, stream=False, on_delta=None),
        system_prompt="Model: x\n",
    )
    svc = ConnectionService.bootstrap(client.config, client=client, runner=runner)
    snap = svc.apply(provider="grok", model="grok-4.5", require_auth=False)
    assert snap.provider == "grok"
    assert load_preferred_provider(home=tmp_path) == "grok"
