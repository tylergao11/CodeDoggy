"""Context window must follow the connected model, not a blind 32k default."""

from __future__ import annotations

import pytest

from codedoggy.model.context_limits import (
    DEFAULT_CONTEXT_WINDOW,
    lookup_model_context,
    resolve_context_window,
)
from codedoggy.model.registry import model_config_from_env


def test_lookup_flagship_models() -> None:
    assert lookup_model_context("grok-4.5") == 256_000
    assert lookup_model_context("claude-opus-4-5") == 200_000
    assert lookup_model_context("gpt-5.6-sol") == 256_000
    assert lookup_model_context("deepseek-reasoner") == 128_000
    assert lookup_model_context("qwen3:8b") == 131_072


def test_resolve_prefers_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEDOGGY_CONTEXT_WINDOW", raising=False)
    monkeypatch.delenv("CODEDOGGY_CONTEXT_MAX_TOKENS", raising=False)
    assert (
        resolve_context_window("grok", "grok-4.5", explicit=99_000, probe=False)
        == 99_000
    )


def test_resolve_uses_model_not_default_32k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEDOGGY_CONTEXT_WINDOW", raising=False)
    monkeypatch.delenv("CODEDOGGY_CONTEXT_MAX_TOKENS", raising=False)
    win = resolve_context_window("grok", "grok-4.5", probe=False)
    assert win == 256_000
    assert win != DEFAULT_CONTEXT_WINDOW


def test_model_config_from_env_follows_provider_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEDOGGY_CONTEXT_WINDOW", raising=False)
    monkeypatch.delenv("CODEDOGGY_CONTEXT_MAX_TOKENS", raising=False)
    monkeypatch.setenv("CODEDOGGY_PROVIDER", "grok")
    monkeypatch.setenv("CODEDOGGY_MODEL", "grok-4.5")
    cfg = model_config_from_env(provider="grok", model="grok-4.5")
    assert cfg.context_window == 256_000


def test_model_config_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEDOGGY_CONTEXT_WINDOW", raising=False)
    monkeypatch.delenv("CODEDOGGY_CONTEXT_MAX_TOKENS", raising=False)
    cfg = model_config_from_env(provider="claude", model="claude-opus-4-5")
    assert cfg.context_window == 200_000


def test_env_still_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEDOGGY_CONTEXT_WINDOW", "50000")
    cfg = model_config_from_env(provider="grok", model="grok-4.5")
    assert cfg.context_window == 50_000


def test_ensure_rewrites_stale_32k(monkeypatch: pytest.MonkeyPatch) -> None:
    from codedoggy.model.context_limits import ensure_model_context_window
    from codedoggy.model.types import ModelConfig

    monkeypatch.delenv("CODEDOGGY_CONTEXT_WINDOW", raising=False)
    monkeypatch.delenv("CODEDOGGY_CONTEXT_MAX_TOKENS", raising=False)
    stale = ModelConfig(
        provider="grok",
        model="grok-4.5",
        base_url="https://api.x.ai/v1",
        context_window=32_768,
    )
    fixed = ensure_model_context_window(stale)
    assert fixed.context_window == 256_000


def test_connection_apply_rewrites_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from codedoggy.context.compactor import ContextCompactor
    from codedoggy.model.connection import ConnectionService

    monkeypatch.delenv("CODEDOGGY_CONTEXT_WINDOW", raising=False)
    monkeypatch.delenv("CODEDOGGY_CONTEXT_MAX_TOKENS", raising=False)
    cfg = model_config_from_env(provider="claude", model="claude-opus-4-5")
    comp = ContextCompactor.from_env(
        provider="claude",
        model="claude-opus-4-5",
        context_window=cfg.context_window,
    )
    svc = ConnectionService.bootstrap(cfg)
    svc.bind_runtime(context=comp)
    snap = svc.apply(provider="grok", model="grok-4.5", require_auth=False)
    assert snap.context_window == 256_000
    assert comp.budget.context_window == 256_000
