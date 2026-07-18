"""Provider registry — register by name, create clients from ModelConfig.

Inspired by Hermes provider profiles + Grok SamplerConfig construction:
config is pure data; factories produce clients. Only one factory per name.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from codedoggy.model.openai_compat import OpenAICompatClient
from codedoggy.model.provider import ChatClient
from codedoggy.model.types import ModelConfig

logger = logging.getLogger(__name__)

ClientFactory = Callable[[ModelConfig], ChatClient]

_REGISTRY: dict[str, ClientFactory] = {}


def register_provider(name: str, factory: ClientFactory, *, replace: bool = False) -> None:
    """Register a provider factory under ``name`` (e.g. ``ollama``)."""
    key = name.strip().lower()
    if not key:
        raise ValueError("provider name must be non-empty")
    if key in _REGISTRY and not replace:
        raise ValueError(f"provider already registered: {key}")
    _REGISTRY[key] = factory
    logger.debug("registered model provider %s", key)


def unregister_provider(name: str) -> None:
    _REGISTRY.pop(name.strip().lower(), None)


def list_providers() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_factory(name: str) -> ClientFactory | None:
    return _REGISTRY.get(name.strip().lower())


def create_client(config: ModelConfig) -> ChatClient:
    """Build a ChatClient for ``config.provider`` (fallback: openai_compat)."""
    key = (config.provider or "openai_compat").strip().lower()
    factory = _REGISTRY.get(key)
    if factory is None:
        if key not in {"openai_compat", "openai", "custom"}:
            logger.warning("unknown provider %r — using openai_compat transport", key)
        factory = _REGISTRY.get("openai_compat")
    if factory is None:
        # Always available default.
        return OpenAICompatClient(config)
    return factory(config)


def model_config_from_env(
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ModelConfig:
    """Resolve config from args + environment (Ollama-friendly defaults)."""
    prov = (
        provider
        or os.environ.get("CODEDOGGY_PROVIDER")
        or "ollama"
    ).strip().lower()
    if prov == "ollama":
        default_base = "http://127.0.0.1:11434/v1"
        default_model = "qwen3:8b"
        default_key = os.environ.get("OLLAMA_API_KEY") or "ollama"
    else:
        default_base = "http://127.0.0.1:11434/v1"
        default_model = "gpt-4o-mini"
        default_key = os.environ.get("OPENAI_API_KEY") or ""

    return ModelConfig(
        provider=prov,
        model=(model or os.environ.get("CODEDOGGY_MODEL") or default_model).strip(),
        base_url=(
            base_url
            or os.environ.get("CODEDOGGY_BASE_URL")
            or os.environ.get("OLLAMA_HOST")
            or default_base
        ).strip(),
        api_key=(
            api_key
            if api_key is not None
            else (os.environ.get("CODEDOGGY_API_KEY") or default_key or None)
        ),
        temperature=_env_float("CODEDOGGY_TEMPERATURE", 0.2),
        max_tokens=_env_int("CODEDOGGY_MAX_TOKENS", None),
        timeout_s=_env_float("CODEDOGGY_TIMEOUT_S", 120.0) or 120.0,
        context_window=_env_int("CODEDOGGY_CONTEXT_WINDOW", None)
        or _env_int("CODEDOGGY_CONTEXT_MAX_TOKENS", 32768),
    )


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _ollama_factory(config: ModelConfig) -> ChatClient:
    # Normalize host-only URLs to OpenAI-compatible /v1 root.
    base = config.base_url.rstrip("/")
    if base.endswith(":11434"):
        base = base + "/v1"
    elif "/v1" not in base and "11434" in base and not base.endswith("/v1"):
        # e.g. http://127.0.0.1:11434
        if base.count("/") <= 2 or base.rstrip("/").endswith("11434"):
            base = base.rstrip("/") + "/v1"
    cfg = ModelConfig(
        provider="ollama",
        model=config.model,
        base_url=base,
        api_key=config.api_key or "ollama",
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
        extra_headers=dict(config.extra_headers),
        extra=dict(config.extra),
    )
    return OpenAICompatClient(cfg)


def _openai_compat_factory(config: ModelConfig) -> ChatClient:
    return OpenAICompatClient(config)


def register_builtin_providers() -> None:
    """Idempotent registration of stock providers."""
    if "openai_compat" not in _REGISTRY:
        register_provider("openai_compat", _openai_compat_factory)
    if "openai" not in _REGISTRY:
        register_provider("openai", _openai_compat_factory)
    if "custom" not in _REGISTRY:
        register_provider("custom", _openai_compat_factory)
    if "ollama" not in _REGISTRY:
        register_provider("ollama", _ollama_factory)


# Register on import so create_client always works.
register_builtin_providers()
