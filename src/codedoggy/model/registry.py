"""Provider registry — auth → profile → transport (OpenAI | Anthropic)."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from codedoggy.model.anthropic_messages import AnthropicMessagesClient
from codedoggy.model.bedrock import BedrockConverseClient
from codedoggy.model.codex_app_server import CodexAppServerClient
from codedoggy.model.codex_responses import CodexResponsesClient
from codedoggy.model.openai_compat import OpenAICompatClient
from codedoggy.model.profile import (
    API_ANTHROPIC_MESSAGES,
    API_CHAT_COMPLETIONS,
    API_CODEX_RESPONSES,
)
from codedoggy.model.vertex import VertexClient
from codedoggy.model.profile_registry import (
    get_profile,
    list_profiles,
    resolve_profile_name,
)
from codedoggy.model.provider import ChatClient
from codedoggy.model.types import ModelConfig

logger = logging.getLogger(__name__)

ClientFactory = Callable[[ModelConfig], ChatClient]

_REGISTRY: dict[str, ClientFactory] = {}


def register_provider(name: str, factory: ClientFactory, *, replace: bool = False) -> None:
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
    names = set(_REGISTRY.keys()) | set(list_profiles())
    return sorted(names)


def get_factory(name: str) -> ClientFactory | None:
    return _REGISTRY.get(name.strip().lower())


def create_client(config: ModelConfig, *, require_auth: bool | None = None) -> ChatClient:
    """Auth layer fills credentials, then api_mode picks transport.

    ``require_auth`` defaults to True for OAuth providers (Grok/Claude/Codex),
    False for api_key providers. Pass explicitly to override.
    """
    from codedoggy.model.auth.resolve import apply_auth_to_config
    from codedoggy.model.context_limits import ensure_model_context_window

    cfg = apply_auth_to_config(config, require=require_auth)
    # Always re-derive window from provider+model (never trust a stale 32k).
    cfg = ensure_model_context_window(cfg)
    key = (cfg.provider or "openai_compat").strip().lower()
    canon = resolve_profile_name(key) or key
    profile = get_profile(key) or get_profile(canon)

    factory = _REGISTRY.get(key) or _REGISTRY.get(canon or "")
    if factory is not None:
        return factory(cfg)

    # Profile-driven transport
    if profile is not None:
        mode = profile.api_mode
        if mode == API_ANTHROPIC_MESSAGES:
            return AnthropicMessagesClient(cfg, profile=profile)
        if mode == API_CODEX_RESPONSES:
            return CodexResponsesClient(cfg, profile=profile)
        if mode == "bedrock_converse":
            return BedrockConverseClient(cfg, profile=profile)
        if mode == "codex_app_server":
            return CodexAppServerClient(cfg, profile=profile)
        if profile.name in {"vertex", "vertex-ai", "google-vertex"}:
            return VertexClient(cfg, profile=profile)
        return OpenAICompatClient(cfg, profile=profile)

    if key not in {"openai_compat", "openai", "custom"}:
        logger.warning("unknown provider %r — using openai_compat transport", key)
    return OpenAICompatClient(cfg, profile=get_profile("openai_compat"))


def model_config_from_env(
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ModelConfig:
    """Resolve config from args + environment, using provider profiles + auth."""
    _ensure_profiles()

    env_provider = (os.environ.get("CODEDOGGY_PROVIDER") or "").strip().lower()
    explicit_provider = bool((provider or "").strip() or env_provider)
    if (provider or "").strip():
        prov = str(provider).strip().lower()
    elif env_provider:
        prov = env_provider
    else:
        from codedoggy.model.preferred_provider import resolve_startup_provider

        prov = resolve_startup_provider()
    profile = get_profile(prov)

    if profile is not None:
        default_base = profile.resolve_base_url(None) or profile.base_url
        default_model = profile.default_model or "gpt-4o-mini"
    elif prov == "ollama":
        default_base = "http://127.0.0.1:11434/v1"
        default_model = "qwen3:8b"
    else:
        default_base = "http://127.0.0.1:11434/v1"
        default_model = "gpt-4o-mini"

    # Auto-picked imperial must not inherit a leftover local Ollama BASE_URL.
    env_base = os.environ.get("CODEDOGGY_BASE_URL")
    if (
        not explicit_provider
        and prov != "ollama"
        and env_base
        and _looks_like_local_ollama_url(env_base)
    ):
        env_base = None

    resolved_base = (
        base_url
        or env_base
        or (profile.resolve_base_url(None) if profile else None)
        or (os.environ.get("OLLAMA_HOST") if prov == "ollama" else None)
        or default_base
        or ""
    )
    if prov == "ollama":
        resolved_base = _normalize_ollama_base(resolved_base)
    elif _looks_like_local_ollama_url(resolved_base):
        # Explicit grok/claude/codex + stale 11434 — fail closed to profile URL.
        resolved_base = str(
            (profile.resolve_base_url(None) if profile else None) or default_base or ""
        )

    env_model = os.environ.get("CODEDOGGY_MODEL")
    if (
        not explicit_provider
        and prov != "ollama"
        and env_model
        and _looks_like_ollama_model_tag(env_model)
    ):
        env_model = None

    resolved_model = (
        model
        or env_model
        or default_model
    ).strip()
    from codedoggy.model.context_limits import ensure_model_context_window

    # Leave api_key as explicit only; create_client / apply_auth fills OAuth.
    cfg = ModelConfig(
        provider=prov,
        model=resolved_model,
        base_url=str(resolved_base).strip(),
        api_key=api_key if api_key is not None else os.environ.get("CODEDOGGY_API_KEY"),
        temperature=_env_float("CODEDOGGY_TEMPERATURE", 0.2),
        max_tokens=_env_int("CODEDOGGY_MAX_TOKENS", None),
        timeout_s=_env_float("CODEDOGGY_TIMEOUT_S", 120.0) or 120.0,
        context_window=None,  # filled by ensure_model_context_window
        extra=_reasoning_extra_from_env(),
    )
    # Soft hydrate: fill tokens when present; do not raise LoginRequired here
    # (callers that need hard gate use create_client / apply_auth with require).
    try:
        from codedoggy.model.auth.resolve import apply_auth_to_config

        cfg = apply_auth_to_config(cfg, require=False)
    except Exception:  # noqa: BLE001
        logger.debug("auth hydrate skipped", exc_info=True)
    return ensure_model_context_window(cfg)


def _reasoning_extra_from_env() -> dict[str, Any]:
    """Product default: reasoning ON at maximum effort (``high``).

    Override with env:
      CODEDOGGY_REASONING_EFFORT=low|medium|high|xhigh
      CODEDOGGY_REASONING_ENABLED=0   # force off when supported
    """
    extra: dict[str, Any] = {}
    enabled_raw = os.environ.get("CODEDOGGY_REASONING_ENABLED")
    effort_raw = os.environ.get("CODEDOGGY_REASONING_EFFORT")

    # Default max when unset (user can lower via env).
    if enabled_raw is None and effort_raw is None:
        return {"reasoning": {"enabled": True, "effort": "high"}}

    rc: dict[str, Any] = {}
    if enabled_raw is not None:
        rc["enabled"] = enabled_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        rc["enabled"] = True
    if effort_raw is not None and effort_raw.strip():
        rc["effort"] = effort_raw.strip().lower()
    elif rc.get("enabled") is not False:
        rc["effort"] = "high"
    if rc.get("enabled") is False:
        extra["reasoning"] = {"enabled": False}
    else:
        extra["reasoning"] = rc
    return extra


def _normalize_ollama_base(base: str) -> str:
    b = (base or "").rstrip("/")
    if not b:
        return "http://127.0.0.1:11434/v1"
    if b.endswith(":11434"):
        return b + "/v1"
    if "/v1" not in b and "11434" in b and not b.endswith("/v1"):
        if b.count("/") <= 2 or b.rstrip("/").endswith("11434"):
            return b.rstrip("/") + "/v1"
    if "://" not in b and "11434" in b:
        return f"http://{b}/v1" if not b.endswith("/v1") else f"http://{b}"
    return b


def _looks_like_local_ollama_url(base: str | None) -> bool:
    raw = (base or "").strip().lower()
    if not raw:
        return False
    if ":11434" in raw:
        return True
    return "127.0.0.1" in raw and "ollama" in raw


def _looks_like_ollama_model_tag(model: str | None) -> bool:
    """Heuristic for leftover local tags (e.g. ``qwen3:8b``) on imperial providers."""
    name = (model or "").strip().lower()
    if not name or ":" not in name:
        return False
    # Imperial cloud ids rarely use ``name:size`` tags.
    size = name.rsplit(":", 1)[-1]
    return size.endswith("b") and any(ch.isdigit() for ch in size)


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


def _openai_factory(config: ModelConfig) -> ChatClient:
    return OpenAICompatClient(
        config, profile=get_profile(config.provider) or get_profile("openai_compat")
    )


def _anthropic_factory(config: ModelConfig) -> ChatClient:
    return AnthropicMessagesClient(config, profile=get_profile(config.provider) or get_profile("claude"))


def _codex_factory(config: ModelConfig) -> ChatClient:
    return CodexResponsesClient(
        config, profile=get_profile(config.provider) or get_profile("codex")
    )


def _bedrock_factory(config: ModelConfig) -> ChatClient:
    return BedrockConverseClient(
        config, profile=get_profile(config.provider) or get_profile("bedrock")
    )


def _vertex_factory(config: ModelConfig) -> ChatClient:
    return VertexClient(
        config, profile=get_profile(config.provider) or get_profile("vertex")
    )


def _codex_app_server_factory(config: ModelConfig) -> ChatClient:
    return CodexAppServerClient(
        config, profile=get_profile(config.provider) or get_profile("codex_app_server")
    )


def _ollama_factory(config: ModelConfig) -> ChatClient:
    base = _normalize_ollama_base(config.base_url)
    cfg = ModelConfig(
        provider="ollama",
        model=config.model,
        base_url=base,
        api_key=config.api_key or "ollama",
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
        context_window=config.context_window,
        extra_headers=dict(config.extra_headers),
        extra=dict(config.extra),
    )
    return OpenAICompatClient(cfg, profile=get_profile("ollama"))


def _ensure_profiles() -> None:
    from codedoggy.model.builtin_profiles import register_builtin_profiles

    register_builtin_profiles()
    # Ensure auth providers registered
    from codedoggy.model.auth import resolve as _auth_resolve

    _auth_resolve._bootstrap()  # noqa: SLF001


def register_builtin_providers() -> None:
    _ensure_profiles()
    factories: dict[str, ClientFactory] = {
        "openai_compat": _openai_factory,
        "openai": _openai_factory,
        "custom": _openai_factory,
        "ollama": _ollama_factory,
        "deepseek": _openai_factory,
        # OAuth session providers (Hermes: xai + codex → Responses)
        "grok": _codex_factory,
        "xai": _codex_factory,
        "codex": _codex_factory,
        "openai-codex": _codex_factory,
        "claude": _anthropic_factory,
        "anthropic": _anthropic_factory,
        "bedrock": _bedrock_factory,
        "aws-bedrock": _bedrock_factory,
        "vertex": _vertex_factory,
        "vertex-ai": _vertex_factory,
        "codex_app_server": _codex_app_server_factory,
        "codex-app-server": _codex_app_server_factory,
    }
    for name, factory in factories.items():
        register_provider(name, factory, replace=True)


register_builtin_providers()
