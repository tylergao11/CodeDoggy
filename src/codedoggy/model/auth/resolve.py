"""Resolve auth for a provider and apply onto ModelConfig."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codedoggy.model.auth.api_key import ApiKeyAuth
from codedoggy.model.auth.base import (
    AUTH_API_KEY,
    AUTH_OAUTH,
    AuthCredential,
    AuthKind,
    AuthStatus,
    LoginRequired,
)
from codedoggy.model.auth.claude_oauth import ClaudeOAuthAuth
from codedoggy.model.auth.codex_oauth import CodexOAuthAuth
from codedoggy.model.auth.grok_oauth import GrokOAuthAuth
from codedoggy.model.auth.registry import (
    get_auth_provider,
    list_auth_providers,
    register_auth_provider,
    resolve_auth_name,
)

if TYPE_CHECKING:
    from codedoggy.model.types import ModelConfig

logger = logging.getLogger(__name__)

# Grok / Claude / Codex — browser/session auth first
IMPERIAL_OAUTH: frozenset[str] = frozenset(
    {"grok", "xai", "claude", "anthropic", "codex"}
)

_BOOTSTRAPPED = False


def _bootstrap() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    register_auth_provider(
        GrokOAuthAuth(), aliases=("xai", "x-ai", "x.ai"), replace=True
    )
    register_auth_provider(ClaudeOAuthAuth(), aliases=("anthropic",), replace=True)
    register_auth_provider(CodexOAuthAuth(), aliases=("openai-codex",), replace=True)
    register_auth_provider(
        ApiKeyAuth("deepseek", env_vars=("DEEPSEEK_API_KEY",), display_name="DeepSeek"),
        replace=True,
    )
    register_auth_provider(
        ApiKeyAuth(
            "openai",
            env_vars=("OPENAI_API_KEY", "CODEDOGGY_API_KEY"),
            display_name="OpenAI API",
        ),
        replace=True,
    )
    register_auth_provider(
        ApiKeyAuth("ollama", env_vars=("OLLAMA_API_KEY",), display_name="Ollama"),
        replace=True,
    )
    register_auth_provider(
        ApiKeyAuth(
            "custom",
            env_vars=("CODEDOGGY_API_KEY", "OPENAI_API_KEY"),
            display_name="Custom",
        ),
        replace=True,
    )
    register_auth_provider(
        ApiKeyAuth(
            "openai_compat",
            env_vars=("OPENAI_API_KEY", "CODEDOGGY_API_KEY"),
            display_name="OpenAI-compatible",
        ),
        replace=True,
    )
    _BOOTSTRAPPED = True


def is_imperial(provider: str | None) -> bool:
    _bootstrap()
    name = (provider or "").strip().lower()
    canon = resolve_auth_name(name) or name
    return name in IMPERIAL_OAUTH or canon in {"grok", "claude", "codex"}


def auth_kind_for_provider(provider: str | None) -> AuthKind:
    _bootstrap()
    name = (provider or "").strip().lower()
    canon = resolve_auth_name(name) or name
    if is_imperial(name):
        return AUTH_OAUTH
    auth = get_auth_provider(name)
    if auth is not None:
        return auth.kind  # type: ignore[return-value]
    if canon:
        auth = get_auth_provider(canon)
        if auth is not None:
            return auth.kind  # type: ignore[return-value]
    return AUTH_API_KEY


def resolve_credential(
    provider: str,
    *,
    explicit_token: str | None = None,
    require: bool = False,
) -> AuthCredential | None:
    _bootstrap()
    auth = get_auth_provider(provider)
    if auth is None:
        if explicit_token and str(explicit_token).strip():
            return AuthCredential(
                provider=provider,
                kind=AUTH_API_KEY,
                token=str(explicit_token).strip(),
                source="explicit",
            )
        if require:
            raise LoginRequired(provider, f"no auth provider for {provider!r}")
        return None
    try:
        cred = auth.resolve(explicit_token=explicit_token)
    except LoginRequired:
        if require:
            raise
        return None
    if cred is None and require:
        kind = auth_kind_for_provider(provider)
        if kind == AUTH_OAUTH:
            raise LoginRequired(
                provider,
                f"{provider} requires login — call begin_login({provider!r}) "
                f"or provide credentials",
            )
        raise LoginRequired(provider, f"{provider} requires an API key")
    return cred


def apply_auth_to_config(
    config: ModelConfig,
    *,
    require: bool | None = None,
) -> ModelConfig:
    """Fill api_key / headers from the auth layer.

    ``require``:
      * None (default) — require credentials for OAuth providers, soft for others
      * True / False — force
    """
    from codedoggy.model.types import ModelConfig

    _bootstrap()
    if require is None:
        require = is_imperial(config.provider)

    cred = resolve_credential(
        config.provider,
        explicit_token=config.api_key,
        require=require,
    )
    if cred is None:
        return config

    headers = dict(config.extra_headers)
    headers.update(cred.headers)
    extra = dict(config.extra)
    extra["auth_kind"] = cred.kind
    extra["auth_source"] = cred.source
    return ModelConfig(
        provider=config.provider,
        model=config.model,
        base_url=config.base_url,
        api_key=cred.token,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
        context_window=config.context_window,
        extra_headers=headers,
        extra=extra,
    )


def auth_status(provider: str) -> AuthStatus:
    _bootstrap()
    auth = get_auth_provider(provider)
    if auth is None:
        return AuthStatus(
            provider=provider,
            kind=auth_kind_for_provider(provider),
            logged_in=False,
            detail="unknown provider",
        )
    return auth.status()


def begin_login(provider: str) -> AuthStatus:
    _bootstrap()
    auth = get_auth_provider(provider)
    if auth is None:
        return AuthStatus(
            provider=provider,
            kind=AUTH_API_KEY,
            logged_in=False,
            detail="no login flow — use API key",
        )
    return auth.begin_login()


__all__ = [
    "IMPERIAL_OAUTH",
    "apply_auth_to_config",
    "auth_kind_for_provider",
    "auth_status",
    "begin_login",
    "get_auth_provider",
    "is_imperial",
    "list_auth_providers",
    "resolve_credential",
]
