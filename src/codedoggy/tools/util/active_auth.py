"""Resolve tool credentials from the session ActiveConnection.

Product rule: once the user picks a login (Grok / Claude / Codex OAuth or any
API key provider), *all* extras (image, video, web search, …) must use that
same provider's credential and base_url — never a different env key or another
provider's OAuth session.
"""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import urlparse

# Endpoint wire families for media tools (derived from base_url / profile.api_mode,
# not from a hand-maintained provider denylist).
ImageApiFamily = Literal["xai", "openai", "unsupported"]
VideoApiFamily = Literal["xai", "unsupported"]
SearchApiFamily = Literal["responses", "custom", "unsupported"]


def connection_snapshot(connection: Any) -> Any | None:
    """Accept ConnectionService, ActiveConnection, or a mapping-like object."""
    if connection is None:
        return None
    snap_fn = getattr(connection, "snapshot", None)
    if callable(snap_fn):
        try:
            return snap_fn()
        except Exception:  # noqa: BLE001
            return None
    return connection


def connection_from_extra(extra: dict[str, Any] | None) -> Any | None:
    """Find ConnectionService / ActiveConnection on a tool ``ctx.extra`` bag."""
    if not extra:
        return None
    conn = extra.get("connection")
    if conn is not None:
        return conn
    kernel = extra.get("kernel")
    if kernel is not None:
        kconn = getattr(kernel, "connection", None)
        if kconn is not None:
            return kconn
    return None


def connection_fields(connection: Any) -> tuple[str, str, str]:
    """Return ``(provider, base_url, model)`` from a connection object."""
    snap = connection_snapshot(connection)
    if snap is None:
        return "", "", ""
    if isinstance(snap, dict):
        provider = str(snap.get("provider") or "").strip().lower()
        base = str(snap.get("base_url") or "").strip().rstrip("/")
        model = str(snap.get("model") or "").strip()
        return provider, base, model
    provider = str(getattr(snap, "provider", "") or "").strip().lower()
    base = str(getattr(snap, "base_url", "") or "").strip().rstrip("/")
    model = str(getattr(snap, "model", "") or "").strip()
    return provider, base, model


def provider_and_base(connection: Any) -> tuple[str, str]:
    """Return ``(provider, base_url)`` — thin wrapper over :func:`connection_fields`."""
    provider, base, _model = connection_fields(connection)
    return provider, base


def profile_base_url(provider: str) -> str:
    """Profile-declared default base for *this* provider (empty if unknown)."""
    name = (provider or "").strip().lower()
    if not name:
        return ""
    try:
        from codedoggy.model.profile_registry import get_profile

        profile = get_profile(name)
        if profile is None:
            return ""
        return str(profile.resolve_base_url(None) or "").strip().rstrip("/")
    except Exception:  # noqa: BLE001
        return ""


def profile_api_mode(provider: str) -> str:
    name = (provider or "").strip().lower()
    if not name:
        return ""
    try:
        from codedoggy.model.profile_registry import get_profile

        profile = get_profile(name)
        if profile is None:
            return ""
        return str(getattr(profile, "api_mode", "") or "").strip().lower()
    except Exception:  # noqa: BLE001
        return ""


def resolve_base_for_provider(provider: str, conn_base: str = "") -> str:
    """Connection base first, then that provider's profile — never another vendor."""
    base = (conn_base or "").strip().rstrip("/")
    if base:
        return base
    return profile_base_url(provider)


def resolve_provider_token(provider: str) -> tuple[str | None, str]:
    """Resolve the login/API credential for *this* provider only.

    Returns ``(token, source_label)``. Never falls back to another provider.
    """
    name = (provider or "").strip().lower()
    if not name:
        return None, ""
    try:
        from codedoggy.model.auth import resolve_credential

        cred = resolve_credential(name)
    except Exception:  # noqa: BLE001
        return None, ""
    if cred is None:
        return None, ""
    token = (getattr(cred, "token", None) or "").strip() or None
    if not token:
        return None, ""
    source = str(getattr(cred, "source", None) or f"auth:{name}")
    return token, source


def endpoint_host(base_url: str) -> str:
    return (urlparse(base_url or "").hostname or "").lower()


def is_xai_endpoint(base_url: str) -> bool:
    """Whether *this base URL* is an xAI host (endpoint-derived)."""
    host = endpoint_host(base_url)
    return host == "api.x.ai" or host.endswith(".x.ai")


def is_openai_official_endpoint(base_url: str) -> bool:
    host = endpoint_host(base_url)
    return host == "api.openai.com" or host.endswith(".openai.com")


def is_local_endpoint(base_url: str) -> bool:
    host = endpoint_host(base_url)
    return host in {"127.0.0.1", "localhost", "::1"} or host.endswith(".local")


def image_api_family(base_url: str, *, provider: str = "") -> ImageApiFamily:
    """Which image wire this connection endpoint supports.

    Derived from base_url host + profile ``api_mode`` — not a provider name list.
    """
    if not (base_url or "").strip():
        return "unsupported"
    if is_xai_endpoint(base_url):
        return "xai"
    if is_openai_official_endpoint(base_url):
        return "openai"

    host = endpoint_host(base_url)
    mode = profile_api_mode(provider)

    # Anthropic Messages API has no OpenAI-style /images/* route.
    if mode == "anthropic_messages" or host.endswith("anthropic.com"):
        return "unsupported"
    # Local Ollama / loopback rarely expose Imagine-compatible images.
    if is_local_endpoint(base_url):
        return "unsupported"
    # Official DeepSeek chat host has no public /images/generations.
    if host.endswith("deepseek.com"):
        return "unsupported"

    # Custom / openai_compat proxies: assume OpenAI images shape; HTTP 404 if wrong.
    if mode in {"chat_completions", "codex_responses", "openai_compat", ""}:
        return "openai"
    return "unsupported"


def video_api_family(base_url: str) -> VideoApiFamily:
    """Video client only implements xAI ``/videos/*`` wire today."""
    if is_xai_endpoint(base_url):
        return "xai"
    return "unsupported"


def search_api_family(base_url: str, *, provider: str = "", custom_url: str | None = None) -> SearchApiFamily:
    """Responses-style web_search vs custom GET backend."""
    if custom_url:
        return "custom"
    if not (base_url or "").strip():
        return "unsupported"
    if is_xai_endpoint(base_url) or is_openai_official_endpoint(base_url):
        return "responses"
    mode = profile_api_mode(provider)
    host = endpoint_host(base_url)
    if mode == "anthropic_messages" or host.endswith("anthropic.com"):
        return "unsupported"
    if is_local_endpoint(base_url):
        return "unsupported"
    # OpenAI-compat / codex_responses proxies may expose /responses + web_search.
    if mode in {"chat_completions", "codex_responses", "openai_compat", ""}:
        return "responses"
    return "unsupported"


def unsupported_image_reason(provider: str, base_url: str) -> str:
    return (
        f"Image generation follows your active connection ({provider or 'unknown'}), "
        f"but this endpoint does not expose an OpenAI/xAI-compatible /images API "
        f"({base_url or 'no base_url'}). Switch to Grok, OpenAI/Codex, or a custom "
        f"OpenAI-compatible image endpoint, or set CODEDOGGY_IMAGINE_BASE_URL."
    )


def unsupported_video_reason(provider: str, base_url: str) -> str:
    return (
        f"Video generation follows your active connection ({provider or 'unknown'}), "
        f"but only xAI video endpoints are implemented "
        f"({base_url or 'no base_url'}). Log in with Grok / an xAI base_url to use video."
    )


def unsupported_search_reason(provider: str, base_url: str) -> str:
    return (
        f"Web search follows your active connection ({provider or 'unknown'}), "
        f"but this endpoint does not support the Responses web_search tool "
        f"({base_url or 'no base_url'}). Use Grok/OpenAI, a compatible proxy, "
        f"or set CODEDOGGY_WEB_SEARCH_URL for a custom GET backend."
    )
