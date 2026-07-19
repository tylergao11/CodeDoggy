"""Declarative inference provider profiles (Hermes-style).

A profile describes *how* to talk to a vendor: endpoints, auth env vars,
message preprocessing, and request-time kwargs quirks.  The transport
reads the profile; it does not own credentials or streaming.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# api_mode — protocol family (layer 2)
API_CHAT_COMPLETIONS = "chat_completions"  # OpenAI 系 chat
API_ANTHROPIC_MESSAGES = "anthropic_messages"  # Anthropic 系
API_CODEX_RESPONSES = "codex_responses"  # OpenAI Responses (Codex / xAI OAuth)

# auth_mode — who you are (layer 1)
AUTH_MODE_OAUTH = "oauth"  # grok / claude / codex session login
AUTH_MODE_API_KEY = "api_key"  # everyone else

# How assistant ``reasoning_content`` is handled on *input* history replay:
#   strip   — never send (strict OpenAI-compat: Mistral/Groq/… reject the key)
#   require — always echo/pad for thinking models (DeepSeek V4 / reasoner)
#   pass    — keep if present, never invent
REASONING_STRIP = "strip"
REASONING_REQUIRE = "require"
REASONING_PASS = "pass"


@dataclass
class ProviderProfile:
    """One inference vendor's wire-format contract."""

    name: str
    api_mode: str = API_CHAT_COMPLETIONS
    auth_mode: str = AUTH_MODE_API_KEY
    aliases: tuple[str, ...] = ()
    display_name: str = ""
    description: str = ""

    # Auth / endpoints
    env_vars: tuple[str, ...] = ()  # API key candidates, first wins
    base_url: str = ""
    base_url_env_var: str = ""
    default_model: str = ""
    default_aux_model: str = ""
    hostname: str = ""

    # Reasoning / thinking history policy for chat completions input
    # Override via :meth:`reasoning_policy_for_model` when it depends on model id.
    reasoning_policy: str = REASONING_STRIP

    # Anthropic-style cache_control (system_and_3). DeepSeek disk cache is
    # automatic prefix — keep system stable; no markers needed.
    prompt_cache: bool = False
    prompt_cache_ttl: str = "5m"

    # Optional static headers (e.g. custom User-Agent)
    default_headers: dict[str, str] = field(default_factory=dict)

    def get_hostname(self) -> str:
        if self.hostname:
            return self.hostname
        if self.base_url:
            return urlparse(self.base_url).hostname or ""
        return ""

    def reasoning_policy_for_model(self, model: str | None) -> str:
        """Return strip | require | pass for this model on this provider."""
        return self.reasoning_policy

    def prepare_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Provider-specific message preprocessing before the HTTP body.

        Applies reasoning echo/strip + optional Anthropic prompt-cache markers
        (Hermes protocol_context).
        """
        from codedoggy.model.protocol_context import prepare_wire_messages

        return prepare_wire_messages(
            messages,
            profile=self,
            model=model,
            enable_prompt_cache=bool(self.prompt_cache),
            cache_ttl=self.prompt_cache_ttl or "5m",
        )

    def build_extra_body(
        self,
        *,
        model: str | None = None,
        session_id: str | None = None,
        **context: Any,
    ) -> dict[str, Any]:
        """Provider-specific ``extra_body`` fields (merged into request body)."""
        return {}

    def build_api_kwargs_extras(
        self,
        *,
        model: str | None = None,
        reasoning_config: dict[str, Any] | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return ``(extra_body_additions, top_level_kwargs)``.

        Some vendors put thinking knobs in ``extra_body`` (DeepSeek),
        others top-level (Kimi ``reasoning_effort``).  Default: empty.
        """
        return {}, {}

    def resolve_api_key(self, explicit: str | None = None) -> str | None:
        """Resolve API key from explicit value or env_vars."""
        if explicit is not None and str(explicit).strip():
            return str(explicit).strip()
        import os

        for name in self.env_vars:
            raw = os.environ.get(name)
            if raw is not None and str(raw).strip():
                return str(raw).strip()
        return None

    def resolve_base_url(self, explicit: str | None = None) -> str:
        """Resolve base URL: explicit → env → profile default."""
        if explicit is not None and str(explicit).strip():
            return str(explicit).strip().rstrip("/")
        import os

        if self.base_url_env_var:
            raw = os.environ.get(self.base_url_env_var)
            if raw is not None and str(raw).strip():
                return str(raw).strip().rstrip("/")
        return (self.base_url or "").rstrip("/")
