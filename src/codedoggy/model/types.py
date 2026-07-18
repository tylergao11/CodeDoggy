"""Model client config and completion types.

Shape mirrors the useful core of Grok ``SamplerConfig`` (base_url / model /
api_key / sampling knobs) without the product-specific header machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    """How to reach a chat-completions endpoint.

    Providers (ollama, openai_compat, …) are registered by name; the
    transport is usually OpenAI-compatible HTTP.
    """

    provider: str
    model: str
    base_url: str
    api_key: str | None = None
    temperature: float | None = 0.2
    max_tokens: int | None = None
    timeout_s: float = 120.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Opaque bag for provider-specific options (num_ctx, etc.).
    extra: dict[str, Any] = field(default_factory=dict)

    def normalized_base_url(self) -> str:
        return self.base_url.rstrip("/")


@dataclass(slots=True)
class ChatMessage:
    role: str  # system | user | assistant | tool
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class CompletionResult:
    """One non-streaming chat completion."""

    content: str | None
    model: str
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
