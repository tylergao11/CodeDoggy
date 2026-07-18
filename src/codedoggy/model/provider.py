"""Model provider protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from codedoggy.model.types import ChatMessage, CompletionResult, ModelConfig


@runtime_checkable
class ChatClient(Protocol):
    """Minimal chat surface used by turn Sampler and model-brain Auditor."""

    @property
    def config(self) -> ModelConfig:
        ...

    def complete(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResult:
        ...
