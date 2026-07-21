"""Model registry, OpenAI-compat client, ChatSampler."""

from __future__ import annotations

from typing import Any

import pytest

from codedoggy.model import (
    ChatSampler,
    CompletionResult,
    ModelConfig,
    create_client,
    list_providers,
    model_config_from_env,
    register_provider,
)
from codedoggy.model.openai_compat import scrub_model_content
from codedoggy.model.registry import unregister_provider


class FakeClient:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[Any] = []
        self.config = ModelConfig(
            provider="fake", model="fake", base_url="http://x", api_key="k"
        )

    def complete(self, messages, **kwargs):
        self.calls.append(messages)
        return CompletionResult(content=self._content, model="fake")


def test_scrub_model_content_strips_think() -> None:
    assert scrub_model_content("<think>secret</think>pong") == "pong"
    assert scrub_model_content("<think>only thinking</think>") is None
    assert scrub_model_content("<think>unclosed") is None
    assert scrub_model_content("  hi  ") == "hi"
    assert scrub_model_content(None) is None


def test_builtin_providers_registered() -> None:
    names = list_providers()
    assert "ollama" in names
    assert "openai_compat" in names
    # OAuth session providers present
    assert "grok" in names or "xai" in names
    assert "claude" in names
    assert "codex" in names


def test_register_custom_provider() -> None:
    def factory(cfg: ModelConfig):
        return FakeClient('{"ok": true}')

    register_provider("testfake", factory, replace=True)
    try:
        assert "testfake" in list_providers()
        client = create_client(
            ModelConfig(provider="testfake", model="m", base_url="http://x")
        )
        assert isinstance(client, FakeClient)
    finally:
        unregister_provider("testfake")


def test_model_config_from_env_ollama_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEDOGGY_PROVIDER", raising=False)
    monkeypatch.delenv("CODEDOGGY_MODEL", raising=False)
    monkeypatch.delenv("CODEDOGGY_BASE_URL", raising=False)
    # Isolate from the developer's real ~/.grok login + preferred provider.
    monkeypatch.setattr(
        "codedoggy.model.preferred_provider.resolve_startup_provider",
        lambda: "ollama",
    )
    cfg = model_config_from_env()
    assert cfg.provider == "ollama"
    assert "11434" in cfg.base_url or "ollama" in cfg.base_url.lower()
    assert cfg.model


def test_ollama_factory_normalizes_port_url() -> None:
    client = create_client(
        ModelConfig(
            provider="ollama",
            model="qwen3:8b",
            base_url="http://127.0.0.1:11434",
        )
    )
    assert client.config.base_url.rstrip("/").endswith("/v1")


def test_chat_sampler_maps_tool_calls() -> None:
    class ToolClient:
        config = ModelConfig(provider="x", model="m", base_url="http://x")

        def complete(self, messages, **kwargs):
            return CompletionResult(
                content=None,
                model="m",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"target_file": "a.py"}',
                        },
                    }
                ],
            )

    sample = ChatSampler(ToolClient()).sample([], tools=[])  # type: ignore[arg-type]
    assert sample.tool_calls
    assert sample.tool_calls[0].name == "read_file"
