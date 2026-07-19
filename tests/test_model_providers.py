"""Hermes-style provider profiles + DeepSeek reasoning_content contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codedoggy.model.builtin_profiles import DeepSeekProfile, _deepseek_thinking_model
from codedoggy.model.openai_compat import OpenAICompatClient
from codedoggy.model.profile import REASONING_REQUIRE, REASONING_STRIP
from codedoggy.model.profile_registry import get_profile, list_profiles
from codedoggy.model.reasoning import apply_reasoning_echo
from codedoggy.model.registry import create_client, model_config_from_env
from codedoggy.model.types import ChatMessage, ModelConfig


def test_builtin_profiles_registered() -> None:
    names = list_profiles()
    for need in ("openai", "deepseek", "grok", "claude", "codex", "ollama", "custom", "openai_compat"):
        assert need in names, need
    # aliases resolve
    assert get_profile("xai") is not None
    assert get_profile("xai").name == "grok"


def test_deepseek_thinking_model_detect() -> None:
    assert _deepseek_thinking_model("deepseek-reasoner") is True
    assert _deepseek_thinking_model("deepseek-v4-pro") is True
    assert _deepseek_thinking_model("deepseek-chat") is False
    assert _deepseek_thinking_model("deepseek-v3") is False


def test_deepseek_reasoning_policy() -> None:
    p = get_profile("deepseek")
    assert p is not None
    assert p.reasoning_policy_for_model("deepseek-chat") == REASONING_STRIP
    assert p.reasoning_policy_for_model("deepseek-reasoner") == REASONING_REQUIRE
    assert p.reasoning_policy_for_model("deepseek-v4-flash") == REASONING_REQUIRE


def test_reasoning_strip_removes_field() -> None:
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "ok",
            "reasoning_content": "secret chain",
            "tool_calls": [{"id": "1"}],
        },
    ]
    out = apply_reasoning_echo(msgs, policy=REASONING_STRIP)
    assert "reasoning_content" not in out[1]
    assert out[1]["content"] == "ok"


def test_reasoning_require_pads_missing() -> None:
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
        }
    ]
    out = apply_reasoning_echo(msgs, policy=REASONING_REQUIRE)
    assert out[0]["reasoning_content"] == " "


def test_reasoning_require_upgrades_empty_string() -> None:
    msgs = [{"role": "assistant", "content": "a", "reasoning_content": ""}]
    out = apply_reasoning_echo(msgs, policy=REASONING_REQUIRE)
    assert out[0]["reasoning_content"] == " "


def test_reasoning_require_preserves_cot() -> None:
    msgs = [{"role": "assistant", "content": "a", "reasoning_content": "step 1"}]
    out = apply_reasoning_echo(msgs, policy=REASONING_REQUIRE)
    assert out[0]["reasoning_content"] == "step 1"


def test_deepseek_profile_thinking_kwargs() -> None:
    p = DeepSeekProfile(name="deepseek")
    extra, top = p.build_api_kwargs_extras(
        model="deepseek-reasoner",
        reasoning_config={"enabled": True, "effort": "high"},
    )
    assert extra["thinking"] == {"type": "enabled"}
    assert top["reasoning_effort"] == "high"

    extra_off, top_off = p.build_api_kwargs_extras(
        model="deepseek-reasoner",
        reasoning_config={"enabled": False},
    )
    assert extra_off["thinking"] == {"type": "disabled"}
    assert top_off == {}

    # V3 chat — no thinking wire
    e3, t3 = p.build_api_kwargs_extras(model="deepseek-chat")
    assert e3 == {} and t3 == {}


def test_client_build_body_applies_deepseek_prepare() -> None:
    cfg = ModelConfig(
        provider="deepseek",
        model="deepseek-reasoner",
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test",
        extra={"reasoning": {"enabled": True, "effort": "medium"}},
    )
    # Construct without auth gate (unit test of wire format only)
    client = OpenAICompatClient(cfg, profile=get_profile("deepseek"))
    body = client._build_body(
        [
            ChatMessage(role="user", content="hi"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
                # missing reasoning_content — must be padded on wire
            ),
        ],
        stream=False,
        temperature=None,
        max_tokens=None,
        tools=None,
    )
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "medium"
    asst = body["messages"][1]
    assert asst["reasoning_content"] == " "


def test_openai_client_strips_reasoning_on_input() -> None:
    cfg = ModelConfig(
        provider="openai",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
    )
    client = OpenAICompatClient(cfg, profile=get_profile("openai"))
    body = client._build_body(
        [
            ChatMessage(
                role="assistant",
                content="hi",
                reasoning_content="should not go out",
            )
        ],
        stream=False,
        temperature=0.1,
        max_tokens=10,
        tools=None,
    )
    assert "reasoning_content" not in body["messages"][0]
    assert "thinking" not in body


def test_model_config_from_env_deepseek(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEDOGGY_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
    monkeypatch.delenv("CODEDOGGY_BASE_URL", raising=False)
    monkeypatch.delenv("CODEDOGGY_MODEL", raising=False)
    cfg = model_config_from_env()
    assert cfg.provider == "deepseek"
    assert "deepseek.com" in cfg.base_url
    assert cfg.api_key == "sk-ds"
    assert cfg.model  # default


def test_model_config_from_env_xai(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate from real ~/.grok so env key is the only credential
    monkeypatch.setenv("GROK_HOME", str(tmp_path))
    monkeypatch.setenv("CODEDOGGY_PROVIDER", "xai")
    monkeypatch.setenv("XAI_API_KEY", "xai-key")
    monkeypatch.delenv("CODEDOGGY_FORCE_API_KEY", raising=False)
    monkeypatch.delenv("CODEDOGGY_BASE_URL", raising=False)
    cfg = model_config_from_env()
    assert cfg.provider == "xai"
    assert "x.ai" in cfg.base_url
    # No auth.json session → falls back to XAI_API_KEY
    assert cfg.api_key == "xai-key"


def test_create_client_deepseek() -> None:
    client = create_client(
        ModelConfig(
            provider="deepseek",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="k",
        )
    )
    assert isinstance(client, OpenAICompatClient)
    assert client.profile is not None
    assert client.profile.name == "deepseek"


def test_chat_sampler_preserves_reasoning() -> None:
    from codedoggy.model.chat_sampler import ChatSampler
    from codedoggy.model.types import CompletionResult
    from codedoggy.turn.types import Message, Role

    class Fake:
        config = ModelConfig(provider="deepseek", model="deepseek-reasoner", base_url="http://x")

        def complete(self, messages, **kwargs):
            return CompletionResult(
                content="done",
                model="deepseek-reasoner",
                reasoning_content="think hard",
                tool_calls=[],
            )

    sample = ChatSampler(Fake()).sample(
        [Message(role=Role.USER, content="q")],
        tools=[],
    )
    assert sample.reasoning_content == "think hard"
    assert sample.raw.get("reasoning_content") == "think hard"
