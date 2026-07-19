"""Codex Responses API conversion — Hermes-aligned contracts."""

from __future__ import annotations

from codedoggy.model.codex_responses import (
    chat_messages_to_responses_input,
    classify_responses_issuer,
    content_cache_key,
    normalize_responses_payload,
    responses_tools,
    split_system_instructions,
)
from codedoggy.model.profile import API_CODEX_RESPONSES
from codedoggy.model.profile_registry import get_profile
from codedoggy.model.registry import create_client
from codedoggy.model.types import ModelConfig


def test_codex_profile_uses_responses() -> None:
    p = get_profile("codex")
    assert p is not None
    assert p.api_mode == API_CODEX_RESPONSES


def test_grok_profile_uses_responses_like_hermes() -> None:
    # Hermes xai transport = codex_responses
    p = get_profile("grok")
    assert p is not None
    assert p.api_mode == API_CODEX_RESPONSES
    from codedoggy.model.codex_responses import CodexResponsesClient

    client = create_client(
        ModelConfig(
            provider="grok",
            model="grok-3",
            base_url="https://api.x.ai/v1",
            api_key="xai-test",
        )
    )
    assert isinstance(client, CodexResponsesClient)
    assert client._issuer_kind == "xai_responses"


def test_create_client_codex_responses() -> None:
    from codedoggy.model.codex_responses import CodexResponsesClient

    client = create_client(
        ModelConfig(
            provider="codex",
            model="gpt-5.1-codex",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
        )
    )
    assert isinstance(client, CodexResponsesClient)


def test_split_system_instructions() -> None:
    instr, rest = split_system_instructions(
        [
            {"role": "system", "content": "You are Codex."},
            {"role": "user", "content": "hi"},
        ]
    )
    assert "Codex" in instr
    assert rest[0]["role"] == "user"


def test_tools_conversion() -> None:
    tools = responses_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "read",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    assert tools is not None
    assert tools[0]["type"] == "function"
    assert tools[0]["name"] == "read_file"
    assert tools[0]["strict"] is False


def test_chat_to_responses_user_assistant_tools() -> None:
    items = chat_messages_to_responses_input(
        [
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"ls"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "a.py\nb.py"},
            {"role": "user", "content": "thanks"},
        ],
        current_issuer_kind="codex_backend",
    )
    assert items[0] == {"role": "user", "content": "list files"}
    assert items[1]["type"] == "function_call"
    assert items[1]["name"] == "bash"
    assert items[1]["call_id"] == "call_1"
    assert items[2]["type"] == "function_call_output"
    assert items[2]["output"] == "a.py\nb.py"
    assert items[3]["role"] == "user"


def test_replay_encrypted_reasoning_same_issuer() -> None:
    items = chat_messages_to_responses_input(
        [
            {
                "role": "assistant",
                "content": "done",
                "provider_data": {
                    "codex_reasoning_items": [
                        {
                            "type": "reasoning",
                            "encrypted_content": "blob-aaa",
                            "_issuer_kind": "codex_backend",
                        }
                    ]
                },
            }
        ],
        current_issuer_kind="codex_backend",
    )
    # reasoning item + following assistant message
    assert any(i.get("encrypted_content") == "blob-aaa" for i in items)
    # no id/issuer leaked to wire
    for i in items:
        if i.get("encrypted_content"):
            assert "id" not in i or not str(i.get("id") or "").startswith("rs_")
            assert "_issuer_kind" not in i


def test_drop_cross_issuer_reasoning() -> None:
    items = chat_messages_to_responses_input(
        [
            {
                "role": "assistant",
                "content": "hi",
                "codex_reasoning_items": [
                    {
                        "type": "reasoning",
                        "encrypted_content": "xai-blob",
                        "_issuer_kind": "xai_responses",
                    }
                ],
            }
        ],
        current_issuer_kind="codex_backend",
    )
    assert not any(i.get("encrypted_content") == "xai-blob" for i in items)


def test_normalize_responses_payload() -> None:
    result = normalize_responses_payload(
        {
            "model": "gpt-5.1-codex",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "encrypted_content": "enc",
                    "summary": [{"type": "summary_text", "text": "think"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hello"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_9",
                    "name": "read_file",
                    "arguments": '{"p":"a"}',
                },
            ],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 12,
                "input_tokens_details": {"cached_tokens": 40},
            },
        },
        model="fallback",
        issuer_kind="codex_backend",
    )
    assert result.content == "hello"
    assert result.reasoning_content == "think"
    assert result.tool_calls[0]["function"]["name"] == "read_file"
    assert result.finish_reason == "tool_calls"
    assert result.provider_data is not None
    assert result.provider_data["codex_reasoning_items"][0]["_issuer_kind"] == "codex_backend"
    assert result.usage.get("cached_tokens") == 40


def test_issuer_classification() -> None:
    assert classify_responses_issuer(base_url="https://api.x.ai/v1", provider="grok") == "xai_responses"
    assert (
        classify_responses_issuer(base_url="https://api.openai.com/v1", provider="codex")
        == "codex_backend"
    )


def test_prompt_cache_key_stable() -> None:
    tools = [{"type": "function", "name": "a", "parameters": {}}]
    k1 = content_cache_key("sys", tools)
    k2 = content_cache_key("sys", list(reversed(tools)))
    assert k1 == k2
    assert k1 and k1.startswith("pck_")
