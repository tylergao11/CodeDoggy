"""Protocol / context contracts — Hermes-aligned (cache + reasoning + blocks)."""

from __future__ import annotations

from codedoggy.model.anthropic_messages import (
    convert_messages_to_anthropic,
    normalize_anthropic_response,
)
from codedoggy.model.profile_registry import get_profile
from codedoggy.model.prompt_caching import apply_anthropic_cache_control
from codedoggy.model.protocol_context import prepare_wire_messages
from codedoggy.model.reasoning import apply_reasoning_echo
from codedoggy.model.profile import REASONING_REQUIRE, REASONING_STRIP


def test_deepseek_require_pads_assistant() -> None:
    msgs = [{"role": "assistant", "content": "ok", "tool_calls": [{"id": "1"}]}]
    out = apply_reasoning_echo(msgs, policy=REASONING_REQUIRE)
    assert out[0]["reasoning_content"] == " "


def test_openai_strip_drops_reasoning() -> None:
    msgs = [
        {"role": "assistant", "content": "x", "reasoning_content": "secret"},
    ]
    out = apply_reasoning_echo(msgs, policy=REASONING_STRIP)
    assert "reasoning_content" not in out[0]


def test_anthropic_cache_system_and_last_messages() -> None:
    msgs = [
        {"role": "system", "content": "You are stable system."},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ]
    out = apply_anthropic_cache_control(msgs, native_anthropic=True)
    # system marked
    sys = out[0]
    assert sys["role"] == "system"
    assert isinstance(sys["content"], list)
    assert sys["content"][0].get("cache_control", {}).get("type") == "ephemeral"
    # last 3 non-system also marked (up to 4 total)
    marked = 0
    for m in out:
        c = m.get("content")
        if isinstance(c, list) and c and isinstance(c[-1], dict) and c[-1].get("cache_control"):
            marked += 1
        elif m.get("cache_control"):
            marked += 1
    assert marked == 4


def test_cache_skips_empty_assistant_on_envelope() -> None:
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t"}]},
        {"role": "user", "content": "next"},
    ]
    out = apply_anthropic_cache_control(msgs, native_anthropic=False)
    # empty assistant should not waste a breakpoint on envelope layout
    asst = out[2]
    assert asst.get("role") == "assistant"
    assert not asst.get("cache_control")
    content = asst.get("content")
    if isinstance(content, list) and content:
        assert "cache_control" not in content[-1]


def test_claude_profile_prepare_adds_cache() -> None:
    claude = get_profile("claude")
    assert claude is not None and claude.prompt_cache is True
    msgs = [
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "hello"},
    ]
    out = prepare_wire_messages(msgs, profile=claude, model="claude-sonnet-4-5")
    assert isinstance(out[0]["content"], list)
    assert out[0]["content"][0].get("cache_control")


def test_claude_tools_get_last_tool_cache_control() -> None:
    from codedoggy.model.anthropic_messages import AnthropicMessagesClient
    from codedoggy.model.types import ModelConfig

    client = AnthropicMessagesClient(
        ModelConfig(
            provider="claude",
            model="claude-sonnet-4-5",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-api03-x",
        ),
        profile=get_profile("claude"),
    )
    body = client._build_body(
        [{"role": "user", "content": "hi"}],
        temperature=None,
        max_tokens=100,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "a",
                    "description": "a",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "b",
                    "description": "b",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
    )
    assert "cache_control" in body["tools"][-1]
    assert "cache_control" not in body["tools"][0]


def test_deepseek_profile_no_cache_markers() -> None:
    ds = get_profile("deepseek")
    assert ds is not None and ds.prompt_cache is False
    msgs = [
        {"role": "system", "content": "stable"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
    ]
    out = prepare_wire_messages(
        msgs, profile=ds, model="deepseek-reasoner", enable_prompt_cache=True
    )
    # no cache_control, but reasoning pad
    assert out[1].get("reasoning_content") == " "
    assert "cache_control" not in out[0]


def test_anthropic_preserves_thinking_block_order() -> None:
    # Valid tool pair required — orphan tool_use would invalidate signatures
    # (Hermes strip_orphaned_tool_blocks).
    asst = {
        "role": "assistant",
        "content": None,
        "provider_data": {
            "anthropic_content_blocks": [
                {"type": "thinking", "thinking": "plan", "signature": "sig123"},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "bash",
                    "input": {"command": "ls"},
                },
            ]
        },
        "tool_calls": [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"ls"}'},
            }
        ],
    }
    tool = {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "ok",
    }
    system, anth = convert_messages_to_anthropic(
        [{"role": "user", "content": "go"}, asst, tool],
        base_url="https://api.anthropic.com",
    )
    assert anth[1]["role"] == "assistant"
    blocks = anth[1]["content"]
    assert blocks[0]["type"] == "thinking"
    assert blocks[0].get("signature") == "sig123"
    assert blocks[1]["type"] == "tool_use"


def test_third_party_strips_thinking_signatures() -> None:
    from codedoggy.model.anthropic_hygiene import finalize_anthropic_messages

    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "x", "signature": "sig"},
                {"type": "text", "text": "hi"},
            ],
        }
    ]
    # third-party base → strip all thinking
    out = finalize_anthropic_messages(
        msgs, base_url="https://proxy.example.com/anthropic"
    )
    types = [b.get("type") for b in out[0]["content"]]
    assert "thinking" not in types
    assert "text" in types


def test_native_keeps_latest_signed_thinking() -> None:
    from codedoggy.model.anthropic_hygiene import finalize_anthropic_messages

    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "x", "signature": "sig"},
                {"type": "text", "text": "hi"},
            ],
        }
    ]
    out = finalize_anthropic_messages(msgs, base_url="https://api.anthropic.com")
    types = [b.get("type") for b in out[0]["content"]]
    assert "thinking" in types


def test_orphan_tool_use_stripped() -> None:
    from codedoggy.model.anthropic_hygiene import finalize_anthropic_messages

    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
                {"type": "text", "text": "x"},
            ],
        },
        {"role": "user", "content": "no tool result"},
    ]
    out = finalize_anthropic_messages(msgs, base_url="https://api.anthropic.com")
    types = [b.get("type") for b in out[0]["content"] if isinstance(b, dict)]
    assert "tool_use" not in types


def test_rewrite_system_identity() -> None:
    from codedoggy.model.provider_switch import rewrite_system_model_identity

    sp = "You are helpful.\nModel: old\nProvider: oldp\nRules..."
    out = rewrite_system_model_identity(sp, model="new-m", provider="grok")
    assert "Model: new-m" in (out or "")
    assert "Provider: grok" in (out or "")
    assert "You are helpful" in (out or "")


def test_openai_usage_deepseek_cache_fields() -> None:
    from codedoggy.model.openai_compat import normalize_openai_usage

    u = normalize_openai_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
        }
    )
    assert u["cached_tokens"] == 80
    assert u["cache_read_input_tokens"] == 80
    assert u["cache_miss_input_tokens"] == 20


def test_normalize_stores_anthropic_blocks() -> None:
    result = normalize_anthropic_response(
        {
            "model": "claude",
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "x", "signature": "s"},
                {"type": "tool_use", "id": "1", "name": "a", "input": {}},
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_read_input_tokens": 8,
            },
        },
        model="claude",
    )
    assert result.provider_data is not None
    blocks = result.provider_data["anthropic_content_blocks"]
    assert blocks[0]["signature"] == "s"
    assert result.usage.get("cache_read_input_tokens") == 8
