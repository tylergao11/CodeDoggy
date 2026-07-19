"""Redaction + Bedrock conversion + Vertex URL + Codex app-server (no cloud SDKs)."""

from __future__ import annotations

from codedoggy.model.bedrock import (
    build_converse_kwargs,
    convert_messages_to_converse,
    convert_tools_to_converse,
    is_streaming_access_denied_error,
    model_supports_tools,
    normalize_converse_response,
    stream_converse_to_result,
)
from codedoggy.model.codex_app_server import (
    TurnResult,
    _classify_oauth_failure,
    _coerce_turn_input_text,
    check_codex_binary,
    parse_codex_version,
)
from codedoggy.model.codex_event_projector import CodexEventProjector
from codedoggy.model.profile_registry import get_profile
from codedoggy.model.protocol_context import prepare_wire_messages
from codedoggy.model.redact import (
    is_env_dump_command,
    mask_secret,
    redact_messages_for_api,
    redact_sensitive_text,
    redact_terminal_output,
)
from codedoggy.model.registry import create_client
from codedoggy.model.types import ModelConfig
from codedoggy.model.vertex import build_vertex_base_url, resolve_vertex_region


def test_redact_api_keys_and_bearer() -> None:
    text = "key=sk-proj-abcdefghijklmnopqrstuvwxyz Authorization: Bearer ghp_ABCDEFGHIJKLMNOPQRST"
    out = redact_sensitive_text(text)
    assert "sk-proj-" not in (out or "")
    assert "ghp_" not in (out or "")
    assert "«redacted-secret»" in (out or "")


def test_redact_xai_jwt_private_key_db() -> None:
    text = (
        "xai-" + ("a" * 40) + " "
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0In0.sig_payload_here "
        "-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY----- "
        "postgres://user:hunter2@db.example.com/app"
    )
    out = redact_sensitive_text(text) or ""
    assert "xai-" + ("a" * 10) not in out
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in out
    assert "PRIVATE KEY" not in out or "REDACTED" in out
    assert "hunter2" not in out
    assert "postgres://user:***@" in out or "***" in out


def test_redact_yaml_and_json_fields() -> None:
    yaml_txt = "password: supersecretvalue123\n"
    out = redact_sensitive_text(yaml_txt) or ""
    assert "supersecretvalue123" not in out

    js = '{"api_key": "sk-abcdefghijklmnopqrstuv", "count": 1}'
    out2 = redact_sensitive_text(js) or ""
    assert "sk-abcdefghij" not in out2


def test_redact_messages_tool_args() -> None:
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"curl -H \\"Authorization: Bearer sk-abc1234567890xyz\\""}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "1",
            "content": "OPENAI_API_KEY=sk-live-abcdefghijklmnopqrst",
        },
    ]
    out = redact_messages_for_api(msgs)
    args = out[0]["tool_calls"][0]["function"]["arguments"]
    assert "sk-abc" not in args or "redacted" in args.lower() or "«" in args
    assert "sk-live" not in (out[1]["content"] or "")


def test_redact_terminal_env_dump() -> None:
    assert is_env_dump_command("env | grep KEY")
    assert not is_env_dump_command("python -c 'print(1)'")
    out = redact_terminal_output(
        "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuv\n",
        "env",
        force=True,
    )
    assert "sk-abcdefghij" not in out


def test_mask_secret_floor() -> None:
    assert mask_secret("short") == "***"
    assert "..." in mask_secret("this-is-a-long-secret-value")


def test_prepare_wire_redacts_before_reasoning() -> None:
    ds = get_profile("deepseek")
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "x",
                        "arguments": '{"token":"sk-abcdefghijklmnopqrstuv"}',
                    },
                }
            ],
        }
    ]
    out = prepare_wire_messages(msgs, profile=ds, model="deepseek-reasoner")
    args = out[0]["tool_calls"][0]["function"]["arguments"]
    assert "sk-abcdefghij" not in args


def test_bedrock_convert_tools_and_messages() -> None:
    tools = convert_tools_to_converse(
        [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "r",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    assert tools[0]["toolSpec"]["name"] == "read_file"
    system, msgs = convert_messages_to_converse(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t1",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "body"},
        ]
    )
    assert system and system[0]["text"] == "sys"
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert "toolUse" in msgs[1]["content"][0]
    assert msgs[2]["role"] == "user"
    assert "toolResult" in msgs[2]["content"][0]


def test_bedrock_role_alternation_merge() -> None:
    """Converse requires strict alternation — consecutive same roles merge."""
    _, msgs = convert_messages_to_converse(
        [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"},
            {"role": "assistant", "content": "d"},
        ]
    )
    # two users merge → one user; two assistants merge → one assistant; trailing user pad
    roles = [m["role"] for m in msgs]
    assert roles[0] == "user"
    assert "a" in msgs[0]["content"][0]["text"] or any(
        b.get("text") == "a" for b in msgs[0]["content"]
    )
    assert any(m["role"] == "assistant" for m in msgs)
    # last must be user
    assert msgs[-1]["role"] == "user"


def test_bedrock_build_kwargs_strips_tools_for_r1() -> None:
    assert not model_supports_tools("us.deepseek.r1-v1:0")
    kwargs = build_converse_kwargs(
        model="deepseek.r1-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "x",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    assert "toolConfig" not in kwargs


def test_normalize_converse_response() -> None:
    result = normalize_converse_response(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "hello"},
                        {
                            "toolUse": {
                                "toolUseId": "x",
                                "name": "bash",
                                "input": {"command": "ls"},
                            }
                        },
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        },
        model="anthropic.claude",
    )
    assert result.content == "hello"
    assert result.tool_calls[0]["function"]["name"] == "bash"
    assert result.finish_reason == "tool_calls"
    assert result.usage["prompt_tokens"] == 10


def test_stream_converse_to_result() -> None:
    events = {
        "stream": [
            {
                "contentBlockStart": {
                    "start": {"toolUse": {"toolUseId": "t1", "name": "bash"}}
                }
            },
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"c":"ls"}'}}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "tool_use"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 3, "outputTokens": 2}
                }
            },
        ]
    }
    result = stream_converse_to_result(events, model="m")
    assert result.tool_calls[0]["function"]["name"] == "bash"
    assert result.finish_reason == "tool_calls"
    assert result.usage["prompt_tokens"] == 3


def test_streaming_access_denied_detection() -> None:
    class E(Exception):
        pass

    exc = E(
        "User is not authorized to perform: bedrock:InvokeModelWithResponseStream"
    )
    assert is_streaming_access_denied_error(exc)


def test_vertex_base_url_global() -> None:
    url = build_vertex_base_url("my-proj", "global")
    assert "aiplatform.googleapis.com" in url
    assert "my-proj" in url
    assert "/openapi" in url
    regional = build_vertex_base_url("p", "us-central1")
    assert "us-central1-aiplatform.googleapis.com" in regional
    assert resolve_vertex_region("europe-west1") == "europe-west1"


def test_create_client_bedrock_vertex_profiles() -> None:
    from codedoggy.model.bedrock import BedrockConverseClient
    from codedoggy.model.vertex import VertexClient

    b = create_client(
        ModelConfig(
            provider="bedrock",
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            base_url="",
            api_key="unused",
        )
    )
    assert isinstance(b, BedrockConverseClient)
    v = create_client(
        ModelConfig(
            provider="vertex",
            model="google/gemini-2.5-pro",
            base_url="",
            api_key="unused",
        )
    )
    assert isinstance(v, VertexClient)


def test_create_client_codex_app_server_profile() -> None:
    from codedoggy.model.codex_app_server import CodexAppServerClient

    c = create_client(
        ModelConfig(
            provider="codex_app_server",
            model="gpt-5.1-codex",
            base_url="",
            api_key="x",
        )
    )
    assert isinstance(c, CodexAppServerClient)


def test_codex_version_parse() -> None:
    assert parse_codex_version("codex-cli 0.130.0") == (0, 130, 0)
    assert parse_codex_version("garbage") is None


def test_codex_oauth_classify() -> None:
    hint = _classify_oauth_failure("invalid_grant: refresh token expired")
    assert hint is not None
    assert "codex login" in hint
    assert _classify_oauth_failure("connection reset by peer") is None


def test_codex_coerce_input() -> None:
    assert _coerce_turn_input_text("hi") == "hi"
    assert "image" in _coerce_turn_input_text(
        [{"type": "image_url", "image_url": {"url": "x"}}]
    ).lower()
    assert "hello" in _coerce_turn_input_text(
        [{"type": "text", "text": "hello"}]
    )


def test_codex_event_projector_agent_and_exec() -> None:
    proj = CodexEventProjector()
    r1 = proj.project(
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "agentMessage",
                    "id": "a1",
                    "text": "done",
                }
            },
        }
    )
    assert r1.final_text == "done"
    assert r1.messages[0]["role"] == "assistant"

    r2 = proj.project(
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "commandExecution",
                    "id": "c1",
                    "command": "ls",
                    "cwd": "/tmp",
                    "aggregatedOutput": "a\nb",
                    "exitCode": 0,
                }
            },
        }
    )
    assert r2.is_tool_iteration
    assert r2.messages[0]["tool_calls"][0]["function"]["name"] == "exec_command"
    assert r2.messages[1]["role"] == "tool"

    # streaming deltas do not materialize
    r3 = proj.project(
        {"method": "item/agentMessage/delta", "params": {"delta": "x"}}
    )
    assert r3.messages == []


def test_turn_result_defaults() -> None:
    t = TurnResult()
    assert t.final_text == ""
    assert t.should_retire is False
