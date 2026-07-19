"""Audit-backed tests: auth gates, Grok priority, URI safety, honest Claude/Codex."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from codedoggy.model.anthropic_messages import (
    convert_messages_to_anthropic,
    convert_tools_to_anthropic,
    normalize_anthropic_response,
)
from codedoggy.model.auth import (
    AUTH_API_KEY,
    AUTH_OAUTH,
    LoginRequired,
    apply_auth_to_config,
    auth_kind_for_provider,
    auth_status,
    begin_login,
    resolve_credential,
)
from codedoggy.model.auth.device_flow import DeviceFlowError, validate_verification_uri
from codedoggy.model.auth.grok_oauth import GrokOAuthAuth, _load_best_oauth_entry
from codedoggy.model.auth.secure_store import atomic_write_json
from codedoggy.model.profile import API_ANTHROPIC_MESSAGES, API_CHAT_COMPLETIONS
from codedoggy.model.profile_registry import get_profile
from codedoggy.model.registry import create_client
from codedoggy.model.types import ModelConfig


def test_imperial_auth_kinds() -> None:
    assert auth_kind_for_provider("grok") == AUTH_OAUTH
    assert auth_kind_for_provider("claude") == AUTH_OAUTH
    assert auth_kind_for_provider("codex") == AUTH_OAUTH
    assert auth_kind_for_provider("deepseek") == AUTH_API_KEY


def test_profiles_modes() -> None:
    from codedoggy.model.profile import API_CODEX_RESPONSES

    # Hermes: xai + codex → Responses; claude → Anthropic messages
    assert get_profile("grok").api_mode == API_CODEX_RESPONSES
    assert get_profile("claude").api_mode == API_ANTHROPIC_MESSAGES
    assert get_profile("codex").api_mode == API_CODEX_RESPONSES
    assert get_profile("codex").auth_mode == "oauth"


def test_create_client_routes() -> None:
    from codedoggy.model.anthropic_messages import AnthropicMessagesClient
    from codedoggy.model.codex_responses import CodexResponsesClient

    assert isinstance(
        create_client(
            ModelConfig(
                provider="claude",
                model="m",
                base_url="https://api.anthropic.com",
                api_key="sk-ant-api03-x",
            )
        ),
        AnthropicMessagesClient,
    )
    assert isinstance(
        create_client(
            ModelConfig(
                provider="grok",
                model="m",
                base_url="https://api.x.ai/v1",
                api_key="x",
            )
        ),
        CodexResponsesClient,
    )


def test_imperial_create_client_requires_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("GROK_HOME", str(Path("/nonexistent/codedoggy-no-auth")))
    with pytest.raises(LoginRequired):
        create_client(
            ModelConfig(
                provider="grok",
                model="grok-3",
                base_url="https://api.x.ai/v1",
                api_key=None,
            )
        )


def test_claude_without_creds_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)
    # Isolate from real home files by patching path list
    with patch(
        "codedoggy.model.auth.claude_oauth._claude_paths",
        return_value=[],
    ):
        with pytest.raises(LoginRequired):
            create_client(
                ModelConfig(
                    provider="claude",
                    model="claude-sonnet-4-5",
                    base_url="https://api.anthropic.com",
                    api_key=None,
                )
            )


def test_grok_oauth_beats_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROK_HOME", str(tmp_path))
    monkeypatch.setenv("XAI_API_KEY", "payg-key-should-lose")
    monkeypatch.delenv("CODEDOGGY_FORCE_API_KEY", raising=False)
    scope = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                scope: {
                    "key": "oauth-session-token",
                    "auth_mode": "oidc",
                    "create_time": "2026-01-01T00:00:00Z",
                    "refresh_token": "rt",
                    "oidc_issuer": "https://auth.x.ai",
                    "oidc_client_id": "b1a00492-073a-47ea-816f-4c329264a828",
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(hours=1)
                    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                }
            }
        ),
        encoding="utf-8",
    )
    cred = GrokOAuthAuth().resolve()
    assert cred is not None
    assert cred.token == "oauth-session-token"
    assert cred.kind == AUTH_OAUTH


def test_grok_force_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROK_HOME", str(tmp_path))
    monkeypatch.setenv("XAI_API_KEY", "forced-key")
    monkeypatch.setenv("CODEDOGGY_FORCE_API_KEY", "1")
    scope = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"
    (tmp_path / "auth.json").write_text(
        json.dumps({scope: {"key": "oauth", "auth_mode": "oidc"}}),
        encoding="utf-8",
    )
    cred = GrokOAuthAuth().resolve()
    assert cred is not None
    assert cred.token == "forced-key"
    assert cred.kind == AUTH_API_KEY


def test_grok_expired_without_refresh_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROK_HOME", str(tmp_path))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEDOGGY_FORCE_API_KEY", raising=False)
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    scope = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                scope: {
                    "key": "stale",
                    "auth_mode": "oidc",
                    "expires_at": past,
                    # no refresh_token
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LoginRequired):
        GrokOAuthAuth().resolve()


def test_load_best_skips_api_key_scope(tmp_path: Path) -> None:
    data = {
        "xai::api_key": {
            "key": "api-only",
            "auth_mode": "api_key",
            "create_time": "2026-06-01T00:00:00Z",
        },
        "https://auth.x.ai::client": {
            "key": "oauth",
            "auth_mode": "oidc",
            "create_time": "2026-01-01T00:00:00Z",
            "refresh_token": "r",
        },
    }
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    entry = _load_best_oauth_entry(path)
    assert entry is not None
    assert entry["key"] == "oauth"


def test_validate_verification_uri() -> None:
    validate_verification_uri("https://auth.x.ai/device")
    validate_verification_uri("https://accounts.x.ai/sign-in")
    with pytest.raises(DeviceFlowError):
        validate_verification_uri("http://auth.x.ai/device")
    with pytest.raises(DeviceFlowError):
        validate_verification_uri("https://evil.example/phish")


def test_atomic_write_json(tmp_path: Path) -> None:
    path = tmp_path / "cred.json"
    atomic_write_json(path, {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
    atomic_write_json(path, {"a": 2, "b": 3})
    assert json.loads(path.read_text(encoding="utf-8"))["b"] == 3


def test_apply_auth_fills_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROK_HOME", str(tmp_path))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                "scope": {
                    "key": "from-file",
                    "auth_mode": "oidc",
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(hours=2)
                    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                }
            }
        ),
        encoding="utf-8",
    )
    out = apply_auth_to_config(
        ModelConfig(
            provider="grok", model="m", base_url="https://api.x.ai/v1", api_key=None
        )
    )
    assert out.api_key == "from-file"
    assert out.extra.get("auth_kind") == AUTH_OAUTH


def test_claude_begin_login_honest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    with patch("codedoggy.model.auth.claude_oauth._claude_paths", return_value=[]):
        with patch("webbrowser.open", return_value=True):
            st = begin_login("claude")
    assert st.logged_in is False
    assert st.meta.get("closed_loop") is False
    assert "not enough" in st.detail.lower() or "device-code" in st.detail.lower() or "ANTHROPIC" in st.detail


def test_codex_begin_login_honest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("webbrowser.open", return_value=True):
        st = begin_login("codex")
    assert st.logged_in is False
    assert st.meta.get("closed_loop") is False


def test_auth_status_export() -> None:
    st = auth_status("deepseek")
    assert st.provider == "deepseek"


def test_anthropic_message_conversion() -> None:
    system, msgs = convert_messages_to_anthropic(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
    )
    assert system == "sys"
    assert msgs[1]["content"][0]["type"] == "tool_use"


def test_anthropic_tools_and_normalize() -> None:
    tools = convert_tools_to_anthropic(
        [
            {
                "type": "function",
                "function": {
                    "name": "x",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    assert tools[0]["name"] == "x"
    result = normalize_anthropic_response(
        {
            "model": "c",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 1, "output_tokens": 2},
        },
        model="c",
    )
    assert result.content == "hi"
    assert result.finish_reason == "stop"


def test_deepseek_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
    cred = resolve_credential("deepseek")
    assert cred is not None and cred.token == "sk-ds"
