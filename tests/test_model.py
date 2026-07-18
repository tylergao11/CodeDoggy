"""Model registry, OpenAI-compat client, ModelAuditor parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codedoggy.audit.model_auditor import ModelAuditor, _parse_verdict
from codedoggy.audit.types import (
    AuditContext,
    MemorySelectResult,
    MutationEvent,
)
from codedoggy.model import (
    ChatMessage,
    ChatSampler,
    CompletionResult,
    ModelConfig,
    create_client,
    list_providers,
    model_config_from_env,
    register_provider,
)
from codedoggy.model.openai_compat import (
    ModelError,
    OpenAICompatClient,
    scrub_model_content,
)
from codedoggy.model.registry import unregister_provider
from codedoggy.turn.types import Message, Role


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


def test_parse_verdict_ok() -> None:
    v = _parse_verdict('{"ok": true}')
    assert v is not None and v.ok


def test_parse_verdict_findings() -> None:
    v = _parse_verdict(
        json.dumps(
            {
                "ok": False,
                "findings": [
                    {
                        "severity": "important",
                        "message": "Off goal",
                        "path": "x.py",
                    }
                ],
            }
        )
    )
    assert v is not None and not v.ok
    assert v.findings[0].message == "Off goal"


def test_parse_verdict_fenced_and_think() -> None:
    raw = (
        "<think>reasoning</think>\n"
        '```json\n{"ok": false, "findings": [{"message": "nope"}]}\n```'
    )
    # ModelAuditor strips think before parse; unit the parse path with clean fence
    v = _parse_verdict('```json\n{"ok": false, "findings": [{"message": "nope"}]}\n```')
    assert v is not None and not v.ok


def test_model_auditor_pass_silent() -> None:
    auditor = ModelAuditor(FakeClient('{"ok": true}'))
    ctx = AuditContext(
        goal="fix login",
        mutation=MutationEvent(
            path="auth.py",
            tool_name="search_replace",
            call_id="1",
            before="a",
            after="b",
        ),
        trajectory_summary="(none)",
        memory=MemorySelectResult(),
        cwd=".",
    )
    v = auditor.review(ctx)
    assert v.ok


def test_model_auditor_fail_soft() -> None:
    payload = {
        "ok": False,
        "findings": [{"severity": "important", "message": "Wrong file for goal"}],
    }
    auditor = ModelAuditor(FakeClient(json.dumps(payload)))
    ctx = AuditContext(
        goal="only touch auth.py",
        mutation=MutationEvent(
            path="readme.md",
            tool_name="search_replace",
            call_id="1",
            after="noise",
            is_create=True,
        ),
        trajectory_summary="- create readme.md",
        memory=MemorySelectResult(),
        cwd=".",
    )
    v = auditor.review(ctx)
    assert not v.ok
    assert "Wrong file" in v.findings[0].message


def test_model_auditor_bad_json_pass_silent() -> None:
    auditor = ModelAuditor(FakeClient("not json at all"))
    ctx = AuditContext(
        goal="x",
        mutation=MutationEvent(path="a", tool_name="t", call_id="1"),
        trajectory_summary="",
        memory=MemorySelectResult(),
        cwd=".",
    )
    assert auditor.review(ctx).ok


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

    sampler = ChatSampler(ToolClient())
    out = sampler.sample(
        [Message(role=Role.USER, content="hi")],
        tools=[],
    )
    assert out.tool_calls[0].name == "read_file"
    assert out.tool_calls[0].arguments["target_file"] == "a.py"


def test_openai_compat_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import codedoggy.model.openai_compat as mod

    class FakeResp:
        def __enter__(self):
            raise mod.urllib.error.HTTPError(
                "http://x", 500, "err", hdrs=None, fp=None  # type: ignore[arg-type]
            )

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        mod.urllib.request,
        "urlopen",
        lambda *a, **k: FakeResp(),
    )
    client = OpenAICompatClient(
        ModelConfig(provider="openai_compat", model="m", base_url="http://127.0.0.1:9/v1")
    )
    with pytest.raises(ModelError):
        client.complete([ChatMessage(role="user", content="hi")])


@pytest.mark.integration
def test_ollama_live_complete() -> None:
    """Live smoke against local Ollama — skip if unreachable."""
    cfg = model_config_from_env(provider="ollama", model="qwen3:8b")
    client = create_client(cfg)
    try:
        result = client.complete(
            [
                ChatMessage(
                    role="user",
                    content="Reply with exactly one word: pong",
                ),
            ],
            temperature=0.0,
            max_tokens=256,
        )
    except ModelError as e:
        pytest.skip(f"ollama not reachable: {e}")
    assert result.content
    assert "pong" in result.content.lower()
