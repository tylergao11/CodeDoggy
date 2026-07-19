"""Focused tests for Grok-aligned web_search port."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.grok_build.web_search import (
    DESCRIPTION_TEMPLATE,
    NO_RESULTS_CONTENT,
    build_search_request,
    extract_citation_pairs,
    extract_citations,
    extract_output_text,
    format_prompt_output,
)
from codedoggy.tools.runtime import ToolCallContext, ToolError
from codedoggy.tools.util.web_search_api import WebSearchConfig, WebSearchResult


# ── pure logic (grok_build/web_search) ───────────────────────────────


def test_description_matches_grok() -> None:
    assert "Search the web" in DESCRIPTION_TEMPLATE
    assert "coding" in DESCRIPTION_TEMPLATE


def test_build_search_request_shape() -> None:
    body = build_search_request("rust lang", "test-model", None)
    assert body["model"] == "test-model"
    assert body["input"] == "rust lang"
    assert body["store"] is False
    assert body["temperature"] == 0.1
    assert body["top_p"] == 0.95
    assert body["max_output_tokens"] == 8192
    assert body["tools"] == [{"type": "web_search"}]


def test_build_search_request_with_allowed_domains() -> None:
    body = build_search_request(
        "q", "m", allowed_domains=["docs.rs", "rust-lang.org"]
    )
    assert body["tools"][0]["filters"]["allowed_domains"] == [
        "docs.rs",
        "rust-lang.org",
    ]


def test_format_prompt_output_default() -> None:
    out = format_prompt_output("cats", "meow facts")
    assert out == 'Web search results for: "cats"\n\nmeow facts'


def test_format_prompt_output_pre_formatted() -> None:
    out = format_prompt_output("q", "ignored", pre_formatted="Title: x\nContent: y")
    assert out == "Title: x\nContent: y"


def test_extract_output_text_empty() -> None:
    assert extract_output_text({"output": []}) == NO_RESULTS_CONTENT
    assert extract_output_text({}) == NO_RESULTS_CONTENT


def test_extract_citations_with_url_citations() -> None:
    # Mirrors client.rs test_extract_citations_with_url_citations
    response = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1234567890,
        "status": "completed",
        "model": "test-model",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Here is some info about Rust.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://www.rust-lang.org/",
                                "title": "Rust Programming Language",
                                "start_index": 0,
                                "end_index": 10,
                            },
                            {
                                "type": "url_citation",
                                "url": "https://docs.rs/",
                                "title": "Docs.rs",
                                "start_index": 11,
                                "end_index": 20,
                            },
                        ],
                    }
                ],
            }
        ],
    }
    assert extract_output_text(response) == "Here is some info about Rust."
    citations = extract_citations(response)
    assert citations == ["https://www.rust-lang.org/", "https://docs.rs/"]
    pairs = extract_citation_pairs(response)
    assert pairs[0] == ("Rust Programming Language", "https://www.rust-lang.org/")


def test_extract_citations_deduplicates() -> None:
    response = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "dup",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://example.com/page1",
                                "title": "Page 1",
                            },
                            {
                                "type": "url_citation",
                                "url": "https://example.com/page2",
                                "title": "Page 2",
                            },
                            {
                                "type": "url_citation",
                                "url": "https://example.com/page1",
                                "title": "Page 1 Again",
                            },
                        ],
                    }
                ],
            }
        ]
    }
    assert extract_citations(response) == [
        "https://example.com/page1",
        "https://example.com/page2",
    ]


def test_extract_citations_multiple_messages() -> None:
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "First",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://first.com/",
                                "title": "First",
                            }
                        ],
                    }
                ],
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Second",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://second.com/",
                                "title": "Second",
                            }
                        ],
                    }
                ],
            },
        ]
    }
    assert extract_citations(response) == [
        "https://first.com/",
        "https://second.com/",
    ]
    assert extract_output_text(response) == "FirstSecond"


def test_extract_ignores_non_url_annotations() -> None:
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Some text",
                        "annotations": [
                            {"type": "file_citation", "file_id": "f1"},
                            {
                                "type": "url_citation",
                                "url": "https://valid.com/",
                                "title": "Valid",
                            },
                        ],
                    }
                ],
            }
        ]
    }
    assert extract_citations(response) == ["https://valid.com/"]


# ── config ───────────────────────────────────────────────────────────


def test_config_default_disabled_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "CODEDOGGY_WEB_SEARCH_API_KEY",
        "XAI_API_KEY",
        "CODEDOGGY_API_KEY",
        "OPENAI_API_KEY",
        "CODEDOGGY_WEB_SEARCH_URL",
        "CODEDOGGY_PROVIDER",
        "CODEDOGGY_MODEL_PROVIDER",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CODEDOGGY_WEB_SEARCH_ENABLED", "1")
    monkeypatch.setattr(
        "codedoggy.tools.util.web_search_api.resolve_provider_token",
        lambda *_a, **_k: (None, ""),
    )
    cfg = WebSearchConfig.from_env()
    assert not cfg.is_enabled()
    assert "API key" in cfg.reason_disabled or "login" in cfg.reason_disabled.lower()


def test_config_enabled_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEDOGGY_WEB_SEARCH_API_KEY", "sk-test")
    monkeypatch.setenv("CODEDOGGY_WEB_SEARCH_BASE_URL", "https://api.x.ai/v1")
    monkeypatch.setenv("CODEDOGGY_WEB_SEARCH_MODEL", "custom-enterprise-model")
    cfg = WebSearchConfig.from_env()
    assert cfg.enabled
    assert cfg.api_key == "sk-test"
    assert cfg.model == "custom-enterprise-model"
    red = cfg.redacted()
    assert red.api_key == "***REDACTED***"
    assert red.model == "custom-enterprise-model"


def test_config_force_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEDOGGY_WEB_SEARCH_API_KEY", "sk-test")
    monkeypatch.setenv("CODEDOGGY_WEB_SEARCH_ENABLED", "0")
    cfg = WebSearchConfig.from_env()
    assert not cfg.enabled


# ── tool surface ─────────────────────────────────────────────────────


def test_schema_aligned_with_grok() -> None:
    tools = ToolRegistryBuilder.new().finalize()
    defs = {s.name: s for s in tools.tool_definitions()}
    assert "web_search" in defs
    schema = defs["web_search"].parameters or {}
    props = schema.get("properties") or {}
    assert "query" in props
    assert "allowed_domains" in props
    assert "num_results" not in props
    assert schema.get("required") == ["query"]
    desc = defs["web_search"].description or ""
    assert "coding" in desc
    assert "Search the web" in desc


def test_web_search_not_supported_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for k in (
        "CODEDOGGY_WEB_SEARCH_API_KEY",
        "XAI_API_KEY",
        "CODEDOGGY_API_KEY",
        "OPENAI_API_KEY",
        "CODEDOGGY_WEB_SEARCH_URL",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CODEDOGGY_WEB_SEARCH_ENABLED", "1")
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    with pytest.raises(ToolError) as ei:
        tools.call("web_search", {"query": "rust lang"}, ctx)
    assert ei.value.code == "not_supported"
    assert (
        "API key" in ei.value.message
        or "not supported" in ei.value.message.lower()
        or "disabled" in ei.value.message.lower()
    )


def test_web_search_with_mock_client(tmp_path: Path) -> None:
    class Client:
        def search(self, query: str, allowed_domains):
            return WebSearchResult(
                query=query,
                content="Rust is a systems language.",
                citations=["https://www.rust-lang.org/"],
                allowed_domains=allowed_domains,
            )

    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path, extra={"web_search_client": Client()})
    out = tools.call("web_search", {"query": "rust"}, ctx)
    assert out.startswith('Web search results for: "rust"')
    assert "systems language" in out


def test_web_search_requires_query(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path, extra={
        "web_search_client": type("C", (), {"search": staticmethod(lambda q, a: "x")})()
    })
    with pytest.raises(ToolError) as ei:
        tools.call("web_search", {}, ctx)
    assert ei.value.code == "invalid_arguments"


def test_web_search_allowed_domains_validation(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={
            "web_search_client": type(
                "C", (), {"search": staticmethod(lambda q, a: "ok")}
            )()
        },
    )
    with pytest.raises(ToolError) as ei:
        tools.call(
            "web_search",
            {"query": "q", "allowed_domains": "not-a-list"},
            ctx,
        )
    assert ei.value.code == "invalid_arguments"
