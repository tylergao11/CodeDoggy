"""web_search pure logic — source port from Grok.

Ported from grok-build:
  crates/codegen/xai-grok-tools/src/implementations/grok_build/web_search/mod.rs
  crates/codegen/xai-grok-tools/src/implementations/web_search/client.rs
  crates/codegen/xai-grok-tools/src/implementations/web_search/types.rs
  crates/codegen/xai-grok-tools/src/types/output.rs  (WebSearchOutput prompt format)

Function map:
  DESCRIPTION_TEMPLATE          ← ToolMetadata::description_template
  WebSearchInput fields         ← WebSearchInput { query, allowed_domains }
  build_search_request          ← WebSearchClient::search request body
  extract_output_text           ← Response::output_text + "No search results found."
  extract_citations             ← extract_citations
  extract_citation_pairs        ← extract_citation_pairs
  format_prompt_output          ← ToolOutput::WebSearch to_prompt_format
  NO_RESULTS_CONTENT            ← "No search results found."
  DEFAULT_MAX_OUTPUT_TOKENS     ← max_output_tokens 8192
  DEFAULT_TEMPERATURE / TOP_P   ← 0.1 / 0.95
"""

from __future__ import annotations

from typing import Any

# Grok ToolMetadata::description_template
DESCRIPTION_TEMPLATE: str = (
    "Search the web for up-to-date information, tailored for coding "
    "and software development tasks."
)

# Grok schemars descriptions on WebSearchInput
QUERY_PARAM_DESC: str = "The search query to perform."
ALLOWED_DOMAINS_PARAM_DESC: str = "Optional list of domains to restrict search to."

# client.rs CreateResponseArgs defaults
DEFAULT_MAX_OUTPUT_TOKENS: int = 8192
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_TOP_P: float = 0.95
DEFAULT_STORE: bool = False

# client.rs / Response::output_text fallback
NO_RESULTS_CONTENT: str = "No search results found."

# types.rs redacted api_key
REDACTED_API_KEY: str = "***REDACTED***"

# Grok workspace default_web_search_model
DEFAULT_WEB_SEARCH_MODEL: str = "grok-4.20-multi-agent"
DEFAULT_BASE_URL: str = "https://api.x.ai/v1"


def build_search_request(
    query: str,
    model: str,
    allowed_domains: list[str] | None = None,
) -> dict[str, Any]:
    """Build Responses API POST body matching WebSearchClient::search.

    Grok:
      WebSearchToolArgs::default().filters(WebSearchToolFilters { allowed_domains })
      CreateResponseArgs: model, input=query, tools=[WebSearch], store=false,
      temperature=0.1, top_p=0.95, max_output_tokens=8192
    """
    tool: dict[str, Any] = {"type": "web_search"}
    if allowed_domains is not None:
        tool["filters"] = {"allowed_domains": list(allowed_domains)}
    return {
        "model": model,
        "input": query,
        "tools": [tool],
        "store": DEFAULT_STORE,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
    }


def extract_output_text(response: dict[str, Any]) -> str:
    """Join assistant output_text parts; fallback to NO_RESULTS_CONTENT.

    Mirrors async-openai Response::output_text() used in client.rs.
    """
    parts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        # Grok only walks OutputItem::Message
        if item.get("type") is not None and item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            ctype = content.get("type")
            if ctype in ("output_text", "text") and content.get("text") is not None:
                parts.append(str(content["text"]))
    text = "".join(parts).strip()
    if text:
        return text
    # Some providers put convenience field on the root
    root = response.get("output_text")
    if isinstance(root, str) and root.strip():
        return root.strip()
    return NO_RESULTS_CONTENT


def extract_citations(response: dict[str, Any]) -> list[str]:
    """Unique citation URLs in first-seen order (client.rs extract_citations)."""
    return [url for _title, url in extract_citation_pairs(response)]


def extract_citation_pairs(response: dict[str, Any]) -> list[tuple[str, str]]:
    """(title, url) pairs, URL-deduped, first-seen order.

    Port of client.rs extract_citation_pairs: walk message → output_text →
    annotations of type url_citation; empty title when missing.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") is not None and item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            for ann in content.get("annotations") or []:
                if not isinstance(ann, dict):
                    continue
                # Grok: Annotation::UrlCitation only
                if ann.get("type") is not None and ann.get("type") != "url_citation":
                    continue
                url = ann.get("url")
                if not isinstance(url, str) or not url:
                    continue
                if url in seen:
                    continue
                seen.add(url)
                title = ann.get("title")
                title_s = title if isinstance(title, str) else ""
                pairs.append((title_s, url))
    return pairs


def format_prompt_output(
    query: str,
    content: str,
    *,
    pre_formatted: str | None = None,
) -> str:
    """Model-visible string for WebSearchOutput (output.rs ToolOutput::WebSearch).

    Grok:
      if pre_formatted: return pre
      else: format!("Web search results for: \"{}\"\\n\\n{}", query, content)
    """
    if pre_formatted is not None:
        return pre_formatted
    return f'Web search results for: "{query}"\n\n{content}'


def redacted_config_api_key() -> str:
    """types.rs WebSearchConfig::redacted api_key value."""
    return REDACTED_API_KEY
