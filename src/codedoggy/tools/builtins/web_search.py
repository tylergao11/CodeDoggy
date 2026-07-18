"""web_search — Grok WebSearchTool wire + Responses API client.

Core logic: ``codedoggy.tools.grok_build.web_search``
  Ported from implementations/grok_build/web_search/mod.rs
  + implementations/web_search/client.rs + types.rs
  + types/output.rs WebSearchOutput prompt format

HTTP: ``codedoggy.tools.util.web_search_api``
  POST {base}/responses with tools=[{type: web_search}] when API key present.
  not_supported when disabled / no key (no mock ranking default).

Optional test override: ctx.extra['web_search_client'] with .search(query, allowed_domains).
"""

from __future__ import annotations

from typing import Any

from codedoggy.tools.defaults import DEFAULT_TOOL_OUTPUT_CHARS
from codedoggy.tools.grok_build.web_search import (
    ALLOWED_DOMAINS_PARAM_DESC,
    DESCRIPTION_TEMPLATE,
    QUERY_PARAM_DESC,
    format_prompt_output,
)
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolError,
    ToolId,
)
from codedoggy.tools.util.web_search_api import (
    WebSearchError,
    WebSearchNotSupported,
    WebSearchResult,
    format_result,
    search as api_search,
)


class WebSearchTool(Tool):
    def id(self) -> ToolId:
        return ToolId("web_search")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.WebSearch

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="web_search", description=DESCRIPTION_TEMPLATE)

    def parameters_schema(self) -> dict[str, Any]:
        # Grok WebSearchInput (mod.rs): query + optional allowed_domains
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": QUERY_PARAM_DESC,
                },
                "allowed_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": ALLOWED_DOMAINS_PARAM_DESC,
                },
            },
            "required": ["query"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError.invalid_arguments("query is required")
        query = query.strip()

        allowed = args.get("allowed_domains")
        allowed_domains: list[str] | None = None
        if allowed is not None:
            if not isinstance(allowed, list):
                raise ToolError.invalid_arguments(
                    "allowed_domains must be an array of strings"
                )
            domains: list[str] = []
            for d in allowed:
                if not isinstance(d, str) or not d.strip():
                    raise ToolError.invalid_arguments(
                        "allowed_domains must be an array of strings"
                    )
                domains.append(d.strip())
            allowed_domains = domains

        # Optional test/host override (same pattern as image_gen_client)
        client = (ctx.extra or {}).get("web_search_client")
        if client is not None and callable(getattr(client, "search", None)):
            try:
                raw = client.search(query, allowed_domains)
            except Exception as e:  # noqa: BLE001
                raise ToolError(
                    f"web_search failed: {e}",
                    code="search_failed",
                ) from e
            return _coerce_client_result(query, raw, allowed_domains)

        try:
            result = api_search(query, allowed_domains=allowed_domains)
        except WebSearchNotSupported as e:
            raise ToolError(e.message, code=e.code) from e
        except WebSearchError as e:
            raise ToolError(e.message, code=e.code) from e

        text = format_result(result)
        if len(text) > DEFAULT_TOOL_OUTPUT_CHARS:
            text = text[: DEFAULT_TOOL_OUTPUT_CHARS - 20] + "\n…"
        return text


def _coerce_client_result(
    query: str,
    raw: Any,
    allowed_domains: list[str] | None,
) -> str:
    """Accept WebSearchResult, (content, citations), or str from test clients."""
    if isinstance(raw, WebSearchResult):
        text = format_result(raw)
    elif isinstance(raw, tuple) and len(raw) >= 1:
        content = str(raw[0])
        text = format_prompt_output(query, content)
    elif isinstance(raw, dict):
        content = str(raw.get("content") or raw.get("text") or "")
        text = format_prompt_output(query, content or "No search results found.")
    elif isinstance(raw, str):
        # If already formatted with header, pass through; else wrap.
        if raw.startswith("Web search results for:"):
            text = raw
        else:
            text = format_prompt_output(query, raw)
    else:
        text = format_prompt_output(query, str(raw))
    if len(text) > DEFAULT_TOOL_OUTPUT_CHARS:
        text = text[: DEFAULT_TOOL_OUTPUT_CHARS - 20] + "\n…"
    return text
