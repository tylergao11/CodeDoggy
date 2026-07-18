"""search_tool — Grok SearchTool wire surface only.

Source: implementations/search_tool/mod.rs + types/tool_index.rs

Grok injects Arc<dyn ToolSearchIndex> (BM25 lives in xai-grok-shell, not tools).
CodeDoggy does **not** ship a BM25 registry. Host must provide either:

  extra['mcp_tool_index']  — object with search(query, limit=...) or search_snapshot
  extra['mcp_tools']       — list[dict] catalog (simple token filter, not BM25)

No invented mcp_registry.py.
"""

from __future__ import annotations

import json
from typing import Any

from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolError,
    ToolId,
)

_DESC = """\
Search for MCP tools by keyword and retrieve their input schemas.

If status is "partial", some servers may still be connecting.
Include the server name and action for best results (e.g. "linear create issue").
"""

MAX_MCP_DESCRIPTION_LENGTH = 2048
_TRUNC = "… [truncated]"


def truncate_description(s: str) -> str:
    """Port of search_tool::truncate_description (char-boundary)."""
    if len(s) <= MAX_MCP_DESCRIPTION_LENGTH or len(s) <= MAX_MCP_DESCRIPTION_LENGTH:
        # Grok also checks chars().count(); for ASCII-heavy schemas len is fine.
        if sum(1 for _ in s) <= MAX_MCP_DESCRIPTION_LENGTH:
            return s
    budget = MAX_MCP_DESCRIPTION_LENGTH - len(_TRUNC)
    return "".join(list(s)[:budget]) + _TRUNC


class SearchToolTool(Tool):
    def id(self) -> ToolId:
        return ToolId("search_tool")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.SearchTool

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="search_tool", description=_DESC.strip())

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Keywords to match against tool names, server names, and descriptions."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5).",
                    "minimum": 0,
                    "maximum": 50,
                },
            },
            "required": ["query"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        query = args.get("query")
        if not isinstance(query, str):
            raise ToolError.invalid_arguments("query is required")
        limit = args.get("limit")
        if limit is None:
            limit = 5
        limit = max(0, int(limit))

        extra = ctx.extra or {}
        index = extra.get("mcp_tool_index")
        if index is not None:
            snap = getattr(index, "search_snapshot", None)
            if callable(snap):
                try:
                    result = snap(query, limit)
                except Exception as e:  # noqa: BLE001
                    raise ToolError(f"search_tool failed: {e}", code="mcp_error") from e
                return _format_snapshot(result)
            search = getattr(index, "search", None)
            if callable(search):
                try:
                    result = search(query, limit=limit)
                except Exception as e:  # noqa: BLE001
                    raise ToolError(f"search_tool failed: {e}", code="mcp_error") from e
                if isinstance(result, str):
                    return result
                return str(result)

        tools = extra.get("mcp_tools")
        if isinstance(tools, list):
            if not tools:
                return "No MCP tools registered."
            return _filter_catalog(tools, query, limit)

        return "No MCP tools registered."


def _filter_catalog(tools: list[Any], query: str, limit: int) -> str:
    """Host-provided catalog: simple multi-token substring filter (not BM25)."""
    tokens = [t.lower() for t in query.split() if t.strip()]
    scored: list[tuple[int, dict[str, Any]]] = []
    for raw in tools:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "")
        desc = str(raw.get("description") or "")
        server = str(raw.get("server") or "")
        hay = f"{name} {desc} {server}".lower()
        if tokens:
            hits = sum(1 for t in tokens if t in hay)
            if hits == 0:
                continue
            score = hits
        else:
            score = 1
        scored.append((score, raw))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("name") or "")))
    if limit == 0:
        return "No matching MCP tools."
    top = scored[:limit]
    if not top:
        return "No matching MCP tools."
    parts = [f"Found {len(top)} MCP tool(s):\n"]
    for score, t in top:
        name = t.get("name", "")
        desc = truncate_description(str(t.get("description") or ""))
        server = t.get("server") or ""
        schema = t.get("parameters") or t.get("input_schema") or {}
        parts.append(f"\n### {name} (score≈{score}, server: {server})\n")
        if desc:
            parts.append(f"{desc}\n")
        if schema:
            parts.append("input_schema:\n")
            parts.append(json.dumps(schema, ensure_ascii=False, indent=2))
            parts.append("\n")
    return "".join(parts)


def _format_snapshot(result: Any) -> str:
    if isinstance(result, str):
        return result
    results = getattr(result, "results", None)
    if results is None and isinstance(result, dict):
        results = result.get("results")
    if not results:
        return "No matching MCP tools."
    parts = [f"Found {len(results)} MCP tool(s):\n"]
    for r in results:
        if isinstance(r, dict):
            name = r.get("tool_name") or r.get("name") or ""
            server = r.get("server_name") or r.get("server") or ""
            desc = truncate_description(str(r.get("description") or ""))
            score = r.get("score", 0)
            schema = r.get("input_schema") or {}
        else:
            name = getattr(r, "tool_name", getattr(r, "name", ""))
            server = getattr(r, "server_name", getattr(r, "server", ""))
            desc = truncate_description(str(getattr(r, "description", "") or ""))
            score = getattr(r, "score", 0)
            schema = getattr(r, "input_schema", {}) or {}
        parts.append(f"\n### {name} (score: {score:.2f}, server: {server})\n")
        if desc:
            parts.append(f"{desc}\n")
        if schema:
            parts.append("input_schema:\n")
            parts.append(
                json.dumps(schema, ensure_ascii=False, indent=2)
                if not isinstance(schema, str)
                else schema
            )
            parts.append("\n")
    return "".join(parts)
