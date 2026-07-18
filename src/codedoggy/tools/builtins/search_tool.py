"""search_tool — Grok SearchTool (source-level).

Source: implementations/search_tool/mod.rs
  - description_template
  - empty index → JSON note (exact string)
  - search_snapshot → group by server → status ready/partial
Pure: grok_build/search_tool_logic.py
Index: tools/mcp/tool_index.py (Grok shell Bm25ToolSearchIndex)
"""

from __future__ import annotations

import json
from typing import Any

from codedoggy.tools.grok_build.search_tool_logic import (
    MAX_MCP_DESCRIPTION_LENGTH,
    NO_MCP_CONFIGURED_JSON,
    NO_MCP_CONFIGURED_NOTE,
    SEARCH_TOOL_DESCRIPTION,
    ServerSummary,
    build_server_reminder,
    format_search_snapshot_response,
    sanitize_description,
    truncate_description,
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


class SearchToolTool(Tool):
    def id(self) -> ToolId:
        return ToolId("search_tool")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.SearchTool

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="search_tool", description=SEARCH_TOOL_DESCRIPTION)

    def parameters_schema(self) -> dict[str, Any]:
        # Grok SearchToolInput
        return {
            "type": "object",
            "properties": {
                # Grok SearchToolInput
                "query": {
                    "type": "string",
                    "description": (
                        "Keywords to match against tool names, server names, "
                        "and descriptions. Include the server name and action "
                        "for best results (e.g. \"linear create issue\", "
                        '"slack read thread history").'
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5).",
                    "minimum": 0,
                    "maximum": 255,
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
        # Grok: shell injects ToolIndex after MCP init
        try:
            from codedoggy.tools.mcp.tool_index import ensure_mcp_tool_index

            ensure_mcp_tool_index(extra)
        except Exception:  # noqa: BLE001
            pass

        from codedoggy.tools.mcp.types import unwrap_tool_index

        index = unwrap_tool_index(extra.get("mcp_tool_index"))
        if index is None:
            # Grok: no ToolIndex in resources
            return json.dumps(NO_MCP_CONFIGURED_JSON, indent=2, ensure_ascii=False)

        # Grok ToolSearchIndex::search_snapshot only
        try:
            result = index.search_snapshot(query, limit)
        except TypeError:
            try:
                result = index.search_snapshot(query, limit=limit)  # type: ignore[call-arg]
            except Exception as e:  # noqa: BLE001
                raise ToolError(f"search_tool failed: {e}", code="mcp_error") from e
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"search_tool failed: {e}", code="mcp_error") from e
        return _format_grok_snapshot(result)


def _format_grok_snapshot(result: Any) -> str:
    """Map SearchSnapshot / dict → Grok grouped JSON (mod.rs run)."""
    if isinstance(result, str):
        return result
    results = getattr(result, "results", None)
    total_hidden = getattr(result, "total_hidden_tools", 0)
    is_ready = getattr(result, "is_ready", True)
    if results is None and isinstance(result, dict):
        results = result.get("results")
        total_hidden = result.get("total_hidden_tools", 0)
        is_ready = result.get("is_ready", True)
        if result.get("note") == NO_MCP_CONFIGURED_NOTE:
            return json.dumps(result, indent=2, ensure_ascii=False)
    if not results:
        return format_search_snapshot_response(
            [],
            total_hidden_tools=int(total_hidden or 0),
            is_ready=bool(is_ready),
        )
    return format_search_snapshot_response(
        list(results),
        total_hidden_tools=int(total_hidden or 0),
        is_ready=bool(is_ready),
    )


def mcp_server_reminder_from_extra(extra: dict[str, Any] | None) -> str | None:
    bag = extra or {}
    index = bag.get("mcp_tool_index")
    if index is not None:
        list_fn = getattr(index, "list_server_summaries", None)
        if callable(list_fn):
            try:
                summaries = list_fn()
                from codedoggy.tools.mcp.tool_index import ServerSummary as IdxSum

                servers: list[ServerSummary] = []
                for s in summaries or []:
                    if isinstance(s, ServerSummary):
                        servers.append(s)
                    elif hasattr(s, "name"):
                        servers.append(
                            ServerSummary(
                                name=s.name,
                                tool_count=int(getattr(s, "tool_count", 0) or 0),
                                description=getattr(s, "description", None),
                                tool_names=list(getattr(s, "tool_names", []) or []),
                            )
                        )
                return build_server_reminder(servers)
            except Exception:  # noqa: BLE001
                pass
    raw = bag.get("mcp_servers")
    if not isinstance(raw, list) or not raw:
        return None
    servers = []
    for item in raw:
        if isinstance(item, dict) and item.get("name"):
            servers.append(
                ServerSummary(
                    name=str(item["name"]),
                    tool_count=int(item.get("tool_count") or item.get("count") or 0),
                    description=item.get("description"),
                    tool_names=item.get("tool_names") or [],
                )
            )
    return build_server_reminder(servers)


__all__ = [
    "MAX_MCP_DESCRIPTION_LENGTH",
    "SearchToolTool",
    "mcp_server_reminder_from_extra",
    "truncate_description",
    "sanitize_description",
]
