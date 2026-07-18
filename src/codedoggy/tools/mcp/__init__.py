"""MCP interfaces — Grok tool_index + shell Bm25 (public surface).

Grok layout:
  xai-grok-tools types/tool_index.rs  →  types.py (trait + DTOs + ToolIndex wrap)
  xai-grok-shell session/tool_index.rs →  tool_index.py (Bm25ToolSearchIndex)
"""

from codedoggy.tools.mcp.tool_index import (
    Bm25ToolSearchIndex,
    ServerMetadata,
    ToolMetadata,
    ToolMetadataSnapshot,
    ensure_mcp_tool_index,
    index_from_mcp_tools,
    normalize_query,
    split_identifier,
    tools_from_mcp_catalog,
)
from codedoggy.tools.mcp.types import (
    McpDispatch,
    SearchSnapshot,
    ServerSummary,
    ToolIndex,
    ToolSearchIndex,
    ToolSearchResult,
    unwrap_tool_index,
)

__all__ = [
    "Bm25ToolSearchIndex",
    "McpDispatch",
    "SearchSnapshot",
    "ServerMetadata",
    "ServerSummary",
    "ToolIndex",
    "ToolMetadata",
    "ToolMetadataSnapshot",
    "ToolSearchIndex",
    "ToolSearchResult",
    "ensure_mcp_tool_index",
    "index_from_mcp_tools",
    "normalize_query",
    "split_identifier",
    "tools_from_mcp_catalog",
    "unwrap_tool_index",
]
