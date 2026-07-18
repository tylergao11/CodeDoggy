"""Grok MCP / tool-index public types — xai-grok-tools ``types/tool_index.rs``.

Every field name matches Grok. Host injects ``ToolIndex`` (or a
``ToolSearchIndex`` implementor) via ``extra['mcp_tool_index']``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolSearchResult:
    """Grok ``ToolSearchResult``."""

    tool_name: str
    server_name: str
    description: str
    score: float
    parameters: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchSnapshot:
    """Grok ``SearchSnapshot``."""

    results: list[ToolSearchResult]
    total_hidden_tools: int = 0
    is_ready: bool = True


@dataclass
class ServerSummary:
    """Grok ``ServerSummary``."""

    name: str
    description: str | None = None
    tool_count: int = 0
    tool_names: list[str] = field(default_factory=list)


@runtime_checkable
class ToolSearchIndex(Protocol):
    """Grok ``ToolSearchIndex`` trait — backend-agnostic."""

    def search_snapshot(self, query: str, limit: int) -> SearchSnapshot:
        """Search and return results + metadata from one consistent snapshot."""
        ...

    def list_server_summaries(self) -> list[ServerSummary]:
        """List unique MCP servers with tool counts (system-reminder)."""
        ...


@dataclass
class ToolIndex:
    """Grok ``ToolIndex`` resource wrapper around a ``ToolSearchIndex``.

    Shell injects this after MCP init. ``extra['mcp_tool_index']`` may be either
    a bare ``ToolSearchIndex`` or a ``ToolIndex`` wrapping one.
    """

    index: ToolSearchIndex

    def search_snapshot(self, query: str, limit: int) -> SearchSnapshot:
        return self.index.search_snapshot(query, limit)

    def list_server_summaries(self) -> list[ServerSummary]:
        return self.index.list_server_summaries()

    # Convenience for use_tool schema resolution (host glue)
    def get(self, tool_name: str) -> ToolSearchResult | None:
        get = getattr(self.index, "get", None)
        if callable(get):
            return get(tool_name)
        # Fall back: exact search
        snap = self.index.search_snapshot(tool_name, 1)
        for r in snap.results:
            if r.tool_name == tool_name or r.tool_name.endswith(f"__{tool_name}"):
                return r
        return None

    def lookup(self, tool_name: str) -> ToolSearchResult | None:
        return self.get(tool_name)

    def get_schema(self, tool_name: str) -> dict[str, Any] | None:
        r = self.get(tool_name)
        return dict(r.input_schema) if r and r.input_schema else None

    def schema_for(self, tool_name: str) -> dict[str, Any] | None:
        return self.get_schema(tool_name)


@runtime_checkable
class McpDispatch(Protocol):
    """Host dispatch for ``use_tool`` — Grok InnerDispatch / gateway spirit.

    ``callable(tool_name, tool_input) -> str | dict``
    """

    def __call__(self, tool_name: str, tool_input: dict[str, Any]) -> Any: ...


def unwrap_tool_index(obj: Any) -> ToolSearchIndex | None:
    """Accept bare index or ``ToolIndex`` wrapper (Grok ToolIndex.0)."""
    if obj is None:
        return None
    if isinstance(obj, ToolIndex):
        return obj.index
    if hasattr(obj, "search_snapshot") and callable(obj.search_snapshot):
        return obj  # type: ignore[return-value]
    if hasattr(obj, "index") and hasattr(obj.index, "search_snapshot"):
        return obj.index
    return None
