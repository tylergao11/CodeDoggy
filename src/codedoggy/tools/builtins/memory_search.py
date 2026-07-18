"""memory_search — Grok MemorySearchImpl wire surface only.

Source: implementations/memory/search_tool.rs

Without a real MemoryBackend (extra['memory_backend']), return Grok soft text.
No invented token-scoring or FTS-as-memory-backend.

Host injects extra['memory_backend'] with:
  search(query, max_results=..., min_score=...) -> list of result objects/dicts
  fields: score, source, path, start_line, end_line, snippet, optional staleness_note
"""

from __future__ import annotations

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
Search cross-session memory for relevant knowledge chunks. Returns ranked results \
from global, workspace, and session memory files.

Use this proactively when:
- A question references prior work, decisions, or context you don't have
- You need project conventions, coding patterns, or user preferences
- The user mentions something discussed or decided in a previous session
- Starting work in an unfamiliar part of the codebase
- After compaction when prior context may have been lost
"""

_DISABLED = "Memory is not enabled. Use --experimental-memory to enable."


class MemorySearchTool(Tool):
    def id(self) -> ToolId:
        return ToolId("memory_search")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.MemorySearch

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="memory_search", description=_DESC.strip())

    def parameters_schema(self) -> dict[str, Any]:
        # Grok MemorySearchInput
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query string. Use specific technical terms rather "
                        "than conversational language. Good: \"authentication middleware patterns\". "
                        "Bad: \"that thing we discussed about auth\"."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        "Maximum number of results to return. When omitted the "
                        "backend-configured value is used."
                    ),
                },
                "min_score": {
                    "type": "number",
                    "description": (
                        "Minimum relevance score threshold. When omitted the "
                        "backend-configured value is used."
                    ),
                },
            },
            "required": ["query"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError.invalid_arguments("query is required")

        backend = (ctx.extra or {}).get("memory_backend")
        if backend is None:
            # Grok search_tool.rs when MemoryBackend absent
            return _DISABLED

        search = getattr(backend, "search", None)
        if not callable(search):
            return _DISABLED

        max_results = args.get("max_results")
        min_score = args.get("min_score")
        try:
            results = search(
                query.strip(),
                max_results=max_results,
                min_score=min_score,
            )
        except TypeError:
            # allow backends that take positional only
            try:
                results = search(query.strip(), max_results, min_score)
            except Exception as e:  # noqa: BLE001
                raise ToolError(f"memory search failed: {e}", code="memory_error") from e
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"memory search failed: {e}", code="memory_error") from e

        if not results:
            return "No memory results found for query."

        # Grok format (search_tool.rs)
        parts = [f"Found {len(results)} memory result(s):\n"]
        for i, r in enumerate(results, 1):
            if isinstance(r, dict):
                score = float(r.get("score", 0))
                source = r.get("source", "")
                path = r.get("path", "")
                start = r.get("start_line", 1)
                end = r.get("end_line", 1)
                snippet = r.get("snippet", "")
                staleness = r.get("staleness_note", "") or ""
            else:
                score = float(getattr(r, "score", 0))
                source = getattr(r, "source", "")
                path = getattr(r, "path", "")
                start = getattr(r, "start_line", 1)
                end = getattr(r, "end_line", 1)
                snippet = getattr(r, "snippet", "")
                staleness = getattr(r, "staleness_note", "") or ""
            parts.append(
                f"\n### Result {i} (score: {score:.2f}, source: {source})\n"
                f"**File:** {path} (lines {start}-{end})\n"
                f"{staleness}"
                f"```\n{snippet}\n```\n"
            )
        return "".join(parts)
