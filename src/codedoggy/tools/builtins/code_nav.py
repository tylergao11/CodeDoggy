"""code_nav — GitHub-style goto-definition / find-references over ScopeGraphIndex.

Mirrors Navigator APIs from xai-codebase-graph.
"""

from __future__ import annotations

import json
from typing import Any

from codedoggy.graph.types import NavigationError, navigation_result_to_dict
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolError,
    ToolId,
)

_DESCRIPTION = """\
Navigate the workspace code graph (GitHub-style go-to-definition / find-references).

Actions:
  - definition: find where a symbol is defined (read-only)
  - references: find call/use sites (optionally include definition; read-only)
  - at_position: resolve identifier at file:line:col then definition (read-only)
  - stats: index size files/defs/refs (read-only)
  - reindex: full rebuild; writes/updates workspace cache (.goto_index.json)

Query actions are Search-kind (allowed under read-only explore). reindex mutates
the on-disk graph cache under the workspace root when caching is enabled.

Prefer symbol name when you know it. Use at_position when you have a cursor
from a prior read_file line. Paths are relative to session cwd.
"""


class CodeNavTool(Tool):
    def id(self) -> ToolId:
        return ToolId("code_nav")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Search

    def description(self, ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="code_nav", description=_DESCRIPTION)

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "definition",
                        "references",
                        "at_position",
                        "stats",
                        "reindex",
                    ],
                    "description": "Navigation action",
                },
                "symbol": {
                    "type": "string",
                    "description": "Symbol name (for definition/references)",
                },
                "file_path": {
                    "type": "string",
                    "description": "Relative path (for at_position / context ranking)",
                },
                "line": {
                    "type": "integer",
                    "description": "1-indexed line (at_position)",
                },
                "col": {
                    "type": "integer",
                    "description": "1-indexed column (at_position); default 1",
                },
                "include_definition": {
                    "type": "boolean",
                    "description": "Include definition sites in references (default true)",
                },
            },
            "required": ["action"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        action = (args.get("action") or "").strip()
        graph = (ctx.extra or {}).get("graph")
        if graph is None:
            raise ToolError(
                "code graph not available on this session (enable_graph / extensions.graph)",
                code="not_available",
            )

        if action == "reindex":
            # reindex mutates the in-memory graph and may write .goto_index.json.
            # ToolKind is Search (read-only explore), so gate only check_read —
            # enforce write policy here, fail closed when policy denies writes.
            policy = (ctx.extra or {}).get("policy")
            if policy is not None:
                check_w = getattr(policy, "check_write", None)
                if callable(check_w):
                    # Sentinel = actual cache path name (not .codedoggy/ which is deny-listed)
                    from codedoggy.graph.cache import CACHE_FILE_NAME

                    wd = check_w(CACHE_FILE_NAME)
                    if wd is not None and not getattr(wd, "allowed", True):
                        raise ToolError(
                            getattr(wd, "reason", None)
                            or f"reindex denied by policy (write to {CACHE_FILE_NAME})",
                            code=getattr(wd, "code", None) or "policy_denied",
                        )
            reindex = getattr(graph, "reindex", None)
            if not callable(reindex):
                raise ToolError("graph.reindex not supported", code="not_available")
            stats = reindex()
            return json.dumps({"ok": True, "stats": stats}, ensure_ascii=False, indent=2)

        if action == "stats":
            stats = getattr(graph, "stats", None)
            if callable(stats):
                s = stats()
                return json.dumps(
                    {
                        "files": getattr(s, "files", s.get("files") if isinstance(s, dict) else 0),
                        "definitions": getattr(
                            s, "definitions", s.get("definitions") if isinstance(s, dict) else 0
                        ),
                        "references": getattr(
                            s, "references", s.get("references") if isinstance(s, dict) else 0
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            raise ToolError("graph.stats not supported", code="not_available")

        nav = getattr(graph, "navigator", None)
        if nav is None:
            get_nav = getattr(graph, "get_navigator", None)
            nav = get_nav() if callable(get_nav) else None
        if nav is None:
            raise ToolError("graph navigator not ready", code="not_available")

        include_def = args.get("include_definition")
        if include_def is None:
            include_def = True

        try:
            if action == "definition":
                symbol = (args.get("symbol") or "").strip()
                if not symbol:
                    raise ToolError.invalid_arguments("symbol is required for definition")
                ctx_file = args.get("file_path")
                result = nav.goto_definition_by_name(symbol, context_file=ctx_file)
                return json.dumps(navigation_result_to_dict(result), ensure_ascii=False, indent=2)

            if action == "references":
                symbol = (args.get("symbol") or "").strip()
                if not symbol:
                    raise ToolError.invalid_arguments("symbol is required for references")
                ctx_file = args.get("file_path")
                result = nav.goto_references_by_name(
                    symbol, context_file=ctx_file, include_definition=bool(include_def)
                )
                return json.dumps(navigation_result_to_dict(result), ensure_ascii=False, indent=2)

            if action == "at_position":
                file_path = (args.get("file_path") or "").strip()
                line = args.get("line")
                if not file_path or line is None:
                    raise ToolError.invalid_arguments(
                        "file_path and line are required for at_position"
                    )
                col = int(args.get("col") or 1)
                line = int(line)
                # Prefer definition; optional include_definition=false → references only
                if args.get("include_definition") is False:
                    result = nav.goto_references(
                        file_path, line, col, include_definition=False
                    )
                else:
                    result = nav.goto_definition(file_path, line, col)
                return json.dumps(navigation_result_to_dict(result), ensure_ascii=False, indent=2)

            raise ToolError.invalid_arguments(f"unknown action: {action}")
        except NavigationError as e:
            raise ToolError(str(e), code=e.kind) from e
