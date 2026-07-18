"""lsp — Grok LspTool wire surface only.

Source description: grok_build/lsp/mod.rs description_template.
Runtime: Grok requires LspBackend in Resources. We do **not** invent a graph
fallback. Without extra['lsp_backend'], return Grok's unavailable wording.
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

# Grok LspOperation display names (lsp/types.rs)
_OPS = (
    "goToDefinition",
    "findReferences",
    "hover",
    "goToImplementation",
    "documentSymbol",
    "workspaceSymbol",
)

# Grok description_template (template vars expanded to plain tool names)
_DESC = """\
Code intelligence via language servers. Prefer over grep/read_file for understanding code.
Operations: goToDefinition (jump to where a symbol is defined), findReferences (all usages of a symbol), hover (type info/docs at a position), goToImplementation (trait/interface implementations), documentSymbol (list all symbols in a file), workspaceSymbol (search symbols by name across the workspace — requires query parameter, not file_path).
Requires file_path + line + character for position-based operations.
"""

# Grok tool error when backend missing (lsp/mod.rs)
_UNAVAILABLE = (
    "LSP tool is unavailable. Configure ~/.grok/lsp.json or <cwd>/.grok/lsp.json "
    "and ensure the language server can start."
)


class LspTool(Tool):
    def id(self) -> ToolId:
        return ToolId("lsp")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Lsp

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="lsp", description=_DESC.strip())

    def parameters_schema(self) -> dict[str, Any]:
        # Grok LspToolInput (types.rs)
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": list(_OPS),
                    "description": "The LSP operation to perform.",
                },
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file.",
                },
                "line": {
                    "type": "integer",
                    "description": "0-indexed line number.",
                },
                "character": {
                    "type": "integer",
                    "description": "0-indexed column number.",
                },
                "query": {
                    "type": "string",
                    "description": "Symbol name or partial name (workspaceSymbol only).",
                },
            },
            "required": ["operation"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        op = str(args.get("operation") or "").strip()
        if op not in _OPS:
            raise ToolError.invalid_arguments(
                f"operation must be one of {list(_OPS)}, got {op!r}"
            )

        backend = (ctx.extra or {}).get("lsp_backend")
        if backend is None:
            # Grok: custom process_manager error with this message
            raise ToolError(_UNAVAILABLE, code="process_manager")

        dispatch = getattr(backend, "dispatch", None) or getattr(backend, "run", None)
        if not callable(dispatch):
            raise ToolError(_UNAVAILABLE, code="process_manager")

        try:
            result = dispatch(args)
        except Exception as e:  # noqa: BLE001
            raise ToolError(str(e), code="process_manager") from e

        if isinstance(result, str):
            return result
        if isinstance(result, dict) and result.get("is_error"):
            raise ToolError(str(result.get("text") or "lsp error"), code="process_manager")
        return json.dumps(result, ensure_ascii=False, indent=2)
