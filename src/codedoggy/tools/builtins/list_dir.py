"""list_dir — Grok ListDirTool wire + source-ported tree/BFS.

Core algorithm: ``codedoggy.tools.grok_build.list_dir``
  Ported from implementations/grok_build/list_dir/mod.rs
"""

from __future__ import annotations

from typing import Any

from codedoggy.tools.defaults import LIST_DIR_MAX_OUTPUT_CHARS
from codedoggy.tools.grok_build.list_dir import (
    compute_display_path,
    render_list_dir,
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
from codedoggy.tools.util.paths import resolve_model_path

# Grok description_template (params placeholder expanded)
_DESCRIPTION = """\
Lists files and directories in a given path.
The 'target_directory' parameter can be relative to the workspace root or absolute.

Other details:
    - The result does not display dot-files and dot-directories.
    - Respects .gitignore patterns (files/directories ignored by git are not shown).
    - Large directories are summarized with file counts and extension breakdowns instead of listing all files.
"""


class ListDirTool(Tool):
    def id(self) -> ToolId:
        return ToolId("list_dir")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.ListDir

    def description(self, ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="list_dir", description=_DESCRIPTION.strip())

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_directory": {
                    "type": "string",
                    "description": (
                        "Path to directory to list contents of, relative to the "
                        "workspace root or absolute."
                    ),
                },
            },
            "required": ["target_directory"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        raw = args.get("target_directory")
        if not isinstance(raw, str):
            raise ToolError.invalid_arguments("target_directory is required")

        display_path = compute_display_path(ctx.cwd, raw)
        path = resolve_model_path(ctx.cwd, raw.strip() or ".")

        if not path.exists():
            raise ToolError(
                f"Error: {display_path} does not exist or is not a valid directory.",
                code="not_found",
            )
        if path.is_file():
            raise ToolError(
                f"Error: {display_path} is a file, not a directory.",
                code="invalid_arguments",
            )
        if not path.is_dir():
            raise ToolError(
                f"Error: {display_path} is not a valid directory.",
                code="invalid_arguments",
            )

        respect = True
        if ctx.extra and "respect_gitignore" in ctx.extra:
            respect = bool(ctx.extra["respect_gitignore"])

        max_chars = LIST_DIR_MAX_OUTPUT_CHARS
        if ctx.extra and ctx.extra.get("list_dir_max_output_chars") is not None:
            max_chars = int(ctx.extra["list_dir_max_output_chars"])

        return render_list_dir(
            path,
            display_path,
            max_output_chars=max_chars,
            respect_gitignore=respect,
        )
