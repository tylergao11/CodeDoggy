"""write — OpenCode WriteTool as used in Grok workspace_grok_build_toolset.

Ported from:
  crates/codegen/xai-grok-tools/src/implementations/opencode/write/mod.rs
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
from codedoggy.tools.util.paths import resolve_model_path, validate_path_component_lengths

# Grok DESCRIPTION with ${{ tools.by_kind.read }} expanded to product name
_DESCRIPTION = """\
Create or overwrite a file.

- Writing to an existing path replaces the file — read it first with the read_file tool.
- Parent directories are created for you.
"""


class WriteTool(Tool):
    def id(self) -> ToolId:
        return ToolId("write")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Write

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="write", description=_DESCRIPTION.strip())

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The full file content to write.",
                },
            },
            "required": ["file_path", "content"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        file_path = args.get("file_path")
        content = args.get("content")
        if not isinstance(file_path, str) or not file_path.strip():
            raise ToolError.invalid_arguments("file_path is required")
        if not isinstance(content, str):
            raise ToolError.invalid_arguments("content is required")

        path_err = validate_path_component_lengths(file_path)
        if path_err:
            raise ToolError(path_err, code="filename_too_long")

        # CodeDoggy host policy (same pattern as search_replace); Grok uses
        # permission manager outside the tool body.
        policy = (ctx.extra or {}).get("policy")
        if policy is not None:
            check = getattr(policy, "check_write", None)
            if callable(check):
                decision = check(file_path)
                if decision is not None and not getattr(decision, "allowed", True):
                    raise ToolError(
                        getattr(decision, "reason", None) or "write denied by policy",
                        code=getattr(decision, "code", None) or "policy_denied",
                    )

        # Grok: resolve_model_path(&cwd, display_cwd.as_deref(), &input.file_path)
        path = resolve_model_path(ctx.cwd, file_path)

        # Grok: match fs.read_file — Ok => (true, lossy utf-8), Err => (false, None)
        before = None
        existed = path.is_file()
        if existed:
            try:
                before = path.read_bytes().decode("utf-8", errors="replace")
            except OSError:
                before = None
                existed = False

        # Grok: create_dir_all(parent) then write_file (bytes)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode("utf-8"))
        except OSError as e:
            # Grok ComputerError Display: "IO Error: {0}"
            raise ToolError(f"IO Error: {e}", code="io_error") from e

        # CodeDoggy mutation audit (Grok emits FileWritten notification)
        ctx.set_mutation(
            path=file_path,
            before=before,
            after=content,
            is_create=not existed,
            tool_name="write",
            args=dict(args),
        )

        # Exact Grok tool_output_for_prompt strings (path.display())
        display = str(path)
        if existed:
            return f"Wrote file successfully to {display}."
        return f"The file {display} has been created."
