"""list_dir — list directory tree with char budget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codedoggy.tools.defaults import LIST_DIR_MAX_DEPTH, LIST_DIR_MAX_OUTPUT_CHARS
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

_DESCRIPTION = """\
Lists files and directories in a given path.
The 'target_directory' parameter can be relative to the workspace root or absolute.

Other details:
    - The result does not display dot-files and dot-directories.
    - Does not apply .gitignore; only hides names starting with '.'.
    - Expansion is depth-limited (max depth 3) and character-budgeted (~10k chars);
      very large trees are truncated with a notice. Prefer a deeper path when you need
      more detail under a large directory.
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

        display_path = _compute_display_path(ctx.cwd, raw)
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

        body = _render_tree(path, depth=0, budget=LIST_DIR_MAX_OUTPUT_CHARS)
        trimmed = body.rstrip("\n")
        return f"- {display_path}/\n{trimmed}" if trimmed else f"- {display_path}/"


def _compute_display_path(cwd: Path, target: str) -> str:
    t = target.strip()
    if not t or t in {".", "./"}:
        return str(cwd)
    p = Path(target)
    if p.is_absolute():
        return str(p.resolve())
    return str((cwd / target).resolve())


def _render_tree(root: Path, depth: int, budget: int) -> str:
    if depth >= LIST_DIR_MAX_DEPTH or budget <= 0:
        return ""

    try:
        children = list(root.iterdir())
    except OSError:
        return ""

    items: list[tuple[str, Path, bool]] = []
    for child in children:
        name = child.name
        if name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        items.append((name, child, is_dir))
    items.sort(key=lambda t: t[0].lower())

    out = ""
    for idx, (name, child, is_dir) in enumerate(items):
        label = f"{name}/" if is_dir else name
        indent = "  " * (depth + 1)
        line = f"{indent}- {label}\n"
        if len(out) + len(line) > budget:
            remaining = len(items) - idx
            notice = f"{indent}… ({remaining} more entries truncated)\n"
            if len(out) + len(notice) <= budget:
                out += notice
            break
        out += line
        if is_dir and depth + 1 < LIST_DIR_MAX_DEPTH:
            sub = _render_tree(child, depth + 1, budget - len(out))
            if sub:
                if len(out) + len(sub) > budget:
                    room = budget - len(out)
                    out += sub[:room]
                    break
                out += sub
    return out
