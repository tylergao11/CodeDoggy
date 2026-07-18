"""apply_patch tool shell.

Description from Grok codex/apply_patch/tool.rs DESCRIPTION.
Engine: source-ported parser/apply/seek_sequence under tools/codex/apply_patch/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codedoggy.tools.codex.apply_patch.apply_logic import (
    ApplyPatchError,
    derive_new_contents,
)
from codedoggy.tools.codex.apply_patch.parser import (
    AddFile,
    DeleteFile,
    ParseError,
    UpdateFile,
    parse_patch,
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

# Grok tool.rs DESCRIPTION (verbatim core)
_DESCRIPTION = """\
Use the `apply_patch` tool to edit files.
Your patch language is a stripped-down, file-oriented diff format designed to be easy to parse and safe to apply. You can think of it as a high-level envelope:

*** Begin Patch
[ one or more file sections ]
*** End Patch

Within that envelope, you get a sequence of file operations.
You MUST include a header to specify the action you are taking.
Each operation starts with one of three headers:

*** Add File: <path> - create a new file. Every following line is a + line (the initial contents).
*** Delete File: <path> - remove an existing file. Nothing follows.
*** Update File: <path> - patch an existing file in place (optionally with a rename).

May be immediately followed by *** Move to: <new path> if you want to rename the file.
Then one or more “hunks”, each introduced by @@ (optionally followed by a hunk header).
Within a hunk each line starts with:

For instructions on [context_before] and [context_after]:
- By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change’s [context_after] lines in the second change’s [context_before] lines.
- If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs.

It is important to remember:

- You must include a header with your intended action (Add/Delete/Update)
- You must prefix new lines with `+` even when creating a new file
- File references can only be relative, NEVER ABSOLUTE.
"""


class ApplyPatchTool(Tool):
    def id(self) -> ToolId:
        return ToolId("apply_patch")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Edit

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="apply_patch", description=_DESCRIPTION.strip())

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "The patch text in codex apply_patch format.",
                },
            },
            "required": ["patch"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        raw = args.get("patch")
        if not isinstance(raw, str) or not raw.strip():
            raw = args.get("input")
        if not isinstance(raw, str) or not raw.strip():
            raise ToolError.invalid_arguments("patch is required")

        try:
            parsed = parse_patch(raw)
        except ParseError as e:
            raise ToolError(e.message, code="patch_parse_error") from e

        if not parsed.hunks:
            return "Success. (empty patch)"

        results: list[str] = []
        for hunk in parsed.hunks:
            if isinstance(hunk, AddFile):
                results.append(_apply_add(ctx, hunk))
            elif isinstance(hunk, DeleteFile):
                results.append(_apply_delete(ctx, hunk))
            elif isinstance(hunk, UpdateFile):
                results.append(_apply_update(ctx, hunk))
        return "Success. " + " ".join(results)


def _check_rel(path: str) -> None:
    if Path(path).is_absolute():
        raise ToolError(
            f"File references can only be relative, NEVER ABSOLUTE: {path}",
            code="invalid_arguments",
        )


def _resolve_in_workspace(ctx: ToolCallContext, path: str) -> Path:
    """Resolve *path* under session cwd; fail closed on workspace escape."""
    _check_rel(path)
    base = Path(ctx.cwd).resolve()
    target = resolve_model_path(base, path)
    try:
        target.relative_to(base)
    except ValueError as e:
        raise ToolError(
            f"path escapes workspace: {path}",
            code="path_escape",
        ) from e
    return target


def _policy_write(ctx: ToolCallContext, path: str) -> None:
    """Policy check **before** any disk mutation. Raises on deny."""
    policy = (ctx.extra or {}).get("policy")
    if policy is None:
        return
    check = getattr(policy, "check_write", None)
    if callable(check):
        d = check(path)
        if d is not None and not getattr(d, "allowed", True):
            raise ToolError(
                getattr(d, "reason", None) or "write denied by policy",
                code=getattr(d, "code", None) or "policy_denied",
            )


def _apply_add(ctx: ToolCallContext, hunk: AddFile) -> str:
    path = _resolve_in_workspace(ctx, hunk.path)
    _policy_write(ctx, hunk.path)
    before = None
    if path.is_file():
        before = path.read_bytes().decode("utf-8", errors="replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(hunk.contents.encode("utf-8"))
    ctx.set_mutation(
        path=hunk.path,
        before=before,
        after=hunk.contents,
        is_create=before is None,
        tool_name="apply_patch",
        args={"path": hunk.path, "op": "add"},
    )
    return f"A {hunk.path}"


def _apply_delete(ctx: ToolCallContext, hunk: DeleteFile) -> str:
    path = _resolve_in_workspace(ctx, hunk.path)
    _policy_write(ctx, hunk.path)
    if not path.is_file():
        raise ToolError(f"File not found for delete: {hunk.path}", code="not_found")
    before = path.read_bytes().decode("utf-8", errors="replace")
    path.unlink()
    ctx.set_mutation(
        path=hunk.path,
        before=before,
        after=None,
        is_delete=True,
        tool_name="apply_patch",
        args={"path": hunk.path, "op": "delete"},
    )
    return f"D {hunk.path}"


def _apply_update(ctx: ToolCallContext, hunk: UpdateFile) -> str:
    """Update in place, or Move: policy **both** paths before any write/delete.

    Move mutations (Shadow can restore):
      1. source delete (before=old content, after=None, is_delete)
      2. dest create/edit (before=prior dest or None, after=new content)
    """
    path = _resolve_in_workspace(ctx, hunk.path)
    if hunk.move_path:
        _resolve_in_workspace(ctx, hunk.move_path)
        _policy_write(ctx, hunk.path)
        _policy_write(ctx, hunk.move_path)
    else:
        _policy_write(ctx, hunk.path)

    if not path.is_file():
        raise ToolError(f"File not found for update: {hunk.path}", code="not_found")
    text = path.read_bytes().decode("utf-8", errors="replace")
    try:
        new_content = derive_new_contents(text, hunk.path, hunk.chunks)
    except ApplyPatchError as e:
        raise ToolError(e.message, code="patch_no_match") from e

    if hunk.move_path:
        dest = _resolve_in_workspace(ctx, hunk.move_path)
        # Re-check both after resolve (belt + suspenders)
        _policy_write(ctx, hunk.path)
        _policy_write(ctx, hunk.move_path)
        dest_before: str | None = None
        if dest.is_file():
            dest_before = dest.read_bytes().decode("utf-8", errors="replace")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(new_content.encode("utf-8"))
        path.unlink(missing_ok=True)
        # Source removed
        ctx.set_mutation(
            path=hunk.path,
            before=text,
            after=None,
            is_delete=True,
            tool_name="apply_patch",
            args={"path": hunk.path, "op": "move_delete", "move_to": hunk.move_path},
        )
        # Destination created or overwritten
        ctx.set_mutation(
            path=hunk.move_path,
            before=dest_before,
            after=new_content,
            is_create=dest_before is None,
            tool_name="apply_patch",
            args={"path": hunk.move_path, "op": "move_create", "from": hunk.path},
        )
        return f"M {hunk.move_path}"

    path.write_bytes(new_content.encode("utf-8"))
    ctx.set_mutation(
        path=hunk.path,
        before=text,
        after=new_content,
        is_create=False,
        tool_name="apply_patch",
        args={"path": hunk.path, "op": "update"},
    )
    return f"M {hunk.path}"
