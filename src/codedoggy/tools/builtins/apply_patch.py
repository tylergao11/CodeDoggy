"""apply_patch tool shell.

Description from Grok codex/apply_patch/tool.rs DESCRIPTION.
Engine: source-ported parser/apply/seek_sequence under tools/codex/apply_patch/.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(slots=True)
class _FileChange:
    """One fully computed change, ready for the write phase."""

    action: str
    path: Path
    rel_path: str
    before: str | None
    after: str | None
    dest_path: Path | None = None
    dest_rel_path: str | None = None
    dest_before: str | None = None


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

        # Grok codex/apply_patch/tool.rs: compute every file change first.
        # Path resolution, policy checks, file reads, and hunk matching all
        # complete before the first filesystem mutation.
        changes = _compute_all_changes(ctx, parsed.hunks)
        results = _apply_all_changes(ctx, changes)
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


def _compute_all_changes(
    ctx: ToolCallContext,
    hunks: list[Any],
) -> list[_FileChange]:
    """Phase 1: resolve, authorize, read, and derive every hunk in memory.

    A tiny virtual file map preserves sequential semantics when one patch has
    multiple sections for the same path.  Any later not-found/no-match/policy
    failure aborts this phase before an earlier hunk can touch disk.
    """
    virtual: dict[Path, str | None] = {}
    changes: list[_FileChange] = []

    def read_virtual(path: Path, rel_path: str, *, operation: str) -> str:
        if path in virtual:
            content = virtual[path]
            if content is None:
                raise ToolError(
                    f"File not found for {operation}: {rel_path}",
                    code="not_found",
                )
            return content
        if path.exists() and not path.is_file():
            raise ToolError(
                f"Path is not a file for {operation}: {rel_path}",
                code="invalid_arguments",
            )
        if not path.is_file():
            raise ToolError(
                f"File not found for {operation}: {rel_path}",
                code="not_found",
            )
        content = path.read_bytes().decode("utf-8", errors="replace")
        virtual[path] = content
        return content

    def existing_or_none(
        path: Path,
        rel_path: str,
        *,
        operation: str,
    ) -> str | None:
        if path in virtual:
            return virtual[path]
        if path.exists() and not path.is_file():
            raise ToolError(
                f"Path is not a file for {operation}: {rel_path}",
                code="invalid_arguments",
            )
        if not path.is_file():
            virtual[path] = None
            return None
        content = path.read_bytes().decode("utf-8", errors="replace")
        virtual[path] = content
        return content

    for hunk in hunks:
        if isinstance(hunk, AddFile):
            path = _resolve_in_workspace(ctx, hunk.path)
            _policy_write(ctx, hunk.path)
            before = existing_or_none(path, hunk.path, operation="add")
            changes.append(
                _FileChange(
                    action="add",
                    path=path,
                    rel_path=hunk.path,
                    before=before,
                    after=hunk.contents,
                )
            )
            virtual[path] = hunk.contents
            continue

        if isinstance(hunk, DeleteFile):
            path = _resolve_in_workspace(ctx, hunk.path)
            _policy_write(ctx, hunk.path)
            before = read_virtual(path, hunk.path, operation="delete")
            changes.append(
                _FileChange(
                    action="delete",
                    path=path,
                    rel_path=hunk.path,
                    before=before,
                    after=None,
                )
            )
            virtual[path] = None
            continue

        if not isinstance(hunk, UpdateFile):
            continue

        path = _resolve_in_workspace(ctx, hunk.path)
        _policy_write(ctx, hunk.path)
        before = read_virtual(path, hunk.path, operation="update")
        try:
            after = derive_new_contents(before, hunk.path, hunk.chunks)
        except ApplyPatchError as e:
            raise ToolError(e.message, code="patch_no_match") from e

        if not hunk.move_path:
            changes.append(
                _FileChange(
                    action="update",
                    path=path,
                    rel_path=hunk.path,
                    before=before,
                    after=after,
                )
            )
            virtual[path] = after
            continue

        dest = _resolve_in_workspace(ctx, hunk.move_path)
        if dest == path:
            raise ToolError(
                f"Move destination must differ from source: {hunk.path}",
                code="invalid_arguments",
            )
        _policy_write(ctx, hunk.move_path)
        dest_before = existing_or_none(dest, hunk.move_path, operation="move")
        changes.append(
            _FileChange(
                action="move",
                path=path,
                rel_path=hunk.path,
                before=before,
                after=after,
                dest_path=dest,
                dest_rel_path=hunk.move_path,
                dest_before=dest_before,
            )
        )
        virtual[path] = None
        virtual[dest] = after

    return changes


def _apply_all_changes(
    ctx: ToolCallContext,
    changes: list[_FileChange],
) -> list[str]:
    """Phase 2: apply only changes that survived the complete preflight."""
    results: list[str] = []
    for change in changes:
        try:
            if change.action == "add":
                change.path.parent.mkdir(parents=True, exist_ok=True)
                change.path.write_bytes((change.after or "").encode("utf-8"))
                ctx.set_mutation(
                    path=change.rel_path,
                    before=change.before,
                    after=change.after,
                    is_create=change.before is None,
                    tool_name="apply_patch",
                    args={"path": change.rel_path, "op": "add"},
                )
                results.append(f"A {change.rel_path}")
                continue

            if change.action == "delete":
                change.path.unlink()
                ctx.set_mutation(
                    path=change.rel_path,
                    before=change.before,
                    after=None,
                    is_delete=True,
                    tool_name="apply_patch",
                    args={"path": change.rel_path, "op": "delete"},
                )
                results.append(f"D {change.rel_path}")
                continue

            if change.action == "update":
                change.path.write_bytes((change.after or "").encode("utf-8"))
                ctx.set_mutation(
                    path=change.rel_path,
                    before=change.before,
                    after=change.after,
                    tool_name="apply_patch",
                    args={"path": change.rel_path, "op": "update"},
                )
                results.append(f"M {change.rel_path}")
                continue

            if change.action == "move":
                dest = change.dest_path
                dest_rel = change.dest_rel_path
                if dest is None or not dest_rel:
                    raise ToolError("invalid precomputed move", code="internal")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((change.after or "").encode("utf-8"))
                change.path.unlink()
                ctx.set_mutation(
                    path=change.rel_path,
                    before=change.before,
                    after=None,
                    is_delete=True,
                    tool_name="apply_patch",
                    args={
                        "path": change.rel_path,
                        "op": "move_delete",
                        "move_to": dest_rel,
                    },
                )
                ctx.set_mutation(
                    path=dest_rel,
                    before=change.dest_before,
                    after=change.after,
                    is_create=change.dest_before is None,
                    tool_name="apply_patch",
                    args={
                        "path": dest_rel,
                        "op": "move_create",
                        "from": change.rel_path,
                    },
                )
                results.append(f"M {dest_rel}")
        except ToolError:
            raise
        except OSError as e:
            raise ToolError(
                f"Failed to apply {change.action} for {change.rel_path}: {e}",
                code="execution_error",
            ) from e
    return results
