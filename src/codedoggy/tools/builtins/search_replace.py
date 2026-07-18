"""search_replace — exact string replace / create file."""

from __future__ import annotations

from pathlib import Path
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
from codedoggy.tools.util.paths import (
    NAME_MAX,
    resolve_model_path,
    validate_path_component_lengths,
)

_DESCRIPTION = """\
Replace an exact string in a file.

- Read the file with `read_file` before editing it.
- `read_file` may prefix lines with "LINE_NUMBER→" (first visible line and every 10th line number). That prefix is not part of the file: match only what comes after the →, with its exact indentation.
- `old_string` must match exactly one place in the file. If it appears more than once, add surrounding lines to make it unique, or set `replace_all` to change every occurrence (handy for renaming an identifier).
- Empty `old_string` creates (or overwrites) the file with `new_string`.
- CRLF files: matching ignores \\r (LF-only old_string works); after edit, original CRLF line endings are preserved.
"""


class SearchReplaceTool(Tool):
    def id(self) -> ToolId:
        return ToolId("search_replace")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Edit

    def description(self, ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="search_replace", description=_DESCRIPTION.strip())

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "The path to the file to modify. You can use either a relative "
                        "path in the workspace or an absolute path."
                    ),
                },
                "old_string": {
                    "type": "string",
                    "description": "The text to replace",
                },
                "new_string": {
                    "type": "string",
                    "description": (
                        "The text to replace it with (must be different from old_string)"
                    ),
                },
                "replace_all": {
                    "type": "boolean",
                    "description": (
                        "Replace all occurrences of old_string (default false)"
                    ),
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        file_path = args.get("file_path")
        old = args.get("old_string")
        new = args.get("new_string")
        replace_all = _as_bool(args.get("replace_all", False))

        if not isinstance(file_path, str) or not file_path.strip():
            raise ToolError.invalid_arguments("file_path is required")
        if not isinstance(old, str):
            raise ToolError.invalid_arguments("old_string is required")
        if not isinstance(new, str):
            raise ToolError.invalid_arguments("new_string is required")

        path_err = validate_path_component_lengths(file_path)
        if path_err:
            raise ToolError(path_err, code="filename_too_long")

        if old == new:
            raise ToolError(
                "Old string and new string are the same",
                code="invalid_arguments",
            )

        # Workspace policy (tool layer) — deny before mutation/audit
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

        path = resolve_model_path(ctx.cwd, file_path)
        if path.is_dir():
            raise ToolError("File path is a directory", code="invalid_arguments")

        # Empty old_string → create / overwrite file
        if old == "":
            before = None
            if path.is_file():
                try:
                    before = path.read_bytes().decode("utf-8", errors="replace")
                except OSError:
                    before = None
            msg = _create_file(path, file_path, new)
            ctx.set_mutation(
                path=file_path,
                before=before,
                after=new,
                is_create=before is None,
                tool_name="search_replace",
                args=dict(args),
            )
            return msg

        if not path.is_file():
            raise ToolError(
                f"File not found: {file_path}. Please check the path and try again.",
                code="not_found",
            )

        try:
            text = path.read_bytes().decode("utf-8", errors="replace")
        except OSError as e:
            if e.errno == 13:
                raise ToolError(
                    f"Error: permission denied reading {file_path}.",
                    code="permission_denied",
                ) from e
            raise ToolError(f"Error: failed to read {file_path}: {e}", code="io_error") from e

        # read_file strips \r from displayed lines; models usually pass LF-only
        # old_string. Match on LF-normalized text and preserve original endings.
        has_crlf = "\r\n" in text
        match_text = text.replace("\r\n", "\n") if has_crlf else text
        old_n = old.replace("\r\n", "\n")
        new_n = new.replace("\r\n", "\n")

        count = match_text.count(old_n)
        if count == 0:
            hint = _nearest_match_hint(match_text, old_n)
            user_hint = (
                " The user may have changed the file since you last read it;"
                " re-read before retrying."
            )
            raise ToolError(
                "The string to replace was not found in the file"
                f"{hint}.{user_hint}",
                code="edit_no_match",
            )
        if count > 1 and not replace_all:
            raise ToolError(
                f"The string to replace was found {count} times in the file. "
                "Use replace_all=true to replace all occurrences, or provide a "
                "more specific old_string that matches exactly once.",
                code="edit_ambiguous",
            )

        if replace_all:
            updated = match_text.replace(old_n, new_n)
            msg = (
                f"The file {file_path} has been updated. "
                "All occurrences were successfully replaced."
            )
        else:
            updated = match_text.replace(old_n, new_n, 1)
            msg = f"The file {file_path} has been updated successfully."

        if has_crlf:
            # Normalize any residual \r\n then re-emit consistent CRLF.
            updated = updated.replace("\r\n", "\n").replace("\n", "\r\n")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Bytes write preserves exact line endings (text mode rewrites \n on Windows).
            path.write_bytes(updated.encode("utf-8"))
        except OSError as e:
            raise ToolError(
                f"Error: failed to write {file_path}: {e}",
                code="io_error",
            ) from e

        ctx.set_mutation(
            path=file_path,
            before=text,
            after=updated,
            is_create=False,
            tool_name="search_replace",
            args=dict(args),
        )
        return msg


def _create_file(path: Path, file_path: str, content: str) -> str:
    """Create or fully overwrite when old_string is empty."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
    except OSError as e:
        raise ToolError(
            f"Error: failed to write {file_path}: {e}",
            code="io_error",
        ) from e
    return f"The file {file_path} has been created successfully."


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)


def _nearest_match_hint(file_text: str, old_string: str) -> str:
    """Short nearest-line hint when exact match fails."""
    keyword = old_string.strip().split("\n", 1)[0][:40]
    if not keyword:
        return ""
    for i, line in enumerate(file_text.splitlines(), start=1):
        if keyword in line:
            snippet = line.strip()
            if len(snippet) > 120:
                snippet = snippet[:120] + "…"
            return f"\n\nNearest match: line {i}: {snippet}"
    return ""
