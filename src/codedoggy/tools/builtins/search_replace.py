"""search_replace — Grok SearchReplaceTool wire + normalized confusable fallback.

Match helpers: ``codedoggy.tools.grok_build.search_replace_logic``
  Ported from search_replace/helpers.rs + unicode_confusables.rs

Error/success strings match Grok search_replace/mod.rs (template names expanded).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codedoggy.tools.defaults import SEARCH_REPLACE_INCLUDE_USER_EDIT_HINT
from codedoggy.tools.grok_build.search_replace_logic import (
    NormalizedMatchResultKind,
    build_nearest_match_hint,
    find_normalized_match_positions,
    replace_normalized_matches,
    replace_using_positions,
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
from codedoggy.tools.util.paths import (
    resolve_model_path,
    validate_path_component_lengths,
)
from codedoggy.tools.util.unicode_confusables import has_confusables

# Grok DESCRIPTION_FULL with templates expanded to product names
_DESCRIPTION = """\
Replace an exact string in a file.

- Read the file with `read_file` before editing it.
- `read_file` prefixes each line with "LINE_NUMBER→". That prefix is not part of the file: match only what comes after the →, with its exact indentation.
- `old_string` must match exactly one place in the file. If it appears more than once, add surrounding lines to make it unique, or set `replace_all` to change every occurrence (handy for renaming an identifier).
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

        # Grok: "Old string and new string are the same"
        if old == new:
            raise ToolError(
                "Old string and new string are the same",
                code="invalid_arguments",
            )

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
            raise ToolError(
                f"Error: {file_path} is a directory, not a file.",
                code="invalid_arguments",
            )

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
        except PermissionError as e:
            raise ToolError(
                f"Error: permission denied reading {file_path}.",
                code="permission_denied",
            ) from e
        except OSError as e:
            raise ToolError(f"Error: failed to read {file_path}: {e}", code="io_error") from e

        has_crlf = "\r\n" in text
        match_text = text.replace("\r\n", "\n") if has_crlf else text
        # Grok matches against old_string as-is on match_text (CRLF already
        # normalized to LF). Do not rewrite old_string CRLF separately beyond that.
        old_n = old.replace("\r\n", "\n")
        new_n = new.replace("\r\n", "\n")

        positions = [i for i, _ in _match_indices(match_text, old_n)]
        used_normalized = False
        norm_matches = None

        if not positions:
            # Grok: unicode_normalized_fallback (default enabled in practice)
            fallback_on = True
            if ctx.extra is not None and "unicode_normalized_fallback" in ctx.extra:
                fallback_on = bool(ctx.extra["unicode_normalized_fallback"])
            if fallback_on:
                result = find_normalized_match_positions(match_text, old_n)
                if result.kind is NormalizedMatchResultKind.Matches and result.matches:
                    if len(result.matches) > 1 and not replace_all:
                        raise ToolError(
                            "The string to replace was found multiple times in the file "
                            "(via Unicode normalization). Use replace_all to replace all "
                            "occurrences, or include more context to only edit one occurrence.",
                            code="edit_ambiguous",
                        )
                    positions = [m.original_start for m in result.matches]
                    norm_matches = result.matches
                    used_normalized = True
                elif result.kind is NormalizedMatchResultKind.Ambiguous:
                    raise ToolError(
                        "The string to replace was found via Unicode normalization but the "
                        "match is ambiguous (partial or overlapping). Use a more specific "
                        "old_string that avoids lines with Unicode typography characters.",
                        code="edit_ambiguous",
                    )

        if not positions:
            user_edit_hint = ""
            if SEARCH_REPLACE_INCLUDE_USER_EDIT_HINT:
                # Grok exact: no "re-read before retrying"
                user_edit_hint = (
                    " The user may have changed the file since you last read it."
                )
            hint = build_nearest_match_hint(match_text, old_n)
            conf_hint = ""
            if has_confusables(match_text):
                conf_hint = (
                    "\n\nNote: the file contains typography confusables "
                    "(smart quotes/dashes/ellipsis). Exact match failed; "
                    "normalized fallback also found no unique match. "
                    "Copy old_string from read_file output carefully."
                )
            raise ToolError(
                "The string to replace was not found in the file, use the read_file "
                f"tool to see the correct string.{user_edit_hint}{hint}{conf_hint}",
                code="edit_no_match",
            )

        if len(positions) > 1 and not replace_all:
            raise ToolError(
                "The string to replace was found multiple times in the file. "
                "Use replace_all to replace all occurrences, or include more context "
                "to only edit one occurrence.",
                code="edit_ambiguous",
            )

        if used_normalized and norm_matches is not None:
            updated, new_positions = replace_normalized_matches(
                match_text, norm_matches if replace_all else norm_matches[:1], new_n
            )
        else:
            pos_list = positions if replace_all else positions[:1]
            updated, new_positions = replace_using_positions(
                match_text, pos_list, old_n, new_n
            )

        # Grok success strings (no "confusable" suffix)
        if len(new_positions) == 1:
            msg = f"The file {file_path} has been updated successfully."
        else:
            msg = (
                f"The file {file_path} has been updated. "
                "All occurrences were successfully replaced."
            )

        return _finish_write(
            ctx,
            path=path,
            file_path=file_path,
            text=text,
            updated=updated,
            has_crlf=has_crlf,
            msg=msg,
            args=args,
            snippet_focus=new_n,
        )


def _match_indices(text: str, needle: str) -> list[tuple[int, str]]:
    if not needle:
        return []
    out: list[tuple[int, str]] = []
    start = 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            break
        out.append((i, needle))
        start = i + len(needle)
    return out


def _finish_write(
    ctx: ToolCallContext,
    *,
    path: Path,
    file_path: str,
    text: str,
    updated: str,
    has_crlf: bool,
    msg: str,
    args: dict[str, Any],
    snippet_focus: str,
) -> str:
    if has_crlf:
        updated = updated.replace("\r\n", "\n").replace("\n", "\r\n")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
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
    snippet = _edit_snippet(updated.replace("\r\n", "\n"), snippet_focus)
    if snippet:
        return f"{msg}\n\n{snippet}"
    return msg


def _edit_snippet(file_text: str, focus: str, *, context_lines: int = 3) -> str:
    """Lightweight context card (Grok CONTEXT_LINES=3; Doggy display aid)."""
    if not focus:
        return ""
    idx = file_text.find(focus)
    if idx < 0:
        first = focus.split("\n", 1)[0]
        idx = file_text.find(first) if first else -1
    if idx < 0:
        return ""
    line_start = file_text.count("\n", 0, idx) + 1
    lines = file_text.splitlines()
    i0 = max(0, line_start - 1 - context_lines)
    i1 = min(len(lines), line_start - 1 + focus.count("\n") + 1 + context_lines)
    out = ["# Edit context"]
    for i in range(i0, i1):
        mark = "→" if i == line_start - 1 else " "
        out.append(f"{i + 1:4}{mark}| {lines[i]}")
    return "\n".join(out)


def _create_file(path: Path, file_path: str, content: str) -> str:
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
