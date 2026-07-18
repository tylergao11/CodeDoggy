"""grep — Grok GrepTool wire + source-ported finalize formatters.

Format/finalize: ``codedoggy.tools.grok_build.grep_format``
  Ported from implementations/grok_build/grep/mod.rs

rg invocation mirrors prepare_grep: ``--heading --with-filename --line-number``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from codedoggy.tools.defaults import (
    GREP_TIMEOUT_DEFAULT_SECS,
    GREP_TIMEOUT_WSL_SECS,
)
from codedoggy.tools.grok_build.grep_format import (
    DEFAULT_MAX_CHARS_PER_LINE,
    DEFAULT_TOOL_OUTPUT_BYTES,
    MAX_STDOUT_BYTES,
    OutputMode,
    finalize_grep_body,
    no_matches_card,
    resolve_effective_head_limit,
    rg_exit2_message,
    rg_unknown_exit_message,
    wrap_workspace_result,
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

# Re-export for tests
__all__ = [
    "GrepTool",
    "OutputMode",
    "resolve_effective_head_limit",
    "format_content_output",
]

from codedoggy.tools.grok_build.grep_format import format_content_output  # noqa: E402

# Grok description_template (exact product wording; Doggy note about fallback appended)
_DESCRIPTION = """\
Search file contents with regular expressions (ripgrep).

- Full regex syntax, so escape literal special characters: `functionCall\\(`, or `interface\\{\\}` to find interface{} in Go.
- Pass the pattern as a raw regex string — no surrounding quotes.
- Respects .gitignore unless you pass a broad glob like '--glob *'.
- Only filter by 'type' or 'glob' when you are sure of the file type; import paths may not match source file types (.js vs .ts).
- Output is ripgrep-style: ':' marks match lines, '-' marks context lines, grouped by file. Large results are capped and report "at least" counts.
"""


def _is_wsl() -> bool:
    if sys.platform != "linux":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text(
            encoding="utf-8", errors="ignore"
        ).lower()
    except OSError:
        return False


def grep_timeout_secs() -> int:
    return GREP_TIMEOUT_WSL_SECS if _is_wsl() else GREP_TIMEOUT_DEFAULT_SECS


class GrepTool(Tool):
    def id(self) -> ToolId:
        return ToolId("grep")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Search

    def description(self, ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="grep", description=_DESCRIPTION.strip())

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "The regular expression pattern to search for in file contents "
                        "(rg --regexp)"
                    ),
                },
                "path": {
                    "type": "string",
                    "description": (
                        "File or directory to search in (rg pattern -- PATH). "
                        "Defaults to workspace path."
                    ),
                },
                "glob": {
                    "type": "string",
                    "description": (
                        'Glob pattern (rg --glob GLOB -- PATH) to filter files '
                        '(e.g. "*.js", "*.{ts,tsx}").'
                    ),
                },
                "-B": {
                    "type": "integer",
                    "description": "Number of lines to show before each match (rg -B).",
                },
                "-A": {
                    "type": "integer",
                    "description": "Number of lines to show after each match (rg -A).",
                },
                "-C": {
                    "type": "integer",
                    "description": (
                        "Number of lines to show before and after each match (rg -C)."
                    ),
                },
                "-i": {
                    "type": "boolean",
                    "description": "Case insensitive search (rg -i). Defaults to false.",
                },
                "type": {
                    "type": "string",
                    "description": (
                        "File type to search (rg --type). Common types: js, py, rust, go, "
                        "java, etc. More efficient than glob for standard file types."
                    ),
                },
                "head_limit": {
                    "type": "integer",
                    "description": (
                        'Limit output to first N matching content lines, equivalent to "| head -N". '
                        "Defaults to 200 (hard cap 2000)."
                    ),
                },
                "multiline": {
                    "type": "boolean",
                    "description": (
                        "Enable multiline mode where . matches newlines and patterns can "
                        "span lines (rg -U --multiline-dotall). Default: false."
                    ),
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": (
                        "content (default): path:line:text; "
                        "files_with_matches: paths only; "
                        "count: path:count per file."
                    ),
                },
            },
            "required": ["pattern"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ToolError.invalid_arguments("pattern is required")

        search_path = args.get("path")
        root = (
            resolve_model_path(ctx.cwd, search_path)
            if isinstance(search_path, str) and search_path.strip()
            else ctx.cwd
        )
        workspace_label = str(ctx.cwd)
        if not root.exists():
            raise ToolError(f"Path not found: {root}", code="not_found")

        mode = _parse_output_mode(args.get("output_mode"))
        head_limit = _optional_int(args.get("head_limit"), "head_limit")
        effective = resolve_effective_head_limit(head_limit, mode)

        case_i = bool(args.get("-i") or args.get("case_insensitive"))
        multiline = bool(args.get("multiline"))
        before = _optional_int(
            args.get("-B") if args.get("-B") is not None else args.get("before_context"),
            "-B",
        )
        after = _optional_int(
            args.get("-A") if args.get("-A") is not None else args.get("after_context"),
            "-A",
        )
        context_n = _optional_int(
            args.get("-C") if args.get("-C") is not None else args.get("context"),
            "-C",
        )
        glob = args.get("glob") if isinstance(args.get("glob"), str) else None
        ftype = args.get("type") if isinstance(args.get("type"), str) else None

        rg = shutil.which("rg") or shutil.which("rg.exe")
        if rg:
            raw_lines, hit_limit, err = _run_rg(
                rg,
                pattern=pattern,
                root=root,
                cwd_display=workspace_label,
                case_i=case_i,
                multiline=multiline,
                before=before,
                after=after,
                context_n=context_n,
                glob=glob,
                ftype=ftype,
                head_limit=effective,
                mode=mode,
            )
            if err is not None:
                raise ToolError(err, code="rg_error")
        else:
            _reject_python_fallback_unsupported(
                multiline=multiline,
                before=before,
                after=after,
                context_n=context_n,
                ftype=ftype,
                glob=glob,
            )
            raw_lines, hit_limit = _run_python_grep(
                pattern=pattern,
                root=root,
                case_i=case_i,
                head_limit=effective,
                glob=glob,
                mode=mode,
            )

        if not raw_lines:
            return no_matches_card(workspace_label)

        body = finalize_grep_body(
            raw_lines,
            mode=mode,
            is_truncated=hit_limit,
            effective_head_limit=effective,
            max_chars_per_line=DEFAULT_MAX_CHARS_PER_LINE,
            max_output_bytes=DEFAULT_TOOL_OUTPUT_BYTES,
        )
        return wrap_workspace_result(workspace_label, body)


def _parse_output_mode(raw: Any) -> OutputMode:
    if raw is None or raw == "":
        return OutputMode.Content
    s = str(raw).strip().lower()
    for m in OutputMode:
        if m.value == s:
            return m
    raise ToolError.invalid_arguments(
        f"output_mode must be content|files_with_matches|count, got {raw!r}"
    )


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ToolError.invalid_arguments(f"invalid {name}: {value}") from e


def _is_simple_ext_glob(glob: str) -> bool:
    g = glob.strip()
    if not g.startswith("*.") or len(g) < 3:
        return False
    ext = g[2:]
    if any(ch in ext for ch in "*?[]{}\\/"):
        return False
    return True


def _reject_python_fallback_unsupported(
    *,
    multiline: bool,
    before: Any,
    after: Any,
    context_n: Any,
    ftype: str | None,
    glob: str | None,
) -> None:
    unsupported: list[str] = []
    if multiline:
        unsupported.append("multiline")
    if before is not None:
        unsupported.append("-B / before_context")
    if after is not None:
        unsupported.append("-A / after_context")
    if context_n is not None:
        unsupported.append("-C / context")
    if ftype:
        unsupported.append("type")
    if glob and not _is_simple_ext_glob(glob):
        unsupported.append(
            f"glob={glob!r} (python fallback only supports simple *.ext patterns)"
        )
    if unsupported:
        raise ToolError(
            "ripgrep (`rg`) is not available; the Python fallback cannot honor: "
            + ", ".join(unsupported)
            + ". Install rg, or omit those arguments.",
            code="unsupported_without_rg",
        )


def _run_rg(
    rg: str,
    *,
    pattern: str,
    root: Path,
    cwd_display: str,
    case_i: bool,
    multiline: bool,
    before: Any,
    after: Any,
    context_n: Any,
    glob: str | None,
    ftype: str | None,
    head_limit: int,
    mode: OutputMode = OutputMode.Content,
) -> tuple[list[str], bool, str | None]:
    """Returns (lines, hit_head_limit, error_message_or_None)."""
    # Grok prepare_grep base flags
    if mode is OutputMode.FilesWithMatches:
        cmd = [rg, "-l", "--color=never"]
    elif mode is OutputMode.Count:
        cmd = [rg, "-c", "--color=never"]
    else:
        cmd = [
            rg,
            "--heading",
            "--with-filename",
            "--line-number",
            "--color=never",
            "--max-columns",
            "1000",
            "--max-columns-preview",
        ]
    if case_i:
        cmd.append("--ignore-case")
    if multiline:
        cmd.extend(["-U", "--multiline-dotall"])
    if mode is OutputMode.Content:
        if context_n is not None and int(context_n) > 0:
            cmd.extend(["-C", str(int(context_n))])
        else:
            if before is not None and int(before) > 0:
                cmd.extend(["-B", str(int(before))])
            if after is not None and int(after) > 0:
                cmd.extend(["-A", str(int(after))])
    if glob:
        cmd.extend(["--glob", glob])
    if ftype:
        cmd.extend(["--type", ftype])
    cmd.extend(["-e", pattern, str(root)])
    cmd.extend(["--max-filesize", "5M"])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=grep_timeout_secs(),
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise ToolError(
            f"grep timed out after {grep_timeout_secs()}s",
            code="timeout",
        ) from e
    except OSError as e:
        raise ToolError(f"Error calling tool: {e}", code="io_error") from e

    raw = proc.stdout[:MAX_STDOUT_BYTES]
    text = raw.decode("utf-8", errors="replace")
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
    code = proc.returncode if proc.returncode is not None else -1

    # finalize_grep exit handling
    if (code == 1 and not text.strip()) or (
        code == 2 and "No files were searched" in stderr
    ):
        return [], False, None
    if code == 2:
        return [], False, rg_exit2_message(stderr.strip() or "(no stderr)", cwd_display)
    if code not in (0, 1):
        return [], False, rg_unknown_exit_message(code, cwd_display)

    # str::lines() style: drop trailing empties from splitlines
    lines = text.splitlines()
    # Read head_limit+1 to detect truncation (Grok)
    probe = head_limit + 1 if head_limit >= 0 else head_limit
    hit = len(lines) > head_limit
    if hit and head_limit >= 0:
        lines = lines[:head_limit]
    return lines, hit, None


def _run_python_grep(
    *,
    pattern: str,
    root: Path,
    case_i: bool,
    head_limit: int,
    glob: str | None,
    mode: OutputMode = OutputMode.Content,
) -> tuple[list[str], bool]:
    """Heading-style pure-Python fallback (content mode)."""
    flags = re.MULTILINE
    if case_i:
        flags |= re.IGNORECASE
    try:
        cre = re.compile(pattern, flags)
    except re.error as e:
        raise ToolError.invalid_arguments(f"invalid regex: {e}") from e

    out: list[str] = []
    paths: list[Path] = []
    if root.is_file():
        paths = [root]
    else:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                if name.startswith("."):
                    continue
                p = Path(dirpath) / name
                if glob and glob.startswith("*.") and not name.endswith(glob[1:]):
                    continue
                paths.append(p)

    for path in paths:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8192]:
            continue
        text = data.decode("utf-8", errors="replace")
        try:
            rel = path.relative_to(root if root.is_dir() else path.parent)
        except ValueError:
            rel = path
        rel_s = str(rel).replace("\\", "/")

        if mode is OutputMode.FilesWithMatches:
            if any(cre.search(line) for line in text.splitlines()):
                out.append(rel_s)
                if head_limit >= 0 and len(out) >= head_limit:
                    return out, True
            continue
        if mode is OutputMode.Count:
            n = sum(1 for line in text.splitlines() if cre.search(line))
            if n:
                out.append(f"{rel_s}:{n}")
                if head_limit >= 0 and len(out) >= head_limit:
                    return out, True
            continue

        # Content: heading format
        file_lines: list[str] = []
        for i, line in enumerate(text.splitlines(), start=1):
            if cre.search(line):
                file_lines.append(f"{i}:{line}")
        if file_lines:
            # path header + matches (Grok heading style)
            chunk = [rel_s, *file_lines, ""]
            for ln in chunk:
                out.append(ln)
                if head_limit >= 0 and len(out) >= head_limit:
                    return out, True
    return out, False
