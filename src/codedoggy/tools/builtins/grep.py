"""grep — content search via ripgrep when available, else pure Python."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any

from codedoggy.tools.defaults import (
    DEFAULT_TOOL_OUTPUT_BYTES,
    GREP_CONTENT_LINE_DEFAULT,
    GREP_CONTENT_LINE_LIMIT,
    GREP_DEFAULT_MAX_CHARS_PER_LINE,
    GREP_FILE_COUNT_DEFAULT,
    GREP_FILE_COUNT_LIMIT,
    GREP_MAX_STDOUT_BYTES,
    GREP_TIMEOUT_DEFAULT_SECS,
    GREP_TIMEOUT_WSL_SECS,
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

_DESCRIPTION = """\
Search file contents with regular expressions.

Usage:
- Prefer this tool over shell pipelines for content search.
- Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+").
- Pass the pattern as a raw regex string — no surrounding quotes.
- Filter files with glob (e.g., "*.js", "*.{ts,tsx}").
- When the `rg` binary is available it is used; otherwise a pure-Python fallback
  runs (slower). Without `rg`, context flags (-A/-B/-C), `type`, and multiline
  are rejected rather than silently ignored — install ripgrep or omit those args.
- Python fallback only supports simple `*.ext` globs (not brace globs like `*.{ts,tsx}`).
- This tool returns content matches only (path:line:text). Results are wrapped
  in a <workspace_result> block with a Found N matching lines summary.
- head_limit defaults to 200 matching content lines (hard cap 2000).
- Multiline mode can match across lines when enabled (requires `rg`, or is rejected
  on the Python fallback).
"""


class OutputMode(str, Enum):
    Content = "content"
    FilesWithMatches = "files_with_matches"
    Count = "count"


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


def resolve_effective_head_limit(head_limit: int | None, mode: OutputMode) -> int:
    if mode is OutputMode.Content:
        default, cap = GREP_CONTENT_LINE_DEFAULT, GREP_CONTENT_LINE_LIMIT
    else:
        default, cap = GREP_FILE_COUNT_DEFAULT, GREP_FILE_COUNT_LIMIT
    raw = default if head_limit is None else head_limit
    # Clamp to at least 1 so head_limit=0 does not look like "No matches found".
    return min(max(1, int(raw)), cap)


def truncate_line(line: str, max_chars: int = GREP_DEFAULT_MAX_CHARS_PER_LINE) -> str:
    if len(line) <= max_chars:
        return line
    return line[: max_chars - 1] + "…"


def format_content_output(
    output_lines: list[str],
    *,
    is_truncated: bool,
    max_chars_per_line: int = GREP_DEFAULT_MAX_CHARS_PER_LINE,
    max_output_bytes: int = DEFAULT_TOOL_OUTPUT_BYTES,
) -> str:
    """Model-facing card body (before workspace_result wrapper)."""
    at_least = "at least " if is_truncated else ""
    n = len(output_lines)
    lines = [f"Found {at_least}{n} matching lines"]
    trimmed = [truncate_line(ln, max_chars_per_line) for ln in output_lines]
    cut = _first_idx_exceed_cum_limit(trimmed, max_output_bytes)
    lines.extend(trimmed[:cut])
    remaining = len(trimmed) - cut
    if remaining > 0:
        lines.append(f"... [{at_least}{remaining} lines truncated] ...")
    return "\n".join(lines)


def wrap_workspace_result(workspace_path: str, body: str) -> str:
    return (
        f'<workspace_result workspace_path="{workspace_path}">\n'
        f"{body}\n"
        f"</workspace_result>"
    )


def _first_idx_exceed_cum_limit(lines: list[str], max_bytes: int) -> int:
    total = 0
    for i, line in enumerate(lines):
        add = len(line.encode("utf-8", errors="replace")) + (1 if i else 0)
        if total + add > max_bytes:
            return i
        total += add
    return len(lines)


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

        mode = OutputMode.Content
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
            raw_lines, hit_limit = _run_rg(
                rg,
                pattern=pattern,
                root=root,
                case_i=case_i,
                multiline=multiline,
                before=before,
                after=after,
                context_n=context_n,
                glob=glob,
                ftype=ftype,
                head_limit=effective,
            )
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
                multiline=False,
                head_limit=effective,
                glob=glob,
            )

        if not raw_lines:
            body = "No matches found"
            return wrap_workspace_result(workspace_label, body)

        body = format_content_output(
            raw_lines,
            is_truncated=hit_limit,
            max_chars_per_line=GREP_DEFAULT_MAX_CHARS_PER_LINE,
            max_output_bytes=DEFAULT_TOOL_OUTPUT_BYTES,
        )
        return wrap_workspace_result(workspace_label, body)


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ToolError.invalid_arguments(f"invalid {name}: {value}") from e


def _is_simple_ext_glob(glob: str) -> bool:
    """True for patterns like ``*.py`` / ``*.tsx`` that the Python walker can honor."""
    g = glob.strip()
    if not g.startswith("*.") or len(g) < 3:
        return False
    ext = g[2:]
    # No nested globs, braces, path seps, or multi-segment patterns.
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
    """Fail closed: do not silently drop flags the Python path cannot honor."""
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
    case_i: bool,
    multiline: bool,
    before: Any,
    after: Any,
    context_n: Any,
    glob: str | None,
    ftype: str | None,
    head_limit: int,
) -> tuple[list[str], bool]:
    cmd = [rg, "--line-number", "--color", "never", "--no-heading"]
    if case_i:
        cmd.append("-i")
    if multiline:
        cmd.extend(["-U", "--multiline-dotall"])
    if context_n is not None:
        cmd.extend(["-C", str(int(context_n))])
    else:
        if before is not None:
            cmd.extend(["-B", str(int(before))])
        if after is not None:
            cmd.extend(["-A", str(int(after))])
    if glob:
        cmd.extend(["--glob", glob])
    if ftype:
        cmd.extend(["--type", ftype])
    cmd.extend(["--regexp", pattern, "--", str(root)])

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
        raise ToolError(f"failed to run rg: {e}", code="io_error") from e

    raw = proc.stdout[:GREP_MAX_STDOUT_BYTES]
    text = raw.decode("utf-8", errors="replace")
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
    code = proc.returncode if proc.returncode is not None else -1

    # rg: 0 = matches, 1 = no matches, 2 = error (bad regex, type, I/O, …).
    # Do not report exit-2 as "No matches found".
    if code == 1 and not text.strip():
        return [], False
    if code == 2 and "No files were searched" in stderr:
        return [], False
    if code == 2:
        msg = stderr.strip() or "rg failed with exit 2"
        raise ToolError(f"grep error: {msg}", code="rg_error")
    if code not in (0, 1):
        msg = stderr.strip() or f"rg failed with exit {code}"
        raise ToolError(f"grep error: {msg}", code="rg_error")

    lines = [ln for ln in text.splitlines() if ln != ""]
    hit = len(lines) > head_limit
    if hit:
        lines = lines[:head_limit]
    return lines, hit


def _run_python_grep(
    *,
    pattern: str,
    root: Path,
    case_i: bool,
    multiline: bool,
    head_limit: int,
    glob: str | None,
) -> tuple[list[str], bool]:
    flags = re.MULTILINE
    if case_i:
        flags |= re.IGNORECASE
    if multiline:
        flags |= re.DOTALL
    try:
        cre = re.compile(pattern, flags)
    except re.error as e:
        raise ToolError.invalid_arguments(f"invalid regex: {e}") from e

    matches: list[str] = []
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
        for i, line in enumerate(text.splitlines(), start=1):
            if cre.search(line):
                matches.append(f"{rel}:{i}:{line}")
                if len(matches) >= head_limit:
                    return matches, True
    return matches, False
