"""read_file — line-numbered file read."""

from __future__ import annotations

from typing import Any

from codedoggy.tools.defaults import MAX_LINES_READ_DEFAULT
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolError,
    ToolId,
)
from codedoggy.tools.util.binary import is_binary
from codedoggy.tools.util.paths import resolve_model_path

# Approximate token guard (chars/4) when content exceeds this after decode.
MAX_NUM_TOKENS = 25_000
MAX_CONTENT_CHARS_FOR_TOKEN_GUARD = MAX_NUM_TOKENS * 4

_DESCRIPTION = f"""\
Read a file.

Usage:
- The target_file parameter can be a relative path in the workspace or an absolute path
- By default, it reads up to {MAX_LINES_READ_DEFAULT} lines starting from the beginning of the file
- Results are returned with line numbers starting at 1. The format is: LINE_NUMBER→LINE_CONTENT
- Line-number prefixes appear on the first visible line and every line whose number is divisible by 10 (sparse numbering to save tokens).
- offset is 1-based; 0 means 1; negative values count from the last content line (e.g. -1 starts at the last line).
- Only plain text files are supported. Binary files are rejected.
- For large files, pass offset and limit to page through content, or use grep to find regions of interest first.
"""


def resolve_read_start_line(file_content: str, offset: int | None) -> int:
    """1-indexed start line. None/0 → 1. Negative counts from the last content line.

    Content lines use the same ``split_inclusive`` view as extraction (not a
    phantom field past EOF). So ``offset=-1`` always starts at the last line
    the model can see — including files with no trailing newline.
    """
    if offset is None or offset == 0:
        return 1
    if offset > 0:
        return int(offset)
    lines = split_inclusive_newline(file_content)
    n = len(lines)
    if n == 0:
        return 1
    return max(1, n + int(offset) + 1)


def split_inclusive_newline(s: str) -> list[str]:
    if not s:
        return []
    parts: list[str] = []
    start = 0
    for i, ch in enumerate(s):
        if ch == "\n":
            parts.append(s[start : i + 1])
            start = i + 1
    if start < len(s):
        parts.append(s[start:])
    return parts


def strip_line_ending(s: str) -> str:
    if s.endswith("\n"):
        s = s[:-1]
    if s.endswith("\r"):
        s = s[:-1]
    return s


def extract_file_content_lines(
    file_content: str,
    offset: int | None,
    limit: int | None,
) -> str:
    """Windowed line view; prefix on first visible line and every 10th line number."""
    skip = max(0, resolve_read_start_line(file_content, offset) - 1)
    # Always clamp to max window (explicit limit is min(limit, max_lines)).
    max_lines = MAX_LINES_READ_DEFAULT
    take = max_lines if limit is None else min(max(0, limit), max_lines)

    if not file_content:
        return ""

    # split_inclusive already yields one entry per line including a real final
    # blank line when the file ends with "\n\n". Do NOT invent an extra empty
    # line solely because the file ends with a single "\n" (almost all source
    # files) — that phantom disagrees with offset=-1 and confuses models.
    lines_inc = split_inclusive_newline(file_content)

    output: list[str] = []
    first_line: int | None = None
    taken = 0

    for i, line_with_nl in enumerate(lines_inc):
        if i < skip:
            continue
        if taken >= take:
            break
        line = strip_line_ending(line_with_nl)
        line_num = i + 1
        is_first_visible = first_line is None
        if is_first_visible:
            first_line = line_num
        else:
            output.append("\n")
        if is_first_visible or line_num % 10 == 0:
            output.append(f"{line_num}→{line}")
        else:
            output.append(line)
        taken += 1

    return "".join(output)


class ReadFileTool(Tool):
    def id(self) -> ToolId:
        return ToolId("read_file")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Read

    def description(self, ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="read_file", description=_DESCRIPTION.strip())

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_file": {
                    "type": "string",
                    "description": (
                        "The path of the file to read. You can use either a relative path "
                        "in the workspace or an absolute path. If an absolute path is "
                        "provided, it will be preserved as is."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "The line number to start reading from. Only provide if the file "
                        "is too large to read at once."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "The number of lines to read. Only provide if the file is too "
                        "large to read at once. Capped at the max window size."
                    ),
                },
            },
            "required": ["target_file"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        path_arg = args.get("target_file")
        if not isinstance(path_arg, str) or not path_arg.strip():
            raise ToolError.invalid_arguments("target_file is required")

        path = resolve_model_path(ctx.cwd, path_arg)
        if not path.exists():
            raise ToolError(f"File not found: {path}", code="not_found")
        if not path.is_file():
            raise ToolError(f"Not a file: {path}", code="invalid_arguments")

        raw = path.read_bytes()
        ext = path.suffix.lstrip(".").lower()
        if is_binary(ext, raw):
            raise ToolError(f"Cannot read binary file: {path}", code="binary_file")

        file_content = raw.decode("utf-8", errors="replace")
        if not file_content:
            return ""

        offset = _parse_optional_int(args.get("offset"), "offset")
        limit = _parse_optional_int(args.get("limit"), "limit")
        if limit is not None and limit < 0:
            raise ToolError.invalid_arguments("limit must be non-negative")

        # Window first, then guard the projected observation (not whole file).
        window = extract_file_content_lines(file_content, offset, limit)
        if len(window) > MAX_CONTENT_CHARS_FOR_TOKEN_GUARD:
            raise ToolError(
                "Requested range is too large for one read "
                f"({len(window)} characters after formatting). "
                f"Use a smaller limit (max {MAX_LINES_READ_DEFAULT} lines) or "
                "use grep to locate a narrower region first.",
                code="file_too_large",
            )
        return window


def _parse_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ToolError.invalid_arguments(f"invalid {name}: {value}") from e
