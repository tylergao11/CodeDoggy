"""read_file — Grok ReadFileTool wire + source-ported extract.

Extract core: ``codedoggy.tools.grok_build.read_file_extract``
  Ported from implementations/grok_build/read_file/mod.rs

Rich formats: ``util/rich_files`` (pdf/pptx/image subset).
"""

from __future__ import annotations

from typing import Any

from codedoggy.tools.defaults import MAX_LINES_READ_DEFAULT
from codedoggy.tools.grok_build import read_file_extract as _extract_mod
from codedoggy.tools.grok_build.read_file_extract import (
    ExtractedContent,
    resolve_read_start_line,
    split_inclusive_newline,
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
from codedoggy.tools.util.binary import is_binary
from codedoggy.tools.util.paths import resolve_model_path
from codedoggy.tools.util.rich_files import (
    is_image,
    is_pdf,
    is_pptx,
    read_image_meta,
    read_pdf_text,
    read_pptx_text,
)

# Re-export for tests that import from builtins.read_file
__all__ = [
    "ReadFileTool",
    "extract_file_content_lines",
    "resolve_read_start_line",
    "split_inclusive_newline",
    "ExtractedContent",
    "MAX_NUM_TOKENS",
    "MAX_LINES_READ",
]


def extract_file_content_lines(
    file_content: str,
    offset: int | None = None,
    limit: int | None = None,
    total_lines: int = 0,
) -> str:
    """Return formatted content string (Grok ``ExtractedContent.content``)."""
    return _extract_mod.extract_file_content_lines(
        file_content, offset, limit, total_lines
    ).content


# Grok constants
MAX_NUM_TOKENS = 25_000
MAX_LINES_READ = MAX_LINES_READ_DEFAULT
MAX_CONTENT_CHARS_FOR_TOKEN_GUARD = MAX_NUM_TOKENS * 4

# Grok DESCRIPTION_FULL
_DESCRIPTION = f"""\
Read a file.

Usage:
- The target_file parameter can be a relative path in the workspace or an absolute path
- By default, it reads up to {MAX_LINES_READ} lines starting from the beginning of the file
- Results are returned with line numbers starting at 1. The format is: LINE_NUMBER→LINE_CONTENT
- This tool can read PDF files (.pdf), PowerPoint files (.pptx), Jupyter notebooks (.ipynb files), and image files (e.g. PNG, JPG, etc).
- When reading an image file the contents are presented visually as this tool uses multimodal LLMs.
"""


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
                        "large to read at once."
                    ),
                },
                "pages": {
                    "type": "string",
                    "description": (
                        "Page range for PDF files (e.g. '1-5', '3', '10-'). "
                        "Required for PDFs with more than 10 pages. Max 20 pages per call. "
                        "Ignored for non-PDF files."
                    ),
                },
                "format": {
                    "type": "string",
                    "description": (
                        "Output format for PDF files. 'image' (default) renders pages as "
                        "images. 'text' extracts text content. Ignored for non-PDF files."
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
        if path.is_dir():
            raise ToolError(
                f"Error: {path} is a directory, not a file.",
                code="is_a_directory",
            )
        if not path.is_file():
            raise ToolError(f"Not a file: {path}", code="invalid_arguments")

        try:
            raw = path.read_bytes()
        except PermissionError as e:
            raise ToolError(f"Permission denied: {path}", code="permission_denied") from e
        except OSError as e:
            raise ToolError(f"Failed to read file: {path}, {e}", code="io_error") from e

        fmt = args.get("format")
        fmt_s = str(fmt).strip().lower() if fmt is not None else None

        # Image first (Grok: bytes_to_metadata → image path)
        if is_image(path, raw) and fmt_s != "text":
            try:
                return read_image_meta(path, raw)
            except ValueError as e:
                raise ToolError(str(e), code="image_error") from e

        if is_pdf(path, raw):
            # Grok default format for PDF is image; without multimodal host we
            # fall back to text extract and note honesty in error paths.
            extract_text = fmt_s == "text" or fmt_s in {None, "auto"}
            if fmt_s == "image":
                # No page renderer in pure Python tools — honest soft path via text note
                try:
                    text = read_pdf_text(
                        raw,
                        pages=args.get("pages") if isinstance(args.get("pages"), str) else None,
                    )
                except ValueError as e:
                    raise ToolError(str(e), code="pdf_error") from e
                return (
                    "[PDF format=image not available without host page renderer; "
                    "showing text extract instead]\n\n"
                    + _window_text(text, args)
                )
            if extract_text:
                try:
                    text = read_pdf_text(
                        raw,
                        pages=args.get("pages") if isinstance(args.get("pages"), str) else None,
                    )
                except ValueError as e:
                    raise ToolError(str(e), code="pdf_error") from e
                return _window_text(text, args)
            if fmt_s is not None and fmt_s not in {"image", "text", "auto"}:
                raise ToolError(
                    f"Invalid format '{fmt_s}'. Supported values: 'image' (default), 'text'.",
                    code="invalid_arguments",
                )

        if is_pptx(path, raw):
            try:
                text = read_pptx_text(raw)
            except ValueError as e:
                raise ToolError(str(e), code="pptx_error") from e
            return _window_text(text, args)

        ext = path.suffix.lstrip(".").lower()
        if is_binary(ext, raw):
            raise ToolError(f"Cannot read binary file: {path}", code="binary_file")

        file_content = raw.decode("utf-8", errors="replace")
        if not file_content:
            return ""
        return _window_text(file_content, args)


def _window_text(file_content: str, args: dict[str, Any]) -> str:
    offset = _parse_optional_int(args.get("offset"), "offset")
    limit = _parse_optional_int(args.get("limit"), "limit")
    if limit is not None and limit < 0:
        raise ToolError.invalid_arguments("limit must be non-negative")
    # Grok tool path clamps to MAX_LINES_READ
    effective_limit = MAX_LINES_READ if limit is None else min(limit, MAX_LINES_READ)
    # Grok tests: matches('\n').count() + 1
    total_lines = file_content.count("\n") + 1 if file_content else 0
    extracted = _extract_mod.extract_file_content_lines(
        file_content,
        offset,
        effective_limit,
        total_lines,
    )
    window = extracted.content
    if len(window) > MAX_CONTENT_CHARS_FOR_TOKEN_GUARD:
        raise ToolError(
            "Requested range is too large for one read "
            f"({len(window)} characters after formatting). "
            f"Use a smaller limit (max {MAX_LINES_READ} lines) or "
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
