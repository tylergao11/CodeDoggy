"""read_file line extract — source port from Grok.

Ported from:
  grok-build/.../implementations/grok_build/read_file/mod.rs
    resolve_read_start_line
    extract_file_content_lines

Harness-compatible negative offset + trailing-empty-line behavior.
Inline base64 image extraction is optional (host multimodal); we keep a
no-op stub so non-data-uri lines round-trip byte-equal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ExtractedContent:
    """Grok ExtractedContent (content_concise == content for now)."""

    content: str
    content_concise: str
    raw_output: str
    extracted_images: list[dict] = field(default_factory=list)


def resolve_read_start_line(file_content: str, offset: int | None) -> int:
    """1-indexed start line (Grok harness-compatible).

    Negatives use ``split('\\n')`` field count plus a phantom field when the
    file is non-empty and has no trailing ``\\n``. Extraction still uses
    split_inclusive, so a start that lands on the phantom-only field yields
    an empty window.
    """
    offset_raw = 1 if offset is None else int(offset)
    if offset_raw == 0:
        return 1
    if offset_raw > 0:
        return offset_raw
    total_fields = len(file_content.split("\n"))
    if file_content and not file_content.endswith("\n"):
        total_fields += 1
    computed = total_fields + offset_raw + 1
    return max(1, computed)


def _strip_line_ending(s: str) -> str:
    """Grok strip: only strip ``\\r`` after a trailing ``\\n`` was removed.

    A lone trailing ``\\r`` (no ``\\n``) is preserved in the model-visible line.
    """
    if not s.endswith("\n"):
        return s
    s = s[:-1]
    if s.endswith("\r"):
        s = s[:-1]
    return s


def split_inclusive_newline(s: str) -> list[str]:
    """Python stand-in for Rust ``str::split_inclusive('\\n')``."""
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


def _try_extract_base64_images(
    line: str,
) -> tuple[str, list[dict]] | None:
    """Optional hook: replace data-uri images with placeholder.

    Full Grok util lives in base64_images.rs; without it, return None so the
    line is used unchanged.
    """
    # Minimal fidelity: leave lines alone (host can inject later).
    _ = line
    return None


def extract_file_content_lines(
    file_content: str,
    offset: int | None,
    limit: int | None,
    total_lines: int = 0,
    *,
    line_transform: Callable[[str], tuple[str, list[dict]] | None] | None = None,
) -> ExtractedContent:
    """Windowed line view; prefix on first visible line and every 10th line number.

    ``limit is None`` → no limit (``usize::MAX`` in Grok). Caller/tool should
    clamp to MAX_LINES_READ.
    """
    transform = line_transform or _try_extract_base64_images
    output: list[str] = []
    start = 0
    end = 0
    first_line: int | None = None
    extracted_images: list[dict] = []

    lines_inc = split_inclusive_newline(file_content)
    split_count = len(lines_inc)
    has_trailing_empty = bool(file_content) and file_content.endswith("\n")
    skip = max(0, resolve_read_start_line(file_content, offset) - 1)
    take = 10**18 if limit is None else max(0, int(limit))

    if not file_content and total_lines > 0 and skip == 0 and take > 0:
        output.append("1→")
        first_line = 1

    pos = 0
    taken = 0
    for i, line_with_nl in enumerate(lines_inc):
        line_len = len(line_with_nl)
        if i < skip:
            pos += line_len
            continue
        if taken >= take:
            break
        line = _strip_line_ending(line_with_nl)
        is_first_visible = first_line is None
        if is_first_visible:
            start = pos
            first_line = i + 1
        else:
            output.append("\n")
        end = pos + line_len

        transformed = transform(line)
        if transformed is not None:
            line, imgs = transformed
            extracted_images.extend(imgs)

        line_num = i + 1
        if is_first_visible or line_num % 10 == 0:
            output.append(f"{line_num}→{line}")
        else:
            output.append(line)
        taken += 1
        pos += line_len

    # Advance pos for skipped remaining only if we stopped early — not needed
    # for trailing empty handling.
    if has_trailing_empty:
        trailing_line_idx = split_count
        if trailing_line_idx >= skip and trailing_line_idx < skip + take:
            line_num = trailing_line_idx + 1
            is_first_visible = first_line is None
            if is_first_visible:
                first_line = line_num
            else:
                output.append("\n")
            if is_first_visible or line_num % 10 == 0:
                output.append(f"{line_num}→")

    content = "".join(output)
    if first_line is None or not file_content:
        raw_output = ""
    else:
        raw_output = file_content[start:end]
        if raw_output.endswith("\r\n"):
            raw_output = raw_output[:-2] + "\n"

    return ExtractedContent(
        content=content,
        content_concise=content,
        raw_output=raw_output,
        extracted_images=extracted_images,
    )
