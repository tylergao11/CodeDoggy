"""grep finalize / formatters — source port from Grok.

Ported from:
  implementations/grok_build/grep/mod.rs
    resolve_effective_head_limit, max_head_limit
    finalize_grep (format path)
    parse_numbered_line_prefix, count_matches
    format_content_output, format_files_with_matches_output, format_count_output
    first_idx_exceed_cum_limit
  util/truncate.rs::truncate_line (grep line clip)
"""

from __future__ import annotations

from enum import Enum

# Grok constants
CONTENT_LINE_LIMIT: int = 2_000
CONTENT_LINE_DEFAULT: int = 200
FILE_COUNT_LIMIT: int = 10_000
FILE_COUNT_DEFAULT: int = 500
DEFAULT_MAX_CHARS_PER_LINE: int = 1_000
DEFAULT_TOOL_OUTPUT_BYTES: int = 40_000  # crate DEFAULT_TOOL_OUTPUT_BYTES
MAX_STDOUT_BYTES: int = 5_000_000


class OutputMode(str, Enum):
    Content = "content"
    FilesWithMatches = "files_with_matches"
    Count = "count"


def resolve_effective_head_limit(head_limit: int | None, mode: OutputMode) -> int:
    """Grok: ``input.head_limit.unwrap_or(default).min(cap)`` — no floor at 1."""
    if mode is OutputMode.Content:
        default, cap = CONTENT_LINE_DEFAULT, CONTENT_LINE_LIMIT
    else:
        default, cap = FILE_COUNT_DEFAULT, FILE_COUNT_LIMIT
    raw = default if head_limit is None else int(head_limit)
    return min(raw, cap)


def max_head_limit(mode: OutputMode) -> int:
    if mode is OutputMode.Content:
        return CONTENT_LINE_LIMIT
    return FILE_COUNT_LIMIT


def truncate_line(line: str, max_chars: int = DEFAULT_MAX_CHARS_PER_LINE) -> str:
    """Grok util/truncate.rs::truncate_line (char-count, long marker)."""
    if len(line.encode("utf-8", errors="replace")) <= max_chars and len(line) <= max_chars:
        # Fast path: byte_len ≤ max ⇒ char_count ≤ max for ASCII; multi-byte falls through
        if sum(1 for _ in line) <= max_chars:
            return line
    char_count = sum(1 for _ in line)
    if char_count <= max_chars:
        return line
    # take first max_chars characters
    taken = 0
    end = 0
    for i, ch in enumerate(line):
        if taken >= max_chars:
            end = i
            break
        taken += 1
        end = i + 1
    else:
        end = len(line)
    return f"{line[:end]} [... truncated ({char_count} chars total)]"


def parse_numbered_line_prefix(line: str) -> tuple[int, str, str] | None:
    """Parse ``123:content`` or ``45-context`` (heading mode body lines)."""
    idx = 0
    n = len(line)
    while idx < n and line[idx].isdigit():
        idx += 1
    if idx == 0 or idx >= n:
        return None
    sep = line[idx]
    if sep not in {":", "-"}:
        return None
    try:
        line_number = int(line[:idx])
    except ValueError:
        return None
    return line_number, sep, line[idx + 1 :]


def count_matches(output_lines: list[str]) -> int:
    """Count content match lines (``:`` sep), not context (``-``)."""
    n = 0
    for line in output_lines:
        parsed = parse_numbered_line_prefix(line)
        if parsed is not None and parsed[1] == ":":
            n += 1
    return n


def first_idx_exceed_cum_limit(lines: list[str], limit: int) -> int:
    """Grok: sum of UTF-8 byte lengths (String::len), no inter-line +1."""
    cum = 0
    for i, line in enumerate(lines):
        line_len = len(line.encode("utf-8", errors="replace"))
        if cum + line_len > limit:
            return i
        cum += line_len
    return len(lines)


def format_content_output(
    output_lines: list[str],
    is_truncated: bool,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
    max_output_bytes: int = DEFAULT_TOOL_OUTPUT_BYTES,
) -> str:
    is_truncated_str = "at least " if is_truncated else ""
    num_matching = count_matches(output_lines)
    final: list[str] = [f"Found {is_truncated_str}{num_matching} matching lines"]
    trimmed = [truncate_line(ln, max_chars_per_line) for ln in output_lines]
    cut = first_idx_exceed_cum_limit(trimmed, max_output_bytes)
    final.extend(trimmed[:cut])
    remaining = count_matches(trimmed[cut:])
    if remaining > 0:
        final.append(f"... [{is_truncated_str}{remaining} lines truncated] ...")
    return "\n".join(final)


def format_files_with_matches_output(
    output_lines: list[str],
    is_truncated: bool,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
    max_output_bytes: int = DEFAULT_TOOL_OUTPUT_BYTES,
) -> str:
    is_truncated_str = "at least " if is_truncated else ""
    final: list[str] = [f"Found {is_truncated_str}{len(output_lines)} files"]
    trimmed = [truncate_line(ln, max_chars_per_line) for ln in output_lines]
    cut = first_idx_exceed_cum_limit(trimmed, max_output_bytes)
    final.extend(trimmed[:cut])
    if len(output_lines) > cut:
        final.append(
            f"... [{is_truncated_str}{len(output_lines) - cut} lines truncated] ..."
        )
    return "\n".join(final)


def format_count_output(
    output_lines: list[str],
    is_truncated: bool,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
    max_output_bytes: int = DEFAULT_TOOL_OUTPUT_BYTES,
) -> str:
    is_truncated_str = "at least " if is_truncated else ""
    sum_matches = 0
    for line in output_lines:
        # Grok: split(':').next_back()
        count_str = line.rsplit(":", 1)[-1] if ":" in line else ""
        try:
            sum_matches += int(count_str)
        except ValueError:
            pass
    final: list[str] = [
        f"Found {sum_matches} across {is_truncated_str}{len(output_lines)} files"
    ]
    trimmed = [truncate_line(ln, max_chars_per_line) for ln in output_lines]
    cut = first_idx_exceed_cum_limit(trimmed, max_output_bytes)
    final.extend(trimmed[:cut])
    if len(output_lines) > cut:
        final.append(
            f"... [{is_truncated_str}{len(output_lines) - cut} lines truncated] ..."
        )
    return "\n".join(final)


def wrap_workspace_result(workspace_path: str, body: str) -> str:
    return (
        f'<workspace_result workspace_path="{workspace_path}">\n'
        f"{body}\n"
        f"</workspace_result>"
    )


def finalize_grep_body(
    output_lines: list[str],
    *,
    mode: OutputMode,
    is_truncated: bool,
    effective_head_limit: int,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
    max_output_bytes: int = DEFAULT_TOOL_OUTPUT_BYTES,
) -> str:
    """Format path of finalize_grep after lines collected."""
    lines = list(output_lines)
    trunc = is_truncated
    if len(lines) > effective_head_limit:
        trunc = True
        lines = lines[:effective_head_limit]
    if mode is OutputMode.Content:
        return format_content_output(lines, trunc, max_chars_per_line, max_output_bytes)
    if mode is OutputMode.FilesWithMatches:
        return format_files_with_matches_output(
            lines, trunc, max_chars_per_line, max_output_bytes
        )
    return format_count_output(lines, trunc, max_chars_per_line, max_output_bytes)


def no_matches_card(cwd_display: str) -> str:
    return wrap_workspace_result(cwd_display, "No matches found")


def rg_exit2_message(stderr: str, cwd_display: str) -> str:
    return f"Error calling tool: {stderr} (exit 2, root: {cwd_display})"


def rg_unknown_exit_message(exit_code: int, cwd_display: str) -> str:
    return (
        f"Error calling tool: unknown error (exit {exit_code}, root: {cwd_display})"
    )
