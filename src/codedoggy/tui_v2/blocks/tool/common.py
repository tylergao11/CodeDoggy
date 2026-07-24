"""Shared header helpers for Grok tool-call scrollback blocks.

Maps ratatui Span/Line paint to prompt_toolkit StyleAndTextTuples rows
at the paint edge. Style classes use the ``class:grok.*`` prefix.
"""

from __future__ import annotations

import os
from pathlib import PurePosixPath
from typing import Sequence

# ---------------------------------------------------------------------------
# Output types (PORT.md contract)
# ---------------------------------------------------------------------------

Fragment = tuple[str, str]
Row = list[Fragment]
Rows = list[Row]

# Re-export for hook/other modules that import Fragment from common.
__all_common_types__ = ("Fragment", "Row", "Rows")

# ---------------------------------------------------------------------------
# Theme / style fallbacks (import from tui_v2 when present)
# ---------------------------------------------------------------------------

from codedoggy.tui_v2.theme import (
    S_BOLD,
    S_COMMAND,
    S_DELETE,
    S_DIM,
    S_EQUAL,
    S_ERROR,
    S_GUTTER,
    S_INSERT,
    S_MUTED,
    S_PANEL,
    S_PATH,
    S_PRIMARY,
    S_SELECTED,
    S_SUCCESS,
)

# Individual tool header verb strings (Grok block source: fixed present verbs).
# Status tense (Reading/Listed/…) lives in verb_group only — not on tool blocks.
HEADER_READ = "Read "
HEADER_READING = "Reading "  # legacy / verb_group; tool blocks use HEADER_READ
HEADER_EDIT = "Edit "
HEADER_EDITED = "Edited "  # legacy / verb_group
HEADER_EDITING = "Editing "  # legacy / verb_group
HEADER_CREATING = "Creating "
HEADER_SEARCH = "Search "
HEADER_SEARCHED = "Searched "  # legacy / verb_group
HEADER_SEARCHING = "Searching "  # legacy / verb_group
HEADER_LIST = "List "
HEADER_LISTED = "Listed "  # legacy / verb_group
HEADER_LISTING = "Listing "  # legacy / verb_group
HEADER_FETCH = "Fetch "
HEADER_FETCHED = "Fetched "  # legacy / verb_group
HEADER_FETCHING = "Fetching "  # legacy / verb_group
HEADER_WEB_SEARCH = "Web Search "
HEADER_SHELL = "$ "

FIRST_LINES_READ = 5
LAST_LINES_READ = 3
FIRST_LINES_EXECUTE = 5
LAST_LINES_EXECUTE = 3
MAX_INLINE_WEB = 10
TRUNCATED_INLINE_WEB = 3

ELLIPSIS = "\u2026"  # …


def is_running(status: str) -> bool:
    return status.lower() in {
        "running",
        "pending",
        "streaming",
        "in_progress",
        "active",
    }


def is_failed(status: str) -> bool:
    return status.lower() in {"failed", "error", "cancelled", "canceled"}


def row(*parts: Fragment | tuple[str, str]) -> Row:
    """One painted row: style/text fragments + trailing newline."""
    out: Row = [p for p in parts if p[1]]
    out.append(("", "\n"))
    return out


def empty_row() -> Row:
    return [("", "\n")]


def text_of(rows: Rows) -> str:
    """Join painted rows to plain text (for tests / debugging)."""
    parts: list[str] = []
    for r in rows:
        for _, t in r:
            parts.append(t)
    return "".join(parts).rstrip("\n")


def digit_count(n: int) -> int:
    """Decimal digits for gutter width (matches Rust ``digit_count``)."""
    if n <= 0:
        return 1
    d = 0
    while n:
        n //= 10
        d += 1
    return d


def basename(path: str) -> str:
    if not path:
        return path
    p = path.replace("\\", "/")
    name = PurePosixPath(p).name
    return name or path


def display_path(path: str, *, collapsed: bool, cwd: str | None = None) -> str:
    """Collapsed → basename; expanded → path relative to cwd when possible."""
    if not path:
        return path
    if collapsed:
        return basename(path)
    if cwd:
        try:
            rel = os.path.relpath(path, cwd)
            if not rel.startswith(".."):
                return rel.replace("\\", "/")
        except (ValueError, TypeError, OSError):
            pass
    return path


def truncate_str(s: str, max_width: int) -> str:
    """Hard-truncate with ellipsis, char-based (Grok ``truncate_str`` spirit)."""
    if max_width <= 0:
        return ""
    if len(s) <= max_width:
        return s
    if max_width <= 1:
        return ELLIPSIS[:max_width]
    return s[: max_width - 1] + ELLIPSIS


def shorten_path(path: str, budget: int) -> str:
    """Fish-shorten a path to fit ``budget`` columns."""
    if budget <= 0:
        return ""
    if len(path) <= budget:
        return path
    if budget <= 3:
        return ELLIPSIS[:budget]
    # Keep head + … + tail (basename preference).
    base = basename(path)
    if len(base) + 2 <= budget:
        # prefix…/basename
        keep = budget - len(base) - 1  # for …
        head = path[: max(0, keep - 1)]
        return f"{head}{ELLIPSIS}{base}" if head else f"{ELLIPSIS}{base}"[:budget]
    return truncate_str(path, budget)


def bold_label(text: str, *, muted: bool = False, selected: bool = False) -> Fragment:
    style = S_MUTED + " bold" if muted else S_BOLD
    if selected:
        style = f"{style} reverse"
    return (style, text)


def path_span(text: str, *, muted: bool = False, selected: bool = False) -> Fragment:
    style = S_MUTED if muted else S_PATH
    if selected:
        style = f"{style} reverse"
    return (style, text)


def primary_span(text: str, *, muted: bool = False) -> Fragment:
    return (S_MUTED if muted else S_PRIMARY, text)


def muted_span(text: str) -> Fragment:
    return (S_MUTED, text)


def dim_span(text: str) -> Fragment:
    return (S_DIM, text)


def error_span(text: str) -> Fragment:
    return (S_ERROR, text)


def panel_span(text: str, style: str = S_PRIMARY) -> Fragment:
    """Content inside a bg_dark panel band."""
    # Combine panel bg with foreground role.
    if style.startswith("class:"):
        return (f"{S_PANEL} {style}", text)
    return (f"{S_PANEL} {style}", text)


def wrap_text(text: str, max_width: int) -> list[str]:
    """Simple word-wrap (Grok ``wrap_text`` in edit.rs)."""
    if max_width <= 0 or not text:
        return [text]
    lines: list[str] = []
    current = ""
    current_w = 0
    # Split keeping whitespace tokens (split_inclusive spirit).
    tokens: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch.isspace():
            tokens.append(buf)
            buf = ""
    if buf:
        tokens.append(buf)
    for word in tokens:
        ww = len(word)
        if current_w + ww > max_width and current:
            lines.append(current)
            current = ""
            current_w = 0
        current += word
        current_w += ww
    if current:
        lines.append(current)
    return lines or [""]


def first_last_lines(
    lines: Sequence[str], first: int, last: int
) -> tuple[list[str], int | None]:
    """Return kept lines + hidden count (None if no truncation)."""
    total = len(lines)
    threshold = first + last
    if total <= threshold:
        return list(lines), None
    kept = list(lines[:first]) + list(lines[total - last :])
    return kept, total - threshold


def body_lines_with_ellipsis(
    lines: Sequence[str],
    first: int,
    last: int,
    *,
    show_hidden_count: bool = False,
) -> list[str]:
    kept, hidden = first_last_lines(lines, first, last)
    if hidden is None:
        return list(lines)
    mid = f"{ELLIPSIS} +{hidden} lines" if show_hidden_count else ELLIPSIS
    return list(lines[:first]) + [mid] + list(lines[len(lines) - last :])


def arg_path(arguments: dict) -> str:
    for key in ("path", "file_path", "target_file", "filename", "file"):
        val = arguments.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def arg_str(arguments: dict, *keys: str, default: str = "") -> str:
    for key in keys:
        val = arguments.get(key)
        if isinstance(val, str):
            return val
    return default


def result_lines(result: str) -> list[str]:
    if not result:
        return []
    # Preserve empty trailing only if intentional; splitlines drops final bare \n.
    return result.splitlines()


def header_row(
    prefix: str,
    target: str,
    *suffixes: Fragment,
    muted: bool = False,
    selected: bool = False,
    prefix_style: str | None = None,
    target_style: str | None = None,
) -> Row:
    """Standard tool header: bold verb + path/query + optional detail spans."""
    if prefix_style is None:
        p_style = S_MUTED + " bold" if muted else S_BOLD
    else:
        p_style = prefix_style
    if target_style is None:
        t_style = S_MUTED if muted else S_PATH
    else:
        t_style = target_style
    if selected:
        p_style = f"{p_style} reverse"
        t_style = f"{t_style} reverse"
    parts: list[Fragment] = [(p_style, prefix), (t_style, target)]
    parts.extend(suffixes)
    return row(*parts)
