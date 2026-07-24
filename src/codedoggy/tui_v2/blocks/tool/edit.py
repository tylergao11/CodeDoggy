"""EditToolCallBlock — file edit diffs with line-number gutters.

Grok source: ``blocks/tool/edit.rs`` + snapshots (esp. ``diff_basic``).

Gutter layout (single column, indent=True default)::

      10  let x = 1;
    ^^--^^
    | |  content gap (2 spaces)
    | right-aligned line number
    2-space indent

Uses ``difflib`` to build hunks from ``old_string`` / ``new_string`` when the
caller has not already supplied structured hunks.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from codedoggy.tui_v2.blocks.tool.common import (
    HEADER_CREATING,
    HEADER_EDIT,
    S_DELETE,
    S_EQUAL,
    S_ERROR,
    S_GUTTER,
    S_INSERT,
    S_MUTED,
    Rows,
    arg_path,
    arg_str,
    display_path,
    empty_row,
    header_row,
    is_running,
    row,
    wrap_text,
)

# Layout constants (edit.rs)
INDENT = "  "
GUTTER_GAP = " "
CONTENT_GAP = "  "


class ChangeTag(str, Enum):
    EQUAL = "equal"
    DELETE = "delete"
    INSERT = "insert"


@dataclass
class DiffLine:
    text: str
    lo: int  # old-file line (0 if insert-only)
    ln: int  # new-file line (0 if delete-only)
    tag: ChangeTag


DiffHunk = list[DiffLine]


@dataclass
class DiffRenderConfig:
    """Configuration for diff rendering (Grok ``DiffRenderConfig``)."""

    indent: bool = True
    gutter_bg: bool = False
    indent_bg: bool = False
    hunk_separator: str = "…"
    dual_line_numbers: bool = False
    # When False, paint solid insert/delete FG (Grok bandless ``diff_uses_line_fg``).
    syntax_highlight: bool = True
    # Optional path for lexer selection (Grok syntect by file path).
    path: str | None = None
    # File-scoped HL map: new-file 1-based line → content fragments
    # (Grok ``by_new_line`` / FileScoped upgrade). Overrides Equal/Insert cold spans.
    by_new_line: dict | None = None


@dataclass
class GutterLayout:
    width_old: int
    width_new: int
    total: int
    dual: bool


@dataclass
class DiffLineOutput:
    fragments: list[tuple[str, str]]
    background: str | None  # style role for insert/delete band
    content_start_col: int
    gutter_span_count: int
    content_text: str
    is_separator: bool = False


def _digit_width(n: int) -> int:
    n = max(1, n)
    w = 0
    while n:
        n //= 10
        w += 1
    return w


def gutter_layout(hunk: DiffHunk, config: DiffRenderConfig) -> GutterLayout:
    max_old = 1
    max_new = 1
    for line in hunk:
        max_old = max(max_old, line.lo if line.lo > 0 else 1)
        max_new = max(max_new, line.ln if line.ln > 0 else 1)

    indent_w = len(INDENT) if config.indent else 0
    if config.dual_line_numbers:
        w_old = _digit_width(max_old)
        w_new = _digit_width(max_new)
        total = indent_w + w_old + len(GUTTER_GAP) + w_new + len(CONTENT_GAP)
        return GutterLayout(w_old, w_new, total, True)

    max_num = max(max_old, max_new)
    w_new = _digit_width(max_num)
    total = indent_w + w_new + len(CONTENT_GAP)
    return GutterLayout(0, w_new, total, False)


def render_gutter(
    line: DiffLine,
    layout: GutterLayout,
    config: DiffRenderConfig,
) -> list[tuple[str, str]]:
    """Port of ``render_gutter`` — returns gutter fragments only."""
    spans: list[tuple[str, str]] = []
    if config.indent:
        spans.append(("", INDENT))

    if layout.dual:
        w_old, w_new = layout.width_old, layout.width_new
        if line.tag is ChangeTag.EQUAL:
            spans.append((S_GUTTER, f"{line.lo:>{w_old}}"))
            spans.append(("", GUTTER_GAP))
            spans.append((S_GUTTER, f"{line.ln:>{w_new}}"))
        elif line.tag is ChangeTag.DELETE:
            spans.append((S_DELETE, f"{line.lo:>{w_old}}"))
            spans.append(("", GUTTER_GAP))
            spans.append((S_GUTTER, " " * w_new))
        else:  # INSERT
            spans.append((S_GUTTER, " " * w_old))
            spans.append(("", GUTTER_GAP))
            spans.append((S_INSERT, f"{line.ln:>{w_new}}"))
    else:
        w = layout.width_new
        if line.tag is ChangeTag.EQUAL:
            spans.append((S_GUTTER, f"{line.ln:>{w}}"))
        elif line.tag is ChangeTag.DELETE:
            spans.append((S_DELETE, f"{line.lo:>{w}}"))
        else:
            spans.append((S_INSERT, f"{line.ln:>{w}}"))

    spans.append(("", CONTENT_GAP))
    return spans


def compute_bg_start(
    config: DiffRenderConfig, gutter_width: int, indent_width: int
) -> int:
    if config.gutter_bg:
        if config.indent and config.indent_bg:
            return indent_width
        return 0
    return gutter_width


def _content_style(tag: ChangeTag) -> str:
    if tag is ChangeTag.DELETE:
        return S_DELETE
    if tag is ChangeTag.INSERT:
        return S_INSERT
    return S_EQUAL


def _bg_role(tag: ChangeTag) -> str | None:
    if tag is ChangeTag.DELETE:
        return "class:grok.diff.delete_bg"
    if tag is ChangeTag.INSERT:
        return "class:grok.diff.insert_bg"
    return None


def _content_fragments(
    line: DiffLine,
    text: str,
    config: DiffRenderConfig,
) -> list[tuple[str, str]]:
    """Content spans: FileScoped map → per-line HL → solid FG (Grok cold paint)."""
    solid = _content_style(line.tag)
    # FileScoped: Equal always; Insert when banded (we always band); Delete never.
    if (
        config.by_new_line
        and line.tag is not ChangeTag.DELETE
        and line.ln > 0
    ):
        mapped = config.by_new_line.get(line.ln)
        if mapped is not None:
            joined = "".join(t for _, t in mapped)
            expanded = text
            if joined == expanded:
                return list(mapped) if mapped else [(solid, text if text else " ")]
            # Text drift — refuse upgrade for this line (Grok).
    if not config.syntax_highlight:
        return [(solid, text if text else " ")]
    try:
        from codedoggy.tui_v2.syntax_hl import highlight_or_solid

        return highlight_or_solid(
            text,
            path=config.path,
            solid_style=solid,
            enable=True,
        )
    except Exception:  # noqa: BLE001
        return [(solid, text if text else " ")]


def compute_file_scoped_styles(
    path: str | None,
    file_text: str,
    hunks: Sequence[DiffHunk],
) -> dict[int, list[tuple[str, str]]] | None:
    """Grok ``compute_file_scoped_styles``: full-file HL for Equal/Insert lines.

    Returns None if any needed new-file line text drifts from ``file_text``
    (refuse upgrade so displayed content never rewrites).
    """
    expected: dict[int, str] = {}
    for hunk in hunks:
        for line in hunk:
            if line.tag is ChangeTag.DELETE or line.ln <= 0:
                continue
            text = line.text.rstrip("\r\n")
            if "\t" in text:
                text = text.expandtabs(4)
            expected[line.ln] = text
    if not expected:
        return {}
    max_needed = max(expected)
    try:
        from codedoggy.tui_v2.syntax_hl import highlight_file_lines

        by_line = highlight_file_lines(
            file_text, path=path, max_line=max_needed
        )
    except Exception:  # noqa: BLE001
        return None

    out: dict[int, list[tuple[str, str]]] = {}
    file_lines = file_text.splitlines()
    for ln, exp in expected.items():
        if ln < 1 or ln > len(file_lines):
            return None
        disk = file_lines[ln - 1]
        if "\t" in disk:
            disk = disk.expandtabs(4)
        if disk != exp:
            return None
        frags = by_line.get(ln)
        if frags is None:
            return None
        joined = "".join(t for _, t in frags)
        # Pygments may drop pure whitespace differences; require exact join.
        if joined != exp and exp:
            # Re-HL single line as fallback for that line only.
            try:
                from codedoggy.tui_v2.syntax_hl import highlight_code_line

                frags = highlight_code_line(exp, path=path)
                if "".join(t for _, t in frags) != exp:
                    frags = [("class:grok.syn.plain", exp)]
            except Exception:  # noqa: BLE001
                frags = [("class:grok.syn.plain", exp)]
        out[ln] = list(frags)
    if len(out) != len(expected):
        return None
    return out


# (path, mtime_ns, size) → file text cache for FileScoped upgrade
_FILE_TEXT_CACHE: dict[tuple[str, int, int], str] = {}
_FILE_STYLE_CACHE: dict[tuple[str, int, int, int], dict] = {}  # + max_ln


def try_load_file_text(path: str | None, *, max_bytes: int = 1_500_000) -> str | None:
    """Read post-edit file for FileScoped upgrade; None on miss/oversize."""
    if not path:
        return None
    try:
        from pathlib import Path

        p = Path(path)
        if not p.is_file():
            return None
        st = p.stat()
        if st.st_size > max_bytes:
            return None
        key = (str(p.resolve()), int(getattr(st, "st_mtime_ns", st.st_mtime * 1e9)), int(st.st_size))
        cached = _FILE_TEXT_CACHE.get(key)
        if cached is not None:
            return cached
        text = p.read_text(encoding="utf-8", errors="replace")
        # Bound cache size
        if len(_FILE_TEXT_CACHE) > 32:
            _FILE_TEXT_CACHE.clear()
        _FILE_TEXT_CACHE[key] = text
        return text
    except Exception:  # noqa: BLE001
        return None


def cached_file_scoped_styles(
    path: str | None,
    file_text: str,
    hunks: Sequence[DiffHunk],
) -> dict[int, list[tuple[str, str]]] | None:
    """FileScoped styles with mtime cache (avoids re-lex on every paint)."""
    if not path or not file_text:
        return compute_file_scoped_styles(path, file_text, hunks)
    try:
        from pathlib import Path

        p = Path(path)
        st = p.stat()
        max_ln = 0
        for hunk in hunks:
            for line in hunk:
                if line.tag is not ChangeTag.DELETE and line.ln > max_ln:
                    max_ln = line.ln
        key = (
            str(p.resolve()),
            int(getattr(st, "st_mtime_ns", st.st_mtime * 1e9)),
            int(st.st_size),
            max_ln,
        )
        hit = _FILE_STYLE_CACHE.get(key)
        if hit is not None:
            return hit
        styles = compute_file_scoped_styles(path, file_text, hunks)
        if styles is not None:
            if len(_FILE_STYLE_CACHE) > 32:
                _FILE_STYLE_CACHE.clear()
            _FILE_STYLE_CACHE[key] = styles
        return styles
    except Exception:  # noqa: BLE001
        return compute_file_scoped_styles(path, file_text, hunks)


def assemble_diff_line(
    line: DiffLine,
    layout: GutterLayout,
    config: DiffRenderConfig,
    content_width: int,
) -> list[DiffLineOutput]:
    text = line.text.rstrip("\r\n")
    # Expand tabs lightly (tab_width=4 default spirit).
    if "\t" in text:
        text = text.expandtabs(4)

    bg = _bg_role(line.tag)
    indent_w = len(INDENT) if config.indent else 0
    bg_start = compute_bg_start(config, layout.total, indent_w)
    content_frags = _content_fragments(line, text, config)

    if content_width <= 0 or len(text) <= content_width:
        gutter = render_gutter(line, layout, config)
        frags = list(gutter)
        if content_frags:
            frags.extend(content_frags)
        else:
            frags.append((_content_style(line.tag), text if text else " "))
        return [
            DiffLineOutput(
                fragments=frags,
                background=bg,
                content_start_col=bg_start,
                gutter_span_count=len(gutter),
                content_text=text,
            )
        ]

    # Wrap long lines: first row has real gutter; continuations pad spaces.
    # Slice HL fragments by char range so wrap keeps styles.
    try:
        from codedoggy.tui_v2.syntax_hl import slice_fragments
    except Exception:  # noqa: BLE001
        slice_fragments = None  # type: ignore[assignment]

    outputs: list[DiffLineOutput] = []
    gutter_padding = " " * layout.total
    pieces = wrap_text(text, content_width)
    # Map pieces back onto original text offsets (wrap_text is sequential).
    cursor = 0
    for i, piece in enumerate(pieces):
        if i == 0:
            gutter = render_gutter(line, layout, config)
            frags = list(gutter)
            gc = len(gutter)
        else:
            frags = [("", gutter_padding)]
            gc = 1
        if not piece:
            frags.append((_content_style(line.tag), " "))
        elif slice_fragments is not None and content_frags:
            # Find piece in remaining text starting at cursor.
            # wrap_text may drop leading spaces on wrap — locate by sequential take.
            start = cursor
            end = start + len(piece)
            # If piece doesn't match at cursor (space skip), resync.
            if text[start:end] != piece:
                found = text.find(piece, cursor)
                if found >= 0:
                    start, end = found, found + len(piece)
            frags.extend(slice_fragments(content_frags, start, end) or [
                (_content_style(line.tag), piece)
            ])
            cursor = end
        else:
            frags.append((_content_style(line.tag), piece))
            cursor += len(piece)
        outputs.append(
            DiffLineOutput(
                fragments=frags,
                background=bg,
                content_start_col=bg_start,
                gutter_span_count=gc,
                content_text=piece,
            )
        )
    return outputs


def render_diff_hunk(
    hunk: DiffHunk,
    width: int,
    config: DiffRenderConfig | None = None,
) -> list[DiffLineOutput]:
    return render_diff_hunks([hunk], width, config)


def render_diff_hunks(
    hunks: Sequence[DiffHunk],
    width: int,
    config: DiffRenderConfig | None = None,
) -> list[DiffLineOutput]:
    config = config or DiffRenderConfig()
    out: list[DiffLineOutput] = []
    for i, hunk in enumerate(hunks):
        if i > 0 and out and config.hunk_separator:
            gap = _hunk_gap_lines(hunks[i - 1], hunk)
            if gap == 1:
                sep_text = f"{config.hunk_separator} 1 unchanged line"
            elif gap and gap > 1:
                sep_text = f"{config.hunk_separator} {gap} unchanged lines"
            else:
                sep_text = config.hunk_separator
            indent = INDENT if config.indent else ""
            out.append(
                DiffLineOutput(
                    fragments=[("", indent), (S_MUTED, sep_text)],
                    background=None,
                    content_start_col=0,
                    gutter_span_count=0,
                    content_text="",
                    is_separator=True,
                )
            )
        if not hunk:
            continue
        layout = gutter_layout(hunk, config)
        content_width = max(0, width - layout.total)
        for line in hunk:
            out.extend(assemble_diff_line(line, layout, config, content_width))
    return out


def _hunk_gap_lines(prev: DiffHunk, nxt: DiffHunk) -> int | None:
    prev_last = next(
        (l.ln for l in reversed(prev) if l.tag is not ChangeTag.DELETE),
        None,
    )
    next_first = next(
        (l.ln for l in nxt if l.tag is not ChangeTag.DELETE),
        None,
    )
    if prev_last is None or next_first is None:
        return None
    gap = next_first - prev_last - 1
    return gap if gap > 0 else None


def build_hunks_from_strings(
    old_string: str,
    new_string: str,
    *,
    old_start: int = 1,
    context: int = 3,
) -> list[DiffHunk]:
    """Build DiffHunk(s) from old/new text via ``difflib.SequenceMatcher``.

    Line numbers: old side starts at ``old_start``; new side mirrors with the
    same base (caller can pass absolute file line of the old_string match).
    """
    old_lines = old_string.splitlines()
    new_lines = new_string.splitlines()
    # Preserve a single trailing empty only when source had trailing newline
    # and no content — splitlines already drops final empty; fine for edits.

    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    # Collect all opcodes then group into context-bounded hunks.
    opcodes = list(sm.get_opcodes())
    if not opcodes:
        return []

    # Map each changed region with context into hunks (merge overlapping).
    regions: list[tuple[int, int, int, int]] = []  # i1,i2,j1,j2 change spans
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        regions.append((i1, i2, j1, j2))
    if not regions:
        # Pure equal — nothing to show as a diff; still paint as equal lines
        # when both sides identical non-empty (rare for search_replace).
        return []

    # Expand regions by context and merge.
    expanded: list[tuple[int, int, int, int]] = []
    for i1, i2, j1, j2 in regions:
        ei1 = max(0, i1 - context)
        ei2 = min(len(old_lines), i2 + context)
        # Align new context by equal-prefix assumption around change.
        # Use matching window sizes from SequenceMatcher groups later.
        ej1 = max(0, j1 - context)
        ej2 = min(len(new_lines), j2 + context)
        if expanded and ei1 <= expanded[-1][1]:
            p = expanded[-1]
            expanded[-1] = (p[0], max(p[1], ei2), p[2], max(p[3], ej2))
        else:
            expanded.append((ei1, ei2, ej1, ej2))

    hunks: list[DiffHunk] = []
    for ei1, ei2, ej1, ej2 in expanded:
        # Re-diff the window so equal lines get correct pairing.
        o_win = old_lines[ei1:ei2]
        n_win = new_lines[ej1:ej2]
        sm_w = difflib.SequenceMatcher(a=o_win, b=n_win, autojunk=False)
        hunk: DiffHunk = []
        for tag, a1, a2, b1, b2 in sm_w.get_opcodes():
            if tag == "equal":
                for k in range(a2 - a1):
                    hunk.append(
                        DiffLine(
                            text=o_win[a1 + k],
                            lo=old_start + ei1 + a1 + k,
                            ln=old_start + ej1 + b1 + k,
                            tag=ChangeTag.EQUAL,
                        )
                    )
            elif tag == "delete":
                for k in range(a1, a2):
                    hunk.append(
                        DiffLine(
                            text=o_win[k],
                            lo=old_start + ei1 + k,
                            ln=0,
                            tag=ChangeTag.DELETE,
                        )
                    )
            elif tag == "insert":
                for k in range(b1, b2):
                    hunk.append(
                        DiffLine(
                            text=n_win[k],
                            lo=0,
                            ln=old_start + ej1 + k,
                            tag=ChangeTag.INSERT,
                        )
                    )
            elif tag == "replace":
                for k in range(a1, a2):
                    hunk.append(
                        DiffLine(
                            text=o_win[k],
                            lo=old_start + ei1 + k,
                            ln=0,
                            tag=ChangeTag.DELETE,
                        )
                    )
                for k in range(b1, b2):
                    hunk.append(
                        DiffLine(
                            text=n_win[k],
                            lo=0,
                            ln=old_start + ej1 + k,
                            tag=ChangeTag.INSERT,
                        )
                    )
        if hunk:
            hunks.append(hunk)
    return hunks


def count_changes(hunks: Sequence[DiffHunk]) -> tuple[int, int]:
    ins = del_ = 0
    for hunk in hunks:
        for line in hunk:
            if line.tag is ChangeTag.INSERT:
                ins += 1
            elif line.tag is ChangeTag.DELETE:
                del_ += 1
    return ins, del_


def diff_outputs_to_string(outputs: Sequence[DiffLineOutput]) -> str:
    """Plain-text dump matching Grok snap ``diff_outputs_to_string``."""
    lines: list[str] = []
    for o in outputs:
        lines.append("".join(t for _, t in o.fragments))
    return "\n".join(lines)


def paint_edit(
    arguments: dict,
    result: str,
    *,
    width: int,
    collapsed: bool,
    status: str,
    selected: bool = False,
    is_write: bool = False,
) -> Rows:
    path = arg_path(arguments)
    old_string = arg_str(arguments, "old_string", "old_str", "old", default="")
    new_string = arg_str(arguments, "new_string", "new_str", "new", "contents", default="")
    if is_write and not new_string:
        new_string = arg_str(arguments, "contents", "content", "text", default="")
        old_string = ""

    running = is_running(status)
    muted = collapsed and not running

    # Fixed present verbs (Grok individual tool block); tense is verb_group only.
    if is_write:
        prefix = HEADER_CREATING
    else:
        prefix = HEADER_EDIT

    shown = display_path(path or "?", collapsed=collapsed)

    # Build hunks
    hunks: list[DiffHunk] = []
    if is_write and new_string and not old_string:
        # Creating: all inserts starting at line 1.
        lines = new_string.splitlines() or ([""] if new_string == "" else [])
        hunk = [
            DiffLine(text=ln, lo=0, ln=i + 1, tag=ChangeTag.INSERT)
            for i, ln in enumerate(lines)
        ]
        if hunk:
            hunks = [hunk]
    elif old_string or new_string:
        hunks = build_hunks_from_strings(old_string, new_string)

    # Diffstat suffix on collapsed header only.
    suffixes: list[tuple[str, str]] = []
    if collapsed and hunks:
        ins, dele = count_changes(hunks)
        if ins > 0 or dele > 0:
            suffixes.append((S_INSERT, f" +{ins}"))
            suffixes.append((S_MUTED, "/"))
            suffixes.append((S_DELETE, f"-{dele}"))

    rows: Rows = [
        header_row(prefix, shown, *suffixes, muted=muted, selected=selected)
    ]

    if collapsed:
        return rows

    if status.lower() in {"failed", "error"} and result:
        rows.append(empty_row())
        for err_line in result.splitlines():
            rows.append(row((S_ERROR, err_line)))

    if not hunks:
        # Fall back: show result text if any.
        if result and status.lower() not in {"failed", "error"}:
            rows.append(empty_row())
            for line in result.splitlines():
                rows.append(row((S_MUTED, line)))
        return rows

    rows.append(empty_row())
    by_new = None
    file_text = try_load_file_text(path or None)
    if file_text is not None:
        by_new = cached_file_scoped_styles(path or None, file_text, hunks)
    config = DiffRenderConfig(
        path=path or None,
        syntax_highlight=True,
        by_new_line=by_new,
    )
    outputs = render_diff_hunks(hunks, width, config)
    for o in outputs:
        if o.is_separator:
            rows.append(row(*o.fragments))
            continue
        # Apply panel/diff bg on content-oriented style when banded.
        frags = list(o.fragments)
        if o.background:
            # Tint content spans (after gutter) with bg class.
            tinted: list[tuple[str, str]] = []
            for i, (st, tx) in enumerate(frags):
                if i >= o.gutter_span_count and tx:
                    st = f"{o.background} {st}" if st else o.background
                tinted.append((st, tx))
            frags = tinted
        rows.append(row(*frags))

    return rows
