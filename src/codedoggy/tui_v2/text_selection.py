"""Text selection over painted scrollback rows (Grok text_selection spirit).

Coordinates are (row, col) in the *viewport* (what the user sees). Selection
is linear: whole lines between anchor and head, partial first/last line.

When reconstructing copy text, leading quote-bar chrome (``│ `` / ``│`` from
``quote_bar.rs``) is stripped so paste matches source rather than paint.
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt_toolkit.formatted_text import StyleAndTextTuples

S_HL = "class:grok.selected reverse"

# Grok quote_bar.rs: blockquote `>` is rewritten to U+2502 box-drawing bar.
_QUOTE_BAR = "\u2502"  # │


@dataclass(slots=True)
class TextSel:
    """Persistent or active drag selection in viewport coordinates."""

    anchor_row: int
    anchor_col: int
    head_row: int
    head_col: int
    active: bool = False  # True while mouse button held

    def normalized(self) -> tuple[int, int, int, int]:
        """Return (r0, c0, r1, c1) with (r0,c0) <= (r1,c1) in reading order."""
        a = (self.anchor_row, self.anchor_col)
        b = (self.head_row, self.head_col)
        if a <= b:
            return a[0], a[1], b[0], b[1]
        return b[0], b[1], a[0], a[1]

    def is_empty(self) -> bool:
        return (
            self.anchor_row == self.head_row and self.anchor_col == self.head_col
        )


def row_plain_text(row: StyleAndTextTuples) -> str:
    return "".join(t for _, t in row).rstrip("\n")


def strip_quote_bar_prefix(line: str) -> str:
    """Strip leading Grok quote-bar chrome from a selection line.

    Repeatedly peels a leading ``│ `` (U+2502 + space) or bare ``│`` so nested
    bars (``│ │ deep`` → ``deep``) match ``quote_bar.rs`` decoration rather
    than source. Mid-line bars are left untouched.
    """
    while True:
        if line.startswith(_QUOTE_BAR + " "):
            line = line[2:]
        elif line.startswith(_QUOTE_BAR):
            line = line[1:]
        else:
            break
    return line


def reconstruct_selection_text(
    viewport_rows: list[StyleAndTextTuples],
    sel: TextSel,
) -> str:
    """Join selected cells from viewport rows into plain text (with newlines).

    Each line (or partial first/last slice) is passed through
    :func:`strip_quote_bar_prefix` so copied text omits quote-bar paint.
    """
    if not viewport_rows or sel.is_empty():
        return ""
    r0, c0, r1, c1 = sel.normalized()
    r0 = max(0, r0)
    r1 = min(len(viewport_rows) - 1, r1)
    if r0 > r1:
        return ""
    parts: list[str] = []
    for r in range(r0, r1 + 1):
        plain = row_plain_text(viewport_rows[r])
        if r == r0 and r == r1:
            chunk = plain[c0:c1]
        elif r == r0:
            chunk = plain[c0:]
        elif r == r1:
            chunk = plain[:c1]
        else:
            chunk = plain
        parts.append(strip_quote_bar_prefix(chunk))
    return "\n".join(parts)


def apply_text_selection_highlight(
    viewport_rows: list[StyleAndTextTuples],
    sel: TextSel | None,
) -> list[StyleAndTextTuples]:
    """Return new rows with reverse highlight on the selected range."""
    if sel is None or sel.is_empty() or not viewport_rows:
        return viewport_rows
    r0, c0, r1, c1 = sel.normalized()
    r0 = max(0, r0)
    r1 = min(len(viewport_rows) - 1, r1)
    if r0 > r1:
        return viewport_rows

    out: list[StyleAndTextTuples] = []
    for r, row in enumerate(viewport_rows):
        if r < r0 or r > r1:
            out.append(row)
            continue
        plain = row_plain_text(row)
        # Determine [start, end) columns to highlight on this row
        if r == r0 and r == r1:
            s, e = c0, c1
        elif r == r0:
            s, e = c0, len(plain)
        elif r == r1:
            s, e = 0, c1
        else:
            s, e = 0, len(plain)
        s = max(0, min(s, len(plain)))
        e = max(s, min(e, len(plain)))
        if s >= e:
            out.append(row)
            continue
        out.append(_highlight_row_range(row, s, e))
    return out


def _highlight_row_range(
    row: StyleAndTextTuples, start: int, end: int
) -> StyleAndTextTuples:
    """Split fragments so [start, end) gets highlight style."""
    result: StyleAndTextTuples = []
    pos = 0
    for st, tx in row:
        if tx == "\n":
            result.append((st, tx))
            continue
        n = len(tx)
        frag_start = pos
        frag_end = pos + n
        # No overlap
        if frag_end <= start or frag_start >= end:
            result.append((st, tx))
            pos = frag_end
            continue
        # Leading unselected
        if frag_start < start:
            result.append((st, tx[: start - frag_start]))
        # Selected middle
        sel_a = max(start, frag_start) - frag_start
        sel_b = min(end, frag_end) - frag_start
        if sel_a < sel_b:
            mid = tx[sel_a:sel_b]
            style = f"{S_HL} {st}" if st else S_HL
            result.append((style, mid))
        # Trailing unselected
        if frag_end > end:
            result.append((st, tx[end - frag_start :]))
        pos = frag_end
    return result


def col_at_x(row: StyleAndTextTuples, x: int) -> int:
    """Map pixel/column x to character index (cwidth ≈ len for ASCII chrome)."""
    plain = row_plain_text(row)
    return max(0, min(int(x), len(plain)))


__all__ = [
    "TextSel",
    "apply_text_selection_highlight",
    "col_at_x",
    "reconstruct_selection_text",
    "row_plain_text",
    "strip_quote_bar_prefix",
]
