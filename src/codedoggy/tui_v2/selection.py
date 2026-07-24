"""Selection box — port of ``scrollback/selection.rs`` border_chars + box.

Grok draws a box around the selected entry:

- Side borders (``│``) on every content row; dashed ``┆`` when clipped
- Top corners (``┌`` ``┐``) one row above content when not top-clipped
- Bottom corners (``└`` ``┘``) one row below content when not bottom-clipped

At the prompt_toolkit paint edge we materialize corner rows as extra
``StyleAndTextTuples`` lines so the compositor does not need a post-pass
buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt_toolkit.formatted_text import StyleAndTextTuples

from codedoggy.tui_v2.theme import style_class

# border_chars from selection.rs
TOP_LEFT = "\u250c"  # ┌
TOP_RIGHT = "\u2510"  # ┐
BOTTOM_LEFT = "\u2514"  # └
BOTTOM_RIGHT = "\u2518"  # ┘
VERTICAL = "\u2502"  # │
VERTICAL_DASHED = "\u2506"  # ┆

S_BORDER = style_class("selection_border")


@dataclass(slots=True)
class SelectionBox:
    """Grok ``SelectionBox`` (paint-side)."""

    top_clipped: bool = False
    bottom_clipped: bool = False
    style: str = S_BORDER

    def vert_char(self, *, is_first: bool, is_last: bool) -> str:
        if (is_first and self.top_clipped) or (is_last and self.bottom_clipped):
            return VERTICAL_DASHED
        return VERTICAL

    def corner_row(self, width: int, *, top: bool) -> StyleAndTextTuples:
        """One full-width row with only left/right corners filled."""
        w = max(2, int(width))
        left = TOP_LEFT if top else BOTTOM_LEFT
        right = TOP_RIGHT if top else BOTTOM_RIGHT
        mid = " " * (w - 2)
        return [
            (self.style, left),
            ("", mid),
            (self.style, right),
            ("", "\n"),
        ]


def wrap_rows_with_selection_box(
    content_rows: list[StyleAndTextTuples],
    *,
    total_width: int,
    top_clipped: bool = False,
    bottom_clipped: bool = False,
    style: str = S_BORDER,
) -> list[StyleAndTextTuples]:
    """Prepend/append corner rows around already-painted content (with side edges).

    Content rows must already include left/right vertical borders in the first
    and last cells (as ``layout.paint_block_output`` does when ``selected``).
    When clipped, content side edges use ``┆`` and the matching corner is omitted.
    """
    box = SelectionBox(
        top_clipped=top_clipped,
        bottom_clipped=bottom_clipped,
        style=style,
    )
    n = len(content_rows)
    adjusted: list[StyleAndTextTuples] = []
    for i, row in enumerate(content_rows):
        vert = box.vert_char(is_first=(i == 0), is_last=(i == n - 1))
        adjusted.append(_replace_side_edges(list(row), vert, style))
    out: list[StyleAndTextTuples] = []
    if not top_clipped:
        out.append(box.corner_row(total_width, top=True))
    out.extend(adjusted)
    if not bottom_clipped:
        out.append(box.corner_row(total_width, top=False))
    return out


def _replace_side_edges(
    row: StyleAndTextTuples, vert: str, style: str
) -> StyleAndTextTuples:
    """Swap first/last non-newline border cells to ``vert`` when they are edges."""
    if not row:
        return row
    # Strip trailing newline frag for mutation, re-add after.
    nl: StyleAndTextTuples = []
    body = list(row)
    if body and body[-1][1] == "\n":
        nl = [body.pop()]
    if not body:
        return list(row)
    # Left edge: first fragment that is a selection border char
    left_i = None
    right_i = None
    for i, (st, tx) in enumerate(body):
        if tx in {VERTICAL, VERTICAL_DASHED, TOP_LEFT, BOTTOM_LEFT}:
            left_i = i
            break
    for i in range(len(body) - 1, -1, -1):
        st, tx = body[i]
        if tx in {VERTICAL, VERTICAL_DASHED, TOP_RIGHT, BOTTOM_RIGHT}:
            right_i = i
            break
    if left_i is not None:
        body[left_i] = (style or body[left_i][0], vert)
    if right_i is not None and right_i != left_i:
        body[right_i] = (style or body[right_i][0], vert)
    return body + nl


def apply_viewport_selection_clip(
    full_lines: list[StyleAndTextTuples],
    *,
    offset: int,
    height: int,
    line_owners: list[int],
    selected_owners: set[int],
    total_width: int,
    style: str = S_BORDER,
) -> list[StyleAndTextTuples]:
    """After scroll windowing, convert cut selection boxes to dashed edges.

    When a selected block's top/bottom corner rows fall outside the viewport,
    Grok uses ``┆`` on the first/last visible side border rows and drops corners.
    """
    if height <= 0 or not full_lines:
        return []
    end = min(len(full_lines), offset + height)
    window = [list(r) for r in full_lines[offset:end]]
    if not selected_owners or not window:
        return window  # type: ignore[return-value]

    # Map owner → absolute line ranges in full_lines
    ranges: dict[int, tuple[int, int]] = {}
    for abs_i, owner in enumerate(line_owners):
        if owner not in selected_owners:
            continue
        if owner not in ranges:
            ranges[owner] = (abs_i, abs_i + 1)
        else:
            a, _ = ranges[owner]
            ranges[owner] = (a, abs_i + 1)

    for owner, (a, b) in ranges.items():
        top_clipped = a < offset
        bottom_clipped = b > end
        if not top_clipped and not bottom_clipped:
            continue
        # Visible absolute indices for this owner inside window
        for wi, abs_i in enumerate(range(offset, end)):
            if not (a <= abs_i < b):
                continue
            is_first_vis = abs_i == max(a, offset)
            is_last_vis = abs_i == min(b, end) - 1
            row = window[wi]
            text = "".join(t for _, t in row)
            # Drop corner-only rows that are half-visible conceptually — if a
            # corner row itself is the first visible and top is clipped, it
            # shouldn't appear; corners live at a/b-1.
            if abs_i == a and top_clipped:
                # Top corner was cut; this shouldn't happen if a < offset.
                continue
            if abs_i == b - 1 and bottom_clipped and (
                TOP_LEFT in text or BOTTOM_LEFT in text
            ) and VERTICAL not in text and VERTICAL_DASHED not in text:
                # bottom corner outside — skip if we're somehow including it
                pass
            if is_first_vis and top_clipped:
                # If this row is a top corner (only corners), blank it to spaces
                if TOP_LEFT in text and VERTICAL not in text:
                    window[wi] = [
                        (style, " "),
                        ("", " " * max(0, total_width - 2)),
                        (style, " "),
                        ("", "\n"),
                    ]
                else:
                    window[wi] = _replace_side_edges(row, VERTICAL_DASHED, style)
            elif is_last_vis and bottom_clipped:
                if BOTTOM_LEFT in text and VERTICAL not in text:
                    window[wi] = [
                        (style, " "),
                        ("", " " * max(0, total_width - 2)),
                        (style, " "),
                        ("", "\n"),
                    ]
                else:
                    window[wi] = _replace_side_edges(row, VERTICAL_DASHED, style)

    return window  # type: ignore[return-value]


__all__ = [
    "BOTTOM_LEFT",
    "BOTTOM_RIGHT",
    "SelectionBox",
    "TOP_LEFT",
    "TOP_RIGHT",
    "VERTICAL",
    "VERTICAL_DASHED",
    "apply_viewport_selection_clip",
    "wrap_rows_with_selection_box",
]
