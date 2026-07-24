"""Horizontal layout for scrollback entries (Grok ``scrollback/layout.rs``).

Column structure shared by all scrollback entries::

    │A│PL│    Content    │PR│
    │1│ 2│     flex      │ 2│   (defaults from LayoutConfig)

Where:
- A  = Accent line (1 char) — ``glyphs.accent_bar()`` for full height
- PL = Left padding (configurable, default 2)
- Content = Flexible width
- PR = Right padding (configurable, default 2)

Selection borders are drawn INTO the outer viewport padding (1 col each
side of the entry), not as part of the base layout. Scrollbar is separate.

Paint edge (``paint_block_output``) turns ``BlockOutput`` rows into
prompt_toolkit ``StyleAndTextTuples`` rows with accent rail + optional
first-row bullet prepended to content spans (Grok style — bullet does
**not** replace the accent rail).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from prompt_toolkit.formatted_text import StyleAndTextTuples

from codedoggy.tui_v2.types import AccentStyle, BlockLine, BlockOutput, Span

try:
    from codedoggy.tui_v2.glyphs import accent_bar as _glyphs_accent_bar
except Exception:  # noqa: BLE001 — glyphs may be partial during parallel port
    def _glyphs_accent_bar() -> str:
        return "\u2503"  # ┃


try:
    from codedoggy.tui_v2.theme import style_class as _style_class
except Exception:  # noqa: BLE001
    def _style_class(field: str) -> str:
        if field.startswith("class:"):
            return field
        if field.startswith("grok."):
            return f"class:{field}"
        return f"class:grok.{field}"


# Theme class names used at the paint edge (see theme.py class map).
S_ACCENT_DEFAULT = _style_class("accent_tool")
S_BULLET_DEFAULT = _style_class("gray_bright")
S_SELECTION_BORDER = _style_class("selection_border")


def _accent_bar_glyph() -> str:
    return _glyphs_accent_bar()


def _selection_vert_glyph() -> str:
    """Side border for selection box (Grok ``selection::border_chars::VERTICAL``)."""
    return "\u2502"  # │


# ---------------------------------------------------------------------------
# Rect / LayoutConfig (appearance defaults)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Rect:
    """ratatui-compatible rectangle (x, y, width, height)."""

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def new(cls, x: int, y: int, width: int, height: int) -> Rect:
        return cls(x=x, y=y, width=width, height=height)


@dataclass(slots=True)
class LayoutConfig:
    """Scrollback layout padding (Grok ``appearance::LayoutConfig``).

    Defaults from ``LayoutConfig::default`` in pager-render appearance config:
    ``block_pad_left=2``, ``block_pad_right=2``.
    """

    # Minimum value for horizontal padding (room for selection border).
    MIN_HPAD: ClassVar[int] = 1

    outer_vpad: int = 1
    outer_hpad_left: int = 2
    outer_hpad_right: int = 2
    block_pad_left: int = 2
    block_pad_right: int = 2

    def eff_outer_vpad(self, compact: bool) -> int:
        return 0 if compact else self.outer_vpad

    def eff_hpad_left(self, compact: bool) -> int:
        return self.MIN_HPAD if compact else self.outer_hpad_left

    def eff_hpad_right(self, compact: bool) -> int:
        return self.MIN_HPAD if compact else self.outer_hpad_right


# ---------------------------------------------------------------------------
# HorizontalLayout
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HorizontalLayout:
    """Horizontal layout columns for scrollback entries.

    ::

        │A│PL│    Content    │PR│
        │1│ 2│     flex      │ 2│
    """

    # Accent width is always 1 (Grok ``HorizontalLayout::ACCENT``).
    ACCENT: ClassVar[int] = 1

    accent: Rect
    left_padding: Rect
    content: Rect
    right_padding: Rect

    @classmethod
    def new(cls, area: Rect, config: LayoutConfig | None = None) -> HorizontalLayout:
        """Create layout for the given area with config values."""
        cfg = config if config is not None else LayoutConfig()
        accent_w = cls.ACCENT
        pl = cfg.block_pad_left
        pr = cfg.block_pad_right
        # Content takes remaining space (Constraint::Min(1)).
        content_w = max(1, area.width - accent_w - pl - pr)

        x = area.x
        y = area.y
        h = area.height

        accent = Rect(x, y, accent_w, h)
        x += accent_w
        left_padding = Rect(x, y, pl, h)
        x += pl
        content = Rect(x, y, content_w, h)
        x += content_w
        right_padding = Rect(x, y, pr, h)

        return cls(
            accent=accent,
            left_padding=left_padding,
            content=content,
            right_padding=right_padding,
        )

    @classmethod
    def new_default(cls, area: Rect) -> HorizontalLayout:
        """Create layout with default config (backwards compatibility)."""
        return cls.new(area, LayoutConfig())

    @classmethod
    def chrome_width(cls, config: LayoutConfig | None = None) -> int:
        """Total chrome width for a given config (accent + pads)."""
        cfg = config if config is not None else LayoutConfig()
        return cls.ACCENT + cfg.block_pad_left + cfg.block_pad_right

    @classmethod
    def content_width_for(
        cls,
        total_width: int,
        *,
        pad_left: int = 2,
        pad_right: int = 2,
    ) -> int:
        """Content width given total entry width and pad sizes.

        ``total_width`` is the entry area width (``│A│PL│Content│PR│``), not
        including outer viewport padding or selection edges.
        """
        chrome = cls.ACCENT + pad_left + pad_right
        return max(1, total_width - chrome)

    def entry_content_area(self) -> Rect:
        """Area for rendering entry content (accent through right padding).

        Layout: ``│A│PL│Content│PR│``
        """
        return Rect(
            x=self.accent.x,
            y=self.accent.y,
            width=(
                self.accent.width
                + self.left_padding.width
                + self.content.width
                + self.right_padding.width
            ),
            height=self.accent.height,
        )

    def accent_area(self) -> Rect:
        return self.accent

    def content_width(self) -> int:
        """Content column width (for BlockContext)."""
        return self.content.width

    def entry_area(self) -> Rect:
        """Full entry area (same as ``entry_content_area``)."""
        return self.entry_content_area()

    def selection_area(self) -> Rect:
        """Selection area (extends 1 column into outer padding on both sides).

        The selection border is drawn INTO the padding areas:
        - Left edge: 1 column before accent (in ``outer_hpad_left``)
        - Right edge: 1 column after right_padding (gap before scrollbar)

        Returns the area where selection borders should be drawn.
        """
        x = max(0, self.accent.x - 1)
        width = self.entry_content_area().width + 2  # +1 left, +1 right
        return Rect(
            x=x,
            y=self.accent.y,
            width=width,
            height=self.accent.height,
        )

    def for_row(self, y: int, height: int) -> HorizontalLayout:
        """Create a row-specific layout (same columns, different y/height)."""
        return HorizontalLayout(
            accent=Rect(self.accent.x, y, self.accent.width, height),
            left_padding=Rect(
                self.left_padding.x, y, self.left_padding.width, height
            ),
            content=Rect(self.content.x, y, self.content.width, height),
            right_padding=Rect(
                self.right_padding.x, y, self.right_padding.width, height
            ),
        )


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------


def resolve_style(color: str | None, *, fallback: str) -> str:
    """Normalize an AccentStyle.color / hex / class name to a PT style string."""
    if not color:
        return fallback
    if color.startswith("class:") or color.startswith("fg:") or color.startswith("bg:"):
        return color
    if color.startswith("#"):
        return f"fg:{color}"
    return _style_class(color)


def accent_style_pt(accent: AccentStyle | None) -> str:
    """prompt_toolkit style for the accent rail."""
    if accent is None:
        return S_ACCENT_DEFAULT
    return resolve_style(accent.color, fallback=S_ACCENT_DEFAULT)


def bullet_style_pt(accent: AccentStyle | None) -> str:
    """prompt_toolkit style for the first-row bullet (shares accent color)."""
    if accent is None:
        return S_BULLET_DEFAULT
    return resolve_style(accent.color, fallback=S_BULLET_DEFAULT)


# ---------------------------------------------------------------------------
# Paint edge
# ---------------------------------------------------------------------------


def paint_block_output(
    output: BlockOutput,
    *,
    total_width: int,
    accent: AccentStyle | None = None,
    bullet: str | None = None,
    selected: bool = False,
    pad_left: int = 2,
    pad_right: int = 2,
    hide_accent: bool = False,
    top_clipped: bool = False,
    bottom_clipped: bool = False,
) -> list[StyleAndTextTuples]:
    """Turn ``BlockOutput`` rows into prompt_toolkit fragment rows.

    Each painted row (Grok entry chrome)::

        [optional sel edge][accent bar][pad][bullet?][content][pad][optional sel edge]\\n

    - Accent column uses ``glyphs.accent_bar()`` for **full height** when an
      accent is provided (empty space when ``accent is None`` / ``hide_accent``).
    - Bullet/diamond is prepended only on the **first content row** to content
      spans (Grok ``prepend_bullet``) — it does **not** replace the accent rail.
    - Styles use ``class:grok.*`` from theme (via ``AccentStyle.color``).
    - When ``selected``, full Grok selection box: side ``│`` on content rows
      plus corner rows ``┌┐`` / ``└┘`` (``selection.rs``). Clipped edges use
      ``┆`` when ``top_clipped`` / ``bottom_clipped``.
    """
    if total_width <= 0:
        return []

    sel_w = 1 if selected else 0
    accent_w = 0 if hide_accent else HorizontalLayout.ACCENT
    chrome = sel_w + accent_w + pad_left + pad_right + sel_w
    content_w = max(0, total_width - chrome)

    accent_pt = accent_style_pt(accent)
    bullet_pt = bullet_style_pt(accent)
    sel_pt = S_SELECTION_BORDER
    pad_spaces_left = " " * pad_left
    pad_spaces_right = " " * pad_right
    # Accent rail: full height glyph when style present; space clears stale cells
    # when accent is None but the column is kept (Grok entry_renderer).
    if hide_accent:
        bar = ""
    elif accent is not None:
        bar = _accent_bar_glyph()
    else:
        bar = " "
        accent_pt = ""

    bullet_prefix: str | None = None
    if bullet:
        # Grok prepends ``"{char} "`` (char + trailing space).
        bullet_prefix = bullet if bullet.endswith(" ") else f"{bullet} "

    rows: list[StyleAndTextTuples] = []
    for i, line in enumerate(output.lines):
        row: StyleAndTextTuples = []

        # Optional selection left edge (solid; wrap may upgrade to dashed)
        if selected:
            row.append((sel_pt, _selection_vert_glyph()))

        # Accent rail (full height)
        if not hide_accent:
            row.append((accent_pt, bar if bar else " "))

        # Left pad
        if pad_left:
            row.append(("", pad_spaces_left))

        # Content spans (+ bullet only on first row)
        content_spans = list(line.spans)
        if i == 0 and bullet_prefix:
            content_spans = [(bullet_pt, bullet_prefix), *content_spans]

        # Optional line background (semantic shading / panel band).
        bg_style = resolve_style(line.background, fallback="") if line.background else ""

        used = 0
        for style, text in content_spans:
            if content_w and used >= content_w:
                break
            frag = text
            if content_w:
                room = content_w - used
                if len(frag) > room:
                    frag = frag[:room]
            if not frag:
                continue
            row.append((_merge_styles(style, bg_style), frag))
            used += len(frag)

        # Pad content to full content width (Grok paints every column).
        if content_w and used < content_w:
            row.append((bg_style or "", " " * (content_w - used)))

        # Right pad
        if pad_right:
            row.append(("", pad_spaces_right))

        # Optional selection right edge
        if selected:
            row.append((sel_pt, _selection_vert_glyph()))

        row.append(("", "\n"))
        rows.append(row)

    if selected and rows:
        try:
            from codedoggy.tui_v2.selection import wrap_rows_with_selection_box

            rows = wrap_rows_with_selection_box(
                rows,
                total_width=total_width,
                top_clipped=top_clipped,
                bottom_clipped=bottom_clipped,
                style=sel_pt,
            )
        except Exception:  # noqa: BLE001
            pass
    return rows


def paint_block_line(
    line: BlockLine,
    *,
    content_width: int,
    is_first: bool = False,
    bullet: str | None = None,
    bullet_style: str | None = None,
) -> StyleAndTextTuples:
    """Paint a single content line (no chrome) into fragments.

    Bullet is prepended only when ``is_first`` and ``bullet`` is set.
    """
    bstyle = bullet_style if bullet_style is not None else S_BULLET_DEFAULT
    spans: list[Span] = list(line.spans)
    if is_first and bullet:
        prefix = bullet if bullet.endswith(" ") else f"{bullet} "
        spans = [(bstyle, prefix), *spans]

    row: StyleAndTextTuples = []
    used = 0
    for style, text in spans:
        if content_width and used >= content_width:
            break
        frag = text
        if content_width:
            room = content_width - used
            if len(frag) > room:
                frag = frag[:room]
        if frag:
            row.append((style, frag))
            used += len(frag)
    if content_width and used < content_width:
        row.append(("", " " * (content_width - used)))
    return row


def _merge_styles(base: str, extra: str) -> str:
    if not extra:
        return base
    if not base:
        return extra
    # prompt_toolkit accepts space-separated style fragments.
    return f"{base} {extra}"


__all__ = [
    "HorizontalLayout",
    "LayoutConfig",
    "Rect",
    "accent_style_pt",
    "bullet_style_pt",
    "paint_block_line",
    "paint_block_output",
    "resolve_style",
]
