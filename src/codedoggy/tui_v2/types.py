"""Core types for pager scrollback (Grok `scrollback/types.rs`).

Maps ratatui ``Line``/``Span`` to ``(style, text)`` tuples used at the
prompt_toolkit paint edge. Style strings are ``class:grok.*`` (or raw
``fg:#rrggbb``) — never raw ANSI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Iterable, Literal, Sequence


# ---------------------------------------------------------------------------
# Wrap / display / selection
# ---------------------------------------------------------------------------


class WrapMode(Enum):
    """How to wrap content that exceeds the available width."""

    WORD = "word"
    CHARACTER = "character"
    TRUNCATE = "truncate"

    @classmethod
    def default(cls) -> WrapMode:
        return cls.WORD


class DisplayMode(Enum):
    """How a block is currently displayed."""

    COLLAPSED = "collapsed"
    TRUNCATED = "truncated"
    EXPANDED = "expanded"

    @classmethod
    def default(cls) -> DisplayMode:
        return cls.EXPANDED


@dataclass(frozen=True, slots=True)
class Selectable:
    """Which parts of a line can be selected for copying.

    Grok variants:
    - ``All`` — every span selectable (default)
    - ``Spans(Range)`` — contiguous span index range
    - ``None`` — decoration / region boundary (not selectable)
    """

    kind: Literal["all", "spans", "none"] = "all"
    start: int = 0
    end: int = 0

    ALL: ClassVar[Selectable]
    NONE: ClassVar[Selectable]

    @classmethod
    def all(cls) -> Selectable:
        return cls(kind="all")

    @classmethod
    def spans(cls, start: int, end: int) -> Selectable:
        return cls(kind="spans", start=start, end=end)

    @classmethod
    def none(cls) -> Selectable:
        return cls(kind="none")

    def clamped_span_range(self, length: int) -> tuple[int, int] | None:
        """Return ``(start, end)`` clamped to ``length``, or ``None`` if not Spans."""
        if self.kind != "spans":
            return None
        end = min(self.end, length)
        start = min(self.start, end)
        return start, end


Selectable.ALL = Selectable.all()
Selectable.NONE = Selectable.none()


# ---------------------------------------------------------------------------
# Accent
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AccentStyle:
    """Accent/bullet color style for a block.

    Used by both ``accent()`` and ``bullet()`` trait methods.
    When ``animated`` is true, the renderer uses a wave animation effect.

    ``color`` is a prompt_toolkit style fragment: either a theme class
    (``class:grok.accent_tool``) or a raw color (``fg:#787878`` / ``#787878``).
    """

    color: str
    animated: bool = False

    @classmethod
    def static_color(cls, color: str) -> AccentStyle:
        """Create a static (non-animated) accent style."""
        return cls(color=color, animated=False)

    @classmethod
    def animated_color(cls, color: str) -> AccentStyle:
        """Create an animated accent style (wave effect for running blocks).

        Named ``animated_color`` (not ``animated``) so it does not shadow
        the ``animated`` field.
        """
        return cls(color=color, animated=True)


# Alias matching Grok's ``AccentStyle::animated`` constructor name at call sites.
def accent_animated(color: str) -> AccentStyle:
    return AccentStyle.animated_color(color)


# ---------------------------------------------------------------------------
# Spans / lines / output
# ---------------------------------------------------------------------------

# ``(style, text)`` — style is a prompt_toolkit class or fg fragment.
Span = tuple[str, str]
Spans = list[Span]


@dataclass
class BlockLine:
    """A single line of block output (Grok ``BlockLine``)."""

    spans: Spans = field(default_factory=list)
    background: str | None = None
    # Decorative "panel" band (tool previews) vs semantic shading (diff/code).
    background_is_panel: bool = False
    # Column where background starts (0 = full width).
    bg_start_col: int = 0
    wrap: WrapMode = WrapMode.WORD
    selectable: Selectable = field(default_factory=Selectable.all)
    selection_range: int | None = None
    selection_text: str | None = None
    # Soft-wrap joiner: None = hard break; Some("") mid-word; Some(" ") word.
    joiner: str | None = None
    link_target: str | None = None

    @classmethod
    def text(cls, s: str) -> BlockLine:
        """Fully selectable plain text line."""
        return cls(spans=[("", s)], selectable=Selectable.ALL)

    @classmethod
    def styled(cls, spans: Sequence[Span]) -> BlockLine:
        """Styled line, fully selectable."""
        return cls(spans=list(spans), selectable=Selectable.ALL)

    @classmethod
    def separator(cls, spans: Sequence[Span] | str) -> BlockLine:
        """Decoration line (not selectable, acts as region boundary)."""
        if isinstance(spans, str):
            span_list: Spans = [("", spans)]
        else:
            span_list = list(spans)
        return cls(spans=span_list, selectable=Selectable.NONE)

    def with_background(self, color: str) -> BlockLine:
        self.background = color
        return self

    def with_panel_background(self, color: str) -> BlockLine:
        self.background = color
        self.background_is_panel = True
        return self

    def with_background_from(self, color: str, start_col: int) -> BlockLine:
        self.background = color
        self.bg_start_col = start_col
        return self

    def with_wrap(self, mode: WrapMode) -> BlockLine:
        self.wrap = mode
        return self

    def with_selection_range(self, range_id: int | None) -> BlockLine:
        self.selection_range = range_id
        return self

    def with_selection_text(self, text: str | None) -> BlockLine:
        self.selection_text = text
        return self

    def with_joiner(self, joiner: str | None) -> BlockLine:
        self.joiner = joiner
        return self

    def plain_text(self) -> str:
        return line_plain_text(self)


@dataclass
class BlockOutput:
    """Complete output produced by a block for rendering."""

    lines: list[BlockLine] = field(default_factory=list)

    @classmethod
    def plain(cls, text: str) -> BlockOutput:
        """Plain text, all lines fully selectable."""
        # str.splitlines() drops a trailing empty line after final newline;
        # match Grok's ``text.lines()`` (no trailing empty from bare trailing \n).
        if text == "":
            return cls(lines=[])
        return cls(lines=[BlockLine.text(line) for line in text.splitlines()])

    def push(self, line: BlockLine) -> None:
        self.lines.append(line)

    def __len__(self) -> int:
        return len(self.lines)

    def is_empty(self) -> bool:
        return not self.lines

    def height(self) -> int:
        return len(self.lines)

    def with_decorations(
        self,
        prefix: Span | None = None,
        suffix: Span | None = None,
    ) -> BlockOutput:
        """Wrap first line with prefix, last line with suffix.

        Decorations are NOT selectable (selection metadata shifts past them).
        """
        if prefix is not None and self.lines:
            first = self.lines[0]
            first.spans = [prefix, *first.spans]
            shift_selection_metadata_for_prefix(first, 1)

        if suffix is not None and self.lines:
            last = self.lines[-1]
            content_end = len(last.spans)
            last.spans.append(suffix)
            if last.selectable.kind == "all":
                last.selectable = Selectable.spans(0, content_end)
            # Spans / None keep their range as-is (suffix is non-selectable).

        return self


# ---------------------------------------------------------------------------
# Helpers (from types.rs)
# ---------------------------------------------------------------------------


def line_plain_text(line: BlockLine) -> str:
    """Flatten a rendered line's spans into the plain text drawn on that row."""
    return "".join(text for _, text in line.spans)


def derive_selection_text(line: BlockLine) -> str:
    """Selectable copy text for a line (override, spans slice, or trimmed All)."""
    if line.selection_text is not None:
        return line.selection_text

    sel = line.selectable
    if sel.kind == "none":
        return ""
    if sel.kind == "all":
        # Strip trailing whitespace so render-only padding never reaches clipboard.
        text = line_plain_text(line)
        return text.rstrip()
    # spans
    r = sel.clamped_span_range(len(line.spans))
    if r is None:
        return ""
    start, end = r
    return "".join(text for _, text in line.spans[start:end])


def shift_selection_metadata_for_prefix(
    line: BlockLine, prefix_span_count: int
) -> None:
    """Shift selectable span indices after prepending ``prefix_span_count`` spans."""
    if prefix_span_count == 0:
        return
    n = len(line.spans)
    sel = line.selectable
    if sel.kind == "all":
        line.selectable = Selectable.spans(prefix_span_count, n)
    elif sel.kind == "spans":
        shifted = Selectable.spans(
            sel.start + prefix_span_count, sel.end + prefix_span_count
        )
        r = shifted.clamped_span_range(n)
        if r is not None:
            line.selectable = Selectable.spans(r[0], r[1])
        else:
            line.selectable = Selectable.NONE
    # none stays none


def selectable_cols(line: BlockLine) -> tuple[int, int] | None:
    """Convert span indices to display columns for hit-testing.

    Returns ``(start_col, end_col)`` or ``None`` if not selectable.
    Uses character length as a display-width stand-in (ASCII-oriented;
    full Unicode width belongs with a dedicated width helper later).
    """
    sel = line.selectable
    if sel.kind == "none":
        return None
    if sel.kind == "all":
        width = sum(len(t) for _, t in line.spans)
        return 0, width
    r = sel.clamped_span_range(len(line.spans))
    if r is None:
        return None
    start_i, end_i = r
    start_col = sum(len(t) for _, t in line.spans[:start_i])
    end_col = sum(len(t) for _, t in line.spans[:end_i])
    return start_col, end_col


def prewrap_index_per_row(lines: Sequence[BlockLine]) -> list[int]:
    """Pre-wrap (logical source) line index for each post-wrap output row.

    A row whose ``joiner`` is ``None`` starts a new pre-wrap line; soft-wrap
    continuations (``joiner is not None``) stay on the current one.
    """
    indices: list[int] = []
    prewrap = 0
    for row, line in enumerate(lines):
        if row > 0 and line.joiner is None:
            prewrap += 1
        indices.append(prewrap)
    return indices


def spans_from_text(text: str, style: str = "") -> Spans:
    """Convenience: single-span list."""
    return [(style, text)]


def concat_spans(parts: Iterable[Span]) -> Spans:
    return list(parts)


__all__ = [
    "AccentStyle",
    "BlockLine",
    "BlockOutput",
    "DisplayMode",
    "Selectable",
    "Span",
    "Spans",
    "WrapMode",
    "accent_animated",
    "concat_spans",
    "derive_selection_text",
    "line_plain_text",
    "prewrap_index_per_row",
    "selectable_cols",
    "shift_selection_metadata_for_prefix",
    "spans_from_text",
]
