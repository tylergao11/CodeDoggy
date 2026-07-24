"""Prompt input chrome — port of Grok ``views/prompt_widget`` (PromptStyle + borders).

Visual (chrome + borders, matching Grok agent-view layout)::

                                            ← top vpad (configurable)
  ╭────────────────────────────────────╮    ← top border (show_borders)
  │ ❯ type here, text wraps            │    ← side borders + prefix + TextArea
  │   continuation of long input...    │
  ╰── grok-3 · yolo ───────────────────╯    ← bottom info divider

Factory returns a prompt_toolkit ``HSplit`` border box wrapping a ``TextArea``.
Does **not** implement Grok OAuth — Doggy login is Ctrl+L elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout import (
    DynamicContainer,
    FormattedTextControl,
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.layout.containers import Container
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import AfterInput, ConditionalProcessor
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import TextArea

# ---------------------------------------------------------------------------
# Glyphs
# ---------------------------------------------------------------------------

try:
    from codedoggy.tui_v2.glyphs import (
        PROMPT_ARROW_WIDTH as _GLYPH_ARROW_WIDTH,
        collapsed_accent as _collapsed_accent,
        prompt_arrow as _glyph_prompt_arrow,
    )
except Exception:  # noqa: BLE001
    _GLYPH_ARROW_WIDTH = 2
    _collapsed_accent = None  # type: ignore[assignment]
    _glyph_prompt_arrow = None  # type: ignore[assignment]


def _prompt_arrow() -> str:
    """``❯ `` normally (Grok ``prompt_arrow``). Always 2 columns."""
    if _glyph_prompt_arrow is not None:
        return _glyph_prompt_arrow()
    return "\u276f "  # ❯␠


def _accent_bar() -> str:
    if _collapsed_accent is not None:
        return _collapsed_accent()
    return "\u2759"  # ❙


# Box-drawing chars used by Grok border pass (U+256D/E/F, U+2570, U+2500, U+2502).
_TL = "\u256d"  # ╭
_TR = "\u256e"  # ╮
_BL = "\u2570"  # ╰
_BR = "\u256f"  # ╯
_H = "\u2500"  # ─
_V = "\u2502"  # │

# Style classes — keys match theme.py ``theme_style_dict`` (class:grok.*).
STYLE_BORDER = "class:grok.prompt_border"
STYLE_BORDER_ACTIVE = "class:grok.prompt_border_active"
STYLE_PREFIX = "class:grok.accent_user"
STYLE_PREFIX_DIM = "class:grok.gray_dim"
STYLE_INPUT = "class:grok.text_primary"
STYLE_PLACEHOLDER = "class:grok.gray"
STYLE_CAPTION = "class:grok.text_secondary"
STYLE_ACCENT = "class:grok.accent_user"
STYLE_PAD = "class:grok.bg_base"
STYLE_WARNING = "class:grok.warning"

PROMPT_ARROW_WIDTH = int(_GLYPH_ARROW_WIDTH)  # Grok PROMPT_ARROW_WIDTH / PREFIX_WIDTH
DEFAULT_PLACEHOLDER = "Build anything"


# ---------------------------------------------------------------------------
# PromptStyle (Grok PromptStyle — all fields ported)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PromptStyle:
    """Visual configuration for prompt rendering (Grok ``PromptStyle``)."""

    #: Whether the prompt is focused (affects prefix color, text dimming).
    focused: bool = True
    #: Whether to show the ❯ prefix character.
    show_prefix: bool = True
    #: Vertical padding above the text content (hosts the top border row).
    vpad_top: int = 1
    #: Whether to render chrome (accent line + hpad + background fill).
    chrome: bool = True
    #: Layout config for chrome mode (padding widths inside the border box).
    chrome_pad_left: int = 2
    chrome_pad_right: int = 1
    #: Override background style class (or ``None`` for theme default).
    bg_override: str | None = None
    #: Override accent line color/class (plan mode golden, etc.).
    accent_color_override: str | None = None
    #: Override border style class (plan mode). When set, used for both
    #: focused and unfocused instead of ``prompt_border_active`` / ``prompt_border``.
    border_color_override: str | None = None
    #: Override the prefix character and its style class.
    #: e.g. bash mode ``("! ", "class:grok.command")``.
    prefix_override: tuple[str, str] | None = None
    #: Override the placeholder text when the textarea is empty.
    placeholder_override: str | None = None
    #: Compact mode (reserved; unused for info_block sizing in Grok today).
    compact: bool = False
    #: Show the accent line (┃ / ❙) on the left edge of the chrome.
    show_accent_line: bool = False
    #: Draw the prompt's border box (top ╭─╮, side │, bottom ╰─╯).
    #: Only consulted when ``chrome`` is true. Defaults to ``True``.
    show_borders: bool = True
    #: Session title inlined in the top border (right-aligned, 2-cell inset).
    title: str | None = None
    #: Paint image-chip overlay (default true; host wires preview separately).
    image_preview: bool = True

    # -- factories (Grok) ---------------------------------------------------

    @classmethod
    def overlay(cls) -> PromptStyle:
        """Style for overlays (no chrome, no vpad)."""
        return cls(chrome=False, vpad_top=0)

    @classmethod
    def inline(cls, bg: str | None = None) -> PromptStyle:
        """Style for inline prompts in overlay panels (permission / question)."""
        return cls(
            focused=True,
            show_prefix=False,
            vpad_top=0,
            chrome=False,
            chrome_pad_left=0,
            chrome_pad_right=0,
            bg_override=bg,
            show_accent_line=False,
            show_borders=False,
        )

    def info_block(self, has_info: bool) -> int:
        """Info block height: rows reserved for the bottom divider line."""
        return 1 if has_info else 0

    def border_style_class(self) -> str:
        """Active/idle border class from focus (or override)."""
        if self.border_color_override is not None:
            return self.border_color_override
        return STYLE_BORDER_ACTIVE if self.focused else STYLE_BORDER

    def accent_style_class(self) -> str:
        """Mode-tinted accent: override, else focus-dependent default."""
        if self.accent_color_override is not None:
            return self.accent_color_override
        return STYLE_ACCENT if self.focused else STYLE_PREFIX_DIM

    def prefix_parts(self) -> tuple[str, str]:
        """Return ``(prefix_str, style_class)`` for the first text row."""
        if self.prefix_override is not None:
            return self.prefix_override
        style = self.accent_style_class() if self.focused else STYLE_PREFIX_DIM
        return _prompt_arrow(), style


@dataclass(slots=True)
class PromptFlag:
    """A flag displayed in the prompt info line (e.g. ``plan``, ``always-approve``)."""

    text: str
    color: str | None = None  # style class override
    bold: bool = False


@dataclass(slots=True)
class PromptInfo:
    """Optional info line rendered on the bottom border (Grok ``PromptInfo``)."""

    model_name: str = ""
    flags: Sequence[PromptFlag] = field(default_factory=tuple)
    multiline: bool = False
    usage_warning: str | None = None
    usage_warning_critical: bool = False


# ---------------------------------------------------------------------------
# Border / caption fragment builders (Grok draw path)
# ---------------------------------------------------------------------------


def _term_width() -> int:
    try:
        import shutil

        return shutil.get_terminal_size(fallback=(80, 24)).columns
    except Exception:  # noqa: BLE001
        return 80


def _truncate_to_width(text: str, max_w: int) -> str:
    if max_w <= 0:
        return ""
    if get_cwidth(text) <= max_w:
        return text
    out: list[str] = []
    w = 0
    for ch in text:
        cw = get_cwidth(ch)
        if w + cw > max_w - 1:
            break
        out.append(ch)
        w += cw
    return "".join(out) + "…"


def render_top_border(
    style: PromptStyle,
    *,
    width: int | None = None,
) -> StyleAndTextTuples:
    """Top divider: ``╭──────────╮`` with optional right-aligned title.

    Grok draws this on the ``vpad_top`` row when ``chrome && show_borders``.
    """
    width = width if width is not None else _term_width()
    if width < 2 or not style.chrome or not style.show_borders:
        # Still reserve the vpad row as blank when borders are off.
        return [(STYLE_PAD, " " * max(0, width))]

    border = style.border_style_class()
    if width == 1:
        return [(border, _TL)]

    # Corners + horizontal rule.
    inner = width - 2

    # Session title inlined in the divider (` title `, right-aligned ending
    # 2 cells before ╮). Corners plus 2-cell insets stay plain border.
    title = (style.title or "").strip()
    if title:
        max_w = width - 6  # corners + 2-cell insets each side
        if max_w >= 6:
            label = f" {title} "
            trunc = _truncate_to_width(label, max_w)
            label_w = get_cwidth(trunc)
            right_rule = 2  # cells of ─ immediately before ╮
            left_rule = max(0, inner - label_w - right_rule)
            return [
                (border, _TL),
                (border, _H * left_rule),
                (STYLE_CAPTION, trunc),
                (border, _H * right_rule),
                (border, _TR),
            ]

    return [
        (border, _TL),
        (border, _H * inner),
        (border, _TR),
    ]


def render_bottom_border(
    style: PromptStyle,
    info: PromptInfo | None = None,
    *,
    width: int | None = None,
) -> StyleAndTextTuples:
    """Bottom divider: ``╰──────────model · flags──╯``.

    Grok paints the full rule then overlays the right-aligned info caption.
    """
    width = width if width is not None else _term_width()
    if width < 2 or not style.chrome or not style.show_borders:
        return [(STYLE_PAD, " " * max(0, width))]

    border = style.border_style_class()
    if width == 1:
        return [(border, _BL)]

    inner = width - 2
    rule = _H * inner

    # Build info caption (right-aligned with 1-cell pads inside corners).
    caption_frags: StyleAndTextTuples = []
    if info is not None:
        parts: list[tuple[str, str]] = []
        if info.usage_warning:
            warn_style = STYLE_WARNING if info.usage_warning_critical else STYLE_CAPTION
            parts.append((warn_style, info.usage_warning))
        if info.model_name:
            parts.append((STYLE_CAPTION, info.model_name))
        for flag in info.flags:
            flag_style = flag.color or STYLE_CAPTION
            parts.append((flag_style, flag.text))
        if parts:
            # Join with " · "
            joined: StyleAndTextTuples = [(STYLE_PAD, " ")]
            for i, (st, text) in enumerate(parts):
                if i:
                    joined.append((STYLE_CAPTION, " · "))
                joined.append((st, text))
            if info.multiline:
                joined.append((STYLE_CAPTION, "  multiline"))
            joined.append((STYLE_PAD, " "))
            caption_frags = joined

    if not caption_frags:
        return [(border, _BL), (border, rule), (border, _BR)]

    cap_text = "".join(t for _, t in caption_frags)
    cap_w = get_cwidth(cap_text)
    if cap_w >= inner:
        # Truncate caption to fit.
        trunc = _truncate_to_width(cap_text, inner)
        return [(border, _BL), (STYLE_CAPTION, trunc), (border, _BR)]

    left_rule = max(0, inner - cap_w)
    return [
        (border, _BL),
        (border, _H * left_rule),
        *caption_frags,
        (border, _BR),
    ]


def render_side_border_cell(style: PromptStyle) -> StyleAndTextTuples:
    """Single-column vertical border ``│`` for text rows."""
    if not style.chrome or not style.show_borders:
        return [(STYLE_PAD, "")]
    return [(style.border_style_class(), _V)]


def render_accent_cell(style: PromptStyle) -> StyleAndTextTuples:
    """Single-column accent bar when ``show_accent_line``."""
    if not style.chrome or not style.show_accent_line:
        return [(STYLE_PAD, "")]
    return [(style.accent_style_class(), _accent_bar())]


# ---------------------------------------------------------------------------
# Factory: TextArea + border HSplit
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PromptChrome:
    """Assembled prompt widget + handles for the host app."""

    container: Container
    text_area: TextArea
    style: PromptStyle
    #: Mutable info line state (host updates fields; border re-reads).
    info: PromptInfo
    #: Call to refresh focus-dependent border classes after focus changes.
    set_focused: Callable[[bool], None]
    #: Call to update the top-border title.
    set_title: Callable[[str | None], None]


def _make_line_prefix(
    style_holder: list[PromptStyle],
) -> Callable[[int, int], StyleAndTextTuples]:
    """``get_line_prefix`` for every logical / wrapped TextArea row.

    Grok paints the prefix only on the first text row; continuation rows
    are padded to the same width so wrapped lines stay aligned.
    """

    def prefix(line_number: int, wrap_count: int) -> StyleAndTextTuples:
        style = style_holder[0]
        if not style.show_prefix:
            return []
        first = line_number == 0 and wrap_count == 0
        # chrome_pad_left columns sit outside the TextArea (VSplit pads);
        # inside the TextArea we only paint the ❯ (or spaces for wrap).
        if first:
            text, st = style.prefix_parts()
            return [(st, text)]
        # Continuation: blank prefix of same display width.
        return [(STYLE_PAD, " " * PROMPT_ARROW_WIDTH)]

    return prefix


def create_prompt(
    *,
    style: PromptStyle | None = None,
    info: PromptInfo | None = None,
    placeholder: str | None = None,
    accept_handler: Callable[[Any], bool] | None = None,
    history: Any = None,
    multiline: bool = True,
    height: Any = None,
    width_provider: Callable[[], int] | None = None,
    focus_on_click: bool = True,
    text: str = "",
) -> PromptChrome:
    """Factory: prompt_toolkit ``TextArea`` + Grok border ``HSplit``.

    Parameters
    ----------
    style:
        Grok :class:`PromptStyle` fields. Defaults match Grok
        ``PromptStyle::default()`` (focused, chrome, borders, pad 2/1, vpad 1).
    info:
        Bottom-border caption (model · flags). Empty ``PromptInfo`` still
        draws the bottom rule when ``show_borders`` is true.
    width_provider:
        Optional callable returning current terminal content width. When
        omitted, uses ``shutil.get_terminal_size``.
    """
    style = style if style is not None else PromptStyle()
    info = info if info is not None else PromptInfo()
    style_holder: list[PromptStyle] = [style]
    info_holder: list[PromptInfo] = [info]
    get_width = width_provider or _term_width

    ph = placeholder or style.placeholder_override or DEFAULT_PLACEHOLDER

    if height is None:
        height = Dimension(min=1, max=12, preferred=1)

    text_area = TextArea(
        text=text,
        height=height,
        multiline=multiline,
        wrap_lines=True,
        scrollbar=False,
        dont_extend_height=True,
        focus_on_click=focus_on_click,
        get_line_prefix=_make_line_prefix(style_holder)
        if style.show_prefix
        else None,
        style=STYLE_INPUT,
        accept_handler=accept_handler,
        history=history,
        input_processors=[
            ConditionalProcessor(
                AfterInput(ph, style=STYLE_PLACEHOLDER),
                Condition(lambda: not text_area.text),
            )
        ],
    )

    def top_fragments() -> StyleAndTextTuples:
        return render_top_border(style_holder[0], width=get_width())

    def bottom_fragments() -> StyleAndTextTuples:
        # Grok always draws the bottom rule when show_borders; info is optional.
        has_info = bool(
            info_holder[0].model_name
            or info_holder[0].flags
            or info_holder[0].usage_warning
            or info_holder[0].multiline
        )
        # Even without caption content, draw the ╰─╯ rule (info_block=1 when
        # borders on — host typically always shows the bottom row).
        return render_bottom_border(
            style_holder[0],
            info_holder[0] if has_info else info_holder[0],
            width=get_width(),
        )

    def left_border_frags() -> StyleAndTextTuples:
        return render_side_border_cell(style_holder[0])

    def right_border_frags() -> StyleAndTextTuples:
        return render_side_border_cell(style_holder[0])

    def accent_frags() -> StyleAndTextTuples:
        return render_accent_cell(style_holder[0])

    def left_pad_frags() -> StyleAndTextTuples:
        # Grok: content_x = area.x + accent_w + chrome_pad_left; left │ at
        # area.x overwrites the first column. Visual gap between │ and
        # content is chrome_pad_left - 1 when borders are on.
        pad = left_pad_width()
        return [(STYLE_PAD, " " * pad)] if pad else []

    def right_pad_frags() -> StyleAndTextTuples:
        # chrome_pad_right sits before the right │; border overwrites last col.
        # Default chrome_pad_right=1 → border owns that column (0-width pad).
        pad = right_pad_width()
        return [(STYLE_PAD, " " * pad)] if pad else []

    def left_pad_width() -> int:
        st = style_holder[0]
        if not st.chrome:
            return 0
        if st.show_borders:
            return max(0, st.chrome_pad_left - 1)
        return st.chrome_pad_left

    def right_pad_width() -> int:
        st = style_holder[0]
        if not st.chrome:
            return 0
        if st.show_borders:
            return max(0, st.chrome_pad_right - 1)
        return st.chrome_pad_right

    top_win = Window(
        FormattedTextControl(top_fragments),
        height=1,
        dont_extend_height=True,
    )
    bottom_win = Window(
        FormattedTextControl(bottom_fragments),
        height=1,
        dont_extend_height=True,
    )
    accent_win = Window(
        FormattedTextControl(accent_frags),
        width=1,
        dont_extend_width=True,
    )
    left_border_win = Window(
        FormattedTextControl(left_border_frags),
        width=1,
        dont_extend_width=True,
    )
    right_border_win = Window(
        FormattedTextControl(right_border_frags),
        width=1,
        dont_extend_width=True,
    )
    left_pad_win = Window(
        FormattedTextControl(left_pad_frags),
        width=left_pad_width,
        dont_extend_width=True,
    )
    right_pad_win = Window(
        FormattedTextControl(right_pad_frags),
        width=right_pad_width,
        dont_extend_width=True,
    )

    def body_row() -> Container:
        st = style_holder[0]
        children: list[Any] = []
        if st.chrome and st.show_accent_line:
            children.append(accent_win)
        if st.chrome and st.show_borders:
            children.append(left_border_win)
        if st.chrome and left_pad_width() > 0:
            children.append(left_pad_win)
        children.append(text_area)
        if st.chrome and right_pad_width() > 0:
            children.append(right_pad_win)
        if st.chrome and st.show_borders:
            children.append(right_border_win)
        return VSplit(children, padding=0)

    def full_container() -> Container:
        st = style_holder[0]
        rows: list[Any] = []
        # Top vpad / border row
        if st.vpad_top > 0:
            rows.append(top_win)
            # Extra blank vpad rows beyond the border row.
            for _ in range(max(0, st.vpad_top - 1)):
                rows.append(Window(height=1, char=" "))
        rows.append(DynamicContainer(body_row))
        # Bottom info / border row — Grok draws when chrome && show_borders.
        if st.chrome and st.show_borders:
            rows.append(bottom_win)
        return HSplit(rows, style="class:grok.root")

    # When chrome is off, just the TextArea (overlay / inline).
    def root() -> Container:
        st = style_holder[0]
        if not st.chrome:
            return text_area
        return full_container()

    container: Container = DynamicContainer(root)

    def set_focused(focused: bool) -> None:
        style_holder[0] = replace(style_holder[0], focused=focused)

    def set_title(title: str | None) -> None:
        style_holder[0] = replace(style_holder[0], title=title)

    return PromptChrome(
        container=container,
        text_area=text_area,
        style=style_holder[0],
        info=info_holder[0],
        set_focused=set_focused,
        set_title=set_title,
    )


def desired_height(
    text: str,
    area_width: int,
    style: PromptStyle,
    *,
    has_info: bool = True,
    max_height: int = 12,
    prefix_width: int = PROMPT_ARROW_WIDTH,
) -> int:
    """How tall the prompt wants to be (Grok ``desired_height``).

    ``height = vpad_top + textarea_rows + info_block``.
    """
    if style.chrome:
        accent_w = 1 if style.show_accent_line else 0
        content_w = max(
            1,
            area_width - accent_w - style.chrome_pad_left - style.chrome_pad_right,
        )
    else:
        content_w = max(1, area_width)
    pw = prefix_width if style.show_prefix else 0
    text_w = max(1, content_w - pw)

    # Soft-wrap estimate (matches Grok textarea.desired_height spirit).
    rows = 0
    parts = (text or "").split("\n") or [""]
    for part in parts:
        w = get_cwidth(part) if part else 0
        if w <= 0:
            rows += 1
        else:
            rows += max(1, (w + text_w - 1) // text_w)
    text_height = max(1, rows)

    vpad = style.vpad_top
    info_block = 1 if (style.chrome and style.show_borders) else style.info_block(has_info)
    total = vpad + text_height + info_block
    minimum = vpad + 1 + info_block
    return max(minimum, min(max_height, total))


__all__ = [
    "DEFAULT_PLACEHOLDER",
    "PROMPT_ARROW_WIDTH",
    "PromptChrome",
    "PromptFlag",
    "PromptInfo",
    "PromptStyle",
    "create_prompt",
    "desired_height",
    "render_bottom_border",
    "render_side_border_cell",
    "render_top_border",
]
