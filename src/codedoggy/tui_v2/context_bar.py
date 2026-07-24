"""Context usage bar helpers — port of Grok ``views/context_bar.rs``.

Provides compact token formatting and usage-color breakpoints used by the
status bar. Paint edge maps to ``class:grok.*`` style classes (see PORT.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# ---------------------------------------------------------------------------
# Formatting utilities (Grok context_bar.rs)
# ---------------------------------------------------------------------------

# Width of the percentage field on hover (``fmt_pct5`` always returns 5 chars).
PCT_WIDTH = 5
# Gap between the progress bar and the percentage on hover.
BAR_PCT_GAP = 1

# Separator between status-bar items (Grok ``SEPARATOR``).
SEPARATOR = "│"


def fmt_pct5(pct: float) -> str:
    """Format a percentage as a fixed-width 5-char string.

    - ``< 10``:  ``"X.XX%"`` (e.g. ``"0.00%"``, ``"5.12%"``)
    - ``10–99``: ``"XX.X%"`` (e.g. ``"20.1%"``, ``"99.9%"``)
    - ``≥ 100``: ``"MAX %"``
    """
    if pct >= 100.0:
        return "MAX %"
    if pct < 10.0:
        return f"{pct:.2f}%"
    return f"{pct:.1f}%"


def fmt_tokens(n: int) -> str:
    """Format a token count as a compact string (≤4 chars).

    - ``0–999``:    raw number
    - ``1K–9.9K``:  ``"1.2K"``
    - ``10K–999K``: ``"12K"`` / ``"999K"``
    - ``1M–9.9M``:  ``"1.2M"``
    - ``10M+``:     ``"12M"``
    """
    n = int(n)
    if n < 1_000:
        return str(n)
    if n < 10_000:
        return f"{n / 1_000.0:.1f}K"
    if n < 1_000_000:
        return f"{n // 1_000}K"
    if n < 10_000_000:
        return f"{n / 1_000_000.0:.1f}M"
    return f"{n // 1_000_000}M"


# ---------------------------------------------------------------------------
# Color breakpoints (Grok default_breakpoints)
# ---------------------------------------------------------------------------

# GrokNight RGB defaults (mirrored from theme.groknight when import fails).
# text_primary → accent_user → warning → accent_error
try:
    from codedoggy.tui_v2.theme import groknight as _groknight

    _t0 = _groknight()
    _DEFAULT_TEXT_PRIMARY = _t0.text_primary
    _DEFAULT_ACCENT_USER = _t0.accent_user
    _DEFAULT_WARNING = _t0.warning
    _DEFAULT_ACCENT_ERROR = _t0.accent_error
except Exception:  # noqa: BLE001
    _DEFAULT_TEXT_PRIMARY = "#e1e1e1"
    _DEFAULT_ACCENT_USER = "#c8c8c8"
    _DEFAULT_WARNING = "#e0af68"
    _DEFAULT_ACCENT_ERROR = "#f7768e"


@dataclass(frozen=True, slots=True)
class ColorBreakpoint:
    """A breakpoint for color blending: at ``pct`` percent, bar color is ``color``."""

    pct: float
    color: str  # hex ``#rrggbb``


def default_breakpoints(
    *,
    text_primary: str = _DEFAULT_TEXT_PRIMARY,
    accent_user: str = _DEFAULT_ACCENT_USER,
    warning: str = _DEFAULT_WARNING,
    accent_error: str = _DEFAULT_ACCENT_ERROR,
) -> list[ColorBreakpoint]:
    """Default breakpoints: text_primary → accent_user → warning → accent_error.

    Matches Grok ``default_breakpoints(theme)`` percentage stops.
    """
    return [
        ColorBreakpoint(0.0, text_primary),
        ColorBreakpoint(50.0, accent_user),
        ColorBreakpoint(65.0, accent_user),
        ColorBreakpoint(75.0, warning),
        ColorBreakpoint(85.0, warning),
        ColorBreakpoint(95.0, accent_error),
    ]


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    if len(c) != 6:
        return (198, 198, 198)
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def lerp_color(a: str, b: str, t: float) -> str:
    """Linear interpolation between two hex colors."""
    t = max(0.0, min(1.0, float(t)))
    ar, ag, ab = _hex_to_rgb(a)
    br, bg, bb = _hex_to_rgb(b)
    r = int(round(ar + (br - ar) * t))
    g = int(round(ag + (bg - ag) * t))
    b_ch = int(round(ab + (bb - ab) * t))
    return _rgb_to_hex(r, g, b_ch)


def blend_color(pct: float, breakpoints: Sequence[ColorBreakpoint]) -> str:
    """Blend between breakpoints for a given percentage. Returns hex color."""
    if not breakpoints:
        return _DEFAULT_TEXT_PRIMARY
    if pct <= breakpoints[0].pct:
        return breakpoints[0].color
    for i in range(1, len(breakpoints)):
        if pct <= breakpoints[i].pct:
            span = breakpoints[i].pct - breakpoints[i - 1].pct
            t = 0.0 if span == 0 else (pct - breakpoints[i - 1].pct) / span
            return lerp_color(breakpoints[i - 1].color, breakpoints[i].color, t)
    return breakpoints[-1].color


def usage_style(pct: float, breakpoints: Sequence[ColorBreakpoint] | None = None) -> str:
    """Return a prompt_toolkit style class for the given usage percentage.

    Discrete classes track Grok breakpoint bands so theme.py can map them
    without needing per-frame hex injection:

    - ``< 50``  → ``class:grok.context.low``
    - ``50–75`` → ``class:grok.context.mid``
    - ``75–95`` → ``class:grok.context.high``
    - ``≥ 95``  → ``class:grok.context.critical``

    When a caller needs the exact blended hex (hover bar fill), use
    :func:`usage_color` instead.
    """
    del breakpoints  # class bands are fixed; hex blend is separate
    # Map Grok breakpoint bands onto theme.py semantic classes.
    if pct >= 95.0:
        return "class:grok.accent_error"
    if pct >= 75.0:
        return "class:grok.warning"
    if pct >= 50.0:
        return "class:grok.accent_user"
    return "class:grok.text_primary"


def usage_color(pct: float, breakpoints: Sequence[ColorBreakpoint] | None = None) -> str:
    """Blended hex color for ``pct`` using Grok default breakpoints."""
    bps = list(breakpoints) if breakpoints is not None else default_breakpoints()
    return blend_color(pct, bps)


def usage_percentage(used: int, total: int) -> float:
    """Token usage as a percentage of the context window."""
    if total <= 0:
        return 0.0
    return (float(used) / float(total)) * 100.0


def format_token_pair(used: int, total: int) -> str:
    """Default non-hovered form: ``8.5K / 1.0M`` (Grok context bar)."""
    return f"{fmt_tokens(used)} / {fmt_tokens(total)}"


def context_bar_fragments(
    used_tokens: int | None,
    total_tokens: int | None,
    *,
    hovered: bool = False,
) -> list[tuple[str, str]] | None:
    """Build styled fragments for the context usage chip.

    Normal: ``8.5K / 1.0M`` colored by usage percentage.
    Hovered: percentage only (progress-bar glyphs deferred to a later paint).

    Returns ``None`` if token data is unavailable.
    """
    if used_tokens is None or total_tokens is None or total_tokens <= 0:
        return None
    pct = usage_percentage(used_tokens, total_tokens)
    style = usage_style(pct)
    if hovered:
        return [(style, fmt_pct5(pct))]
    token_str = format_token_pair(used_tokens, total_tokens)
    natural = len(token_str)
    min_width = BAR_PCT_GAP + PCT_WIDTH
    if natural < min_width:
        token_str = token_str + (" " * (min_width - natural))
    return [(style, token_str)]


__all__ = [
    "BAR_PCT_GAP",
    "PCT_WIDTH",
    "SEPARATOR",
    "ColorBreakpoint",
    "blend_color",
    "context_bar_fragments",
    "default_breakpoints",
    "fmt_pct5",
    "fmt_tokens",
    "format_token_pair",
    "lerp_color",
    "usage_color",
    "usage_percentage",
    "usage_style",
]
