"""Status bar — port of Grok ``views/status_bar.rs``.

Displays left / center / right context info on a single row. Layout matches
Grok: center only when it fits without colliding with left; right is
right-aligned within the content width.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.utils import get_cwidth

Fragment = tuple[str, str]
WidthProvider = Callable[[], int]


def _display_width(text: str) -> int:
    return get_cwidth(text)


def _pad(n: int) -> str:
    return " " * max(0, n)


@dataclass(slots=True)
class StatusBar:
    """Status bar showing context information.

    Grok fields:
    - ``left``: left-aligned content (e.g. path / context label)
    - ``center``: optional center content (e.g. turn indicator)
    - ``right``: optional right-aligned content (e.g. view mode)
    """

    left: str = ""
    center: str | None = None
    right: str | None = None


def render(
    left: str,
    center: str | None = None,
    right: str | None = None,
    *,
    width: int | None = None,
    style: str = "class:grok.gray",
) -> StyleAndTextTuples:
    """Render a single status-bar row as styled fragments.

    Mirrors Grok ``StatusBar::render``:
    - fill the row with background style
    - paint ``left`` from the left edge
    - paint ``center`` only when it fits with a 2-col gap past left
    - paint ``right`` flush right
    - skip entirely when ``width < 10`` (Grok early-return)
    """
    if width is None:
        try:
            import shutil

            width = shutil.get_terminal_size(fallback=(80, 24)).columns
        except Exception:  # noqa: BLE001
            width = 80

    if width < 10:
        return [(style, "")]

    left = left or ""
    center = center or None
    right = right or None

    left_w = _display_width(left)
    right_w = _display_width(right) if right else 0
    center_w = _display_width(center) if center else 0

    # Build a width-wide line with left / optional center / optional right.
    # Grok uses independent set_span paints; we compose one fragment string
    # per region so prompt_toolkit can reflow cleanly.
    fragments: StyleAndTextTuples = []

    # Left
    fragments.append((style, left))
    cursor = left_w

    # Center: only if it sits strictly after left with a 2-col gap (Grok).
    if center and center_w > 0:
        center_x = (width - center_w) // 2
        if center_x > left_w + 2:
            gap = center_x - cursor
            if gap > 0:
                fragments.append((style, _pad(gap)))
                cursor += gap
            fragments.append((style, center))
            cursor += center_w

    # Right
    if right and right_w > 0:
        right_x = max(cursor, width - right_w)
        # Avoid overwriting left/center content.
        if right_x >= cursor:
            gap = right_x - cursor
            if gap > 0:
                fragments.append((style, _pad(gap)))
                cursor += gap
            fragments.append((style, right))
            cursor += right_w
        elif cursor + right_w <= width:
            # Tight fit: append with single space if possible.
            if cursor < width:
                fragments.append((style, " "))
                cursor += 1
            fragments.append((style, right))
            cursor += right_w

    # Trailing pad to full width (background fill).
    if cursor < width:
        fragments.append((style, _pad(width - cursor)))

    return fragments


def render_bar(bar: StatusBar, *, width: int | None = None) -> StyleAndTextTuples:
    """Render a :class:`StatusBar` instance."""
    return render(bar.left, bar.center, bar.right, width=width)


__all__ = [
    "StatusBar",
    "render",
    "render_bar",
]
