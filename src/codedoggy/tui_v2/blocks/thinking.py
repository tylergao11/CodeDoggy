"""ThinkingBlock — agent thinking/reasoning with markdown support.

Port of ``xai-grok-pager/src/scrollback/blocks/thinking.rs``.

Header strings (must match Grok exactly):
- running: ``Thinking…`` (U+2026 ellipsis)
- done + elapsed: ``Thought`` + `` for {time}`` where time is
  ``{secs:.1f}s`` under 60s, else ``{mins}m{remaining:.0f}s``
- done, no elapsed: ``Thought``

Display modes:
- **Collapsed**: header line only
- **Truncated**: optional header + ``…`` + last N body lines
- **Expanded**: optional header + full markdown body

Paint API collapses the three modes into ``collapsed`` / ``running`` /
optional ``truncated_lines``. Full BlockContent trait (accent, fold cycle)
belongs to the layout/scrollback agents.

Canonical layout types live with the layout agent; this module returns
rows of StyleAndTextTuples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.utils import get_cwidth

from codedoggy.tui_v2.blocks.markdown import render_markdown

if TYPE_CHECKING:
    from prompt_toolkit.formatted_text import StyleAndTextTuples
else:
    StyleAndTextTuples = list  # type: ignore[misc, assignment]

# Style classes — theme maps these (primary bold / muted bold / muted).
S_HEADER = "class:grok.thinking.header"  # theme.primary().bold() when bright
S_HEADER_MUTED = "class:grok.thinking.header.muted"  # theme.muted().bold()
S_DETAIL = "class:grok.thinking.detail"  # theme.muted() for " for Xs"
S_ELLIPSIS = "class:grok.thinking.ellipsis"  # theme.muted() "…"
S_BODY = "class:grok.thinking.body"  # blended markdown body hook

# Default truncated tail length when not specified (Grok config truncated_lines).
DEFAULT_TRUNCATED_LINES = 3


def format_elapsed_ms(ms: int) -> str:
    """Format elapsed milliseconds (Grok ``ThinkingBlock::format_time``).

    - ``ms/1000 < 60`` → ``"{secs:.1f}s"`` (e.g. ``1.2s``, ``0.0s``)
    - else → ``"{mins}m{remaining:.0f}s"`` (e.g. ``2m5s``)
    """
    secs = ms / 1000.0
    if secs < 60.0:
        return f"{secs:.1f}s"
    mins = int(secs // 60.0)
    remaining = secs - (mins * 60.0)
    return f"{mins}m{remaining:.0f}s"


def thinking_header_label(
    *,
    running: bool = False,
    elapsed_ms: int | None = None,
) -> str:
    """Plain-text header for tests / copy (no styles).

    Matches Grok header strings:
    - running → ``Thinking…``
    - elapsed → ``Thought for {time}``
    - else → ``Thought``
    """
    if running:
        return "Thinking\u2026"
    if elapsed_ms is not None:
        return f"Thought for {format_elapsed_ms(elapsed_ms)}"
    return "Thought"


def header_line(
    *,
    running: bool = False,
    elapsed_ms: int | None = None,
    selected: bool = False,
    muted: bool = False,
) -> StyleAndTextTuples:
    """Build the header spans (Grok ``header_line``).

    Bright label when selected or not muted; detail span always muted.
    """
    use_bright = (not muted) or selected
    label_style = S_HEADER if use_bright else S_HEADER_MUTED

    if running:
        return [(label_style, "Thinking\u2026")]
    if elapsed_ms is not None:
        return [
            (label_style, "Thought"),
            (S_DETAIL, f" for {format_elapsed_ms(elapsed_ms)}"),
        ]
    return [(label_style, "Thought")]


def _truncate_line(frags: StyleAndTextTuples, width: int) -> StyleAndTextTuples:
    """Clip styled fragments to ``width`` display columns."""
    width = max(1, int(width))
    out: StyleAndTextTuples = []
    used = 0
    for style, text in frags:
        for ch in text:
            cw = get_cwidth(ch) if ch else 0
            if used + cw > width:
                return out or [(style, "")]
            if out and out[-1][0] == style:
                out[-1] = (style, out[-1][1] + ch)
            else:
                out.append((style, ch))
            used += cw
    return out


def paint_thinking(
    text: str,
    *,
    width: int,
    collapsed: bool = False,
    running: bool = False,
    elapsed_ms: int | None = None,
    selected: bool = False,
    truncated_lines: int | None = None,
    show_header: bool = False,
    muted_collapsed: bool = False,
) -> list[StyleAndTextTuples]:
    """Paint a thinking block.

    Parameters
    ----------
    collapsed:
        True → header only (Grok ``DisplayMode::Collapsed``).
    running:
        True → header is ``Thinking…``; body still rendered when not collapsed.
    elapsed_ms:
        When finished, drives ``Thought for Xs`` (server or local freeze).
    truncated_lines:
        If set and not collapsed, show ``…`` + last N markdown lines
        (Grok ``DisplayMode::Truncated``). ``None`` = full expanded body.
    show_header:
        When expanded/truncated, prepend header + blank separator if True
        (Grok ``thinking.header`` appearance config; default off).
    muted_collapsed:
        When collapsed, use muted header style (Grok muted_collapsed).
    """
    width = max(1, int(width))
    muted = bool(collapsed and muted_collapsed and not selected)
    hdr = header_line(
        running=running,
        elapsed_ms=elapsed_ms,
        selected=selected,
        muted=muted,
    )

    if collapsed:
        return [_truncate_line(hdr, width)]

    # Empty body → same placeholder as collapsed (Grok render_empty_placeholder).
    if not (text or "").strip():
        return [_truncate_line(hdr, width)]

    body = render_markdown(text, width=width)

    if truncated_lines is not None:
        n = max(0, int(truncated_lines))
        if n == 0:
            body = [[(S_ELLIPSIS, "\u2026")]]
        elif len(body) > n:
            body = [[(S_ELLIPSIS, "\u2026")]] + body[-n:]
        # else: fits — show all (Grok truncated path when total <= n)

    rows: list[StyleAndTextTuples] = []
    if show_header:
        rows.append(_truncate_line(hdr, width))
        rows.append([("class:grok.thinking.sep", "")])  # blank separator
    rows.extend(body)
    return rows


__all__ = [
    "DEFAULT_TRUNCATED_LINES",
    "format_elapsed_ms",
    "header_line",
    "paint_thinking",
    "thinking_header_label",
]
