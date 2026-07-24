"""SystemBlock — port spirit of ``scrollback/blocks/system.rs``.

Dim one-line / multi-line system notices (session events, notices).
Word-wraps to width; optional ``max_lines`` with trailing `` …``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codedoggy.tui_v2.blocks.tool.common import wrap_text

if TYPE_CHECKING:
    from prompt_toolkit.formatted_text import StyleAndTextTuples
else:
    StyleAndTextTuples = list  # type: ignore[misc, assignment]

S_BODY = "class:grok.muted"
S_ERROR = "class:grok.accent_error"

_ELLIPSIS = " \u2026"  # " …" (space + U+2026), Grok system.rs


def paint_system(
    text: str,
    *,
    width: int,
    failed: bool = False,
    max_lines: int | None = None,
) -> list[StyleAndTextTuples]:
    """Paint system notice content rows (no chrome).

    Word-wraps each logical line to ``width``. When ``max_lines`` is set and
    wrapped output exceeds it, keep ``max_lines - 1`` lines (or 1 when
    ``max_lines == 1``) and append `` …`` to the last kept line.
    """
    style = S_ERROR if failed else S_BODY
    body = (text or "").rstrip("\n")
    if not body:
        return [[("", "")]]
    w = max(8, int(width))

    wrapped: list[str] = []
    for logical in body.splitlines() or [""]:
        pieces = wrap_text(logical, w)
        wrapped.extend(pieces if pieces else [""])

    if max_lines is not None and max_lines > 0 and len(wrapped) > max_lines:
        take = max_lines - 1 if max_lines > 1 else 1
        kept = list(wrapped[:take])
        # Append trailing ellipsis on the last kept line (Grok system.rs).
        if kept:
            last = kept[-1]
            # Prefer fitting " …" within width when possible.
            room = w - len(_ELLIPSIS)
            if room <= 0:
                kept[-1] = _ELLIPSIS.strip()[:w]
            elif len(last) > room:
                kept[-1] = last[:room] + _ELLIPSIS
            else:
                kept[-1] = last + _ELLIPSIS
        wrapped = kept

    return [[(style, line)] for line in wrapped]


__all__ = ["paint_system"]
