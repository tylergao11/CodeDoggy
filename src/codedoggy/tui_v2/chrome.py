"""Wrap content-only rows in Grok layout chrome (accent rail + bullet).

All block painters that emit content StyleAndTextTuples must go through
:func:`with_chrome` so accent/bullet never drift.
"""

from __future__ import annotations

from typing import Sequence

from prompt_toolkit.formatted_text import StyleAndTextTuples

from codedoggy.tui_v2.glyphs import diamond_filled, diamond_hollow, diamond_dotted
from codedoggy.tui_v2.layout import paint_block_output
from codedoggy.tui_v2.types import AccentStyle, BlockLine, BlockOutput


def _norm_style(st: str) -> str:
    if not st:
        return "class:grok.text_secondary"
    if st.startswith("class:") or st.startswith("fg:") or st.startswith("bg:"):
        return st
    if st.startswith("grok."):
        return f"class:{st}"
    return f"class:grok.{st}" if not st.startswith("#") else f"fg:{st}"


def rows_to_block_output(rows: Sequence[StyleAndTextTuples]) -> BlockOutput:
    lines: list[BlockLine] = []
    for row in rows:
        spans: list[tuple[str, str]] = []
        for item in row:
            if not item:
                continue
            st = item[0] if len(item) >= 1 else ""
            tx = item[1] if len(item) >= 2 else ""
            if tx.endswith("\n"):
                tx = tx[:-1]
            if tx == "" and not st:
                continue
            spans.append((_norm_style(str(st)), str(tx)))
        if not spans:
            spans = [("", "")]
        lines.append(BlockLine.styled(spans))
    return BlockOutput(lines=lines)


def with_chrome(
    content_rows: Sequence[StyleAndTextTuples],
    *,
    width: int,
    accent: str | AccentStyle | None,
    bullet: str | None = None,
    selected: bool = False,
    animated: bool = False,
) -> list[StyleAndTextTuples]:
    """Apply Grok horizontal chrome to content-only rows."""
    if not content_rows:
        content_rows = [[("", "")]]
    output = rows_to_block_output(content_rows)
    if isinstance(accent, AccentStyle):
        acc = accent
    elif accent is None:
        acc = None
    elif animated:
        acc = AccentStyle.animated_color(accent)
    else:
        acc = AccentStyle.static_color(accent)
    return paint_block_output(
        output,
        total_width=max(12, width),
        accent=acc,
        bullet=bullet,
        selected=selected,
    )


def accent_for_kind(
    kind: str, *, running: bool = False, failed: bool = False
) -> str | None:
    # Grok: UserPromptBlock / AgentMessageBlock / thinking have accent=None.
    if kind in {"user", "assistant", "thinking"}:
        return None
    if failed:
        return "accent_error"
    if running:
        return "accent_running"
    return {
        "tool": "accent_tool",
        "subagent": "accent_skill",
        "system": "accent_system",
        "error": "accent_error",
        "verb_group": "accent_tool",
    }.get(kind, "accent_tool")


def bullet_for_kind(kind: str, *, running: bool = False) -> str | None:
    if kind == "user":
        return None  # prefix is in content
    if kind == "assistant":
        # Idle: no bullet; running: hollow diamond only (accent still None).
        return diamond_hollow() if running else None
    if kind == "thinking":
        # Grok keeps filled ◆ for thinking (animates color; never hollow).
        return diamond_filled()
    if kind == "verb_group":
        return diamond_dotted()
    if running:
        return diamond_hollow()
    if kind in {"tool", "subagent", "system", "error"}:
        return diamond_filled() if kind != "tool" else diamond_hollow()
    return None
