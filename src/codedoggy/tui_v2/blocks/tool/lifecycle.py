"""LifecycleEventBlock — port of ``blocks/tool/lifecycle.rs``.

Standalone lifecycle hook event names (user_prompt_submit, session_start, …).
Not real tool calls — rendered as bold one-line headers.
"""

from __future__ import annotations

from codedoggy.tui_v2.blocks.tool.common import (
    S_BOLD,
    S_MUTED,
    Rows,
    is_running,
    row,
    truncate_str,
)


def paint_lifecycle(
    name: str,
    *,
    width: int,
    collapsed: bool = True,
    status: str = "completed",
    selected: bool = False,
) -> Rows:
    running = is_running(status)
    muted = collapsed and not running
    style = S_MUTED + " bold" if muted else S_BOLD
    if selected:
        style = f"{style} reverse"
    label = (name or "lifecycle").strip()
    return [row((style, truncate_str(label, max(1, width))))]


__all__ = ["paint_lifecycle"]
