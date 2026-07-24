"""Doggy logo only — not from Grok. Uses shared doggy_brand portrait."""

from __future__ import annotations

import time

from prompt_toolkit.formatted_text import StyleAndTextTuples

from codedoggy.tui.doggy_brand import _render_doggy_empty, _render_doggy_idle_panel


def render_welcome(*, width: int, model_caption: str = "") -> StyleAndTextTuples:
    frags = list(_render_doggy_empty(width, now=time.monotonic()))
    if model_caption:
        frags.append(("", "\n"))
        frags.append(("class:grok.gray", f"  {model_caption}\n"))
    return frags


def render_idle(*, width: int) -> StyleAndTextTuples:
    return _render_doggy_idle_panel(width)
