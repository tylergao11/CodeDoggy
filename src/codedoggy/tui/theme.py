"""TUI visual themes — colors / weight / italic / underline.

Font face (Cascadia, Plex Mono, …) is owned by the host terminal.
CodeDoggy only controls ANSI style tokens below.

Inventory (class → role)
------------------------
Chrome
  root, header, brand, brand.edge.pink, header.rule.dim, meta, separator
Tasks
  task.marker[.active|.selected|.idle]
  task.title, task.status[.running|.reporting|.completed|.failed], task.interject
Prompt / input
  input, input.placeholder, prompt, prompt.border[.focus], prompt.caption
  turn.status, turn.elapsed, turn.stop
  feedback[.info|.success|.warning]
Todo
  todo.badge[.open], todo.pane[.title|.border]
  todo.item[.pending|.progress|.done|.cancelled]
Shortcuts
  shortcut.key, shortcut.label, shortcut.separator, shortcut.pending
Agent modal
  agent-window[.header|.close|.hint]
  modal.border[.left|.right|.dim]
  detail.input[.prompt]
Ask questionnaire
  ask.dialog, ask.border, ask.header, ask.question, ask.meta
  ask.option[.selected], ask.option.desc, ask.hint
Auth
  auth.item[.selected|.active|.logged|.muted], auth.hint, auth.note
HUD
  hud.title, hud.ok, hud.warn, hud.cyan, hud.dim, hud.bg
Scrollbar
  scrollbar.{background,start,button,end,arrow}
Detail transcript (agent_detail)
  detail.{header,meta,active,separator,text,actor*}
  detail.{tool,block,code*,diff*,success,error,warning,link*}
  detail.md.{ol,ul,h1,h2,h3,quote,inline,bold,italic,strike}
  detail.thinking.{header,rail,body,meta}

Weight / slant used historically (GrokNight)
  bold   → ask.header/question/option.selected, detail.code.kw,
           detail.md.h1/h2/h3, detail.md.bold
  italic → detail.code.cmt, detail.md.quote, detail.md.italic
  underline → detail.link
  strike → detail.md.strike

``fresh`` (default): reading-first detail (warm body, visible hairlines, air
between sections) + soft pink/cyan chrome. Not a classic dense TUI grid.
``groknight``: previous TokyoNight-on-gray look.
"""

from __future__ import annotations

import os
from copy import deepcopy

from prompt_toolkit.styles import Style

# ---------------------------------------------------------------------------
# Shared class keys (detail transcript)
# ---------------------------------------------------------------------------

_DETAIL_GROKNIGHT: dict[str, str] = {
    "detail.header": "bg:#141414 #c4789a",
    "detail.meta": "bg:#141414 #6c6c6c",
    "detail.active": "bg:#141414 #e1e1e1",
    "detail.separator": "bg:#141414 #242424",
    "detail.text": "bg:#141414 #e1e1e1",
    "detail.actor": "bg:#141414 #c8c8c8",
    "detail.actor.user": "bg:#141414 #c8c8c8",
    "detail.actor.assistant": "bg:#141414 #c4789a",
    "detail.actor.tool": "bg:#141414 #787878",
    "detail.tool": "bg:#141414 #7aa2f7",
    "detail.block": "bg:#1c1c1c #c8c8c8",
    "detail.code": "bg:#1c1c1c #c8c8c8",
    "detail.code.rail": "bg:#1c1c1c #363636",
    "detail.code.gutter": "bg:#1c1c1c #585858",
    "detail.code.gutter.mark": "bg:#1c1c1c #6c6c6c",
    "detail.code.gutter.sep": "bg:#1c1c1c #242424",
    "detail.code.meta": "bg:#141414 #6c6c6c",
    "detail.code.kw": "bg:#1c1c1c #c4789a bold",
    "detail.code.str": "bg:#1c1c1c #9ece6a",
    "detail.code.cmt": "bg:#1c1c1c #6c6c6c italic",
    "detail.code.num": "bg:#1c1c1c #ff9e64",
    "detail.code.sym": "bg:#1c1c1c #7dcfff",
    "detail.code.plain": "bg:#1c1c1c #e1e1e1",
    "detail.diff.add": "bg:#063806 #9ece6a",
    "detail.diff.remove": "bg:#420e14 #f7768e",
    "detail.diff.hunk": "bg:#1c1c1c #e0af68",
    "detail.diff.gutter": "bg:#1c1c1c #6c6c6c",
    "detail.success": "bg:#1c1c1c #9ece6a",
    "detail.error": "bg:#1c1c1c #f7768e",
    "detail.warning": "bg:#1c1c1c #e0af68",
    "detail.link": "bg:#141414 #7aa6da underline",
    "detail.link.hint": "bg:#141414 #e0af68",
    "detail.fold.active": "bg:#141414 #c8c8c8",
    "detail.md.ol": "bg:#141414 #7aa2f7",
    "detail.md.ul": "bg:#141414 #6c6c6c",
    "detail.md.h1": "bg:#141414 #1abc9c bold",
    "detail.md.h2": "bg:#141414 #7aa2f7 bold",
    "detail.md.h3": "bg:#141414 #a86888 bold",
    "detail.md.quote": "bg:#141414 #6c6c6c italic",
    "detail.md.inline": "bg:#1c1c1c #3A95AB",
    "detail.md.bold": "bg:#141414 #e1e1e1 bold",
    "detail.md.italic": "bg:#141414 #c8c8c8 italic",
    "detail.md.strike": "bg:#141414 #6c6c6c strike",
    "detail.thinking.header": "bg:#141414 #c4789a",
    "detail.thinking.rail": "bg:#1c1c1c #585858",
    "detail.thinking.body": "bg:#1c1c1c #c8c8c8",
    "detail.thinking.meta": "bg:#141414 #6c6c6c",
    "detail.actor.think": "bg:#141414 #c4789a",
}

# Reading-first detail surface (Linear / Notion / Medium dark reading spirit):
# warm paper-on-ink, one body tone, muted chrome, accent only on actors/links.
# Separators are meant to be *seen* but quiet — not neon, not invisible.
_DETAIL_FRESH: dict[str, str] = {
    "detail.header": "bg:#14131a #e8e2dc",
    "detail.meta": "bg:#14131a #8f8a86",
    "detail.active": "bg:#14131a #f0ebe6",
    "detail.separator": "bg:#14131a #5c5754",
    # Body: comfortable warm off-white for long prose (not pastel mush).
    "detail.text": "bg:#14131a #e8e2dc",
    "detail.actor": "bg:#14131a #a8a29e",
    "detail.actor.user": "bg:#14131a #a8a29e",
    "detail.actor.assistant": "bg:#14131a #d4a0b8",
    "detail.actor.tool": "bg:#14131a #8f8a86",
    "detail.tool": "bg:#14131a #8f8a86",
    "detail.block": "bg:#1c1b22 #d8d2cc",
    "detail.code": "bg:#1c1b22 #d8d2cc",
    "detail.code.rail": "bg:#1c1b22 #4a4648",
    "detail.code.gutter": "bg:#1c1b22 #6e6966",
    "detail.code.gutter.mark": "bg:#1c1b22 #8f8a86",
    "detail.code.gutter.sep": "bg:#1c1b22 #3a3638",
    "detail.code.meta": "bg:#14131a #8f8a86",
    "detail.code.kw": "bg:#1c1b22 #d4a0b8",
    "detail.code.str": "bg:#1c1b22 #9cba7a",
    "detail.code.cmt": "bg:#1c1b22 #8f8a86 italic",
    "detail.code.num": "bg:#1c1b22 #d4b896",
    "detail.code.sym": "bg:#1c1b22 #8ab4c8",
    "detail.code.plain": "bg:#1c1b22 #e8e2dc",
    "detail.diff.add": "bg:#1a2a1c #9cba7a",
    "detail.diff.remove": "bg:#2e181c #e09098",
    "detail.diff.hunk": "bg:#1c1b22 #c4a882",
    "detail.diff.gutter": "bg:#1c1b22 #8f8a86",
    "detail.success": "bg:#1c1b22 #9cba7a",
    "detail.error": "bg:#1c1b22 #e09098",
    "detail.warning": "bg:#1c1b22 #c4a882",
    "detail.link": "bg:#14131a #8ab4c8 underline",
    "detail.link.hint": "bg:#14131a #a8a29e",
    "detail.fold.active": "bg:#14131a #d8d2cc",
    "detail.md.ol": "bg:#14131a #a8a29e",
    "detail.md.ul": "bg:#14131a #8f8a86",
    # Headings: same family as body, slightly brighter — no rainbow hierarchy.
    "detail.md.h1": "bg:#14131a #f0ebe6",
    "detail.md.h2": "bg:#14131a #ebe6e0",
    "detail.md.h3": "bg:#14131a #e0dad4",
    "detail.md.quote": "bg:#14131a #8f8a86 italic",
    "detail.md.inline": "bg:#1c1b22 #8ab4c8",
    "detail.md.bold": "bg:#14131a #f0ebe6",
    "detail.md.italic": "bg:#14131a #d8d2cc italic",
    "detail.md.strike": "bg:#14131a #8f8a86 strike",
    "detail.thinking.header": "bg:#14131a #a8a29e",
    "detail.thinking.rail": "bg:#1c1b22 #4a4648",
    "detail.thinking.body": "bg:#1c1b22 #b8b2ac",
    "detail.thinking.meta": "bg:#14131a #8f8a86",
    "detail.actor.think": "bg:#14131a #a8a29e",
}


def _chrome_groknight() -> dict[str, str]:
    return {
        "root": "bg:#141414 #e1e1e1",
        "header": "bg:#141414 #c8c8c8",
        "brand": "#c4789a",
        "brand.edge.pink": "#a86888",
        "header.rule.dim": "#242424",
        "meta": "#6c6c6c",
        "separator": "#242424",
        "task.card": "bg:#141414 #e1e1e1",
        "task.card.active": "bg:#1a1a1a #e1e1e1",
        "task.card.selected": "bg:#1c1c1c #e1e1e1",
        "task.card.border": "bg:#141414 #363636",
        "task.card.border.active": "bg:#1a1a1a #7dcfff",
        "task.card.border.selected": "bg:#1c1c1c #c4789a",
        "task.marker.active": "#7dcfff",
        "task.marker.selected": "#c4789a",
        "task.marker.idle": "#414141",
        "task.title": "#e1e1e1",
        "task.title.active": "#e1e1e1",
        "task.title.selected": "#e1e1e1",
        "task.status": "#6c6c6c",
        "task.status.running": "#c4789a",
        "task.status.reporting": "#7dcfff",
        "task.status.completed": "#6c6c6c",
        "task.status.failed": "#f7768e",
        "agent.border": "#505058",
        "report": "#c8c8c8",
        "report.active": "#c8c8c8",
        "report.selected": "#c8c8c8",
        "report.stream": "#e1e1e1",
        "report.stream.selected": "#e1e1e1",
        "input": "bg:#111111 #e1e1e1",
        "input.placeholder": "bg:#111111 #585858",
        "prompt": "bg:#111111 #FFDB8D",
        "prompt.border": "bg:#141414 #323237",
        "prompt.border.focus": "bg:#141414 #d4a0b8",
        "prompt.caption": "bg:#141414 #6c6c6c",
        "turn.status": "bg:#141414 #c4789a",
        "turn.elapsed": "bg:#141414 #787878",
        "turn.stop": "bg:#141414 #f7768e",
        "task.interject": "bg:#141414 #FFDB8D",
        "feedback.info": "bg:#141414 #7dcfff",
        "feedback.success": "bg:#141414 #9ece6a",
        "feedback.warning": "bg:#141414 #f7768e",
        "todo.badge": "bg:#141414 #787878",
        "todo.badge.open": "bg:#141414 #FFDB8D",
        "todo.pane": "bg:#141414 #c8c8c8",
        "todo.pane.title": "bg:#141414 #FFDB8D",
        "todo.pane.border": "bg:#141414 #363636",
        "todo.item.pending": "bg:#141414 #6c6c6c",
        "todo.item.progress": "bg:#141414 #FFDB8D",
        "todo.item.done": "bg:#141414 #9ece6a",
        "todo.item.cancelled": "bg:#141414 #585858",
        "shortcut.key": "bg:#141414 #c8c8c8",
        "shortcut.label": "bg:#141414 #6c6c6c",
        "shortcut.separator": "bg:#141414 #242424",
        "shortcut.pending": "bg:#141414 #e0af68",
        "agent-window": "bg:#141414 #e1e1e1",
        "agent-window.header": "bg:#141414 #c4789a",
        "agent-window.close": "bg:#1c1c1c #f7768e",
        "agent-window.hint": "bg:#141414 #6c6c6c",
        "ask.dialog": "bg:#1c1c1c #e1e1e1",
        "ask.border": "bg:#1c1c1c #FFDB8D",
        "ask.header": "bg:#1c1c1c #FFDB8D bold",
        "ask.question": "bg:#1c1c1c #e1e1e1 bold",
        "ask.meta": "bg:#1c1c1c #6c6c6c",
        "ask.option": "bg:#1c1c1c #c8c8c8",
        "ask.option.selected": "bg:#242424 #FFDB8D bold",
        "ask.option.desc": "bg:#1c1c1c #6c6c6c",
        "ask.hint": "bg:#1c1c1c #6c6c6c",
        "modal.border.left": "bg:#141414 #363636",
        "modal.border.right": "bg:#141414 #363636",
        "modal.border.dim": "bg:#141414 #242424",
        "detail.input": "bg:#111111 #e1e1e1",
        "detail.input.prompt": "bg:#111111 #FFDB8D",
        "auth.item": "bg:#141414 #6c6c6c",
        "auth.item.selected": "bg:#141414 #7dcfff",
        "auth.item.active": "bg:#141414 #FFDB8D",
        "auth.item.logged": "bg:#141414 #e1e1e1",
        "auth.item.muted": "bg:#141414 #585858",
        "auth.hint": "bg:#141414 #6c6c6c",
        "auth.note": "bg:#141414 #787878",
        "hud.title": "fg:#c4789a bg:#0a0a0a",
        "hud.ok": "fg:#9ece6a bg:#0a0a0a",
        "hud.warn": "fg:#f7768e bg:#0a0a0a",
        "hud.cyan": "fg:#7dcfff bg:#0a0a0a",
        "hud.dim": "fg:#585858 bg:#0a0a0a",
        "hud.bg": "bg:#0a0a0a",
        "scrollbar.background": "bg:#0a0a0a #242424",
        "scrollbar.start": "bg:#0a0a0a #363636",
        "scrollbar.button": "bg:#505058 #6c6c6c",
        "scrollbar.end": "bg:#505058 #6c6c6c",
        "scrollbar.arrow": "bg:#0a0a0a #505058",
    }


def _chrome_fresh() -> dict[str, str]:
    """Reading-first chrome — same tokens as detail (canvas/text/muted/line/brand)."""
    # Shared reading tokens (must match _DETAIL_FRESH canvas/body).
    bg = "#14131a"
    bg_lift = "#1c1b22"
    bg_active = "#18171e"
    text = "#e8e2dc"
    muted = "#8f8a86"
    line = "#5c5754"
    brand = "#d4a0b8"
    accent = "#7eb8c9"
    warn = "#e09098"
    ok = "#9cba7a"
    gold = "#d4b896"
    return {
        "root": f"bg:{bg} {text}",
        "header": f"bg:{bg} {muted}",
        "brand": brand,
        "brand.edge.pink": brand,
        "header.rule.dim": line,
        "meta": muted,
        "separator": line,
        # Pseudo-card surfaces (whole-row bg fill).
        "task.card": f"bg:{bg} {text}",
        "task.card.active": f"bg:{bg_active} {text}",
        "task.card.selected": f"bg:{bg_lift} {text}",
        "task.card.border": f"bg:{bg} {line}",
        "task.card.border.active": f"bg:{bg_active} {accent}",
        "task.card.border.selected": f"bg:{bg_lift} {brand}",
        "task.marker.active": accent,
        "task.marker.selected": brand,
        "task.marker.idle": line,
        "task.title": f"bg:{bg} {text}",
        "task.title.active": f"bg:{bg_active} {text}",
        "task.title.selected": f"bg:{bg_lift} {text}",
        "task.status": muted,
        "task.status.running": brand,
        "task.status.reporting": accent,
        "task.status.completed": muted,
        "task.status.failed": warn,
        "agent.border": line,
        "report": f"bg:{bg} {muted}",
        "report.active": f"bg:{bg_active} {muted}",
        "report.selected": f"bg:{bg_lift} {muted}",
        "report.stream": f"bg:{bg_active} {text}",
        "report.stream.selected": f"bg:{bg_lift} {text}",
        "input": f"bg:{bg_lift} {text}",
        "input.placeholder": f"bg:{bg_lift} {muted}",
        "prompt": f"bg:{bg_lift} {gold}",
        "prompt.border": f"bg:{bg} {line}",
        "prompt.border.focus": f"bg:{bg} {brand}",
        "prompt.caption": f"bg:{bg} {muted}",
        "turn.status": f"bg:{bg} {brand}",
        "turn.elapsed": f"bg:{bg} {muted}",
        "turn.stop": f"bg:{bg} {warn}",
        "task.interject": f"bg:{bg} {gold}",
        "feedback.info": f"bg:{bg} {accent}",
        "feedback.success": f"bg:{bg} {ok}",
        "feedback.warning": f"bg:{bg} {warn}",
        "todo.badge": f"bg:{bg} {muted}",
        "todo.badge.open": f"bg:{bg} {brand}",
        "todo.pane": f"bg:{bg} {text}",
        "todo.pane.title": f"bg:{bg} {muted}",
        "todo.pane.border": f"bg:{bg} {line}",
        "todo.item.pending": f"bg:{bg} {muted}",
        "todo.item.progress": f"bg:{bg} {brand}",
        "todo.item.done": f"bg:{bg} {ok}",
        "todo.item.cancelled": f"bg:{bg} {line}",
        "shortcut.key": f"bg:{bg} {text}",
        "shortcut.label": f"bg:{bg} {muted}",
        "shortcut.separator": f"bg:{bg} {line}",
        "shortcut.pending": f"bg:{bg} {gold}",
        "agent-window": f"bg:{bg} {text}",
        "agent-window.header": f"bg:{bg} {text}",
        "agent-window.close": f"bg:{bg_lift} {warn}",
        "agent-window.hint": f"bg:{bg} {muted}",
        "ask.dialog": f"bg:{bg_lift} {text}",
        "ask.border": f"bg:{bg_lift} {gold}",
        "ask.header": f"bg:{bg_lift} {gold}",
        "ask.question": f"bg:{bg_lift} {text}",
        "ask.meta": f"bg:{bg_lift} {muted}",
        "ask.option": f"bg:{bg_lift} {text}",
        "ask.option.selected": f"bg:{bg} {gold}",
        "ask.option.desc": f"bg:{bg_lift} {muted}",
        "ask.hint": f"bg:{bg_lift} {muted}",
        "modal.border.left": f"bg:{bg} {line}",
        "modal.border.right": f"bg:{bg} {line}",
        "modal.border.dim": f"bg:{bg} {line}",
        "detail.input": f"bg:{bg_lift} {text}",
        "detail.input.prompt": f"bg:{bg_lift} {gold}",
        "auth.item": f"bg:{bg} {muted}",
        "auth.item.selected": f"bg:{bg} {accent}",
        "auth.item.active": f"bg:{bg} {gold}",
        "auth.item.logged": f"bg:{bg} {text}",
        "auth.item.muted": f"bg:{bg} {line}",
        "auth.hint": f"bg:{bg} {muted}",
        "auth.note": f"bg:{bg} {muted}",
        "hud.title": f"fg:{brand} bg:#0c0c10",
        "hud.ok": f"fg:{ok} bg:#0c0c10",
        "hud.warn": f"fg:{warn} bg:#0c0c10",
        "hud.cyan": f"fg:{accent} bg:#0c0c10",
        "hud.dim": f"fg:{muted} bg:#0c0c10",
        "hud.bg": "bg:#0c0c10",
        "scrollbar.background": f"bg:#0c0c10 {line}",
        "scrollbar.start": f"bg:#0c0c10 {line}",
        "scrollbar.button": f"bg:{line} {muted}",
        "scrollbar.end": f"bg:{line} {muted}",
        "scrollbar.arrow": f"bg:#0c0c10 {line}",
    }


THEMES: dict[str, dict[str, str]] = {
    "fresh": {**_chrome_fresh(), **_DETAIL_FRESH},
    "groknight": {**_chrome_groknight(), **_DETAIL_GROKNIGHT},
    # aliases
    "cute": {**_chrome_fresh(), **_DETAIL_FRESH},
    "quiet": {**_chrome_fresh(), **_DETAIL_FRESH},
    "dark": {**_chrome_groknight(), **_DETAIL_GROKNIGHT},
}

DEFAULT_THEME = "fresh"


def resolve_theme_name(environ: dict[str, str] | None = None) -> str:
    env = environ if environ is not None else os.environ
    raw = str(env.get("CODEDOGGY_THEME", "") or "").strip().lower()
    if not raw:
        return DEFAULT_THEME
    if raw in THEMES:
        return raw
    return DEFAULT_THEME


def style_dict(name: str | None = None) -> dict[str, str]:
    key = (name or resolve_theme_name()).strip().lower()
    base = THEMES.get(key) or THEMES[DEFAULT_THEME]
    return deepcopy(base)


def build_style(name: str | None = None) -> Style:
    return Style.from_dict(style_dict(name))


