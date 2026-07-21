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
Fleet (parallel agents)
  fleet.badge[.open], fleet.pane[.title|.border|.meta]
  fleet.item[.running|.selected]
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

``fresh`` (default): GrokBuild / GrokNight spirit — neutral pure-black canvas,
soft body text (neither dim nor harsh white), muted mauve only as a light
accent. No ink-blue base, no neon pink/cyan chrome.
``groknight``: denser legacy TokyoNight-on-gray look (still available via env).
"""

from __future__ import annotations

import os
from copy import deepcopy

from prompt_toolkit.styles import Style

# ---------------------------------------------------------------------------
# Shared class keys (detail transcript)
# ---------------------------------------------------------------------------

_DETAIL_GROKNIGHT: dict[str, str] = {
    "detail.header": "bg:#141414 #c8c8c8",
    "detail.meta": "bg:#141414 #6c6c6c",
    "detail.active": "bg:#141414 #e1e1e1",
    "detail.separator": "bg:#141414 #2a2a2a",
    "detail.text": "bg:#141414 #d6d6d6",
    "detail.actor": "bg:#141414 #9a9a9a",
    "detail.actor.user": "bg:#141414 #9a9a9a",
    "detail.actor.assistant": "bg:#141414 #e0b0c4",
    "detail.actor.tool": "bg:#141414 #6c6c6c",
    "detail.tool": "bg:#141414 #8a9bb3",
    "detail.block": "bg:#1a1a1a #c8c8c8",
    "detail.code": "bg:#1a1a1a #c8c8c8",
    "detail.code.rail": "bg:#1a1a1a #3a3a3a",
    "detail.code.gutter": "bg:#1a1a1a #585858",
    "detail.code.gutter.mark": "bg:#1a1a1a #6c6c6c",
    "detail.code.gutter.sep": "bg:#1a1a1a #2a2a2a",
    "detail.code.meta": "bg:#141414 #6c6c6c",
    "detail.code.kw": "bg:#1a1a1a #e0b0c4",
    "detail.code.str": "bg:#1a1a1a #8fad7a",
    "detail.code.cmt": "bg:#1a1a1a #6c6c6c italic",
    "detail.code.num": "bg:#1a1a1a #b8a078",
    "detail.code.sym": "bg:#1a1a1a #8a9bb3",
    "detail.code.plain": "bg:#1a1a1a #d6d6d6",
    "detail.diff.add": "bg:#0f1a10 #8fad7a",
    "detail.diff.remove": "bg:#1f0e12 #c97b84",
    "detail.diff.hunk": "bg:#1a1a1a #b8a078",
    "detail.diff.gutter": "bg:#1a1a1a #6c6c6c",
    "detail.success": "bg:#1a1a1a #8fad7a",
    "detail.error": "bg:#1a1a1a #c97b84",
    "detail.warning": "bg:#1a1a1a #b8a078",
    "detail.link": "bg:#141414 #8a9bb3 underline",
    "detail.link.hint": "bg:#141414 #7a7a7a",
    "detail.fold.active": "bg:#141414 #c0c0c0",
    "detail.md.ol": "bg:#141414 #9a9a9a",
    "detail.md.ul": "bg:#141414 #6c6c6c",
    "detail.md.h1": "bg:#141414 #e0e0e0 bold",
    "detail.md.h2": "bg:#141414 #d6d6d6 bold",
    "detail.md.h3": "bg:#141414 #c8c8c8 bold",
    "detail.md.quote": "bg:#141414 #6c6c6c italic",
    "detail.md.inline": "bg:#1a1a1a #8a9bb3",
    "detail.md.bold": "bg:#141414 #e0e0e0 bold",
    "detail.md.italic": "bg:#141414 #c8c8c8 italic",
    "detail.md.strike": "bg:#141414 #6c6c6c strike",
    # Thinking: dim header, readable mid-gray body (Grok header_bright=false).
    "detail.thinking.header": "bg:#141414 #7a7a7a",
    "detail.thinking.rail": "bg:#141414 #333333",
    "detail.thinking.body": "bg:#141414 #a0a0a0",
    "detail.thinking.meta": "bg:#141414 #5a5a5a",
    "detail.actor.think": "bg:#141414 #7a7a7a",
    "detail.jump.fab": "bg:#2a2a2a #d0d0d0 bold",
}

# GrokBuild / GrokNight spirit — pure black canvas, soft text, quiet accents.
# No ink-blue (#14131a) base; no neon pink/cyan chrome.
_DETAIL_FRESH: dict[str, str] = {
    "detail.header": "bg:#0c0c0c #d0d0d0",
    "detail.meta": "bg:#0c0c0c #737373",
    "detail.active": "bg:#0c0c0c #e2e2e2",
    "detail.separator": "bg:#0c0c0c #2e2e2e",
    # Body: soft off-white — readable for long prose, not pure #fff glare.
    "detail.text": "bg:#0c0c0c #d4d4d4",
    "detail.actor": "bg:#0c0c0c #8a8a8a",
    "detail.actor.user": "bg:#0c0c0c #8a8a8a",
    # Assistant stamp: brighter soft pink (readable, not neon).
    "detail.actor.assistant": "bg:#0c0c0c #e0b0c4",
    "detail.actor.tool": "bg:#0c0c0c #6e6e6e",
    "detail.tool": "bg:#0c0c0c #8a8a8a",
    "detail.block": "bg:#141414 #cfcfcf",
    "detail.code": "bg:#141414 #cfcfcf",
    "detail.code.rail": "bg:#141414 #333333",
    "detail.code.gutter": "bg:#141414 #5a5a5a",
    "detail.code.gutter.mark": "bg:#141414 #737373",
    "detail.code.gutter.sep": "bg:#141414 #242424",
    "detail.code.meta": "bg:#0c0c0c #737373",
    "detail.code.kw": "bg:#141414 #e0b0c4",
    "detail.code.str": "bg:#141414 #8fad7a",
    "detail.code.cmt": "bg:#141414 #6e6e6e italic",
    "detail.code.num": "bg:#141414 #b8a078",
    "detail.code.sym": "bg:#141414 #8a9bb3",
    "detail.code.plain": "bg:#141414 #d4d4d4",
    "detail.diff.add": "bg:#0f1810 #8fad7a",
    "detail.diff.remove": "bg:#1c1012 #c97b84",
    "detail.diff.hunk": "bg:#141414 #b8a078",
    "detail.diff.gutter": "bg:#141414 #737373",
    "detail.success": "bg:#141414 #8fad7a",
    "detail.error": "bg:#141414 #c97b84",
    "detail.warning": "bg:#141414 #b8a078",
    "detail.link": "bg:#0c0c0c #8a9bb3 underline",
    "detail.link.hint": "bg:#0c0c0c #737373",
    "detail.fold.active": "bg:#0c0c0c #c0c0c0",
    "detail.md.ol": "bg:#0c0c0c #8a8a8a",
    "detail.md.ul": "bg:#0c0c0c #737373",
    # Headings: same family as body, one step brighter — no rainbow hierarchy.
    "detail.md.h1": "bg:#0c0c0c #e2e2e2",
    "detail.md.h2": "bg:#0c0c0c #dcdcdc",
    "detail.md.h3": "bg:#0c0c0c #d0d0d0",
    "detail.md.quote": "bg:#0c0c0c #737373 italic",
    "detail.md.inline": "bg:#141414 #8a9bb3",
    "detail.md.bold": "bg:#0c0c0c #e2e2e2",
    "detail.md.italic": "bg:#0c0c0c #c8c8c8 italic",
    "detail.md.strike": "bg:#0c0c0c #737373 strike",
    # Thinking: dim header + mid-gray body (secondary to answer prose).
    "detail.thinking.header": "bg:#0c0c0c #6e6e6e",
    "detail.thinking.rail": "bg:#0c0c0c #2e2e2e",
    "detail.thinking.body": "bg:#0c0c0c #9a9a9a",
    "detail.thinking.meta": "bg:#0c0c0c #5a5a5a",
    "detail.actor.think": "bg:#0c0c0c #6e6e6e",
    "detail.jump.fab": "bg:#222222 #d0d0d0",
}


def _chrome_groknight() -> dict[str, str]:
    """Legacy denser chrome — pure black family, less neon than before."""
    return {
        "root": "bg:#141414 #d6d6d6",
        "header": "bg:#141414 #9a9a9a",
        "brand": "#e0b0c4",
        "brand.edge.pink": "#c498b0",
        "header.rule.dim": "#2a2a2a",
        "meta": "#6c6c6c",
        "separator": "#2a2a2a",
        "task.card": "bg:#141414 #d6d6d6",
        "task.card.active": "bg:#1a1a1a #d6d6d6",
        "task.card.selected": "bg:#1c1c1c #e0e0e0",
        "task.card.border": "bg:#141414 #2e2e2e",
        "task.card.border.active": "bg:#1a1a1a #6e6e6e",
        "task.card.border.selected": "bg:#1c1c1c #e0b0c4",
        "task.marker.active": "#b0b0b0",
        "task.marker.selected": "#e0b0c4",
        "task.marker.idle": "#3a3a3a",
        "task.title": "#d6d6d6",
        "task.title.active": "#d6d6d6",
        "task.title.selected": "#e0e0e0",
        "task.status": "#6c6c6c",
        "task.status.running": "#e0b0c4",
        "task.status.reporting": "#9a9a9a",
        "task.status.completed": "#6c6c6c",
        "task.status.failed": "#c97b84",
        "task.agent": "#6c6c6c",
        "task.agent.running": "#a8a8a8",
        "task.agent.selected": "#e0b0c4",
        "agent.border": "#3a3a3a",
        "report": "#a8a8a8",
        "report.active": "#b0b0b0",
        "report.selected": "#b0b0b0",
        "report.stream": "#d6d6d6",
        "report.stream.selected": "#d6d6d6",
        "input": "bg:#111111 #d6d6d6",
        "input.placeholder": "bg:#111111 #555555",
        "prompt": "bg:#111111 #b8a078",
        "prompt.border": "bg:#141414 #2e2e2e",
        "prompt.border.focus": "bg:#141414 #8a7a84",
        "prompt.caption": "bg:#141414 #6c6c6c",
        "turn.status": "bg:#141414 #e0b0c4",
        "turn.elapsed": "bg:#141414 #6c6c6c",
        "turn.stop": "bg:#141414 #c97b84",
        "task.interject": "bg:#141414 #b8a078",
        "feedback.info": "bg:#141414 #8a9bb3",
        "feedback.success": "bg:#141414 #8fad7a",
        "feedback.warning": "bg:#141414 #c97b84",
        "todo.badge": "bg:#141414 #6c6c6c",
        "todo.badge.open": "bg:#141414 #b8a078",
        "todo.pane": "bg:#141414 #c8c8c8",
        "todo.pane.title": "bg:#141414 #b8a078",
        "todo.pane.border": "bg:#141414 #2e2e2e",
        "todo.item.pending": "bg:#141414 #6c6c6c",
        "todo.item.progress": "bg:#141414 #b8a078",
        "todo.item.done": "bg:#141414 #8fad7a",
        "todo.item.cancelled": "bg:#141414 #555555",
        "fleet.badge": "bg:#141414 #6c6c6c",
        "fleet.badge.open": "bg:#141414 #a8a8a8",
        "fleet.pane": "bg:#141414 #c8c8c8",
        "fleet.pane.title": "bg:#141414 #a8a8a8",
        "fleet.pane.border": "bg:#141414 #2e2e2e",
        "fleet.pane.meta": "bg:#141414 #6c6c6c",
        "fleet.item": "bg:#141414 #6c6c6c",
        "fleet.item.running": "bg:#141414 #a8a8a8",
        "fleet.item.selected": "bg:#141414 #e0b0c4",
        "shortcut.key": "bg:#141414 #c0c0c0",
        "shortcut.label": "bg:#141414 #6c6c6c",
        "shortcut.separator": "bg:#141414 #2a2a2a",
        "shortcut.pending": "bg:#141414 #b8a078",
        "agent-window": "bg:#141414 #d6d6d6",
        "agent-window.header": "bg:#141414 #e0b0c4",
        "agent-window.close": "bg:#1a1a1a #c97b84",
        "agent-window.hint": "bg:#141414 #6c6c6c",
        "ask.dialog": "bg:#1a1a1a #d6d6d6",
        "ask.border": "bg:#1a1a1a #b8a078",
        "ask.header": "bg:#1a1a1a #b8a078 bold",
        "ask.question": "bg:#1a1a1a #d6d6d6 bold",
        "ask.meta": "bg:#1a1a1a #6c6c6c",
        "ask.option": "bg:#1a1a1a #c0c0c0",
        "ask.option.selected": "bg:#222222 #b8a078 bold",
        "ask.option.desc": "bg:#1a1a1a #6c6c6c",
        "ask.hint": "bg:#1a1a1a #6c6c6c",
        "modal.border.left": "bg:#141414 #2e2e2e",
        "modal.border.right": "bg:#141414 #2e2e2e",
        "modal.border.dim": "bg:#141414 #2a2a2a",
        "detail.input": "bg:#111111 #d6d6d6",
        "detail.input.prompt": "bg:#111111 #b8a078",
        "auth.item": "bg:#141414 #6c6c6c",
        "auth.item.selected": "bg:#141414 #c0c0c0",
        "auth.item.active": "bg:#141414 #b8a078",
        "auth.item.logged": "bg:#141414 #d6d6d6",
        "auth.item.muted": "bg:#141414 #555555",
        "auth.hint": "bg:#141414 #6c6c6c",
        "auth.note": "bg:#141414 #6c6c6c",
        "hud.title": "fg:#e0b0c4 bg:#0a0a0a",
        "hud.ok": "fg:#8fad7a bg:#0a0a0a",
        "hud.warn": "fg:#c97b84 bg:#0a0a0a",
        "hud.cyan": "fg:#8a9bb3 bg:#0a0a0a",
        "hud.dim": "fg:#555555 bg:#0a0a0a",
        "hud.bg": "bg:#0a0a0a",
        "scrollbar.background": "bg:#0a0a0a #2a2a2a",
        "scrollbar.start": "bg:#0a0a0a #2e2e2e",
        "scrollbar.button": "bg:#3a3a3a #6c6c6c",
        "scrollbar.end": "bg:#3a3a3a #6c6c6c",
        "scrollbar.arrow": "bg:#0a0a0a #3a3a3a",
    }


def _chrome_fresh() -> dict[str, str]:
    """GrokBuild black chrome — pure neutral base, soft text, quiet accents."""
    # Shared tokens (must match _DETAIL_FRESH canvas/body).
    bg = "#0c0c0c"
    bg_lift = "#161616"
    bg_active = "#121212"
    text = "#d4d4d4"
    muted = "#737373"
    dim = "#555555"
    line = "#2e2e2e"
    # Soft pink brand — a step brighter in detail/chrome for legibility.
    brand = "#e0b0c4"
    brand_dim = "#c498b0"
    # Neutral steel for "live" state — replaces loud cyan.
    accent = "#a8a8a8"
    warn = "#c97b84"
    ok = "#8fad7a"
    gold = "#b8a078"
    info = "#8a9bb3"
    return {
        "root": f"bg:{bg} {text}",
        "header": f"bg:{bg} {muted}",
        "brand": brand,
        "brand.edge.pink": brand_dim,
        "header.rule.dim": line,
        "meta": muted,
        "separator": line,
        # Pseudo-card surfaces (whole-row bg fill).
        "task.card": f"bg:{bg} {text}",
        "task.card.active": f"bg:{bg_active} {text}",
        "task.card.selected": f"bg:{bg_lift} {text}",
        "task.card.border": f"bg:{bg} {line}",
        "task.card.border.active": f"bg:{bg_active} {dim}",
        "task.card.border.selected": f"bg:{bg_lift} {brand_dim}",
        "task.marker.active": accent,
        "task.marker.selected": brand,
        "task.marker.idle": dim,
        "task.title": f"bg:{bg} {text}",
        "task.title.active": f"bg:{bg_active} {text}",
        "task.title.selected": f"bg:{bg_lift} {text}",
        "task.status": muted,
        "task.status.running": brand,
        "task.status.reporting": accent,
        "task.status.completed": muted,
        "task.status.failed": warn,
        "task.agent": muted,
        "task.agent.running": accent,
        "task.agent.selected": brand,
        "agent.border": dim,
        "report": f"bg:{bg} {muted}",
        "report.active": f"bg:{bg_active} {muted}",
        "report.selected": f"bg:{bg_lift} {muted}",
        "report.stream": f"bg:{bg_active} {text}",
        "report.stream.selected": f"bg:{bg_lift} {text}",
        "input": f"bg:{bg_lift} {text}",
        "input.placeholder": f"bg:{bg_lift} {dim}",
        "prompt": f"bg:{bg_lift} {gold}",
        "prompt.border": f"bg:{bg} {line}",
        "prompt.border.focus": f"bg:{bg} {brand_dim}",
        "prompt.caption": f"bg:{bg} {muted}",
        "turn.status": f"bg:{bg} {brand}",
        "turn.elapsed": f"bg:{bg} {muted}",
        "turn.stop": f"bg:{bg} {warn}",
        "task.interject": f"bg:{bg} {gold}",
        "feedback.info": f"bg:{bg} {info}",
        "feedback.success": f"bg:{bg} {ok}",
        "feedback.warning": f"bg:{bg} {warn}",
        "todo.badge": f"bg:{bg} {muted}",
        "todo.badge.open": f"bg:{bg} {gold}",
        "todo.pane": f"bg:{bg} {text}",
        "todo.pane.title": f"bg:{bg} {muted}",
        "todo.pane.border": f"bg:{bg} {line}",
        "todo.item.pending": f"bg:{bg} {muted}",
        "todo.item.progress": f"bg:{bg} {gold}",
        "todo.item.done": f"bg:{bg} {ok}",
        "todo.item.cancelled": f"bg:{bg} {dim}",
        "fleet.badge": f"bg:{bg} {muted}",
        "fleet.badge.open": f"bg:{bg} {accent}",
        "fleet.pane": f"bg:{bg} {text}",
        "fleet.pane.title": f"bg:{bg} {muted}",
        "fleet.pane.border": f"bg:{bg} {line}",
        "fleet.pane.meta": f"bg:{bg} {muted}",
        "fleet.item": f"bg:{bg} {muted}",
        "fleet.item.running": f"bg:{bg} {accent}",
        "fleet.item.selected": f"bg:{bg} {brand}",
        "shortcut.key": f"bg:{bg} {text}",
        "shortcut.label": f"bg:{bg} {muted}",
        "shortcut.separator": f"bg:{bg} {line}",
        "shortcut.pending": f"bg:{bg} {gold}",
        "agent-window": f"bg:{bg} {text}",
        "agent-window.header": f"bg:{bg} {brand}",
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
        "auth.item.muted": f"bg:{bg} {dim}",
        "auth.hint": f"bg:{bg} {muted}",
        "auth.note": f"bg:{bg} {muted}",
        "hud.title": f"fg:{brand} bg:#000000",
        "hud.ok": f"fg:{ok} bg:#000000",
        "hud.warn": f"fg:{warn} bg:#000000",
        "hud.cyan": f"fg:{info} bg:#000000",
        "hud.dim": f"fg:{dim} bg:#000000",
        "hud.bg": "bg:#000000",
        "scrollbar.background": f"bg:#000000 {line}",
        "scrollbar.start": f"bg:#000000 {line}",
        "scrollbar.button": f"bg:{dim} {muted}",
        "scrollbar.end": f"bg:{dim} {muted}",
        "scrollbar.arrow": f"bg:#000000 {line}",
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
