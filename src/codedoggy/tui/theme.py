"""TUI visual themes — colors / weight / italic / underline.

Font face (Cascadia, Plex Mono, …) is owned by the host terminal.
CodeDoggy only controls ANSI style tokens below.

Inventory (class → role)
------------------------
Chrome
  root, header, brand, header.rule.dim, meta, separator
Tasks
  task.marker[.active|.selected|.idle]
  task.title, task.status[.running|.reporting|.completed|.failed], task.interject
  task.plan.approve — card 批准 CTA
  plan.status.{draft,review,actions,approved,abandoned} — detail plan strip
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
  detail.thinking.{rail,body}

``fresh`` (default): neutral high-contrast canvas, bright readable prose,
quiet surfaces, and one restrained rose accent.  Hierarchy comes from
contrast and spacing rather than blue-gray text or decorative borders.
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
    "detail.tab": "bg:#141414 #6c6c6c",
    "detail.tab.active": "bg:#222222 #e1e1e1 bold",
    "detail.text": "bg:#141414 #d6d6d6",
    "detail.actor": "bg:#141414 #9a9a9a",
    "detail.actor.user": "bg:#141414 #9a9a9a",
    "detail.actor.assistant": "bg:#141414 #e0b0c4",
    "detail.actor.tool": "bg:#141414 #6c6c6c",
    "detail.tool": "bg:#141414 #8a9bb3",
    "detail.tool.link": "bg:#141414 #aab9c9 underline",
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
    "detail.error.link": "bg:#1a1a1a #d9919a underline",
    "detail.warning": "bg:#1a1a1a #b8a078",
    "detail.link": "bg:#141414 #8a9bb3 underline",
    "detail.link.hint": "bg:#141414 #7a7a7a",
    "detail.fold.collapsed": "bg:#141414 #9aa8b8",
    "detail.fold.collapsed.link": "bg:#141414 #afbdcc underline",
    "detail.fold.expanded": "bg:#141414 #e0b0c4 bold",
    "detail.fold.expanded.link": "bg:#141414 #efc3d5 bold underline",
    "detail.fold.footer": "bg:#141414 #9aa8b8",
    "detail.tool.section": "bg:#141414 #b8c2ce bold",
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
    "detail.thinking.rail": "bg:#141414 #333333",
    "detail.thinking.body": "bg:#141414 #a0a0a0",
    "detail.actor.think": "bg:#141414 #7a7a7a",
    "detail.jump.fab": "bg:#2a2a2a #d0d0d0 bold",
}

# ``fresh`` has one palette shared by chrome, transcript and plan rendering.
# Keeping the values here prevents each surface from inventing a slightly
# different dark blue/gray hierarchy.
_FRESH_CANVAS = "#0f0f0f"
_FRESH_SURFACE = "#171717"
_FRESH_SURFACE_ACTIVE = "#1d1d1d"
_FRESH_SURFACE_SELECTED = "#242424"
_FRESH_CODE = "#1b1b1b"
_FRESH_TEXT = "#f2f2f2"
_FRESH_BODY = "#dedede"
_FRESH_SECONDARY = "#b8b8b8"
_FRESH_MUTED = "#8b8b8b"
_FRESH_DIM = "#626262"
_FRESH_LINE = "#303030"
_FRESH_ACCENT = "#e7b4c8"
_FRESH_LINK = "#edb7cc"
_FRESH_OK = "#8fcf9d"
_FRESH_GOLD = "#e1c077"
_FRESH_ERROR = "#ea929c"
_FRESH_INFO = "#b8b8b8"


_DETAIL_FRESH: dict[str, str] = {
    "detail.header": f"bg:{_FRESH_SURFACE} {_FRESH_TEXT}",
    "detail.meta": f"bg:{_FRESH_SURFACE} {_FRESH_MUTED}",
    "detail.active": f"bg:{_FRESH_SURFACE} {_FRESH_TEXT} bold",
    "detail.separator": f"bg:{_FRESH_SURFACE} {_FRESH_LINE}",
    "detail.tab": f"bg:{_FRESH_SURFACE} {_FRESH_MUTED}",
    "detail.tab.active": f"bg:{_FRESH_SURFACE_SELECTED} {_FRESH_TEXT} bold",
    "detail.text": f"bg:{_FRESH_SURFACE} {_FRESH_BODY}",
    "detail.actor": f"bg:{_FRESH_SURFACE} {_FRESH_SECONDARY}",
    "detail.actor.user": f"bg:{_FRESH_SURFACE} {_FRESH_SECONDARY}",
    "detail.actor.assistant": f"bg:{_FRESH_SURFACE} {_FRESH_ACCENT}",
    "detail.actor.tool": f"bg:{_FRESH_SURFACE} {_FRESH_MUTED}",
    "detail.tool": f"bg:{_FRESH_SURFACE} {_FRESH_SECONDARY}",
    "detail.tool.link": f"bg:{_FRESH_SURFACE} {_FRESH_LINK} underline",
    "detail.block": f"bg:{_FRESH_CODE} {_FRESH_BODY}",
    "detail.code": f"bg:{_FRESH_CODE} {_FRESH_BODY}",
    "detail.code.rail": f"bg:{_FRESH_CODE} {_FRESH_LINE}",
    "detail.code.gutter": f"bg:{_FRESH_CODE} {_FRESH_MUTED}",
    "detail.code.gutter.mark": f"bg:{_FRESH_CODE} {_FRESH_SECONDARY}",
    "detail.code.gutter.sep": f"bg:{_FRESH_CODE} {_FRESH_LINE}",
    "detail.code.meta": f"bg:{_FRESH_SURFACE} {_FRESH_MUTED}",
    "detail.code.kw": f"bg:{_FRESH_CODE} {_FRESH_ACCENT}",
    "detail.code.str": f"bg:{_FRESH_CODE} {_FRESH_OK}",
    "detail.code.cmt": f"bg:{_FRESH_CODE} {_FRESH_MUTED} italic",
    "detail.code.num": f"bg:{_FRESH_CODE} {_FRESH_GOLD}",
    "detail.code.sym": f"bg:{_FRESH_CODE} {_FRESH_SECONDARY}",
    "detail.code.plain": f"bg:{_FRESH_CODE} {_FRESH_BODY}",
    "detail.diff.add": f"bg:#132018 {_FRESH_OK}",
    "detail.diff.remove": f"bg:#241417 {_FRESH_ERROR}",
    "detail.diff.hunk": f"bg:{_FRESH_CODE} {_FRESH_GOLD}",
    "detail.diff.gutter": f"bg:{_FRESH_CODE} {_FRESH_MUTED}",
    "detail.success": f"bg:{_FRESH_CODE} {_FRESH_OK}",
    "detail.error": f"bg:{_FRESH_CODE} {_FRESH_ERROR}",
    "detail.error.link": f"bg:{_FRESH_CODE} {_FRESH_ERROR} underline",
    "detail.warning": f"bg:{_FRESH_CODE} {_FRESH_GOLD}",
    "detail.link": f"bg:{_FRESH_SURFACE} {_FRESH_LINK} underline",
    "detail.link.hint": f"bg:{_FRESH_SURFACE} {_FRESH_MUTED}",
    "detail.fold.collapsed": f"bg:{_FRESH_SURFACE} {_FRESH_SECONDARY}",
    "detail.fold.collapsed.link": (
        f"bg:{_FRESH_SURFACE} {_FRESH_LINK} underline"
    ),
    "detail.fold.expanded": f"bg:{_FRESH_SURFACE} {_FRESH_TEXT} bold",
    "detail.fold.expanded.link": (
        f"bg:{_FRESH_SURFACE} {_FRESH_LINK} bold underline"
    ),
    "detail.fold.footer": f"bg:{_FRESH_SURFACE} {_FRESH_MUTED}",
    "detail.tool.section": f"bg:{_FRESH_SURFACE} {_FRESH_SECONDARY} bold",
    "detail.md.ol": f"bg:{_FRESH_SURFACE} {_FRESH_SECONDARY}",
    "detail.md.ul": f"bg:{_FRESH_SURFACE} {_FRESH_MUTED}",
    "detail.md.h1": f"bg:{_FRESH_SURFACE} {_FRESH_TEXT} bold",
    "detail.md.h2": f"bg:{_FRESH_SURFACE} {_FRESH_TEXT} bold",
    "detail.md.h3": f"bg:{_FRESH_SURFACE} {_FRESH_BODY} bold",
    "detail.md.quote": f"bg:{_FRESH_SURFACE} {_FRESH_SECONDARY} italic",
    "detail.md.inline": f"bg:{_FRESH_CODE} {_FRESH_BODY}",
    "detail.md.bold": f"bg:{_FRESH_SURFACE} {_FRESH_TEXT} bold",
    "detail.md.italic": f"bg:{_FRESH_SURFACE} {_FRESH_BODY} italic",
    "detail.md.strike": f"bg:{_FRESH_SURFACE} {_FRESH_MUTED} strike",
    "detail.thinking.rail": f"bg:{_FRESH_SURFACE} {_FRESH_LINE}",
    "detail.thinking.body": f"bg:{_FRESH_SURFACE} {_FRESH_SECONDARY}",
    "detail.actor.think": f"bg:{_FRESH_SURFACE} {_FRESH_MUTED}",
    "detail.jump.fab": f"bg:{_FRESH_SURFACE_SELECTED} {_FRESH_TEXT} bold",
}


def _chrome_groknight() -> dict[str, str]:
    """Legacy denser chrome — pure black family, less neon than before."""
    return {
        "root": "bg:#141414 #d6d6d6",
        "header": "bg:#141414 #9a9a9a",
        "brand": "#e0b0c4",
        "header.rule.dim": "#2a2a2a",
        "meta": "#6c6c6c",
        "separator": "#2a2a2a",
        "task.card": "bg:#141414 #d6d6d6",
        "task.card.active": "bg:#1a1a1a #d6d6d6",
        "task.card.selected": "bg:#1c1c1c #e0e0e0",
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
        "task.plan.approve": "bg:#2a2228 #e0b0c4 bold",
        "plan.surface": "bg:#101010 #d8d8d8",
        "plan.chrome": "bg:#151515 #d8d8d8",
        "plan.body": "bg:#101010 #d8d8d8",
        "plan.meta": "bg:#151515 #939393",
        "plan.empty": "bg:#151515 #a0a0a0 italic",
        "plan.status.draft": "bg:#151515 #cfb681 bold",
        "plan.status.review": "bg:#151515 #efbfd2 bold",
        "plan.status.actions": "bg:#151515 #c0c0c0",
        "plan.status.approved": "bg:#151515 #9fc48e bold",
        "plan.status.abandoned": "bg:#151515 #939393",
        "plan.action.approve": "bg:#1b2a1e #a9ce97 bold",
        "plan.action.revise": "bg:#2b251a #d7bd88 bold",
        "plan.action.abandon": "bg:#2d1c20 #df9aa3 bold",
        "plan.marker": "bg:#101010 #9eabc0 bold",
        "plan.heading.h1": "bg:#101010 #f0d4e0 bold",
        "plan.heading.h2": "bg:#101010 #e1e1e1 bold",
        "plan.heading.h3": "bg:#101010 #cfcfcf bold",
        "plan.quote.marker": "bg:#101010 #777777",
        "plan.quote": "bg:#101010 #aaaaaa italic",
        "plan.rule": "bg:#101010 #3d3d3d",
        "plan.code": "bg:#191919 #d6d6d6",
        "plan.code.fence": "bg:#191919 #888888",
        "plan.code.inline": "bg:#1d1d1d #cbd6e3",
        "plan.link": "bg:#101010 #a7bfda underline",
        "plan.strong": "bg:#101010 #eeeeee bold",
        "plan.checkbox.pending": "bg:#101010 #d0b47d",
        "plan.checkbox.done": "bg:#101010 #9bc18a",
        "task.agent": "#6c6c6c",
        "task.agent.running": "#a8a8a8",
        "task.agent.selected": "#e0b0c4",
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
        "todo.item.pending": "bg:#141414 #6c6c6c",
        "todo.item.progress": "bg:#141414 #b8a078",
        "todo.item.done": "bg:#141414 #8fad7a",
        "todo.item.cancelled": "bg:#141414 #555555",
        "fleet.badge": "bg:#141414 #6c6c6c",
        "fleet.badge.open": "bg:#141414 #a8a8a8",
        "fleet.pane": "bg:#141414 #c8c8c8",
        "fleet.pane.title": "bg:#141414 #a8a8a8",
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
        "detail.input": "bg:#111111 #d6d6d6",
        "detail.input.prompt": "bg:#111111 #b8a078",
        "detail.input.placeholder": "bg:#111111 #6c6c6c",
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
    """Neutral high-contrast chrome backed by the shared fresh palette."""
    bg = _FRESH_CANVAS
    surface = _FRESH_SURFACE
    active = _FRESH_SURFACE_ACTIVE
    selected = _FRESH_SURFACE_SELECTED
    text = _FRESH_TEXT
    body = _FRESH_BODY
    secondary = _FRESH_SECONDARY
    muted = _FRESH_MUTED
    dim = _FRESH_DIM
    line = _FRESH_LINE
    brand = _FRESH_ACCENT
    link = _FRESH_LINK
    warn = _FRESH_ERROR
    ok = _FRESH_OK
    gold = _FRESH_GOLD
    info = _FRESH_INFO
    return {
        "root": f"bg:{bg} {body}",
        "header": f"bg:{bg} {secondary}",
        "brand": f"{brand} bold",
        "header.rule.dim": line,
        "meta": muted,
        "separator": line,
        "task.card": f"bg:{bg} {body}",
        "task.card.active": f"bg:{active} {body}",
        "task.card.selected": f"bg:{selected} {text}",
        "task.marker.active": secondary,
        "task.marker.selected": brand,
        "task.marker.idle": dim,
        "task.title": f"bg:{bg} {body}",
        "task.title.active": f"bg:{active} {text}",
        "task.title.selected": f"bg:{selected} {text} bold",
        "task.status": muted,
        "task.status.running": brand,
        "task.status.reporting": secondary,
        "task.status.completed": muted,
        "task.status.failed": warn,
        "task.plan.approve": f"bg:{selected} {ok} bold",
        "plan.surface": f"bg:{surface} {body}",
        "plan.chrome": f"bg:{surface} {body}",
        "plan.body": f"bg:{surface} {body}",
        "plan.meta": f"bg:{surface} {muted}",
        "plan.empty": f"bg:{surface} {secondary} italic",
        "plan.status.draft": f"bg:{surface} {gold} bold",
        "plan.status.review": f"bg:{surface} {brand} bold",
        "plan.status.actions": f"bg:{surface} {secondary}",
        "plan.status.approved": f"bg:{surface} {ok} bold",
        "plan.status.abandoned": f"bg:{surface} {muted}",
        "plan.action.approve": f"bg:#1b2a1e {ok} bold",
        "plan.action.revise": f"bg:#2b251a {gold} bold",
        "plan.action.abandon": f"bg:#2d1c20 {warn} bold",
        "plan.marker": f"bg:{surface} {brand} bold",
        "plan.heading.h1": f"bg:{surface} {text} bold",
        "plan.heading.h2": f"bg:{surface} {text} bold",
        "plan.heading.h3": f"bg:{surface} {body} bold",
        "plan.quote.marker": f"bg:{surface} {dim}",
        "plan.quote": f"bg:{surface} {secondary} italic",
        "plan.rule": f"bg:{surface} {line}",
        "plan.code": f"bg:{_FRESH_CODE} {body}",
        "plan.code.fence": f"bg:{_FRESH_CODE} {muted}",
        "plan.code.inline": f"bg:{_FRESH_CODE} {body}",
        "plan.link": f"bg:{surface} {link} underline",
        "plan.strong": f"bg:{surface} {text} bold",
        "plan.checkbox.pending": f"bg:{surface} {gold}",
        "plan.checkbox.done": f"bg:{surface} {ok}",
        "task.agent": muted,
        "task.agent.running": secondary,
        "task.agent.selected": brand,
        "report": f"bg:{bg} {secondary}",
        "report.active": f"bg:{active} {secondary}",
        "report.selected": f"bg:{selected} {secondary}",
        "report.stream": f"bg:{active} {body}",
        "report.stream.selected": f"bg:{selected} {body}",
        "input": f"bg:{surface} {text}",
        "input.placeholder": f"bg:{surface} {muted}",
        "prompt": f"bg:{surface} {brand}",
        "prompt.border": f"bg:{bg} {line}",
        "prompt.border.focus": f"bg:{bg} {secondary}",
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
        "todo.pane": f"bg:{surface} {body}",
        "todo.pane.title": f"bg:{surface} {text} bold",
        "todo.item.pending": f"bg:{surface} {secondary}",
        "todo.item.progress": f"bg:{surface} {gold}",
        "todo.item.done": f"bg:{surface} {ok}",
        "todo.item.cancelled": f"bg:{surface} {muted}",
        "fleet.badge": f"bg:{bg} {muted}",
        "fleet.badge.open": f"bg:{bg} {secondary}",
        "fleet.pane": f"bg:{surface} {body}",
        "fleet.pane.title": f"bg:{surface} {text} bold",
        "fleet.pane.meta": f"bg:{surface} {muted}",
        "fleet.item": f"bg:{surface} {secondary}",
        "fleet.item.running": f"bg:{surface} {body}",
        "fleet.item.selected": f"bg:{selected} {text} bold",
        "shortcut.key": f"bg:{bg} {secondary} bold",
        "shortcut.label": f"bg:{bg} {muted}",
        "shortcut.separator": f"bg:{bg} {line}",
        "shortcut.pending": f"bg:{bg} {gold}",
        "agent-window": f"bg:{surface} {body}",
        "agent-window.header": f"bg:{surface} {text} bold",
        "agent-window.close": f"bg:{surface} {muted}",
        "agent-window.hint": f"bg:{surface} {muted}",
        "ask.dialog": f"bg:{surface} {body}",
        "ask.border": f"bg:{surface} {line}",
        "ask.header": f"bg:{surface} {text} bold",
        "ask.question": f"bg:{surface} {text}",
        "ask.meta": f"bg:{surface} {muted}",
        "ask.option": f"bg:{surface} {body}",
        "ask.option.selected": f"bg:{selected} {text} bold",
        "ask.option.desc": f"bg:{surface} {secondary}",
        "ask.hint": f"bg:{surface} {muted}",
        "detail.input": f"bg:{selected} {text}",
        "detail.input.prompt": f"bg:{selected} {brand}",
        "detail.input.placeholder": f"bg:{selected} {muted}",
        "auth.item": f"bg:{surface} {secondary}",
        "auth.item.selected": f"bg:{selected} {text} bold",
        "auth.item.active": f"bg:{surface} {gold}",
        "auth.item.logged": f"bg:{surface} {body}",
        "auth.item.muted": f"bg:{surface} {muted}",
        "auth.hint": f"bg:{surface} {muted}",
        "auth.note": f"bg:{surface} {secondary}",
        "hud.title": f"fg:{text} bg:{surface}",
        "hud.ok": f"fg:{ok} bg:{surface}",
        "hud.warn": f"fg:{warn} bg:{surface}",
        "hud.cyan": f"fg:{info} bg:{surface}",
        "hud.dim": f"fg:{muted} bg:{surface}",
        "hud.bg": f"bg:{surface}",
        "scrollbar.background": f"bg:{surface} {line}",
        "scrollbar.start": f"bg:{surface} {line}",
        "scrollbar.button": f"bg:{dim} {secondary}",
        "scrollbar.end": f"bg:{dim} {secondary}",
        "scrollbar.arrow": f"bg:{surface} {line}",
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
