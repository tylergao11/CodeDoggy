"""Single semantic palette for the obsidian reading surface.

The terminal owns font family and size. CodeDoggy uses color, underline and
italic only. Weight is deliberately absent: hierarchy comes from restrained
role colors instead of large uninterrupted bright or heavy text.

Interaction grammar
-------------------
* underlined mint: opens a file or URL
* peach italic with a diamond marker: previews a tool record
* lavender: code/keywords, never an interaction promise
"""

from __future__ import annotations

from prompt_toolkit.styles import Style


CANVAS = "#101011"
SURFACE = "#151517"
SURFACE_ACTIVE = "#1b1b1e"
SURFACE_SELECTED = "#222226"
CODE = "#171719"

# Bright white is a scarce focal color, never the default paragraph color.
TEXT = "#F8F8F2"
BODY = "#d6d2cb"
SECONDARY = "#aaa59e"
MUTED = "#79756f"
DIM = "#4b4946"
LINE = "#303034"

# Exact swatches sampled from the supplied reference UI.
BLUSH = "#fde5e4"
LAVENDER = "#f1e9ff"
PEACH = "#fee7cc"
LILAC = "#f6e5f9"
MINT = "#d8f2d5"


def _style_tokens() -> dict[str, str]:
    """Return every style class from the single semantic palette."""
    return {
        # Canvas / chrome
        "root": f"bg:{CANVAS} {BODY}",
        "header": f"bg:{CANVAS} {SECONDARY}",
        "brand": TEXT,
        "brand.dog": BLUSH,
        "header.rule": f"bg:{CANVAS} {LINE}",
        "meta": MUTED,
        "separator": LINE,

        # One-axis task structure
        "task.stream": f"bg:{CANVAS} {BODY}",
        "task.selection": f"bg:{SURFACE_SELECTED}",
        "task.section.label": PEACH,
        "task.section.selected": f"{PEACH} underline",
        "task.section.meta": MUTED,
        "task.title": TEXT,
        "task.status": MUTED,
        "task.status.running": PEACH,
        "task.status.reporting": LAVENDER,
        "task.status.completed": MINT,
        "task.status.failed": BLUSH,
        "task.interject": LILAC,
        "task.action": f"bg:{SURFACE_ACTIVE} {LAVENDER}",
        "task.actor.agent": LAVENDER,
        "task.actor.thinking": f"{SECONDARY} italic",
        "task.chat.gutter": f"bg:{CANVAS}",
        "task.chat.body": BODY,
        "task.chat.muted": MUTED,
        "task.chat.marker": PEACH,
        "task.chat.h1": LILAC,
        "task.chat.h2": PEACH,
        "task.chat.h3": LAVENDER,
        "task.chat.code": LAVENDER,
        "task.chat.path": f"{MINT} underline",
        "task.chat.link": f"{MINT} underline",
        "task.chat.strong": TEXT,
        "task.chat.italic": f"{SECONDARY} italic",
        "task.chat.quote": f"{SECONDARY} italic",
        "task.chat.rule": LINE,
        "task.chat.code.block": f"bg:{CODE} {BODY}",
        "task.chat.code.meta": f"bg:{CODE} {PEACH} italic",
        "task.chat.code.rail": f"bg:{CODE} {DIM}",
        "task.chat.code.kw": f"bg:{CODE} {LILAC}",
        "task.chat.code.fn": f"bg:{CODE} {MINT}",
        "task.chat.code.type": f"bg:{CODE} {PEACH}",
        "task.chat.code.str": f"bg:{CODE} {BLUSH}",
        "task.chat.code.cmt": f"bg:{CODE} {MUTED} italic",
        "task.chat.code.num": f"bg:{CODE} {PEACH}",
        "task.chat.code.sym": f"bg:{CODE} {SECONDARY}",
        "task.chat.code.plain": f"bg:{CODE} {BODY}",
        "task.thinking.rail": DIM,
        "task.thinking.body": f"{SECONDARY} italic",
        # Previewable: diamond marker + peach italic title.
        "task.tool.title": f"{PEACH} italic",
        "task.tool.running": PEACH,
        "task.tool.done": MINT,
        "task.tool.failed": BLUSH,
        "task.plan.ready": PEACH,
        "task.plan.review": f"bg:{PEACH} {CANVAS}",

        # Composer / live status
        "input": f"bg:{SURFACE} {TEXT}",
        "input.placeholder": f"bg:{SURFACE} {MUTED}",
        "prompt": f"bg:{SURFACE} {LAVENDER}",
        "prompt.border": f"bg:{CANVAS} {LINE}",
        "prompt.border.focus": f"bg:{CANVAS} {DIM}",
        "prompt.caption": f"bg:{CANVAS} {MUTED}",
        "turn.status": f"bg:{CANVAS} {PEACH}",
        "turn.elapsed": f"bg:{CANVAS} {MUTED}",
        "turn.stop": f"bg:{CANVAS} {BLUSH}",
        "feedback.info": f"bg:{CANVAS} {LAVENDER}",
        "feedback.success": f"bg:{CANVAS} {MINT}",
        "feedback.warning": f"bg:{CANVAS} {PEACH}",

        # Dedicated plan approval
        "plan.surface": f"bg:{CANVAS} {BODY}",
        "plan.chrome": f"bg:{SURFACE} {BODY}",
        "plan.body": f"bg:{CANVAS} {BODY}",
        "plan.meta": f"bg:{SURFACE} {MUTED}",
        "plan.empty": f"bg:{SURFACE} {SECONDARY} italic",
        "plan.status.review": f"bg:{SURFACE} {PEACH}",
        "plan.action.approve": f"bg:{MINT} {CANVAS}",
        "plan.action.revise": f"bg:{PEACH} {CANVAS}",
        "plan.action.abandon": f"bg:{BLUSH} {CANVAS}",
        "plan.marker": f"bg:{CANVAS} {PEACH}",
        "plan.heading.h1": f"bg:{CANVAS} {LILAC}",
        "plan.heading.h2": f"bg:{CANVAS} {PEACH}",
        "plan.heading.h3": f"bg:{CANVAS} {LAVENDER}",
        "plan.quote.marker": f"bg:{CANVAS} {MUTED}",
        "plan.quote": f"bg:{CANVAS} {SECONDARY} italic",
        "plan.rule": f"bg:{CANVAS} {LINE}",
        "plan.code": f"bg:{CODE} {BODY}",
        "plan.code.fence": f"bg:{CODE} {MUTED} italic",
        "plan.code.inline": f"bg:{CANVAS} {LAVENDER}",
        "plan.code.kw": f"bg:{CODE} {LILAC}",
        "plan.code.fn": f"bg:{CODE} {MINT}",
        "plan.code.type": f"bg:{CODE} {PEACH}",
        "plan.code.str": f"bg:{CODE} {BLUSH}",
        "plan.code.cmt": f"bg:{CODE} {MUTED} italic",
        "plan.code.num": f"bg:{CODE} {PEACH}",
        "plan.code.sym": f"bg:{CODE} {SECONDARY}",
        "plan.code.plain": f"bg:{CODE} {BODY}",
        "plan.link": f"bg:{CANVAS} {MINT} underline",
        "plan.strong": f"bg:{CANVAS} {TEXT}",
        "plan.checkbox.pending": f"bg:{CANVAS} {PEACH}",
        "plan.checkbox.done": f"bg:{CANVAS} {MINT}",

        # Tool preview
        "tool.preview": f"bg:{SURFACE} {BODY}",
        "tool.preview.header": f"bg:{SURFACE} {BODY}",
        "tool.preview.title": f"bg:{SURFACE} {PEACH} italic",
        "tool.preview.meta": f"bg:{SURFACE} {MUTED}",
        "tool.preview.rule": f"bg:{SURFACE} {MINT}",
        "tool.preview.body": f"bg:{SURFACE} {BODY}",
        "tool.preview.footer": f"bg:{SURFACE} {MUTED}",
        "tool.preview.running": f"bg:{SURFACE} {PEACH}",
        "tool.preview.done": f"bg:{SURFACE} {MINT}",
        "tool.preview.failed": f"bg:{SURFACE} {BLUSH}",

        # Todo / fleet
        "todo.badge": f"bg:{CANVAS} {MUTED}",
        "todo.badge.open": f"bg:{CANVAS} {PEACH}",
        "todo.pane": f"bg:{SURFACE} {BODY}",
        "todo.pane.title": f"bg:{SURFACE} {PEACH}",
        "todo.item.pending": f"bg:{SURFACE} {SECONDARY}",
        "todo.item.progress": f"bg:{SURFACE} {PEACH}",
        "todo.item.done": f"bg:{SURFACE} {MINT}",
        "todo.item.cancelled": f"bg:{SURFACE} {MUTED}",
        "fleet.badge": f"bg:{CANVAS} {MUTED}",
        "fleet.badge.open": f"bg:{CANVAS} {LAVENDER}",
        "fleet.pane": f"bg:{SURFACE} {BODY}",
        "fleet.pane.title": f"bg:{SURFACE} {LAVENDER}",
        "fleet.pane.meta": f"bg:{SURFACE} {MUTED}",
        "fleet.item": f"bg:{SURFACE} {SECONDARY}",
        "fleet.item.running": f"bg:{SURFACE} {PEACH}",
        "fleet.item.selected": f"bg:{SURFACE_SELECTED} {TEXT}",

        # Footer / auth / questionnaire
        "shortcut.key": f"bg:{CANVAS} {TEXT}",
        "shortcut.label": f"bg:{CANVAS} {MUTED}",
        "shortcut.separator": f"bg:{CANVAS} {LINE}",
        "shortcut.pending": f"bg:{CANVAS} {PEACH}",
        "agent-window": f"bg:{SURFACE} {BODY}",
        "agent-window.header": f"bg:{SURFACE} {TEXT}",
        "agent-window.close": f"bg:{SURFACE} {BLUSH}",
        "agent-window.hint": f"bg:{SURFACE} {MUTED}",
        "detail.input": f"bg:{SURFACE_ACTIVE} {TEXT}",
        "detail.input.prompt": f"bg:{SURFACE_ACTIVE} {LAVENDER}",
        "detail.input.placeholder": f"bg:{SURFACE_ACTIVE} {MUTED}",
        "ask.dialog": f"bg:{SURFACE} {BODY}",
        "ask.border": f"bg:{SURFACE} {LINE}",
        "ask.header": f"bg:{SURFACE} {LILAC}",
        "ask.question": f"bg:{SURFACE} {TEXT}",
        "ask.meta": f"bg:{SURFACE} {MUTED}",
        "ask.option": f"bg:{SURFACE} {BODY}",
        "ask.option.selected": f"bg:{SURFACE_SELECTED} {PEACH}",
        "ask.option.desc": f"bg:{SURFACE} {SECONDARY}",
        "ask.hint": f"bg:{SURFACE} {MUTED}",
        "auth.item": f"bg:{SURFACE} {SECONDARY}",
        "auth.item.selected": f"bg:{SURFACE_SELECTED} {TEXT}",
        "auth.item.active": f"bg:{SURFACE} {LAVENDER}",
        "auth.item.logged": f"bg:{SURFACE} {MINT}",
        "auth.item.muted": f"bg:{SURFACE} {MUTED}",
        "auth.hint": f"bg:{SURFACE} {MUTED}",
        "auth.note": f"bg:{SURFACE} {SECONDARY}",
        "hud.title": f"fg:{TEXT} bg:{SURFACE}",
        "hud.ok": f"fg:{MINT} bg:{SURFACE}",
        "hud.warn": f"fg:{BLUSH} bg:{SURFACE}",
        "hud.accent": f"fg:{LAVENDER} bg:{SURFACE}",
        "hud.dim": f"fg:{MUTED} bg:{SURFACE}",
        "hud.bg": f"bg:{SURFACE}",

        # Shared record renderer
        "detail.header": f"bg:{SURFACE} {TEXT}",
        "detail.meta": f"bg:{SURFACE} {MUTED}",
        "detail.active": f"bg:{SURFACE} {PEACH}",
        "detail.separator": f"bg:{SURFACE} {LINE}",
        "detail.text": f"bg:{SURFACE} {BODY}",
        "detail.actor": f"bg:{SURFACE} {SECONDARY}",
        "detail.actor.user": f"bg:{SURFACE} {TEXT}",
        "detail.actor.assistant": f"bg:{SURFACE} {BLUSH}",
        "detail.actor.tool": f"bg:{SURFACE} {MINT}",
        "detail.actor.think": f"bg:{SURFACE} {SECONDARY} italic",
        "detail.tool": f"bg:{SURFACE} {PEACH} italic",
        "detail.tool.link": f"bg:{SURFACE} {MINT} underline",
        "detail.block": f"bg:{CODE} {BODY}",
        "detail.code.rail": f"bg:{CODE} {DIM}",
        "detail.code.gutter": f"bg:{CODE} {MUTED}",
        "detail.code.gutter.mark": f"bg:{CODE} {SECONDARY}",
        "detail.code.gutter.sep": f"bg:{CODE} {LINE}",
        "detail.code.meta": f"bg:{SURFACE} {MUTED} italic",
        "detail.code.kw": f"bg:{CODE} {LILAC}",
        "detail.code.fn": f"bg:{CODE} {MINT}",
        "detail.code.type": f"bg:{CODE} {PEACH}",
        "detail.code.str": f"bg:{CODE} {BLUSH}",
        "detail.code.cmt": f"bg:{CODE} {MUTED} italic",
        "detail.code.num": f"bg:{CODE} {PEACH}",
        "detail.code.sym": f"bg:{CODE} {SECONDARY}",
        "detail.code.plain": f"bg:{CODE} {BODY}",
        "detail.diff.add": f"bg:{CODE} {MINT}",
        "detail.diff.remove": f"bg:{CODE} {BLUSH}",
        "detail.diff.hunk": f"bg:{CODE} {PEACH}",
        "detail.diff.gutter": f"bg:{CODE} {MUTED}",
        "detail.success": f"bg:{CODE} {MINT}",
        "detail.error": f"bg:{CODE} {BLUSH}",
        "detail.error.link": f"bg:{CODE} {BLUSH} underline",
        "detail.warning": f"bg:{CODE} {PEACH}",
        "detail.link": f"bg:{SURFACE} {MINT} underline",
        "detail.fold.collapsed": f"bg:{SURFACE} {LAVENDER}",
        "detail.fold.collapsed.link": f"bg:{SURFACE} {MINT} underline",
        "detail.fold.expanded": f"bg:{SURFACE} {PEACH}",
        "detail.fold.expanded.link": f"bg:{SURFACE} {MINT} underline",
        "detail.fold.footer": f"bg:{SURFACE} {MUTED}",
        "detail.tool.section": f"bg:{SURFACE} {PEACH} italic",
        "detail.md.ol": f"bg:{SURFACE} {PEACH}",
        "detail.md.ul": f"bg:{SURFACE} {MINT}",
        "detail.md.h1": f"bg:{SURFACE} {LILAC}",
        "detail.md.h2": f"bg:{SURFACE} {PEACH}",
        "detail.md.h3": f"bg:{SURFACE} {LAVENDER}",
        "detail.md.quote": f"bg:{SURFACE} {SECONDARY} italic",
        "detail.md.inline": f"bg:{SURFACE} {LAVENDER}",
        "detail.md.strong": f"bg:{SURFACE} {TEXT}",
        "detail.md.italic": f"bg:{SURFACE} {SECONDARY} italic",
        "detail.md.strike": f"bg:{SURFACE} {MUTED} strike",
        "detail.thinking.rail": f"bg:{SURFACE} {DIM}",
        "detail.thinking.body": f"bg:{SURFACE} {SECONDARY} italic",

        # Scrollbar
        "scrollbar.background": f"bg:{SURFACE} {LINE}",
        "scrollbar.start": f"bg:{SURFACE} {LINE}",
        "scrollbar.button": f"bg:{DIM} {SECONDARY}",
        "scrollbar.end": f"bg:{DIM} {SECONDARY}",
        "scrollbar.arrow": f"bg:{SURFACE} {LINE}",
    }


def build_style() -> Style:
    """Build the sole CodeDoggy visual system."""
    return Style.from_dict(_style_tokens())
