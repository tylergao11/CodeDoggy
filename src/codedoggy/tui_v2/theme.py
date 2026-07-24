"""Grok pager theme — GrokNight palette and prompt_toolkit Style builder.

Ported from:

* ``xai-grok-pager-render/src/theme/tokyonight.rs`` — ``Theme`` field names
* ``xai-grok-pager-render/src/theme/groknight.rs`` — exact RGB palette
* ``xai-grok-pager-render/src/theme/mod.rs`` — default = GrokNight

SOURCE_REV ``95d84f443eddcbed6cbfd6eed22e2eafe6b3939d``.

All colors are truecolor RGB hex (``#rrggbb``). Paint edges use
``prompt_toolkit`` style strings; fragments reference ``class:grok.*``.

Class map (``Style.from_dict`` key → Theme field / role)
--------------------------------------------------------
::

    grok                         bg:bg_base + text_primary   (root canvas)
    grok.bg_base                 bg:bg_base
    grok.bg_light                bg:bg_light
    grok.bg_dark                 bg:bg_dark
    grok.bg_highlight            bg:bg_highlight
    grok.bg_hover                bg:bg_hover
    grok.bg_terminal             bg:bg_terminal
    grok.bg_visual               bg:bg_visual

    grok.accent_user             accent_user
    grok.accent_assistant        accent_assistant
    grok.accent_thinking         accent_thinking
    grok.accent_tool             accent_tool
    grok.accent_system           accent_system
    grok.accent_error            accent_error
    grok.accent_success          accent_success
    grok.accent_running          accent_running
    grok.accent_skill            accent_skill
    grok.accent_plan             accent_plan
    grok.accent_verify           accent_verify
    grok.accent_feedback         accent_feedback
    grok.accent_remember         accent_remember
    grok.accent_model            accent_model

    grok.text_primary            text_primary
    grok.text_secondary          text_secondary
    grok.gray_dim                gray_dim
    grok.gray                    gray
    grok.gray_bright             gray_bright

    grok.command                 command
    grok.path                    path
    grok.running                 running
    grok.warning                 warning
    grok.fuzzy_accent            fuzzy_accent

    grok.selection_border        selection_border
    grok.hover_border            hover_border
    grok.prompt_border           prompt_border
    grok.prompt_border_active    prompt_border_active

    grok.scrollbar_bg            bg:scrollbar_bg
    grok.scrollbar_fg            scrollbar_fg

    grok.diff_delete_bg          bg:diff_delete_bg
    grok.diff_delete_fg          diff_delete_fg
    grok.diff_insert_bg          bg:diff_insert_bg
    grok.diff_insert_fg          diff_insert_fg
    grok.diff_equal_fg           diff_equal_fg
    grok.diff_gutter_fg          diff_gutter_fg

    grok.paste_bg                bg:paste_bg
    grok.paste_fg                paste_fg
    grok.paste_dim               paste_dim

    grok.md_heading_h1 … h6      md_heading_h* (+ bold when set)
    grok.md_code                 md_code
    grok.md_task_checked         md_task_checked
    grok.md_task_unchecked       md_task_unchecked
    grok.md_muted                md_muted
    grok.md_code_bg              bg:md_code_bg
    grok.md_text                 md_text
    grok.link_fg                 link_fg   (underline)

Composite helpers used by layout/blocks::

    grok.root                    bg:bg_base text_primary
    grok.code_block              bg:md_code_bg md_text
    grok.diff_delete             bg:diff_delete_bg diff_delete_fg
    grok.diff_insert             bg:diff_insert_bg diff_insert_fg
    grok.paste                   bg:paste_bg paste_fg
    grok.prompt_chrome           prompt_border
    grok.prompt_chrome_active    prompt_border_active
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from prompt_toolkit.styles import Style


# ── GrokNight palette (exact RGB from groknight.rs) ─────────────────────────


def _rgb(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


# Backgrounds
_BG = _rgb(10, 10, 10)  # #0a0a0a — Night (terminal bg)
_BG_DARK = _rgb(12, 12, 12)  # #0c0c0c — darkest
_BG_STORM_DARK = _rgb(17, 17, 17)  # #111111 — dark bg
_BG_STORM = _rgb(20, 20, 20)  # #141414 — main bg
_BG_HIGHLIGHT = _rgb(36, 36, 36)  # #242424 — highlight bg

# Text / grays
_FG = _rgb(225, 225, 225)  # #e1e1e1 — primary text
_FG_DARK = _rgb(200, 200, 200)  # #c8c8c8 — secondary text
_FG_GUTTER = _rgb(65, 65, 65)  # #414141 — dim
_COMMENT = _rgb(108, 108, 108)  # #6c6c6c — muted
_DARK3 = _rgb(90, 90, 90)  # #5a5a5a — medium gray
_DARK5 = _rgb(120, 120, 120)  # #787878 — bright gray

# Accent colors (TokyoNight Night)
_BLUE = _rgb(122, 162, 247)  # #7aa2f7
_BLUE0 = _rgb(61, 89, 161)  # #3d59a1
_BLUE1 = _rgb(58, 149, 171)  # #3A95AB
_CYAN = _rgb(125, 207, 255)  # #7dcfff
_GREEN = _rgb(158, 206, 106)  # #9ece6a
_GREEN1 = _rgb(115, 218, 202)  # #73daca
_MAGENTA = _rgb(187, 154, 247)  # #bb9af7
_ORANGE = _rgb(255, 158, 100)  # #ff9e64
_PURPLE = _rgb(157, 124, 216)  # #9d7cd8
_RED = _rgb(247, 118, 142)  # #f7768e
_RED1 = _rgb(219, 75, 75)  # #db4b4b
_TEAL = _rgb(26, 188, 156)  # #1abc9c
_YELLOW = _rgb(224, 175, 104)  # #e0af68

_RED_DARK = _rgb(66, 14, 20)  # #420e14
_GREEN_DARK = _rgb(6, 56, 6)  # #063806


@dataclass(frozen=True, slots=True)
class Theme:
    """Semantic color roles for pager rendering (Grok ``Theme`` struct).

    Every field is a ``#rrggbb`` hex string suitable for prompt_toolkit
    style attributes (``fg:`` / ``bg:``). Heading ``*_mod`` fields are
    space-separated modifier tokens (``"bold"``, ``""``, …).
    """

    # Backgrounds
    bg_base: str
    bg_light: str
    bg_dark: str
    bg_highlight: str
    bg_hover: str
    bg_terminal: str

    # Accent colors (vertical rails / state)
    accent_user: str
    accent_assistant: str
    accent_thinking: str
    accent_tool: str
    accent_system: str
    accent_error: str
    accent_success: str
    accent_running: str
    accent_skill: str

    # Text
    text_primary: str
    text_secondary: str

    # Gray scale (dim → medium → bright)
    gray_dim: str
    gray: str
    gray_bright: str

    # Semantic
    command: str
    path: str
    running: str
    warning: str

    # Search
    fuzzy_accent: str

    # Mode accents
    accent_plan: str
    accent_verify: str
    accent_feedback: str
    accent_remember: str

    # Selection / chrome borders
    selection_border: str
    hover_border: str
    prompt_border: str
    prompt_border_active: str

    # Prompt info
    accent_model: str

    # Scrollbar
    scrollbar_bg: str
    scrollbar_fg: str

    # Diff
    diff_delete_bg: str
    diff_delete_fg: str
    diff_insert_bg: str
    diff_insert_fg: str
    diff_equal_fg: str
    diff_gutter_fg: str

    # Visual selection
    bg_visual: str

    # Paste chip
    paste_bg: str
    paste_fg: str
    paste_dim: str

    # Markdown
    md_heading_h1: str
    md_heading_h1_mod: str
    md_heading_h2: str
    md_heading_h2_mod: str
    md_heading_h3: str
    md_heading_h3_mod: str
    md_heading_h4: str
    md_heading_h4_mod: str
    md_heading_h5: str
    md_heading_h5_mod: str
    md_heading_h6: str
    md_heading_h6_mod: str
    md_code: str
    md_task_checked: str
    md_task_unchecked: str
    md_muted: str
    md_code_bg: str
    md_text: str
    link_fg: str

    def fg_style(self, color: str) -> str:
        """prompt_toolkit style fragment for a foreground color."""
        return color if color.startswith("#") else f"fg:{color}"

    def is_dark(self) -> bool:
        """BT.709-ish luminance of ``bg_base``; True when dark canvas."""
        hex_s = self.bg_base.lstrip("#")
        if len(hex_s) != 6:
            return True
        r, g, b = int(hex_s[0:2], 16), int(hex_s[2:4], 16), int(hex_s[4:6], 16)
        # Relative luminance (sRGB approx used by Grok osc11 classify).
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        return lum < 0.5


def groknight() -> Theme:
    """GrokNight — neutral gray base with TokyoNight accent colors.

    Exact RGB values from ``theme/groknight.rs`` ``Theme::groknight()``.
    """
    return Theme(
        bg_base=_BG_STORM,
        bg_light=_BG_HIGHLIGHT,
        bg_dark=_rgb(28, 28, 28),  # lighter than bg_base for visible code blocks
        bg_highlight=_BG_HIGHLIGHT,
        bg_hover=_rgb(44, 44, 44),
        bg_terminal=_BG,
        accent_user=_FG_DARK,
        accent_assistant=_MAGENTA,
        accent_thinking=_MAGENTA,
        accent_tool=_DARK5,
        accent_system=_BLUE,
        accent_error=_RED,
        accent_success=_GREEN,
        accent_running=_MAGENTA,
        accent_skill=_BLUE,
        text_primary=_FG,
        text_secondary=_FG_DARK,
        gray_dim=_rgb(88, 88, 88),  # #585858 — slightly brighter than FG_GUTTER
        gray=_COMMENT,
        gray_bright=_DARK5,
        command=_YELLOW,
        path=_ORANGE,
        running=_CYAN,
        warning=_YELLOW,
        fuzzy_accent=_BLUE,
        accent_plan=_rgb(255, 219, 141),  # #FFDB8D — golden
        accent_verify=_rgb(187, 154, 247),  # #bb9af7 — violet
        accent_feedback=_GREEN1,  # #73daca
        accent_remember=_rgb(139, 195, 74),  # #8BC34A — Material Design light green
        selection_border=_rgb(60, 60, 65),
        prompt_border=_rgb(50, 50, 55),  # #323237 — dimmer prompt chrome
        prompt_border_active=_rgb(80, 80, 88),  # #505058 — brighter when focused
        hover_border=_rgb(30, 30, 34),
        accent_model=_TEAL,
        scrollbar_bg=_BG_STORM_DARK,
        scrollbar_fg=_BG_HIGHLIGHT,
        diff_delete_bg=_RED_DARK,
        diff_delete_fg=_RED,
        diff_insert_bg=_GREEN_DARK,
        diff_insert_fg=_GREEN,
        diff_equal_fg=_COMMENT,
        diff_gutter_fg=_COMMENT,
        bg_visual=_rgb(54, 54, 54),
        paste_bg=_BG_STORM_DARK,
        paste_fg=_FG_DARK,
        paste_dim=_FG_GUTTER,
        md_heading_h1=_TEAL,
        md_heading_h1_mod="bold",
        md_heading_h2=_BLUE,
        md_heading_h2_mod="bold",
        md_heading_h3=_PURPLE,
        md_heading_h3_mod="bold",
        md_heading_h4=_DARK5,
        md_heading_h4_mod="bold",
        md_heading_h5=_COMMENT,
        md_heading_h5_mod="bold",
        md_heading_h6=_DARK3,
        md_heading_h6_mod="",
        md_code=_BLUE1,
        md_task_checked=_GREEN,
        md_task_unchecked=_FG_DARK,
        md_muted=_COMMENT,
        md_code_bg=_rgb(28, 28, 28),
        md_text=_FG_DARK,
        link_fg=_rgb(122, 166, 218),  # #7aa6da — soft blue for dark bg
    )


def _mod_suffix(mod: str) -> str:
    mod = (mod or "").strip()
    return f" {mod}" if mod else ""


def theme_style_dict(theme: Theme) -> dict[str, str]:
    """Map Theme fields → prompt_toolkit ``Style.from_dict`` entries.

    Keys are class names **without** the ``class:`` prefix (prompt_toolkit
    convention). Fragments use ``("class:grok.text_primary", text)``.
    """
    t = theme
    d: dict[str, str] = {
        # Root / canvas
        "grok": f"bg:{t.bg_base} {t.text_primary}",
        "grok.root": f"bg:{t.bg_base} {t.text_primary}",
        # Backgrounds
        "grok.bg_base": f"bg:{t.bg_base}",
        "grok.bg_light": f"bg:{t.bg_light}",
        "grok.bg_dark": f"bg:{t.bg_dark}",
        "grok.bg_highlight": f"bg:{t.bg_highlight}",
        "grok.bg_hover": f"bg:{t.bg_hover}",
        "grok.bg_terminal": f"bg:{t.bg_terminal}",
        "grok.bg_visual": f"bg:{t.bg_visual}",
        # Accents
        "grok.accent_user": t.accent_user,
        "grok.accent_assistant": t.accent_assistant,
        "grok.accent_thinking": t.accent_thinking,
        "grok.accent_tool": t.accent_tool,
        "grok.accent_system": t.accent_system,
        "grok.accent_error": t.accent_error,
        "grok.accent_success": t.accent_success,
        "grok.accent_running": t.accent_running,
        "grok.accent_skill": t.accent_skill,
        "grok.accent_plan": t.accent_plan,
        "grok.accent_verify": t.accent_verify,
        "grok.accent_feedback": t.accent_feedback,
        "grok.accent_remember": t.accent_remember,
        "grok.accent_model": t.accent_model,
        # Text / gray
        "grok.text_primary": t.text_primary,
        "grok.text_secondary": t.text_secondary,
        "grok.gray_dim": t.gray_dim,
        "grok.gray": t.gray,
        "grok.gray_bright": t.gray_bright,
        # Semantic
        "grok.command": t.command,
        "grok.path": t.path,
        "grok.running": t.running,
        "grok.warning": t.warning,
        "grok.fuzzy_accent": t.fuzzy_accent,
        # Borders
        "grok.selection_border": t.selection_border,
        "grok.hover_border": t.hover_border,
        "grok.prompt_border": t.prompt_border,
        "grok.prompt_border_active": t.prompt_border_active,
        "grok.prompt_chrome": t.prompt_border,
        "grok.prompt_chrome_active": t.prompt_border_active,
        # Scrollbar
        "grok.scrollbar_bg": f"bg:{t.scrollbar_bg}",
        "grok.scrollbar_fg": t.scrollbar_fg,
        # Diff
        "grok.diff_delete_bg": f"bg:{t.diff_delete_bg}",
        "grok.diff_delete_fg": t.diff_delete_fg,
        "grok.diff_insert_bg": f"bg:{t.diff_insert_bg}",
        "grok.diff_insert_fg": t.diff_insert_fg,
        "grok.diff_equal_fg": t.diff_equal_fg,
        "grok.diff_gutter_fg": t.diff_gutter_fg,
        "grok.diff_delete": f"bg:{t.diff_delete_bg} {t.diff_delete_fg}",
        "grok.diff_insert": f"bg:{t.diff_insert_bg} {t.diff_insert_fg}",
        # Paste
        "grok.paste_bg": f"bg:{t.paste_bg}",
        "grok.paste_fg": t.paste_fg,
        "grok.paste_dim": t.paste_dim,
        "grok.paste": f"bg:{t.paste_bg} {t.paste_fg}",
        # Markdown
        "grok.md_heading_h1": f"{t.md_heading_h1}{_mod_suffix(t.md_heading_h1_mod)}",
        "grok.md_heading_h2": f"{t.md_heading_h2}{_mod_suffix(t.md_heading_h2_mod)}",
        "grok.md_heading_h3": f"{t.md_heading_h3}{_mod_suffix(t.md_heading_h3_mod)}",
        "grok.md_heading_h4": f"{t.md_heading_h4}{_mod_suffix(t.md_heading_h4_mod)}",
        "grok.md_heading_h5": f"{t.md_heading_h5}{_mod_suffix(t.md_heading_h5_mod)}",
        "grok.md_heading_h6": f"{t.md_heading_h6}{_mod_suffix(t.md_heading_h6_mod)}",
        "grok.md_code": t.md_code,
        "grok.md_task_checked": t.md_task_checked,
        "grok.md_task_unchecked": t.md_task_unchecked,
        "grok.md_muted": t.md_muted,
        "grok.md_code_bg": f"bg:{t.md_code_bg}",
        "grok.md_text": t.md_text,
        "grok.code_block": f"bg:{t.md_code_bg} {t.md_text}",
        "grok.link_fg": f"{t.link_fg} underline",
        "grok.link": f"{t.link_fg} underline",
        # ── Painter aliases (block modules use short / dotted names) ──
        # Tools (common.py S_*)
        "grok.primary": t.text_primary,
        "grok.muted": t.gray,
        "grok.dim": t.gray_dim,
        "grok.selected": f"bg:{t.bg_highlight}",
        "grok.diff.insert": f"bg:{t.diff_insert_bg} {t.diff_insert_fg}",
        "grok.diff.delete": f"bg:{t.diff_delete_bg} {t.diff_delete_fg}",
        "grok.diff.equal": t.diff_equal_fg,
        "grok.diff.gutter": t.diff_gutter_fg,
        "grok.diff.insert_bg": f"bg:{t.diff_insert_bg}",
        "grok.diff.delete_bg": f"bg:{t.diff_delete_bg}",
        # User prompt
        "grok.prompt.prefix": t.accent_user,
        "grok.prompt.body": t.text_primary,
        "grok.prompt.skill": t.accent_skill,
        "grok.prompt.band": f"bg:{t.bg_light}",
        # Thinking
        "grok.thinking.header": f"{t.text_primary} bold",
        "grok.thinking.header.muted": f"{t.gray} bold",
        "grok.thinking.detail": t.gray,
        "grok.thinking.ellipsis": t.gray,
        "grok.thinking.body": t.text_secondary,
        "grok.thinking.sep": t.gray_dim,
        # Markdown (dotted aliases for blocks/markdown.py)
        "grok.md.text": t.md_text,
        "grok.md.h1": f"{t.md_heading_h1}{_mod_suffix(t.md_heading_h1_mod)}",
        "grok.md.h2": f"{t.md_heading_h2}{_mod_suffix(t.md_heading_h2_mod)}",
        "grok.md.h3": f"{t.md_heading_h3}{_mod_suffix(t.md_heading_h3_mod)}",
        "grok.md.h4": f"{t.md_heading_h4}{_mod_suffix(t.md_heading_h4_mod)}",
        "grok.md.h5": f"{t.md_heading_h5}{_mod_suffix(t.md_heading_h5_mod)}",
        "grok.md.h6": f"{t.md_heading_h6}{_mod_suffix(t.md_heading_h6_mod)}",
        "grok.md.strong": f"{t.text_primary} bold",
        "grok.md.em": f"{t.text_secondary} italic",
        "grok.md.code": t.md_code,
        "grok.md.code.block": f"bg:{t.md_code_bg} {t.md_text}",
        "grok.md.code.lang": t.gray,
        "grok.md.quote": t.md_muted,
        "grok.md.quote.bar": f"{t.md_muted} dim",
        "grok.md.list.marker": t.md_muted,
        "grok.md.rule": t.md_muted,
        "grok.md.link": f"{t.link_fg} underline",
        "grok.md.task.checked": t.md_task_checked,
        "grok.md.task.unchecked": t.md_task_unchecked,
        # Misc block / chrome
        "grok.tool.header": f"{t.text_primary} bold",
        "grok.context.low": t.text_primary,
        "grok.context.mid": t.accent_user,
        "grok.context.high": t.warning,
        "grok.context.critical": t.accent_error,
        # Syntax highlight tokens (edit/read — Grok syntect spirit)
        "grok.syn.kw": t.accent_skill,  # blue-ish keywords
        "grok.syn.fn": t.accent_running,  # magenta functions
        "grok.syn.type": t.accent_model,  # teal types
        "grok.syn.str": t.command,  # yellow strings
        "grok.syn.cmt": t.gray,  # muted comments
        "grok.syn.num": t.warning,  # numbers
        "grok.syn.sym": t.gray_bright,  # operators
        "grok.syn.plain": t.md_text,  # default code text
    }
    return d


def build_style(theme: Theme | None = None) -> Style:
    """Build a prompt_toolkit ``Style`` from *theme* (default GrokNight)."""
    return Style.from_dict(theme_style_dict(theme or groknight()))


def style_class(field: str) -> str:
    """Return ``class:grok.<field>`` for fragment tuples."""
    if field.startswith("class:"):
        return field
    if field.startswith("grok."):
        return f"class:{field}"
    return f"class:grok.{field}"


# ── Style class constants for block painters (class:grok.*) ─────────────────
# Prefer these over inventing ad-hoc names; theme_style_dict maps them.

S_PRIMARY = "class:grok.primary"
S_BOLD = "class:grok.primary bold"
S_MUTED = "class:grok.muted"
S_DIM = "class:grok.dim"
S_PATH = "class:grok.path"
S_ERROR = "class:grok.accent_error"
S_INSERT = "class:grok.diff.insert"
S_DELETE = "class:grok.diff.delete"
S_GUTTER = "class:grok.diff.gutter"
S_EQUAL = "class:grok.diff.equal"
S_COMMAND = "class:grok.command"
S_SUCCESS = "class:grok.accent_success"
S_PANEL = "class:grok.bg_dark"
S_SELECTED = "class:grok.selected"


# Palette constants exported for block painters that need raw hex.
PALETTE: Mapping[str, str] = {
    "BG": _BG,
    "BG_DARK": _BG_DARK,
    "BG_STORM_DARK": _BG_STORM_DARK,
    "BG_STORM": _BG_STORM,
    "BG_HIGHLIGHT": _BG_HIGHLIGHT,
    "FG": _FG,
    "FG_DARK": _FG_DARK,
    "FG_GUTTER": _FG_GUTTER,
    "COMMENT": _COMMENT,
    "DARK3": _DARK3,
    "DARK5": _DARK5,
    "BLUE": _BLUE,
    "BLUE0": _BLUE0,
    "BLUE1": _BLUE1,
    "CYAN": _CYAN,
    "GREEN": _GREEN,
    "GREEN1": _GREEN1,
    "MAGENTA": _MAGENTA,
    "ORANGE": _ORANGE,
    "PURPLE": _PURPLE,
    "RED": _RED,
    "RED1": _RED1,
    "TEAL": _TEAL,
    "YELLOW": _YELLOW,
    "RED_DARK": _RED_DARK,
    "GREEN_DARK": _GREEN_DARK,
}


__all__ = [
    "PALETTE",
    "S_BOLD",
    "S_COMMAND",
    "S_DELETE",
    "S_DIM",
    "S_EQUAL",
    "S_ERROR",
    "S_GUTTER",
    "S_INSERT",
    "S_MUTED",
    "S_PANEL",
    "S_PATH",
    "S_PRIMARY",
    "S_SELECTED",
    "S_SUCCESS",
    "Theme",
    "build_style",
    "groknight",
    "style_class",
    "theme_style_dict",
]
