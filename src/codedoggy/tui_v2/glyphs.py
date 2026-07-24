"""Chrome glyphs for the Grok pager scrollback / prompt UI.

Ported from ``xai-grok-pager-render/src/glyphs.rs`` (SOURCE_REV
``95d84f443eddcbed6cbfd6eed22e2eafe6b3939d``).

Legacy-console fallbacks fire when the host is a bare Windows ConHost
(Consolas / Lucida Console lack Dingbats / Braille / heavy box-drawing),
or when forced via environment:

* ``CODEDOGGY_ASCII_GLYPHS=1`` / ``true`` — force legacy fallbacks
* ``GROK_FORCE_LEGACY_CONSOLE=1`` / ``true`` — same (Grok QA hatch)
* ``=0`` / ``false`` on either — force fancy glyphs

Every single-cell chrome glyph is exactly **1 column** wide so layout
never shifts between platforms. ``prompt_arrow`` is **2 columns**.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Sequence

# Display width of :func:`prompt_arrow` in columns.
PROMPT_ARROW_WIDTH: int = 2


def _parse_forced_legacy(value: str | None) -> bool | None:
    """``"1"``/``"true"`` → on, ``"0"``/``"false"`` → off, else ``None``."""
    if value is None:
        return None
    if value in ("1", "true"):
        return True
    if value in ("0", "false"):
        return False
    return None


def _forced_legacy_console_override() -> bool | None:
    """Read CodeDoggy / Grok force-legacy escape hatches."""
    for key in ("CODEDOGGY_ASCII_GLYPHS", "GROK_FORCE_LEGACY_CONSOLE"):
        parsed = _parse_forced_legacy(os.environ.get(key))
        if parsed is not None:
            return parsed
    return None


def _decide_legacy_windows_console() -> bool:
    """Default-deny on Windows: unknown / bare ConHost is legacy."""
    if sys.platform != "win32":
        return False

    # Modern emulators that ship fonts covering chrome glyphs.
    # Mirrors TerminalName allow-list in glyphs.rs.
    if os.environ.get("WT_SESSION"):
        return False  # Windows Terminal
    term_program = (os.environ.get("TERM_PROGRAM") or "").lower()
    if term_program in (
        "vscode",
        "cursor",
        "windsurf",
        "zed",
        "wezterm",
        "ghostty",
        "alacritty",
        "kitty",
        "rio",
        "grokdesktop",
        "grok-desktop",
    ):
        return False
    if os.environ.get("TERM_PROGRAM_VERSION") and term_program:
        # VS Code family often sets TERM_PROGRAM=vscode
        if term_program in ("vscode", "cursor", "windsurf"):
            return False
    if os.environ.get("KITTY_WINDOW_ID") or os.environ.get("KITTY_PID"):
        return False
    if os.environ.get("ALACRITTY_SOCKET") or os.environ.get("ALACRITTY_LOG"):
        return False
    if os.environ.get("WEZTERM_EXECUTABLE") or os.environ.get("WEZTERM_PANE"):
        return False
    if os.environ.get("GHOSTTY_RESOURCES_DIR"):
        return False
    # VS Code integrated terminal
    if os.environ.get("VSCODE_INJECTION") or os.environ.get("TERM_PROGRAM") == "vscode":
        return False
    # Windows Terminal also sets this on some builds
    if (os.environ.get("TERM") or "").lower() in ("xterm-256color", "xterm-ghostty"):
        # Ambiguous alone — only skip legacy if another signal exists.
        # Bare ConHost often has no TERM or TERM=cygwin; keep conservative.
        pass

    return True


@lru_cache(maxsize=1)
def is_legacy_windows_console() -> bool:
    """True when chrome glyphs need ASCII / CP437 fallbacks.

    Cached for process lifetime. Override via ``CODEDOGGY_ASCII_GLYPHS``
    or ``GROK_FORCE_LEGACY_CONSOLE``.
    """
    forced = _forced_legacy_console_override()
    if forced is not None:
        return forced
    return _decide_legacy_windows_console()


def clear_legacy_console_cache() -> None:
    """Drop the cached legacy-console decision (tests / env changes)."""
    is_legacy_windows_console.cache_clear()


# ── Prompt / voice ──────────────────────────────────────────────────────────


def prompt_arrow() -> str:
    """``"❯ "`` normally, ``"> "`` on legacy ConHost. Always 2 columns."""
    if is_legacy_windows_console():
        return "> "
    return "\u276f "


def record_dot(filled: bool) -> str:
    """Recording indicator: FISHEYE / BULLSEYE; ``*``/``o`` on legacy. 1 col."""
    if is_legacy_windows_console():
        return "*" if filled else "o"
    return "\u25c9" if filled else "\u25ce"


# ── Status / buttons ────────────────────────────────────────────────────────


def collapsed_accent() -> str:
    """``"❙"`` normally, ``"|"`` on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "|"
    return "\u2759"


def ballot_x() -> str:
    """``"✗"`` (U+2717) normally, ``"x"`` on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "x"
    return "\u2717"


def check_mark() -> str:
    """``"✓"`` (U+2713) normally, ``"√"`` (U+221A) on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "\u221a"
    return "\u2713"


def enlarge() -> str:
    """``"↗"`` (U+2197) normally, ``"o"`` on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "o"
    return "\u2197"


def copy_icon() -> str:
    """``"⧉"`` (U+29C9) normally, ``"c"`` on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "c"
    return "\u29c9"


def token_arrow() -> str:
    """``"⇣"`` (U+21E3) normally, ``"↓"`` (U+2193) on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "\u2193"
    return "\u21e3"


def ballot_x_button() -> str:
    """``"[✗]"`` normally, ``"[x]"`` on legacy. Always 3 columns."""
    if is_legacy_windows_console():
        return "[x]"
    return "[\u2717]"


def enlarge_button() -> str:
    """``"[↗]"`` normally, ``"[o]"`` on legacy. Always 3 columns."""
    if is_legacy_windows_console():
        return "[o]"
    return "[\u2197]"


# ── Diamonds ────────────────────────────────────────────────────────────────


def diamond_filled() -> str:
    """``"◆"`` (U+25C6) normally, ``"♦"`` (U+2666) on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "\u2666"
    return "\u25c6"


def diamond_hollow() -> str:
    """``"◇"`` (U+25C7) normally, ``"○"`` (U+25CB) on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "\u25cb"
    return "\u25c7"


def diamond_dotted() -> str:
    """``"◈"`` (U+25C8) normally, ``"♦"`` (U+2666) on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "\u2666"
    return "\u25c8"


def diamond_filled_char() -> str:
    """Filled-diamond glyph as a single character (see :func:`diamond_filled`)."""
    return diamond_filled()[0]


def diamond_hollow_char() -> str:
    """Hollow-diamond glyph as a single character (see :func:`diamond_hollow`)."""
    return diamond_hollow()[0]


# ── Spinners / monitor ──────────────────────────────────────────────────────


def braille_spinner_frames() -> Sequence[str]:
    """Braille spinner frames; ASCII ``|/|-|\\`` on legacy. Each frame 1 col."""
    if is_legacy_windows_console():
        return ("|", "/", "-", "\\")
    return (
        "\u280b",
        "\u2819",
        "\u2839",
        "\u2838",
        "\u283c",
        "\u2834",
        "\u2826",
        "\u2827",
    )


def dot_spinner_frames() -> Sequence[str]:
    """Pulsing-dot spinner frames; quiet ``.``/``:``/``·`` on legacy. 1 col."""
    if is_legacy_windows_console():
        return (".", ":", "\u00b7")
    return (
        "\u22c5",
        ":",
        "\u2e2c",
        "\u2059",
        "\u22c5",
        ":",
        "\u2e2c",
        "\u2059",
    )


def monitor_icon_frames() -> Sequence[str]:
    """Monitor-indicator pulse frames; CP437-safe dots on legacy. 1 col."""
    if is_legacy_windows_console():
        return ("\u00b7", "\u25cb", "\u2022", "\u25cb")
    return ("\u25cb", "\u25ce", "\u25c9", "\u25ce")


# ── Box-drawing / rails ─────────────────────────────────────────────────────


def accent_bar() -> str:
    """``"┃"`` (U+2503) normally, ``"│"`` (U+2502) on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "\u2502"
    return "\u2503"


def heavy_horizontal() -> str:
    """``"━"`` (U+2501) normally, ``"─"`` (U+2500) on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "\u2500"
    return "\u2501"


def light_horizontal() -> str:
    """``"─"`` (U+2500). Always 1 column; present on every target."""
    return "\u2500"


def timeline_tick_active() -> str:
    """Precomposed 2-col active tick: ``"━━"`` or ``"══"`` on legacy."""
    if is_legacy_windows_console():
        return "\u2550\u2550"
    return "\u2501\u2501"


def timeline_tick_hover() -> str:
    """Precomposed 2-col hover tick: ``"──"``."""
    return "\u2500\u2500"


def timeline_chevron_up() -> str:
    """``"▴"`` (U+25B4) normally, ``"▲"`` (U+25B2) on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "\u25b2"
    return "\u25b4"


def timeline_chevron_down() -> str:
    """``"▾"`` (U+25BE) normally, ``"▼"`` (U+25BC) on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "\u25bc"
    return "\u25be"


# ── Selection / chevrons / disclosure ───────────────────────────────────────


def filled_dot() -> str:
    """``"●"`` (U+25CF) normally, ``"•"`` (U+2022) on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "\u2022"
    return "\u25cf"


def selection_bar() -> str:
    """``"▏"`` (U+258F) normally, ``"│"`` (U+2502) on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "\u2502"
    return "\u258f"


def chevron() -> str:
    """``"›"`` (U+203A) normally, ``">"`` on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return ">"
    return "\u203a"


def chevron_left() -> str:
    """``"‹"`` (U+2039) normally, ``"<"`` on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "<"
    return "\u2039"


def chevron_down() -> str:
    """``"⌄"`` (U+2304) normally, ``"v"`` on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "v"
    return "\u2304"


def disclosure_open() -> str:
    """``"▾"`` (U+25BE) normally, ``"v"`` on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return "v"
    return "\u25be"


def disclosure_closed() -> str:
    """``"▸"`` (U+25B8) normally, ``">"`` on legacy. Always 1 column."""
    if is_legacy_windows_console():
        return ">"
    return "\u25b8"


# ── Free-text toast scrubber ────────────────────────────────────────────────


def _to_legacy_glyphs(s: str) -> str:
    """Map tofu chrome glyphs to legacy-safe stand-ins."""
    return "".join(
        {
            "\u2713": "\u221a",  # ✓ → √
            "\u2717": "x",  # ✗ → x
            "\u26a0": "!",  # ⚠ → !
        }.get(c, c)
        for c in s
    )


def legacy_glyph_fallback(s: str) -> str:
    """Substitute legacy-unsafe chrome glyphs in free-flowing status text.

    On non-legacy platforms returns ``s`` unchanged. Maps ``✓``→``√``,
    ``✗``→``x``, ``⚠``→``!`` when legacy fallbacks are active.
    """
    if not is_legacy_windows_console():
        return s
    if not any(c in s for c in ("\u2713", "\u2717", "\u26a0")):
        return s
    return _to_legacy_glyphs(s)


__all__ = [
    "PROMPT_ARROW_WIDTH",
    "accent_bar",
    "ballot_x",
    "ballot_x_button",
    "braille_spinner_frames",
    "check_mark",
    "chevron",
    "chevron_down",
    "chevron_left",
    "clear_legacy_console_cache",
    "collapsed_accent",
    "copy_icon",
    "diamond_dotted",
    "diamond_filled",
    "diamond_filled_char",
    "diamond_hollow",
    "diamond_hollow_char",
    "disclosure_closed",
    "disclosure_open",
    "dot_spinner_frames",
    "enlarge",
    "enlarge_button",
    "filled_dot",
    "heavy_horizontal",
    "is_legacy_windows_console",
    "legacy_glyph_fallback",
    "light_horizontal",
    "monitor_icon_frames",
    "prompt_arrow",
    "record_dot",
    "selection_bar",
    "timeline_chevron_down",
    "timeline_chevron_up",
    "timeline_tick_active",
    "timeline_tick_hover",
    "token_arrow",
]
