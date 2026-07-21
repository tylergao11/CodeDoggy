"""TUI theme inventory + GrokBuild black default."""

from __future__ import annotations

from codedoggy.tui.theme import (
    DEFAULT_THEME,
    THEMES,
    build_style,
    resolve_theme_name,
    style_dict,
)


def test_default_theme_is_fresh() -> None:
    assert DEFAULT_THEME == "fresh"
    assert resolve_theme_name({}) == "fresh"


def test_env_theme_aliases() -> None:
    assert resolve_theme_name({"CODEDOGGY_THEME": "groknight"}) == "groknight"
    assert resolve_theme_name({"CODEDOGGY_THEME": "dark"}) == "dark"
    assert resolve_theme_name({"CODEDOGGY_THEME": "quiet"}) == "quiet"
    assert resolve_theme_name({"CODEDOGGY_THEME": "nope"}) == "fresh"


def test_fresh_is_pure_black_not_ink_blue() -> None:
    """GrokBuild canvas: neutral black, soft body text, no neon pink/cyan."""
    fresh = style_dict("fresh")
    # Pure black family — never ink-blue #14131a.
    assert "#0c0c0c" in fresh["root"]
    assert "#0c0c0c" in fresh["detail.text"]
    assert "#14131a" not in fresh["root"]
    assert "#14131a" not in fresh["detail.text"]
    # Soft body (not pure white glare).
    assert "#d4d4d4" in fresh["detail.text"]
    # Quiet hairline separator.
    assert "#2e2e2e" in fresh["detail.separator"]
    # Soft bright pink brand (detail legibility), not neon cyan.
    assert "#e0b0c4" in fresh["brand"]
    assert "#e0b0c4" in fresh["detail.actor.assistant"]
    assert "#7dcfff" not in fresh["task.marker.active"]
    assert "#7eb8c9" not in fresh["task.marker.active"]
    # No bold shout on ask / md / code (reading-first).
    assert "bold" not in fresh["ask.header"]
    assert "bold" not in fresh["detail.md.h1"]
    assert "bold" not in fresh["detail.code.kw"]
    # Thinking is dim, not pink.
    assert "#6e6e6e" in fresh["detail.thinking.header"]
    assert "#9a9a9a" in fresh["detail.thinking.body"]


def test_section_break_uses_air_not_box_rails() -> None:
    from codedoggy.tui.agent_detail import _section_break

    fr = _section_break(40)
    text = "".join(p[1] for p in fr)
    assert "┈" not in text
    assert "╾" not in text
    assert "─" in text
    # Leading/trailing blank lines for reading rhythm.
    assert text.startswith("\n")
    assert text.count("\n") >= 3


def test_groknight_keeps_legacy_bold() -> None:
    dark = style_dict("groknight")
    assert "bold" in dark["ask.header"]
    assert "bold" in dark["detail.md.h1"]
    # Still pure-black family after the calm pass.
    assert "#141414" in dark["root"]


def test_themes_share_class_keys() -> None:
    keys = set(THEMES["fresh"])
    assert set(THEMES["groknight"]) == keys
    assert "detail.text" in keys
    assert "prompt.border.focus" in keys
    assert "task.card.selected" in keys
    assert "task.card.border.selected" in keys
    assert "report.stream" in keys


def test_fresh_card_surface_uses_black_canvas() -> None:
    fresh = style_dict("fresh")
    assert "#0c0c0c" in fresh["task.card"]
    assert "#161616" in fresh["task.card.selected"]
    assert "#c498b0" in fresh["task.card.border.selected"] or "#e0b0c4" in fresh[
        "task.card.border.selected"
    ]


def test_doggy_splash_void_is_theme_black() -> None:
    """Couple art background matches TUI black — no gray plate."""
    from codedoggy.tui.app import _DOGGY_ART_PALETTE

    assert _DOGGY_ART_PALETTE["."] == "#0c0c0c"


def test_build_style_constructs() -> None:
    style = build_style("fresh")
    assert style is not None
