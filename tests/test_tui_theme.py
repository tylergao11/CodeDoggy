"""TUI theme inventory + fresh default."""

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


def test_fresh_detail_separator_is_visible() -> None:
    """Hairline must be mid-gray, not near-invisible on the canvas."""
    fresh = style_dict("fresh")
    assert "#5c5754" in fresh["detail.separator"]
    assert "#e8e2dc" in fresh["detail.text"]
    assert "#f0ebe6" in fresh["detail.md.h1"]
    # Still quiet (no bold shout) on ask / md / code.
    assert "bold" not in fresh["ask.header"]
    assert "bold" not in fresh["detail.md.h1"]
    assert "bold" not in fresh["detail.code.kw"]
    assert "#d4a0b8" in fresh["brand"]
    assert "#7eb8c9" in fresh["task.spine.active"]


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
    assert "#7dcfff" in dark["task.spine.active"]


def test_themes_share_class_keys() -> None:
    keys = set(THEMES["fresh"])
    assert set(THEMES["groknight"]) == keys
    assert "detail.text" in keys
    assert "prompt.border.focus" in keys


def test_build_style_constructs() -> None:
    style = build_style("fresh")
    assert style is not None
