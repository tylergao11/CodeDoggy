"""Grok source-level system prompts + CodeDoggy product appendix."""

from __future__ import annotations

from codedoggy.bootstrap import _default_system_prompt
from codedoggy.prompt.grok_system import (
    COMPACT_SYSTEM_PROMPT,
    build_main_system_prompt,
    build_subagent_system_prompt,
    codedoggy_product_appendix,
    render_grok_base_prompt,
    render_grok_subagent_base,
)


def test_grok_base_has_action_safety_and_tool_calling() -> None:
    text = render_grok_base_prompt()
    assert "<action_safety>" in text
    assert "Confirming is cheap" in text
    assert "<tool_calling>" in text
    assert "read_file" in text
    assert "search_replace" in text
    assert "<output_efficiency>" in text
    assert "<formatting>" in text
    assert "GitHub-flavored markdown" in text


def test_grok_base_non_interactive_wording() -> None:
    text = render_grok_base_prompt(is_non_interactive=True)
    assert "autonomous agent" in text
    assert "interactive CLI" not in text


def test_compact_prompt_matches_grok() -> None:
    assert "AI coding agent" in COMPACT_SYSTEM_PROMPT
    assert "<user_query>" in COMPACT_SYSTEM_PROMPT


def test_main_prompt_is_grok_plus_product() -> None:
    text = build_main_system_prompt(None)
    # Grok structure
    assert "<action_safety>" in text
    assert "read_file" in text
    # Product appendix (CodeDoggy)
    assert "<codedoggy_product>" in text
    assert "parallel_tasks" in text
    assert "nothing auto-fans-out" in text or "auto-fans-out" in text
    assert "code_nav" in text


def test_default_system_prompt_wrapper() -> None:
    text = _default_system_prompt("fix login")
    assert "Session goal: fix login" in text
    assert "<action_safety>" in text
    assert "<codedoggy_product>" in text


def test_product_appendix_separate_from_grok_sections() -> None:
    product = codedoggy_product_appendix()
    assert product.startswith("<codedoggy_product>")
    assert "<action_safety>" not in product


def test_subagent_base_parallelize_tool_calls() -> None:
    text = render_grok_subagent_base(working_directory="/tmp/ws")
    assert "Parallelize independent tool calls" in text
    assert "<project_instructions_spec>" in text
    assert "AGENTS.md" in text
    assert "Workspace Path: /tmp/ws" in text or "Workspace Path:" in text
    assert "<memory>" in text
    # Hermes memory surface — not Grok memory_search/get read spam
    assert "session_search" in text
    assert "memory_search" not in text
    assert "memory_get" not in text


def test_subagent_with_role_instructions() -> None:
    text = build_subagent_system_prompt(
        "Role: explore only.",
        cwd=".",
    )
    assert "Parallelize independent tool calls" in text
    assert "<role-instructions>" in text
    assert "Role: explore only." in text

