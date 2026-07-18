"""System prompts: Grok source-level base + CodeDoggy product appendix."""

from codedoggy.prompt.grok_system import (
    COMPACT_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT_LABEL,
    build_main_system_prompt,
    build_subagent_system_prompt,
    codedoggy_product_appendix,
    render_grok_base_prompt,
    render_grok_subagent_base,
)

__all__ = [
    "COMPACT_SYSTEM_PROMPT",
    "DEFAULT_SYSTEM_PROMPT_LABEL",
    "build_main_system_prompt",
    "build_subagent_system_prompt",
    "codedoggy_product_appendix",
    "render_grok_base_prompt",
    "render_grok_subagent_base",
]
