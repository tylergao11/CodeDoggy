"""Scrollback content blocks — Grok pager paint helpers.

Ports of ``xai-grok-pager/src/scrollback/blocks/{user,thinking,markdown_content,quote_bar}.rs``.

Exports the public paint API used by scrollback / project layers.
"""

from codedoggy.tui_v2.blocks.markdown import (
    MARKDOWN_BODY_RANGE,
    QUOTE_BAR,
    render_markdown,
)
from codedoggy.tui_v2.blocks.thinking import (
    DEFAULT_TRUNCATED_LINES,
    format_elapsed_ms,
    header_line,
    paint_thinking,
    thinking_header_label,
)
from codedoggy.tui_v2.blocks.user import (
    COLLAPSED_MAX_LINES,
    PROMPT_ARROW_WIDTH,
    USER_PROMPT_BODY_RANGE,
    UserPromptBlock,
    paint_user_prompt,
    sanitize_token_ranges,
)

__all__ = [
    # user
    "COLLAPSED_MAX_LINES",
    "PROMPT_ARROW_WIDTH",
    "USER_PROMPT_BODY_RANGE",
    "UserPromptBlock",
    "paint_user_prompt",
    "sanitize_token_ranges",
    # thinking
    "DEFAULT_TRUNCATED_LINES",
    "format_elapsed_ms",
    "header_line",
    "paint_thinking",
    "thinking_header_label",
    # markdown
    "MARKDOWN_BODY_RANGE",
    "QUOTE_BAR",
    "render_markdown",
]
