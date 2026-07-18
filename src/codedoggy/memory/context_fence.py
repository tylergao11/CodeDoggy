"""Hermes memory-context fencing — port of agent/memory_manager.py helpers.

Source of truth: C:\\Ai\\hermes-agent\\agent\\memory_manager.py
  - sanitize_context
  - build_memory_context_block
  - fence tags: <memory-context> … </memory-context>

Do not invent alternate tags. Prefetch is injected into the *current user
message* at sample time only (conversation_loop.py); never into SYSTEM and
never persisted into the live/archive transcript.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Ported from hermes-agent/agent/memory_manager.py
_FENCE_TAG_RE = re.compile(r"</?\s*memory-context\s*>", re.IGNORECASE)
_INTERNAL_CONTEXT_RE = re.compile(
    r"<\s*memory-context\s*>[\s\S]*?</\s*memory-context\s*>",
    re.IGNORECASE,
)
_INTERNAL_NOTE_RE = re.compile(
    r"\[System note:\s*The following is recalled memory context,\s*NOT new user input\.\s*"
    r"Treat as (?:informational background data|authoritative reference data[^\]]*)\.\]\s*",
    re.IGNORECASE,
)

_HERMES_SYSTEM_NOTE = (
    "[System note: The following is recalled memory context, "
    "NOT new user input. Treat as authoritative reference data — "
    "this is the agent's persistent memory and should inform all responses.]"
)


def sanitize_context(text: str) -> str:
    """Strip fence tags / injected blocks / system notes from provider output.

    Hermes: providers must return raw recall text, not pre-wrapped fences.
    """
    if not text:
        return text
    text = _INTERNAL_CONTEXT_RE.sub("", text)
    text = _INTERNAL_NOTE_RE.sub("", text)
    text = _FENCE_TAG_RE.sub("", text)
    return text


def build_memory_context_block(raw_context: str) -> str:
    """Wrap prefetched memory in Hermes ``<memory-context>`` fence.

    Exact shape from hermes-agent MemoryManager.build_memory_context_block.
    """
    if not raw_context or not str(raw_context).strip():
        return ""
    clean = sanitize_context(str(raw_context))
    if clean != str(raw_context):
        logger.warning("memory provider returned pre-wrapped context; stripped")
    if not clean.strip():
        return ""
    return (
        "<memory-context>\n"
        f"{_HERMES_SYSTEM_NOTE}\n\n"
        f"{clean.strip()}\n"
        "</memory-context>"
    )


def strip_memory_context_from_messages(messages: list[Any]) -> list[Any]:
    """Remove any accidental persisted memory-context spans from user contents."""
    from codedoggy.context.live_history import copy_message
    from codedoggy.turn.types import Role

    out: list[Any] = []
    for m in messages:
        if getattr(m, "role", None) is Role.USER and isinstance(m.content, str):
            cleaned = sanitize_context(m.content).strip()
            if cleaned != (m.content or "").strip():
                cm = copy_message(m)
                cm.content = cleaned
                out.append(cm)
                continue
        out.append(m)
    return out


def messages_with_ephemeral_memory(
    messages: list[Any],
    fenced_block: str | None,
) -> list[Any]:
    """Hermes conversation_loop: append fence to *current turn user* for API only.

    Does not mutate the input list. Finds the last USER message and appends
    the fenced block to its content (same as api_msg content injection).
    """
    if not fenced_block or not str(fenced_block).strip():
        return messages
    from codedoggy.context.live_history import copy_message
    from codedoggy.turn.types import Role

    out = [copy_message(m) for m in messages]
    for i in range(len(out) - 1, -1, -1):
        if out[i].role is Role.USER:
            base = out[i].content or ""
            if not isinstance(base, str):
                base = str(base)
            # Avoid double-inject if already present
            if "<memory-context>" in base:
                return out
            out[i].content = base + "\n\n" + str(fenced_block).strip()
            return out
    return out
