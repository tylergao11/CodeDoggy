"""Cross-prompt live transcript (Hermes session lifetime in-process).

Each ``handle_prompt`` used to start a fresh message list. This module
seeds the next prompt from the previous loop's live window so Grok
compaction and Hermes FTS operate on a continuous session narrative.

Archive fidelity (full tool bodies before prune) is handled separately
via SessionStore incremental append at message-create time.
"""

from __future__ import annotations

from codedoggy.turn.types import Message, Role


def strip_system_messages(messages: list[Message]) -> list[Message]:
    """Drop SYSTEM rows — next turn rebuilds system (+ MEMORY) from scratch."""
    return [m for m in messages if m.role is not Role.SYSTEM]


def seed_messages(
    *,
    system_prompt: str | None,
    user_text: str,
    prior_messages: list[Message] | None = None,
) -> list[Message]:
    """Build the opening transcript for one prompt (Grok continuous session).

    Order:
      [SYSTEM(current)] + prior non-system (tool-pair sanitized) + USER(new prompt)
    """
    from codedoggy.context.select import sanitize_tool_pairs

    out: list[Message] = []
    if system_prompt:
        out.append(Message(role=Role.SYSTEM, content=system_prompt))
    if prior_messages:
        prior = strip_system_messages(prior_messages)
        # Grok: never carry orphan tool_result / broken pairs into next sample
        prior = sanitize_tool_pairs(prior)
        out.extend(prior)
    out.append(Message(role=Role.USER, content=user_text))
    return out


def copy_message(m: Message) -> Message:
    """Shallow copy so later prune/fold cannot mutate archived siblings."""
    return Message(
        role=m.role,
        content=m.content,
        tool_calls=list(m.tool_calls) if m.tool_calls else None,
        tool_call_id=m.tool_call_id,
        name=m.name,
    )
