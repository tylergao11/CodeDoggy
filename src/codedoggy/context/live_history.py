"""Cross-prompt live transcript (Hermes session lifetime in-process).

Each ``handle_prompt`` used to start a fresh message list. This module
seeds the next prompt from the previous loop's live window so Grok
compaction and Hermes FTS operate on a continuous session narrative.

Archive fidelity (full tool bodies before prune) is handled separately
via SessionStore incremental append at message-create time.
"""

from __future__ import annotations

from copy import deepcopy

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


def model_sample_messages(
    messages: list[Message],
    *,
    user_message_prefix: str,
) -> list[Message]:
    """Build Grok's model-facing MAIN view without mutating live history.

    The archive/live transcript intentionally retains the user's clean text.
    For the sampler only, insert Grok's session prefix after SYSTEM rows and
    wrap ordinary user requests in ``<user_query>``.  Synthetic user
    items (compaction handoffs, reminders, rewinds) keep their own framing;
    interjections already contain a canonical ``<user_query>`` block.
    """
    from codedoggy.prompt.user_message import user_query

    out = [copy_message(m) for m in messages]
    for msg in out:
        if msg.role is not Role.USER:
            continue
        content = msg.content or ""
        if not isinstance(content, str):
            content = str(content)
        if _is_synthetic_user_content(content):
            continue
        msg.content = user_query(content)

    prefix = (user_message_prefix or "").strip()
    if prefix:
        insert_at = 0
        while insert_at < len(out) and out[insert_at].role is Role.SYSTEM:
            insert_at += 1
        out.insert(insert_at, Message(role=Role.USER, content=prefix))
    return out


def _is_synthetic_user_content(content: str) -> bool:
    stripped = (content or "").lstrip()
    if not stripped:
        return True
    if "<user_query>" in stripped:
        return True
    return stripped.startswith(
        (
            "<user_info>",
            "<system-reminder>",
            "[end-of-turn notes]",
            "[CONTEXT COMPACTION",
            "[CHECKPOINT REWIND",
        )
    )


def copy_message(m: Message) -> Message:
    """Shallow copy so later prune/fold cannot mutate archived siblings."""
    return Message(
        role=m.role,
        content=m.content,
        tool_calls=deepcopy(m.tool_calls) if m.tool_calls else None,
        tool_call_id=m.tool_call_id,
        name=m.name,
        reasoning_content=m.reasoning_content,
        provider_data=deepcopy(m.provider_data) if m.provider_data else None,
    )
