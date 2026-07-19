"""Tool-result pruning — Grok prune_retained + size soft-cap."""

from __future__ import annotations

from codedoggy.context.budget import ContextBudget
from codedoggy.turn.types import Message, Role

_PRUNE_MARK = (
    "\n… [tool output pruned for live context; "
    "re-read the file/tool if you need the full body] …\n"
)
_RETAINED_CLEARED = (
    "[old tool result cleared by retain-prune to free context; "
    "re-run the tool or re-read sources if needed]"
)


def prune_oversized_tool_results(
    messages: list[Message],
    budget: ContextBudget,
) -> tuple[list[Message], int]:
    """Clip individual tool bodies exceeding tool_result_max_chars."""
    cap = budget.tool_result_max_chars
    if cap <= 0:
        return list(messages), 0
    out: list[Message] = []
    pruned = 0
    for m in messages:
        if m.role is Role.TOOL and m.content and len(m.content) > cap:
            body = m.content
            mark = _PRUNE_MARK
            room = max(32, cap - len(mark))
            head = room // 2
            tail = room - head
            if len(body) > room:
                new_content = body[:head] + mark + body[-tail:]
            else:
                new_content = body
            if len(new_content) > cap:
                new_content = new_content[:cap]
            out.append(_copy_msg(m, new_content))
            pruned += 1
        else:
            out.append(m)
    return out, pruned


def prune_retained_tool_results(
    messages: list[Message],
    *,
    retain_recent_tool_messages: int = 6,
) -> tuple[list[Message], int]:
    """Grok-style: clear *bodies* of old tool results; keep recent ones.

    Counts TOOL messages from the end; older ones become a short placeholder.
    """
    if retain_recent_tool_messages < 0:
        return list(messages), 0
    tool_indices = [i for i, m in enumerate(messages) if m.role is Role.TOOL]
    if len(tool_indices) <= retain_recent_tool_messages:
        return list(messages), 0
    keep = set(tool_indices[-retain_recent_tool_messages:])
    out: list[Message] = []
    cleared = 0
    for i, m in enumerate(messages):
        if i in tool_indices and i not in keep:
            content = m.content or ""
            if content == _RETAINED_CLEARED or content.startswith(_RETAINED_CLEARED):
                out.append(m)
                continue
            out.append(_copy_msg(m, _RETAINED_CLEARED))
            cleared += 1
        else:
            out.append(m)
    return out, cleared


def _copy_msg(m: Message, content: str) -> Message:
    return Message(
        role=m.role,
        content=content,
        tool_calls=m.tool_calls,
        tool_call_id=m.tool_call_id,
        name=m.name,
        reasoning_content=m.reasoning_content,
        provider_data=dict(m.provider_data) if m.provider_data else None,
    )
