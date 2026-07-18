"""Tool-result pruning — Grok prune_retained + size soft-cap.

Shadow P0 footnotes must survive prune/retain so soft-interrupts
remain visible to the next sample. Markers shared with ``audit.format``.
"""

from __future__ import annotations

from codedoggy.audit.format import (
    AUDIT_P0_END,
    AUDIT_P0_START,
    LEGACY_P0_END,
    LEGACY_P0_START,
)
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

# Dedicated reinject note — must not be confused with REFERENCE ONLY fold text.
P0_REINJECT_PREFIX = (
    "[shadow — open P0 still unresolved after context compaction]"
)
# Fold summary marker (see context.compactor.COMPACTION_PREFIX).
_COMPACTION_SUMMARY_MARK = "CONTEXT COMPACTION — REFERENCE ONLY"


def has_audit_p0_footer(content: str | None) -> bool:
    return bool(content) and (
        AUDIT_P0_START in content or LEGACY_P0_START in content
    )


def extract_audit_p0_footer(content: str | None) -> str | None:
    """Return the last P0 red-card block in *content*, or None."""
    if not content:
        return None
    # Prefer new shadow markers; fall back to legacy resident-audit markers
    for start_m, end_m in (
        (AUDIT_P0_START, AUDIT_P0_END),
        (LEGACY_P0_START, LEGACY_P0_END),
    ):
        if start_m not in content:
            continue
        start = content.rfind(start_m)
        if start < 0:
            continue
        end = content.find(end_m, start)
        if end < 0:
            return content[start:].strip()
        return content[start : end + len(end_m)].strip()
    return None


def strip_audit_p0_footers(content: str | None) -> str:
    """Remove P0 red-card blocks so fold sketches cannot swallow them."""
    if not content:
        return content or ""
    out = content
    for start_m, end_m in (
        (AUDIT_P0_START, AUDIT_P0_END),
        (LEGACY_P0_START, LEGACY_P0_END),
    ):
        while start_m in out:
            start = out.find(start_m)
            end = out.find(end_m, start)
            if end < 0:
                out = out[:start].rstrip()
                break
            out = (out[:start] + out[end + len(end_m) :]).strip()
    return out


def prune_oversized_tool_results(
    messages: list[Message],
    budget: ContextBudget,
) -> tuple[list[Message], int]:
    """Clip individual tool bodies exceeding tool_result_max_chars.

    When a body carries a resident-audit P0 footer, the footer is always
    kept (appended after the size clip) so soft-interrupts survive.
    """
    cap = budget.tool_result_max_chars
    if cap <= 0:
        return list(messages), 0
    out: list[Message] = []
    pruned = 0
    for m in messages:
        if m.role is Role.TOOL and m.content and len(m.content) > cap:
            footer = extract_audit_p0_footer(m.content)
            body = m.content
            if footer:
                # Clip the non-footer body only.
                idx = body.rfind(footer)
                body = body[:idx].rstrip() if idx >= 0 else body
            mark = _PRUNE_MARK
            if footer:
                room = max(32, cap - len(mark) - len(footer) - 2)
            else:
                room = max(32, cap - len(mark))
            head = room // 2
            tail = room - head
            if len(body) > room:
                new_content = body[:head] + mark + body[-tail:]
            else:
                new_content = body
            if footer:
                new_content = f"{new_content.rstrip()}\n\n{footer}"
            if len(new_content) > cap and not footer:
                new_content = new_content[:cap]
            # P0 may slightly exceed cap — intentional (audit > budget soft-cap).
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
    P0 footers on cleared tools are preserved so the model still sees the
    soft interrupt after retain-prune.
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
            footer = extract_audit_p0_footer(content)
            if footer:
                out.append(_copy_msg(m, f"{_RETAINED_CLEARED}\n\n{footer}"))
            else:
                out.append(_copy_msg(m, _RETAINED_CLEARED))
            cleared += 1
        else:
            out.append(m)
    return out, cleared


def collect_p0_footers(messages: list[Message]) -> list[str]:
    """Unique P0 footers in message order (for fold re-inject)."""
    seen: set[str] = set()
    out: list[str] = []
    for m in messages:
        footer = extract_audit_p0_footer(m.content)
        if footer and footer not in seen:
            seen.add(footer)
            out.append(footer)
    return out


def _is_binding_p0_carrier(m: Message) -> bool:
    """True if this message can still act as a soft-interrupt (not fold summary)."""
    content = m.content or ""
    if AUDIT_P0_START not in content:
        return False
    # Fold middle becomes a USER message with COMPACTION_PREFIX — reference only.
    if _COMPACTION_SUMMARY_MARK in content:
        return False
    # Dedicated reinject notes count as binding.
    if P0_REINJECT_PREFIX in content:
        return True
    # TOOL observations (or other non-summary messages) with a full footer.
    return extract_audit_p0_footer(content) is not None


def reinject_missing_p0(
    messages: list[Message],
    footers: list[str],
) -> list[Message]:
    """If *footers* are not still binding after compact, append a USER note.

    Presence inside a REFERENCE ONLY fold summary does **not** count — that
    text is explicitly non-actionable and must not suppress reinjection.
    """
    if not footers:
        return messages
    binding_text = "\n".join(
        (m.content or "") for m in messages if _is_binding_p0_carrier(m)
    )
    missing = [f for f in footers if f not in binding_text]
    if not missing:
        return messages
    # De-dupe if a previous reinject already listed the same footer partially.
    note = (
        f"{P0_REINJECT_PREFIX}\n"
        "Address these before continuing in the same direction:\n\n"
        + "\n\n".join(missing)
    )
    return list(messages) + [Message(role=Role.USER, content=note)]


def _copy_msg(m: Message, content: str) -> Message:
    return Message(
        role=m.role,
        content=content,
        tool_calls=m.tool_calls,
        tool_call_id=m.tool_call_id,
        name=m.name,
    )
