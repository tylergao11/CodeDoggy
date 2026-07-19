"""Grok-style turn selection / tool-pair safe split for compaction.

Ported from ``xai-grok-compaction`` ``select.rs`` spirit:

  An assistant message with tool_calls and the following TOOL results must
  stay together. Splitting between them orphans tool results and breaks the
  chat-completions protocol (API 400).

Hermes compressor uses the same invariant when choosing protect head/tail
boundaries (``_find_safe_boundary`` style).
"""

from __future__ import annotations

from codedoggy.turn.types import Message, Role


def has_tool_requests(msg: Message) -> bool:
    return bool(msg.tool_calls)


def is_tool_result(msg: Message) -> bool:
    return msg.role is Role.TOOL


def snap_to_safe_boundary(messages: list[Message], split_idx: int) -> int:
    """Move *split_idx* forward if it would orphan tool results.

    ``split_idx`` means: compact ``messages[:split_idx]``, keep
    ``messages[split_idx:]``. If that cut falls inside a
    ``[assistant+tools, tool, tool, ...]`` run, advance past the tools.
    """
    n = len(messages)
    if split_idx <= 0:
        return 0
    if split_idx >= n:
        return n

    i = split_idx
    # If we land on a tool result, the matching assistant+tools is at or
    # before i-1; walk forward until tools stop.
    if is_tool_result(messages[i]):
        # Walk back to find whether there is an assistant-with-tools before us
        j = i - 1
        while j >= 0 and is_tool_result(messages[j]):
            j -= 1
        if j >= 0 and has_tool_requests(messages[j]):
            # Advance i past consecutive tool results
            while i < n and is_tool_result(messages[i]):
                i += 1
            return i

    # If split is right after assistant-with-tools (i points at first tool),
    # already handled above when i is tool. If i points at assistant that
    # has tools and next is tool, keep assistant with tools in the *keep*
    # side only if we would leave tools behind — i.e. if split is ON the
    # assistant, tools would be kept with it (good). If split is after
    # assistant, tools start at i — need to snap forward.
    if i > 0 and has_tool_requests(messages[i - 1]):
        while i < n and is_tool_result(messages[i]):
            i += 1
    return i


def sanitize_tool_pairs(messages: list[Message]) -> list[Message]:
    """Ensure complete tool-call / tool-result pairing for API validity.

    Rules (stricter than “any one match keeps the assistant”):
    - Every tool_call id on an assistant must have a matching TOOL result, or
      we inject a synthetic cancelled/error result.
    - Orphan TOOL results (no matching call) are dropped.
    - Assistant with *zero* matching results and no content is dropped;
      with content, tool_calls are stripped.
    """
    if not messages:
        return []

    # P0: queue results per id (FIFO) so multi-round reuse of call_N still pairs
    # correctly — never global first-wins that drops later results.
    from collections import defaultdict

    results_by_id: dict[str, list[Message]] = defaultdict(list)
    for m in messages:
        if is_tool_result(m) and m.tool_call_id:
            results_by_id[str(m.tool_call_id)].append(m)

    consumed: set[int] = set()  # id(message) of tool rows already paired
    out: list[Message] = []
    for m in messages:
        if has_tool_requests(m):
            calls = list(m.tool_calls or [])
            ids = [tc.id for tc in calls if tc.id]
            if not ids:
                out.append(
                    Message(
                        role=m.role,
                        content=m.content,
                        tool_calls=None,
                        tool_call_id=None,
                        name=m.name,
                        reasoning_content=m.reasoning_content,
                        provider_data=None,
                    )
                )
                continue
            present = any(results_by_id.get(str(i)) for i in ids)
            if not present:
                if (m.content or "").strip():
                    out.append(
                        Message(
                            role=m.role,
                            content=m.content,
                            tool_calls=None,
                            tool_call_id=None,
                            name=m.name,
                            reasoning_content=m.reasoning_content,
                            provider_data=None,
                        )
                    )
                continue
            out.append(m)
            for tc in calls:
                tid = str(tc.id) if tc.id else ""
                if not tid:
                    continue
                bag = results_by_id.get(tid) or []
                if bag:
                    tr = bag.pop(0)
                    consumed.add(id(tr))
                    out.append(tr)
                else:
                    out.append(
                        Message(
                            role=Role.TOOL,
                            content=(
                                f"Error (cancelled): tool call {tc.name!r} "
                                f"({tid}) never executed (abort/cancel mid-batch)"
                            ),
                            tool_call_id=tid,
                            name=tc.name,
                        )
                    )
        elif is_tool_result(m):
            # Orphan if never consumed by an assistant pairing above
            if id(m) in consumed:
                continue
            # Still queued for a later assistant? leave for that assistant;
            # if no later claim, drop as orphan at end of scan.
            continue
        else:
            out.append(m)
    return out


def hard_trim_safe(
    system: list[Message],
    rest: list[Message],
    *,
    over_budget,
) -> list[Message]:
    """Drop oldest messages until under budget, never orphaning tool pairs.

    ``over_budget(messages) -> bool`` is provided by caller (uses estimate).
    """
    keep = list(rest)
    while keep and over_budget(system + keep):
        if len(keep) <= 2:
            break
        # Drop a whole prefix group: if first is assistant+tools, drop it and
        # following tool results; else drop first message.
        if has_tool_requests(keep[0]):
            ids = {tc.id for tc in (keep[0].tool_calls or []) if tc.id}
            keep.pop(0)
            while keep and is_tool_result(keep[0]):
                tid = keep[0].tool_call_id
                if tid and ids and tid not in ids:
                    break
                keep.pop(0)
        else:
            keep.pop(0)
            # If next are orphan tools, drop them too
            while keep and is_tool_result(keep[0]):
                # orphan if no remaining assistant claims them
                tid = keep[0].tool_call_id
                claimed = any(
                    tc.id == tid
                    for m in keep
                    if m.tool_calls
                    for tc in m.tool_calls
                )
                if claimed:
                    break
                keep.pop(0)
    return sanitize_tool_pairs(system + keep)


def plan_fold_regions(
    rest: list[Message],
    *,
    protect_first_n: int,
    keep_recent: int,
) -> tuple[list[Message], list[Message], list[Message]]:
    """Split non-system *rest* into (head, middle, tail) with safe boundaries.

    - head: first ``protect_first_n`` non-system messages (Hermes)
    - tail: last ``keep_recent`` messages, snapped so tool pairs stay intact
    - middle: everything between (may be empty)

    Returns empty middle when not worth folding.
    """
    if not rest:
        return [], [], []

    protect_first_n = max(0, int(protect_first_n))
    keep_recent = max(2, int(keep_recent))

    # Head: first N messages (do not split tool pairs inside head — extend)
    head_end = min(protect_first_n, len(rest))
    if head_end > 0:
        head_end = snap_to_safe_boundary(rest, head_end)
    head = rest[:head_end]
    body = rest[head_end:]

    if len(body) <= keep_recent + 1:
        return head, [], body

    # Candidate: tail = last keep_recent
    tail_start = len(body) - keep_recent
    # Snap so we don't cut tool pairs: treat tail_start as split between
    # middle and tail (compact middle = body[:tail_start])
    # If tail_start falls mid tool-run, move it forward (shrink middle).
    safe_start = snap_to_safe_boundary(body, tail_start)
    # Also: if safe_start is 0, nothing to fold
    if safe_start <= 0:
        return head, [], body
    if safe_start >= len(body):
        # Everything wanted for middle snapped away
        return head, [], body

    middle = body[:safe_start]
    tail = body[safe_start:]
    if not middle:
        return head, [], body
    return head, middle, tail
