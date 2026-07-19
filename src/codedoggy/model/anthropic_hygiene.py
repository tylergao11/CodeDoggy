"""Anthropic message hygiene — ported from Hermes anthropic_adapter.

Order after conversion (must not skip):
  1. strip orphaned tool_use / tool_result pairs (adjacent match only)
  2. merge consecutive same-role messages
  3. manage thinking signatures (native vs third-party)

Without this: intermittent HTTP 400 on multi-turn tool + thinking.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

_THINKING = frozenset({"thinking", "redacted_thinking"})

# Fields allowed on replayed content blocks (Messages INPUT schema)
_TEXT_KEYS = frozenset({"type", "text", "cache_control", "citations"})
_THINKING_KEYS = frozenset({"type", "thinking", "signature", "cache_control"})
_REDACTED_KEYS = frozenset({"type", "data", "cache_control"})
_TOOL_USE_KEYS = frozenset({"type", "id", "name", "input", "cache_control"})
_TOOL_RESULT_KEYS = frozenset(
    {"type", "tool_use_id", "content", "is_error", "cache_control"}
)


def is_third_party_anthropic(base_url: str | None) -> bool:
    if not base_url:
        return False
    host = (urlparse(str(base_url).strip()).hostname or "").lower()
    if not host:
        return False
    return "anthropic.com" not in host


def sanitize_replay_block(block: Any) -> dict[str, Any] | None:
    """Drop SDK output-only fields that Anthropic INPUT rejects."""
    if not isinstance(block, dict):
        return None
    btype = block.get("type")
    if btype == "text":
        allowed = _TEXT_KEYS
    elif btype == "thinking":
        allowed = _THINKING_KEYS
    elif btype == "redacted_thinking":
        allowed = _REDACTED_KEYS
    elif btype == "tool_use":
        allowed = _TOOL_USE_KEYS
    elif btype == "tool_result":
        allowed = _TOOL_RESULT_KEYS
    else:
        # unknown — pass type-only safe subset
        return {"type": str(btype or "text"), "text": str(block.get("text") or "")}
    return {k: v for k, v in block.items() if k in allowed}


def strip_orphaned_tool_blocks(result: list[dict[str, Any]]) -> None:
    """Require tool_result in the *immediately following* user message."""
    for i, m in enumerate(result):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        tool_use_ids = {
            b.get("id")
            for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_use"
        }
        if not tool_use_ids:
            continue
        adjacent: set[Any] = set()
        if i + 1 < len(result):
            nxt = result[i + 1]
            if nxt.get("role") == "user" and isinstance(nxt.get("content"), list):
                for block in nxt["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        adjacent.add(block.get("tool_use_id"))
        orphaned = tool_use_ids - adjacent
        if not orphaned:
            continue
        kept = [
            b
            for b in m["content"]
            if not (
                isinstance(b, dict)
                and b.get("type") == "tool_use"
                and b.get("id") in orphaned
            )
        ]
        if len(kept) != len(m["content"]) and any(
            isinstance(b, dict) and b.get("type") in _THINKING for b in m["content"]
        ):
            m["_thinking_signature_invalidated"] = True
        m["content"] = kept if kept else [{"type": "text", "text": "(tool call removed)"}]

    surviving: set[Any] = set()
    for m in result:
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    surviving.add(block.get("id"))

    for m in result:
        if m.get("role") != "user" or not isinstance(m.get("content"), list):
            continue
        new_content = [
            b
            for b in m["content"]
            if not (isinstance(b, dict) and b.get("type") == "tool_result")
            or b.get("tool_use_id") in surviving
        ]
        if len(new_content) != len(m["content"]):
            m["content"] = (
                new_content if new_content else [{"type": "text", "text": "(tool result removed)"}]
            )


def merge_consecutive_roles(result: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixed: list[dict[str, Any]] = []
    for m in result:
        if fixed and fixed[-1]["role"] == m["role"]:
            if m["role"] == "user":
                fixed[-1]["content"] = _merge_content(
                    fixed[-1]["content"], m.get("content")
                )
            else:
                if m.get("_thinking_signature_invalidated"):
                    fixed[-1]["_thinking_signature_invalidated"] = True
                # Drop thinking from *second* assistant — wrong turn boundary
                curr = m.get("content")
                if isinstance(curr, list):
                    curr = [
                        b
                        for b in curr
                        if not (isinstance(b, dict) and b.get("type") in _THINKING)
                    ]
                fixed[-1]["content"] = _merge_content(fixed[-1]["content"], curr)
        else:
            fixed.append(m)
    return fixed


def _merge_content(a: Any, b: Any) -> Any:
    if isinstance(a, str) and isinstance(b, str):
        return a + "\n" + b
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    if isinstance(a, str):
        a = [{"type": "text", "text": a}]
    if isinstance(b, str):
        b = [{"type": "text", "text": b}]
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    return a


def manage_thinking_signatures(
    result: list[dict[str, Any]],
    *,
    base_url: str | None,
    preserve_unsigned: bool = False,
) -> None:
    """Strip/preserve thinking blocks per Hermes endpoint rules."""
    third_party = is_third_party_anthropic(base_url)
    last_assistant: int | None = None
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "assistant":
            last_assistant = i
            break

    for idx, m in enumerate(result):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        content = m["content"]

        if preserve_unsigned:
            # Kimi/DeepSeek anthropic: strip signed only
            new_c = []
            for b in content:
                if not isinstance(b, dict) or b.get("type") not in _THINKING:
                    new_c.append(b)
                    continue
                if b.get("signature") or b.get("data"):
                    continue
                new_c.append(b)
            m["content"] = new_c or [{"type": "text", "text": "(empty)"}]
            continue

        invalidated = bool(m.get("_thinking_signature_invalidated"))
        if third_party or idx != last_assistant or invalidated:
            # Strip all thinking (signatures invalid or third-party)
            stripped = [
                b
                for b in content
                if not (isinstance(b, dict) and b.get("type") in _THINKING)
            ]
            # Demote unsigned thinking text so reasoning isn't lost
            if idx == last_assistant and not third_party and invalidated:
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "thinking":
                        th = b.get("thinking")
                        if isinstance(th, str) and th.strip() and not b.get("signature"):
                            stripped.insert(0, {"type": "text", "text": th})
            m["content"] = stripped or [{"type": "text", "text": "(thinking elided)"}]
            m.pop("_thinking_signature_invalidated", None)
            continue

        # Latest assistant on native Anthropic: keep signed; unsigned → text
        new_c = []
        for b in content:
            if not isinstance(b, dict):
                new_c.append(b)
                continue
            if b.get("type") not in _THINKING:
                new_c.append(b)
                continue
            if b.get("type") == "thinking" and not b.get("signature"):
                th = b.get("thinking")
                if isinstance(th, str) and th.strip():
                    new_c.append({"type": "text", "text": th})
                continue
            new_c.append(b)
        m["content"] = new_c or [{"type": "text", "text": "(empty)"}]


def finalize_anthropic_messages(
    result: list[dict[str, Any]],
    *,
    base_url: str | None = None,
    preserve_unsigned_thinking: bool = False,
) -> list[dict[str, Any]]:
    """Full Hermes post-convert pipeline."""
    # sanitize all blocks first
    for m in result:
        content = m.get("content")
        if isinstance(content, list):
            cleaned = []
            for b in content:
                s = sanitize_replay_block(b)
                if s is not None:
                    cleaned.append(s)
            m["content"] = cleaned or [{"type": "text", "text": ""}]
    strip_orphaned_tool_blocks(result)
    result = merge_consecutive_roles(result)
    manage_thinking_signatures(
        result,
        base_url=base_url,
        preserve_unsigned=preserve_unsigned_thinking,
    )
    # drop internal flags
    for m in result:
        m.pop("_thinking_signature_invalidated", None)
    return result
