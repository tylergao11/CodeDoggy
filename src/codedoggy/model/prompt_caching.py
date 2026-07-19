"""Anthropic-style prompt caching (Hermes ``system_and_3`` layout).

Places up to 4 ``cache_control`` breakpoints:
  1. system prompt (stable prefix — critical for hit rate)
  2–4. last 3 non-system messages that can carry a marker

If breakpoints land on empty assistant tool-call turns or empty tool
results (envelope layout), providers ignore them → **0% cache hits** and
full re-billing every turn.  Carrier rules follow Hermes ``prompt_caching.py``.

DeepSeek disk KV cache is **automatic prefix** (no markers); keep system
static and inject ephemeral context only into the *current user* message.
"""

from __future__ import annotations

import copy
from typing import Any


def apply_anthropic_cache_control(
    api_messages: list[dict[str, Any]],
    *,
    cache_ttl: str = "5m",
    native_anthropic: bool = True,
) -> list[dict[str, Any]]:
    """Deep-copy *api_messages* and inject cache_control breakpoints."""
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker = _build_marker(cache_ttl)
    used = 0

    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        used += 1

    remaining = 4 - used
    carriers = [
        i
        for i in range(len(messages))
        if messages[i].get("role") != "system"
        and _can_carry_marker(messages[i], native_anthropic=native_anthropic)
    ]
    for idx in carriers[-remaining:]:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages


def _build_marker(ttl: str) -> dict[str, str]:
    marker: dict[str, str] = {"type": "ephemeral"}
    if ttl == "1h":
        marker["ttl"] = "1h"
    return marker


def _can_carry_marker(msg: dict[str, Any], *, native_anthropic: bool) -> bool:
    if native_anthropic:
        return True
    content = msg.get("content")
    if content is None or content == "":
        return False
    if isinstance(content, list):
        return bool(content) and isinstance(content[-1], dict)
    return isinstance(content, str)


def _apply_cache_marker(
    msg: dict[str, Any],
    cache_marker: dict[str, str],
    *,
    native_anthropic: bool,
) -> None:
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool" and native_anthropic:
        msg["cache_control"] = dict(cache_marker)
        return

    if content is None or content == "":
        if role == "tool" and not native_anthropic:
            return
        if role == "assistant" and not native_anthropic:
            return
        msg["cache_control"] = dict(cache_marker)
        return

    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": dict(cache_marker)}
        ]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = dict(cache_marker)
