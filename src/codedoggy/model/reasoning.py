"""Reasoning / thinking-field handling for multi-provider history replay.

Mirrors the essential contract from Hermes ``copy_reasoning_content_for_api``:

* **require** (DeepSeek thinking / reasoner): every assistant turn on the wire
  needs a string ``reasoning_content``.  Empty string is rejected by DeepSeek
  V4 Pro — use a single space pad.  Real CoT is preserved when present.
* **strip** (strict OpenAI-compat): never send ``reasoning_content``; many
  gateways 400/422 on unknown keys.
* **pass**: keep a non-empty string if already present; otherwise omit.
"""

from __future__ import annotations

from typing import Any

from codedoggy.model.profile import (
    REASONING_PASS,
    REASONING_REQUIRE,
    REASONING_STRIP,
)

# DeepSeek V4 Pro rejects empty-string reasoning_content in thinking mode.
_PAD = " "


def apply_reasoning_echo(
    messages: list[dict[str, Any]],
    *,
    policy: str,
) -> list[dict[str, Any]]:
    """Return a shallow-copied message list with reasoning fields reconciled."""
    if policy == REASONING_STRIP:
        return [_strip_reasoning(m) for m in messages]
    if policy == REASONING_REQUIRE:
        return [_require_reasoning(m) for m in messages]
    if policy == REASONING_PASS:
        return [_pass_reasoning(m) for m in messages]
    # Unknown → safe default (strip)
    return [_strip_reasoning(m) for m in messages]


def _strip_reasoning(msg: dict[str, Any]) -> dict[str, Any]:
    out = dict(msg)
    out.pop("reasoning_content", None)
    out.pop("reasoning", None)
    return out


def _pass_reasoning(msg: dict[str, Any]) -> dict[str, Any]:
    out = dict(msg)
    if out.get("role") != "assistant":
        out.pop("reasoning_content", None)
        return out
    existing = out.get("reasoning_content")
    if isinstance(existing, str) and existing:
        return out
    # Promote internal 'reasoning' only when non-empty
    alt = out.get("reasoning")
    if isinstance(alt, str) and alt.strip():
        out["reasoning_content"] = alt
    else:
        out.pop("reasoning_content", None)
    out.pop("reasoning", None)
    return out


def _require_reasoning(msg: dict[str, Any]) -> dict[str, Any]:
    """DeepSeek / thinking-mode echo rules for one message."""
    out = dict(msg)
    if out.get("role") != "assistant":
        out.pop("reasoning_content", None)
        out.pop("reasoning", None)
        return out

    existing = out.get("reasoning_content")
    if isinstance(existing, str):
        # "" → pad; non-empty keep
        out["reasoning_content"] = existing if existing else _PAD
        out.pop("reasoning", None)
        return out

    # Promote 'reasoning' when present
    alt = out.get("reasoning")
    if isinstance(alt, str) and alt:
        # For tool-call turns with foreign CoT, Hermes pads " " to avoid leaking
        # another provider's chain-of-thought; we keep real content when it
        # looks like same-provider history (no special marker).  Pad only when
        # empty after strip.
        out["reasoning_content"] = alt if alt.strip() else _PAD
        out.pop("reasoning", None)
        return out

    # Thinking mode requires the key on every assistant turn.
    out["reasoning_content"] = _PAD
    out.pop("reasoning", None)
    return out


def extract_reasoning_from_message(msg: dict[str, Any] | Any) -> str | None:
    """Pull reasoning text from a response message dict or SDK object."""
    if isinstance(msg, dict):
        for key in ("reasoning_content", "reasoning"):
            val = msg.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return None
    for key in ("reasoning_content", "reasoning"):
        val = getattr(msg, key, None)
        if isinstance(val, str) and val.strip():
            return val
    return None
