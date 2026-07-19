"""Cross-provider context contracts (Hermes-aligned).

Critical rules (easy to get wrong → 400 or 0% cache):

1. **Reasoning / thinking replay**
   - DeepSeek thinking: every assistant wire message needs
     ``reasoning_content`` (empty string rejected → pad ``" "``).
   - Strict OpenAI-compat: never send ``reasoning_content`` (422).
   - Anthropic: signed ``thinking`` blocks must be replayed in order with
     tool_use on the same turn; strip signatures on third-party endpoints.

2. **Prefix / prompt cache**
   - Anthropic: explicit ``cache_control`` system_and_3 breakpoints.
   - DeepSeek: automatic **byte-prefix** KV cache — system must be stable;
     put ephemeral memory only on the *current user* message (Hermes seam).
   - Changing system mid-session or prepending dynamic junk → full miss.

3. **Provider switch**
   - Rebuild wire messages with the *active* profile's reasoning policy
     (require pad vs strip). Never reuse the previous provider's pad.

Call :func:`prepare_wire_messages` from transports before HTTP.
"""

from __future__ import annotations

from typing import Any

from codedoggy.model.profile import (
    API_ANTHROPIC_MESSAGES,
    ProviderProfile,
)
from codedoggy.model.prompt_caching import apply_anthropic_cache_control
from codedoggy.model.reasoning import apply_reasoning_echo


def prepare_wire_messages(
    messages: list[dict[str, Any]],
    *,
    profile: ProviderProfile | None,
    model: str | None = None,
    enable_prompt_cache: bool = True,
    cache_ttl: str = "5m",
    redact: bool = True,
) -> list[dict[str, Any]]:
    """Apply redaction + reasoning policy + optional Anthropic cache markers."""
    from codedoggy.model.redact import redact_messages_for_api

    prepared = list(messages)
    if redact:
        prepared = redact_messages_for_api(prepared)

    if profile is None:
        return prepared

    policy = profile.reasoning_policy_for_model(model)
    prepared = apply_reasoning_echo(prepared, policy=policy)

    if (
        enable_prompt_cache
        and profile.api_mode == API_ANTHROPIC_MESSAGES
        and getattr(profile, "prompt_cache", True)
    ):
        native = True
        prepared = apply_anthropic_cache_control(
            prepared,
            cache_ttl=cache_ttl,
            native_anthropic=native,
        )

    return prepared


def assistant_blocks_for_anthropic(msg: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return ordered Anthropic content blocks for replay if stored.

    Prefers ``provider_data.anthropic_content_blocks`` (signed thinking +
    tool_use order). Falls back to synthesizing from reasoning + tool_calls.
    """
    pdata = msg.get("provider_data")
    if isinstance(pdata, dict):
        blocks = pdata.get("anthropic_content_blocks")
        if isinstance(blocks, list) and blocks:
            return [b for b in blocks if isinstance(b, dict)]

    # Synthesize unsigned thinking + text + tool_use (no signature)
    blocks: list[dict[str, Any]] = []
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        blocks.append({"type": "thinking", "thinking": reasoning})
    content = msg.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        blocks.extend([b for b in content if isinstance(b, dict)])

    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name") or ""
        args_raw = fn.get("arguments") or tc.get("arguments") or {}
        if isinstance(args_raw, str):
            import json

            try:
                args = json.loads(args_raw) if args_raw.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": str(tc.get("id") or ""),
                "name": str(name),
                "input": args,
            }
        )
    return blocks or None
