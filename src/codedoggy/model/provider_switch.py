"""Provider-switch hygiene (Hermes-aligned).

When the active model/provider changes mid-session:

1. Rebuild wire messages with the *new* profile's reasoning policy
   (DeepSeek pad vs OpenAI strip) — never keep the previous pad shape.
2. Optionally rewrite system ``Model:`` / ``Provider:`` identity lines
   without forcing a cold prefix rewrite of the whole static spine
   (only the *last* occurrence, volatile tail).

Call from TUI reload / model switch paths before the next sample.
"""

from __future__ import annotations

import re
from typing import Any

from codedoggy.model.profile import ProviderProfile
from codedoggy.model.profile_registry import get_profile
from codedoggy.model.protocol_context import prepare_wire_messages


def reprepare_messages_for_provider(
    messages: list[dict[str, Any]],
    *,
    provider: str,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Apply active provider's prepare_messages (reasoning + cache markers)."""
    profile = get_profile(provider)
    return prepare_wire_messages(
        messages,
        profile=profile,
        model=model,
        enable_prompt_cache=bool(profile.prompt_cache) if profile else False,
        cache_ttl=(profile.prompt_cache_ttl if profile else "5m") or "5m",
    )


def rewrite_system_model_identity(
    system_prompt: str | None,
    *,
    model: str | None,
    provider: str | None,
) -> str | None:
    """Rewrite last ``Model:`` / ``Provider:`` lines after a provider switch.

    Hermes: only the LAST match is touched so earlier user content that
    happens to include those labels is not rewritten.  The stored session
    system can keep primary labels; runtime sample uses the rewritten copy.
    """
    if not isinstance(system_prompt, str) or not system_prompt:
        return system_prompt
    sp = system_prompt
    for label, value in (("Model", model), ("Provider", provider)):
        if not value:
            continue
        matches = list(re.finditer(rf"(?m)^{label}: .*$", sp))
        if matches:
            last = matches[-1]
            sp = f"{sp[: last.start()]}{label}: {value}{sp[last.end() :]}"
    return sp


def active_profile(provider: str | None) -> ProviderProfile | None:
    return get_profile(provider) if provider else None
