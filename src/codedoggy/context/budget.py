"""Context window budget — Grok threshold_percent + token-aware pressure.

Primary unit is **tokens** (tiktoken when available, else heuristic).
``max_chars`` remains an env-facing alias: default window ≈ max_chars/4 tokens
so existing CODEDOGGY_CONTEXT_MAX_CHARS configs keep working.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from codedoggy.context.tokens import (
    count_messages_tokens,
    count_text_tokens,
    tokenizer_backend,
)
from codedoggy.turn.types import Message


@dataclass(slots=True)
class ContextBudget:
    """Budget for live context."""

    max_chars: int = 120_000
    """Legacy window size in weighted-char units (≈ tokens × 4)."""

    threshold_percent: int = 72
    """Grok-style 0–100: compact when usage >= max * percent / 100."""

    keep_recent_messages: int = 16
    """Always keep this many newest non-system messages intact after fold (Hermes protect_last_n spirit)."""

    protect_first_n: int = 3
    """Hermes: first N non-system messages kept verbatim in addition to system."""

    tool_result_max_chars: int = 6_000
    """Soft-cap individual tool observations (raw chars)."""

    retain_recent_tool_messages: int = 6
    """Grok prune_retained: keep this many newest tool bodies; clear older."""

    protect_system: bool = True
    """Never drop system messages (MEMORY frozen blocks)."""

    enabled: bool = True

    # Optional last-known model usage (prompt_tokens) to refine pressure.
    last_prompt_tokens: int | None = None
    last_completion_tokens: int | None = None

    @property
    def max_tokens(self) -> int:
        return max(1, self.max_chars // 4)

    @property
    def trigger_ratio(self) -> float:
        return max(0.05, min(0.99, self.threshold_percent / 100.0))

    @property
    def trigger_tokens(self) -> int:
        return max(1, int(self.max_tokens * self.trigger_ratio))

    @property
    def trigger_chars(self) -> int:
        """Compat: weighted-char trigger ≈ tokens × 4 (for flush/should_flush)."""
        return max(1, self.trigger_tokens * 4)

    @property
    def max_tokens_approx(self) -> int:
        return self.max_tokens

    @classmethod
    def from_env(cls) -> ContextBudget:
        pct = _env_int("CODEDOGGY_CONTEXT_THRESHOLD_PERCENT", 0)
        if not pct:
            ratio = _env_float("CODEDOGGY_CONTEXT_TRIGGER", 0.72) or 0.72
            pct = int(ratio * 100)
        # Prefer explicit token window when set
        max_tok = _env_int("CODEDOGGY_CONTEXT_MAX_TOKENS", 0)
        if max_tok > 0:
            max_chars = max_tok * 4
        else:
            max_chars = _env_int("CODEDOGGY_CONTEXT_MAX_CHARS", 120_000) or 120_000
        return cls(
            max_chars=max_chars,
            threshold_percent=pct,
            keep_recent_messages=_env_int("CODEDOGGY_CONTEXT_KEEP_RECENT", 16) or 16,
            protect_first_n=_env_int("CODEDOGGY_CONTEXT_PROTECT_FIRST", 3) or 3,
            tool_result_max_chars=_env_int("CODEDOGGY_TOOL_RESULT_MAX_CHARS", 6_000)
            or 6_000,
            retain_recent_tool_messages=_env_int(
                "CODEDOGGY_RETAIN_RECENT_TOOLS", 6
            )
            or 6,
            enabled=_env_bool("CODEDOGGY_CONTEXT_COMPACT", True),
        )


def weighted_text_len(text: str | None) -> int:
    """Compat helper: weighted chars ≈ tokens × 4."""
    return count_text_tokens(text) * 4


def estimate_chars(messages: list[Message]) -> int:
    """Portable size units (tokens × 4) for flush thresholds and logs."""
    return estimate_tokens(messages) * 4


def estimate_tokens(messages: list[Message]) -> int:
    """Token count via tiktoken or heuristic."""
    return count_messages_tokens(messages)


def needs_compaction(messages: list[Message], budget: ContextBudget) -> bool:
    """True when over pressure. Grok trigger uses strictly-greater-than threshold."""
    if not budget.enabled:
        return False
    usage_tok = estimate_tokens(messages)
    if usage_tok > budget.trigger_tokens:
        return True
    # Trust model-reported prompt_tokens when higher (real API truth).
    if budget.last_prompt_tokens is not None:
        if budget.last_prompt_tokens > budget.trigger_tokens:
            return True
    return False


def budget_status(messages: list[Message], budget: ContextBudget) -> dict[str, object]:
    """Debug snapshot for stress reports / CLI."""
    tok = estimate_tokens(messages)
    return {
        "tokens": tok,
        "trigger_tokens": budget.trigger_tokens,
        "max_tokens": budget.max_tokens,
        "backend": tokenizer_backend(),
        "last_prompt_tokens": budget.last_prompt_tokens,
        "over_trigger": tok > budget.trigger_tokens,
    }


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}
