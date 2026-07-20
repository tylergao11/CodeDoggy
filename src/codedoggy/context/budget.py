"""Context window budget — Grok SamplingConfig.context_window as authority.

Grok: trigger = context_window * threshold_percent / 100
Usable window = context_window - completion_reserve - tools_reserve
Tokens only (no dual char-primary path).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from codedoggy.context.tokens import (
    count_messages_tokens,
    count_text_tokens,
    count_tools_tokens,
    tokenizer_backend,
)
from codedoggy.turn.types import Message

# Last-resort only when provider+model cannot be resolved (see context_limits).
from codedoggy.model.context_limits import DEFAULT_CONTEXT_WINDOW

DEFAULT_COMPLETION_RESERVE = 4_096
DEFAULT_THRESHOLD_PERCENT = 85


@dataclass(slots=True)
class ContextBudget:
    """Grok-aligned token budget for the live sample window."""

    context_window: int = DEFAULT_CONTEXT_WINDOW
    """Total model context window (SamplingConfig.context_window)."""

    completion_reserve: int = DEFAULT_COMPLETION_RESERVE
    """Reserved tokens for the model completion (max_completion_tokens spirit)."""

    threshold_percent: int = DEFAULT_THRESHOLD_PERCENT
    """Compact when usage > context_window * percent / 100 (Grok)."""

    target_threshold_percent: int = 50
    """Post-compact target: aim under this % of window (Grok target_threshold)."""

    keep_recent_messages: int = 16
    protect_first_n: int = 3
    tool_result_max_chars: int = 6_000
    retain_recent_tool_messages: int = 6
    protect_system: bool = True
    enabled: bool = True

    # Live tool-schema reserve (set each sample from tool definitions).
    tools_reserve: int = 0
    # Model-facing user-info / memory-fence overhead not stored in live history.
    ephemeral_reserve: int = 0

    last_prompt_tokens: int | None = None
    last_completion_tokens: int | None = None
    # Ensures spent waiting for real usage after fold (never permanent).
    awaiting_usage_ensures: int = 0

    def __post_init__(self) -> None:
        # Tests historically passed max_chars=… — map once to tokens, never dual path.
        pass

    @classmethod
    def from_max_chars(cls, max_chars: int, **kwargs: Any) -> ContextBudget:
        """Construct from test/legacy char budget (≈ chars/4 tokens).

        completion_reserve is scaled down so small test windows remain usable.
        """
        win = max(64, int(max_chars) // 4)
        # Small windows (tests): reserve ~12% not a fixed 4k which would zero usable.
        reserve = kwargs.pop("completion_reserve", None)
        if reserve is None:
            reserve = max(32, min(512, win // 8))
        return cls(context_window=win, completion_reserve=int(reserve), **kwargs)

    @property
    def usable_window(self) -> int:
        """Tokens available for messages after reserves."""
        return max(
            1,
            int(self.context_window)
            - int(self.completion_reserve)
            - int(self.tools_reserve)
            - int(self.ephemeral_reserve),
        )

    @property
    def max_tokens(self) -> int:
        return self.usable_window

    @property
    def trigger_ratio(self) -> float:
        return max(0.05, min(0.99, self.threshold_percent / 100.0))

    @property
    def trigger_tokens(self) -> int:
        """Compact when live usage exceeds this many tokens.

        Uses **usable_window** (context_window − completion_reserve − tools_reserve)
        so reserves actually affect the decision — not only the raw window.
        """
        return max(1, int(self.usable_window * self.threshold_percent // 100))

    @property
    def target_tokens(self) -> int:
        return max(1, int(self.usable_window * self.target_threshold_percent // 100))

    # Compat properties used by flush/should_flush (char≈token×4 heuristic only for flush soft path)
    @property
    def max_chars(self) -> int:
        return self.usable_window * 4

    @property
    def trigger_chars(self) -> int:
        return self.trigger_tokens * 4

    @classmethod
    def from_env(
        cls,
        *,
        context_window: int | None = None,
        completion_reserve: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> ContextBudget:
        pct = _env_int("CODEDOGGY_CONTEXT_THRESHOLD_PERCENT", 0)
        if not pct:
            pct = DEFAULT_THRESHOLD_PERCENT
        target = _env_int("CODEDOGGY_CONTEXT_TARGET_PERCENT", 50) or 50
        if context_window and int(context_window) > 0:
            win = int(context_window)
        else:
            from codedoggy.model.context_limits import resolve_context_window

            # Prefer explicit connection identity; else env CODEDOGGY_PROVIDER/MODEL.
            win = resolve_context_window(
                provider or os.environ.get("CODEDOGGY_PROVIDER"),
                model or os.environ.get("CODEDOGGY_MODEL"),
                base_url=base_url or os.environ.get("CODEDOGGY_BASE_URL"),
                probe=True,
            )
        # Explicit CODEDOGGY_CONTEXT_MAX_CHARS only as last-resort window≈chars/4
        if (
            not context_window
            and not os.environ.get("CODEDOGGY_CONTEXT_WINDOW")
            and not os.environ.get("CODEDOGGY_CONTEXT_MAX_TOKENS")
        ):
            mc = _env_int("CODEDOGGY_CONTEXT_MAX_CHARS", 0)
            if mc > 0:
                win = max(1024, mc // 4)
        reserve = completion_reserve
        if reserve is None:
            reserve = _env_int(
                "CODEDOGGY_COMPLETION_RESERVE", DEFAULT_COMPLETION_RESERVE
            ) or DEFAULT_COMPLETION_RESERVE
        return cls(
            context_window=max(1024, int(win)),
            completion_reserve=max(256, int(reserve)),
            threshold_percent=int(pct),
            target_threshold_percent=int(target),
            keep_recent_messages=_env_int("CODEDOGGY_CONTEXT_KEEP_RECENT", 16) or 16,
            protect_first_n=_env_int("CODEDOGGY_CONTEXT_PROTECT_FIRST", 3) or 3,
            tool_result_max_chars=_env_int("CODEDOGGY_TOOL_RESULT_MAX_CHARS", 6_000)
            or 6_000,
            retain_recent_tool_messages=_env_int("CODEDOGGY_RETAIN_RECENT_TOOLS", 6)
            or 6,
            enabled=_env_bool("CODEDOGGY_CONTEXT_COMPACT", True),
        )

    def bind_tools(self, tool_specs: list[Any] | None) -> None:
        """Recompute tools_reserve from current sample tool definitions (Grok)."""
        if not tool_specs:
            self.tools_reserve = 0
            return
        self.tools_reserve = count_tools_tokens(tool_specs)

    def bind_model(
        self,
        *,
        context_window: int | None = None,
        max_completion_tokens: int | None = None,
    ) -> None:
        if context_window and context_window > 0:
            self.context_window = int(context_window)
        if max_completion_tokens and max_completion_tokens > 0:
            self.completion_reserve = int(max_completion_tokens)


def weighted_text_len(text: str | None) -> int:
    return count_text_tokens(text) * 4


def estimate_chars(messages: list[Message]) -> int:
    return estimate_tokens(messages) * 4


def estimate_tokens(messages: list[Message]) -> int:
    return count_messages_tokens(messages)


def needs_compaction(messages: list[Message], budget: ContextBudget) -> bool:
    """Grok trigger: usage strictly greater than threshold tokens.

    Prefer last real ``prompt_tokens`` from the API when present; fall back
    to local estimate. If the live local estimate is already under the
    trigger (e.g. after prune), clear sticky API usage so we do not force
    fold after pressure was already relieved.
    """
    if not budget.enabled:
        return False
    usage = estimate_tokens(messages)
    if usage <= budget.trigger_tokens:
        # Local window already safe — drop stale API reading.
        if budget.last_prompt_tokens is not None:
            budget.last_prompt_tokens = None
        return False
    if budget.last_prompt_tokens is not None:
        # API prompt_tokens usually include tools + ephemeral overhead.
        # Usable trigger already excludes those reserves — normalize first so
        # we do not double-subtract and force early fold.
        effective_api = (
            int(budget.last_prompt_tokens)
            - int(budget.tools_reserve)
            - int(budget.ephemeral_reserve)
        )
        if effective_api > budget.trigger_tokens:
            return True
    return usage > budget.trigger_tokens


def still_over_target(messages: list[Message], budget: ContextBudget) -> bool:
    return estimate_tokens(messages) > budget.target_tokens


def budget_status(messages: list[Message], budget: ContextBudget) -> dict[str, object]:
    tok = estimate_tokens(messages)
    return {
        "tokens": tok,
        "trigger_tokens": budget.trigger_tokens,
        "target_tokens": budget.target_tokens,
        "context_window": budget.context_window,
        "usable_window": budget.usable_window,
        "tools_reserve": budget.tools_reserve,
        "ephemeral_reserve": budget.ephemeral_reserve,
        "completion_reserve": budget.completion_reserve,
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}
