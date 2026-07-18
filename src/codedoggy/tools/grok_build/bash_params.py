"""Bash timeout/param helpers — source port from Grok bash/mod.rs.

Ported from:
  effective_max_timeout_ms, effective_default_timeout_ms
  effective_auto_bg_wait_ms (min of timeout and FG budget when auto-bg)
  DEFAULT_MAX_TIMEOUT_MS, DEFAULT_FOREGROUND_BLOCK_BUDGET_MS
"""

from __future__ import annotations

from codedoggy.tools.defaults import (
    BASH_DEFAULT_FOREGROUND_BLOCK_BUDGET_MS,
    BASH_DEFAULT_MAX_TIMEOUT_MS,
    BASH_DEFAULT_TIMEOUT_MS,
)

# bash/mod.rs
DEFAULT_MAX_TIMEOUT_MS = BASH_DEFAULT_MAX_TIMEOUT_MS  # 300_000
ABSOLUTE_MAX_TIMEOUT_MS = 36_000_000  # 10h
DEFAULT_FOREGROUND_BLOCK_BUDGET_MS = BASH_DEFAULT_FOREGROUND_BLOCK_BUDGET_MS
DEFAULT_TIMEOUT_MS = BASH_DEFAULT_TIMEOUT_MS  # 120_000


def resolve_fg_timeout_ms(timeout: int | None) -> int:
    """Foreground timeout: None/0 → default 120s; positive clamped to max 300s."""
    if timeout is None or timeout == 0:
        return DEFAULT_TIMEOUT_MS
    if timeout < 0:
        raise ValueError("timeout must be non-negative")
    return min(int(timeout), DEFAULT_MAX_TIMEOUT_MS)


def resolve_bg_max_runtime_s(timeout: int | None, *, max_s: float = 36_000.0) -> float:
    """Background max runtime. timeout 0/None → session max (model owns via kill)."""
    if timeout is None or timeout == 0:
        return max_s
    if timeout < 0:
        raise ValueError("timeout must be non-negative")
    return min(int(timeout), DEFAULT_MAX_TIMEOUT_MS) / 1000.0


def effective_auto_bg_wait_ms(
    resolved_timeout_ms: int,
    *,
    auto_background_on_timeout: bool,
    foreground_block_budget_ms: int | None = None,
) -> int:
    """Grok: FG wait when auto-bg is on is min(timeout, FG budget).

    When auto-bg is off, return resolved_timeout_ms unchanged.
    """
    if not auto_background_on_timeout:
        return resolved_timeout_ms
    budget = (
        DEFAULT_FOREGROUND_BLOCK_BUDGET_MS
        if foreground_block_budget_ms is None
        else int(foreground_block_budget_ms)
    )
    if budget <= 0:
        return resolved_timeout_ms
    return min(resolved_timeout_ms, budget)
