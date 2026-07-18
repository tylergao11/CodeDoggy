"""Interval parse / humanize for scheduler tools.

Ported from grok-build/crates/codegen/xai-grok-tools/src/implementations/grok_build/scheduler/interval.rs

Maps 1:1:
  MINIMUM_INTERVAL_SECS
  parse_interval
  interval_to_human
  SchedulerError::InvalidInterval message bodies (via SchedulerError.invalid_interval)
"""

from __future__ import annotations

from codedoggy.tools.grok_build.scheduler_types import SchedulerError

MINIMUM_INTERVAL_SECS: int = 60


def _rust_dbg_str(s: str) -> str:
    """Approximate Rust Debug for &str (double-quoted, minimal escapes)."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def parse_interval(s: str) -> int:
    """Parse an interval string like \"5m\", \"2h\", \"30s\", \"1d\" into seconds.

    Minimum interval is 60 seconds; values below are clamped (Grok behaviour).
    Raises SchedulerError with exact Grok ``invalid interval: …`` messages.
    """
    s = s.strip()
    if not s:
        raise SchedulerError.invalid_interval("interval cannot be empty")

    if len(s) < 2:
        # split_at(len-1) with empty digits → parse fails
        raise SchedulerError.invalid_interval(
            f"invalid interval format: {_rust_dbg_str(s)} (expected e.g. 5m, 2h, 1d)"
        )

    digits, suffix = s[:-1], s[-1]
    try:
        value = int(digits)
    except ValueError as e:
        raise SchedulerError.invalid_interval(
            f"invalid interval format: {_rust_dbg_str(s)} (expected e.g. 5m, 2h, 1d)"
        ) from e

    # Reject negatives (Rust u64 parse would fail) and zero
    if value < 0:
        raise SchedulerError.invalid_interval(
            f"invalid interval format: {_rust_dbg_str(s)} (expected e.g. 5m, 2h, 1d)"
        )
    if value == 0:
        raise SchedulerError.invalid_interval("interval value must be greater than 0")

    unit_secs = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
    }.get(suffix)
    if unit_secs is None:
        raise SchedulerError.invalid_interval(
            f"invalid interval suffix: {_rust_dbg_str(suffix)} (expected s, m, h, or d)"
        )

    try:
        secs = value * unit_secs
    except OverflowError as e:
        raise SchedulerError.invalid_interval(
            f"interval too large: {_rust_dbg_str(s)}"
        ) from e

    # Python int is unbounded; mirror Rust checked_mul overflow for huge products
    if secs > (2**64 - 1):
        raise SchedulerError.invalid_interval(f"interval too large: {_rust_dbg_str(s)}")

    return max(secs, MINIMUM_INTERVAL_SECS)


def interval_to_human(secs: int) -> str:
    """Convert seconds to a human-readable interval string.

    e.g. 300 -> \"every 5 minutes\", 3600 -> \"every 1 hour\"
    """
    if secs % 86400 == 0:
        n = secs // 86400
        if n == 1:
            return "every 1 day"
        return f"every {n} days"
    if secs % 3600 == 0:
        n = secs // 3600
        if n == 1:
            return "every 1 hour"
        return f"every {n} hours"
    if secs % 60 == 0:
        n = secs // 60
        if n == 1:
            return "every 1 minute"
        return f"every {n} minutes"
    if secs == 1:
        return "every 1 second"
    return f"every {secs} seconds"
