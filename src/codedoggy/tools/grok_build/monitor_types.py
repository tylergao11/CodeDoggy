"""Monitor constants + input validation — source port from Grok.

Ported from:
  grok-build/crates/codegen/xai-grok-tools/src/implementations/grok_build/monitor/types.rs

Function map:
  LINE_TRUNCATION_LIMIT / BATCH_TRUNCATION_LIMIT / BUFFER_CAP_BYTES
  DEBOUNCE_MS / RATE_LIMIT_* / AUTO_KILL_THRESHOLD_MS
  DEFAULT_TIMEOUT_MS / MAX_TIMEOUT_MS
  MonitorInput.validate / resolved_timeout_ms
  format_monitor_started (to_prompt_format in output.rs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Max characters per individual stdout line before truncation.
LINE_TRUNCATION_LIMIT: int = 500
# Max characters per batched event (multiple lines joined).
BATCH_TRUNCATION_LIMIT: int = 3_000
# Raw stdout buffer cap in bytes.
BUFFER_CAP_BYTES: int = 1_048_576  # 1 MB
# Debounce window for batching concurrent stdout lines (ms).
DEBOUNCE_MS: int = 200
# Token bucket capacity.
RATE_LIMIT_CAPACITY: int = 10
# Token bucket refill interval in milliseconds.
RATE_LIMIT_REFILL_MS: int = 2_000
# Auto-kill after this many ms of continuous rate-limit violations.
AUTO_KILL_THRESHOLD_MS: int = 30_000
# Default / max monitor timeout (non-persistent): 10 hours.
DEFAULT_TIMEOUT_MS: int = 36_000_000
MAX_TIMEOUT_MS: int = 36_000_000
MAX_RESULT_SIZE_CHARS: int = 10_000

# Product-facing kill tool name used in monitor start messages / notices.
DEFAULT_KILL_TOOL_NAME: str = "kill_command_or_subagent"

MONITOR_DESC = """\
Start a background monitor that streams events from a long-running script. Each stdout line is an event - you can keep working and notifications arrive in the chat. Exit ends the watch.

**Output volume**: Every stdout line becomes a message in the conversation, so write selective filters. In pipes use `grep --line-buffered` (plain `grep` buffers and delays events by minutes).

Set `persistent: true` for session-length watches (PR monitoring, log tails) -- the monitor runs until you call kill_command_or_subagent or until the session ends. Otherwise it stops at `timeout_ms` (default 10h).\
"""


class MonitorError(ValueError):
    """Grok MonitorError."""


def validate_monitor_input(
    *,
    timeout_ms: Optional[int],
    persistent: bool,
) -> None:
    """MonitorInput::validate."""
    if (
        timeout_ms is not None
        and not persistent
        and timeout_ms > MAX_TIMEOUT_MS
    ):
        raise MonitorError(
            f"persistent must be true when timeout_ms exceeds {MAX_TIMEOUT_MS}ms"
        )


def resolved_timeout_ms(
    *,
    timeout_ms: Optional[int],
    persistent: bool,
) -> int:
    """0 for persistent / no-deadline; else explicit or DEFAULT_TIMEOUT_MS."""
    if persistent:
        return 0
    if timeout_ms is None:
        return DEFAULT_TIMEOUT_MS
    return int(timeout_ms)


def format_monitor_started(
    task_id: str,
    *,
    timeout_ms: int,
    persistent: bool,
    kill_tool_name: str = DEFAULT_KILL_TOOL_NAME,
) -> str:
    """Model-facing start message (tool.rs + output.rs Monitor branch).

    Grok to_prompt_format hardcodes ``kill_task``; tool.rs prefers the resolved
    product name (default kill_command_or_subagent). We use the product name.
    """
    if persistent or timeout_ms == 0:
        return (
            f"Monitor started (task {task_id}, persistent -- runs until "
            f"{kill_tool_name} or session end).\n"
            "You will be notified on each event. Keep working -- do not poll or sleep.\n"
            "Events may arrive while you are waiting for the user -- an event is not their reply."
        )
    return (
        f"Monitor started (task {task_id}, timeout {timeout_ms}ms).\n"
        "You will be notified on each event. Keep working -- do not poll or sleep.\n"
        "Events may arrive while you are waiting for the user -- an event is not their reply."
    )


@dataclass
class MonitorOutput:
    """Grok MonitorOutput."""

    task_id: str
    timeout_ms: int
    persistent: bool
