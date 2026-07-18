"""Scheduler types and errors — pure data / constructors.

Ported from grok-build/crates/codegen/xai-grok-tools/src/implementations/grok_build/scheduler/types.rs

Maps 1:1 where practical:
  SchedulerError::{InvalidInterval, TaskLimitReached}
  ScheduledTask::{new, with_fire_immediately, next_fire_at, is_expired, is_missed}
  SchedulerState
  MAX_SCHEDULED_TASKS (from actor.rs)

Divergences (documented):
  - Task id: Grok uses uuid v7 hex[:12]; we use uuid4 hex[:12] (same length/charset).
  - No tokio mpsc SchedulerHandle / SchedulerCommand (host store is tools/scheduler.py).
  - Durable persistence: flag stored; Resources/ResourcesPersistence restore is host (X).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

# actor.rs
MAX_SCHEDULED_TASKS: int = 50


class SchedulerError(Exception):
    """Grok SchedulerError — Display strings must match thiserror messages exactly."""

    @classmethod
    def invalid_interval(cls, detail: str) -> "SchedulerError":
        # #[error("invalid interval: {0}")]
        return cls(f"invalid interval: {detail}")

    @classmethod
    def task_limit_reached(cls, limit: int = MAX_SCHEDULED_TASKS) -> "SchedulerError":
        # #[error("maximum of {0} scheduled tasks reached")]
        return cls(f"maximum of {limit} scheduled tasks reached")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_task_id() -> str:
    """12-char hex id (Grok: Uuid::now_v7 strip dashes take 12)."""
    return uuid.uuid4().hex[:12]


@dataclass
class ScheduledTask:
    """A single scheduled recurring or one-shot task (Grok ScheduledTask)."""

    id: str
    interval_secs: int
    prompt: str
    recurring: bool
    durable: bool
    created_at: datetime
    last_fired_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    @classmethod
    def new(
        cls,
        interval_secs: int,
        prompt: str,
        recurring: bool,
        durable: bool,
    ) -> "ScheduledTask":
        return cls.with_fire_immediately(
            interval_secs, prompt, recurring, durable, False
        )

    @classmethod
    def with_fire_immediately(
        cls,
        interval_secs: int,
        prompt: str,
        recurring: bool,
        durable: bool,
        fire_immediately: bool,
    ) -> "ScheduledTask":
        now = _utcnow()
        # When fire_immediately is true, anchor created_at in the past so that
        # next_fire_at() = created_at + interval = now, firing on the first tick.
        created_at = (
            now - timedelta(seconds=interval_secs) if fire_immediately else now
        )
        return cls(
            id=_new_task_id(),
            interval_secs=interval_secs,
            prompt=prompt,
            recurring=recurring,
            durable=durable,
            created_at=created_at,
            last_fired_at=None,
            expires_at=(now + timedelta(days=7)) if recurring else None,
        )

    def next_fire_at(self) -> datetime:
        """Next fire time from last_fired_at (or created_at if never fired)."""
        anchor = self.last_fired_at if self.last_fired_at is not None else self.created_at
        return anchor + timedelta(seconds=self.interval_secs)

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """Whether this task has expired (recurring tasks only)."""
        if self.expires_at is None:
            return False
        now = _utcnow() if now is None else now
        return now >= self.expires_at

    def is_missed(self, now: Optional[datetime] = None) -> bool:
        """One-shot: fire time already passed, never fired."""
        now = _utcnow() if now is None else now
        return (
            (not self.recurring)
            and self.last_fired_at is None
            and self.next_fire_at() < now
        )


@dataclass
class SchedulerState:
    """Persisted state for the scheduler (only durable tasks would be saved by host)."""

    tasks: list[ScheduledTask] = field(default_factory=list)

    def durable_only(self) -> "SchedulerState":
        """Filter non-durable tasks (Grok save path)."""
        return SchedulerState(tasks=[t for t in self.tasks if t.durable])


def to_rfc3339(dt: datetime) -> str:
    """ISO-8601 / RFC3339-ish string for list wire (Grok DateTime::to_rfc3339)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # Prefer Z for UTC
    s = dt.isoformat()
    if s.endswith("+00:00"):
        return s[:-6] + "Z"
    return s


def floor_char_boundary(s: str, index: int) -> int:
    """Byte-oriented floor char boundary (Grok util::floor_char_boundary subset).

    Python str is Unicode; we treat index as a character index (sufficient for
    list prompt truncation at 80 where ASCII prompts dominate).
    """
    if index >= len(s):
        return len(s)
    if index <= 0:
        return 0
    return index


def truncate_prompt(prompt: str, max_chars: int = 80) -> str:
    """List summary prompt truncation (list.rs: floor_char_boundary + ...)."""
    if len(prompt) > max_chars:
        cut = floor_char_boundary(prompt, max_chars)
        return f"{prompt[:cut]}..."
    return prompt
