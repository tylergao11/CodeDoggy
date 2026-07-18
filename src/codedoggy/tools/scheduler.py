"""In-process scheduler store — Grok SchedulerActor command subset (no timer loop).

Ported from grok-build/crates/codegen/xai-grok-tools/src/implementations/grok_build/scheduler/
  actor.rs   — Create/Delete/List, fire_next_task, handle_missed_tasks, MAX_SCHEDULED_TASKS
  types.rs   — ScheduledTask / SchedulerState (via grok_build.scheduler_types)
  interval.rs — parse_interval / interval_to_human (via grok_build.scheduler_interval)

Maps 1:1 (pure store / poll):
  handle_command Create / Delete / List
  fire_next_task semantics (advance last_fired_at; remove one-shot / expired recurring)
  handle_missed_tasks (one-shot past due, never fired)
  TaskLimitReached

Honest gaps (timer host path):
  - No tokio select! timer actor, no CancellationToken shutdown chip cleanup.
  - No ToolNotificationHandle (ScheduledTaskCreated/Fired/Removed) — host may hook.
  - Durable ResourcesPersistence restore / re-announce is host (X).
  - Host must poll ``due_tasks()`` / ``fire_due()`` and inject prompts.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Optional

from codedoggy.tools.grok_build.scheduler_interval import (
    interval_to_human,
    parse_interval,
)
from codedoggy.tools.grok_build.scheduler_types import (
    MAX_SCHEDULED_TASKS,
    ScheduledTask,
    SchedulerError,
    SchedulerState,
)

# Re-exports for callers that imported from this module
__all__ = [
    "Scheduler",
    "ScheduledTask",
    "SchedulerError",
    "SchedulerState",
    "MAX_SCHEDULED_TASKS",
    "parse_interval",
    "interval_to_human",
    "ensure_scheduler",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Scheduler:
    """In-memory scheduled-task registry (Grok SchedulerState + command handlers).

    Not a full actor: no background sleep loop. Host polls ``due_tasks``.
    """

    def __init__(self, *, max_tasks: int = MAX_SCHEDULED_TASKS) -> None:
        self._lock = threading.RLock()
        self._state = SchedulerState()
        self._max_tasks = max_tasks

    # ── Create / Delete / List (actor handle_command) ─────────────────

    def create(
        self,
        *,
        interval: str,
        prompt: str,
        recurring: bool = True,
        durable: bool = False,
        fire_immediately: bool = False,
    ) -> ScheduledTask:
        """Parse interval, build task, push (raises SchedulerError)."""
        interval_secs = parse_interval(interval)
        task = ScheduledTask.with_fire_immediately(
            interval_secs,
            prompt,
            recurring,
            durable,
            fire_immediately,
        )
        return self.push_task(task)

    def push_task(self, task: ScheduledTask) -> ScheduledTask:
        """Grok SchedulerCommand::Create body (limit + push)."""
        with self._lock:
            if len(self._state.tasks) >= self._max_tasks:
                raise SchedulerError.task_limit_reached(self._max_tasks)
            self._state.tasks.append(task)
            return task

    def delete(self, task_id: str) -> bool:
        """Grok SchedulerCommand::Delete — True if removed."""
        with self._lock:
            before = len(self._state.tasks)
            self._state.tasks = [t for t in self._state.tasks if t.id != task_id]
            return before != len(self._state.tasks)

    def list(self) -> list[ScheduledTask]:
        """Grok SchedulerCommand::List — all tasks, no expiry prune."""
        with self._lock:
            return list(self._state.tasks)

    # ── Fire / miss (actor fire_next_task / handle_missed_tasks) ──────

    def fire_next_task(
        self, now: Optional[datetime] = None
    ) -> Optional[ScheduledTask]:
        """Fire one due task; advance last_fired_at; remove one-shot/expired.

        Returns a snapshot of the fired task (with last_fired_at set). None if none due.
        """
        now = _utcnow() if now is None else now
        with self._lock:
            idx = next(
                (
                    i
                    for i, t in enumerate(self._state.tasks)
                    if t.next_fire_at() <= now
                ),
                None,
            )
            if idx is None:
                return None
            task = self._state.tasks[idx]
            task.last_fired_at = now
            should_remove = (not task.recurring) or task.is_expired(now)
            # Snapshot for host before possible remove
            fired = ScheduledTask(
                id=task.id,
                interval_secs=task.interval_secs,
                prompt=task.prompt,
                recurring=task.recurring,
                durable=task.durable,
                created_at=task.created_at,
                last_fired_at=task.last_fired_at,
                expires_at=task.expires_at,
            )
            if should_remove:
                del self._state.tasks[idx]
            return fired

    def due_tasks(self, now: Optional[datetime] = None) -> list[ScheduledTask]:
        """Host poll: fire all currently due tasks (repeated fire_next_task)."""
        now = _utcnow() if now is None else now
        due: list[ScheduledTask] = []
        while True:
            fired = self.fire_next_task(now)
            if fired is None:
                break
            due.append(fired)
        return due

    def handle_missed_tasks(
        self, now: Optional[datetime] = None
    ) -> list[ScheduledTask]:
        """Fire and remove missed one-shots (actor startup path)."""
        now = _utcnow() if now is None else now
        fired: list[ScheduledTask] = []
        with self._lock:
            missed = [t for t in self._state.tasks if t.is_missed(now)]
            missed_ids = {t.id for t in missed}
            for t in missed:
                fired.append(
                    ScheduledTask(
                        id=t.id,
                        interval_secs=t.interval_secs,
                        prompt=t.prompt,
                        recurring=t.recurring,
                        durable=t.durable,
                        created_at=t.created_at,
                        last_fired_at=now,
                        expires_at=t.expires_at,
                    )
                )
            self._state.tasks = [
                t for t in self._state.tasks if t.id not in missed_ids
            ]
        return fired

    def durable_snapshot(self) -> SchedulerState:
        """Tasks that would be persisted (Grok durable filter)."""
        with self._lock:
            return self._state.durable_only()

    def restore_tasks(self, tasks: list[ScheduledTask]) -> None:
        """Host restore into state (no re-announce; host may list)."""
        with self._lock:
            self._state.tasks = list(tasks)


def ensure_scheduler(extra: dict[str, Any] | None) -> Scheduler:
    """Resolve session Scheduler from extra / kernel (CodeDoggy storage bag)."""
    bag = extra if extra is not None else {}
    s = bag.get("scheduler")
    if isinstance(s, Scheduler):
        return s
    kernel = bag.get("kernel")
    if kernel is not None:
        existing = getattr(kernel, "scheduler", None)
        if isinstance(existing, Scheduler):
            bag["scheduler"] = existing
            return existing
        s = Scheduler()
        try:
            kernel.scheduler = s
        except Exception:  # noqa: BLE001
            pass
        bag["scheduler"] = s
        return s
    s = Scheduler()
    bag["scheduler"] = s
    return s
