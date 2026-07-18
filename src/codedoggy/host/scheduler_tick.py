"""LIGHT host poll/tick helper for the in-process Scheduler store.

Purpose
-------
Grok's real scheduler is a tokio actor with a timer loop and
``ToolNotificationHandle`` (ScheduledTaskCreated / Fired / Removed). CodeDoggy
ships only the pure store in ``codedoggy.tools.scheduler.Scheduler`` (Create /
Delete / List / ``fire_next_task`` / ``due_tasks`` / missed). **This module is
not that actor** — it is a thin, optional host poller so a session host can
advance schedule state without inventing notifications or timer parity.

Honesty grades
--------------
- Poll / fire via existing ``Scheduler.due_tasks`` / ``fire_next_task``: **A**
  (compatible host path; not source-mapped 1:1 to actor.rs select! loop).
- Tokio timer actor / CancellationToken shutdown: **X** (not implemented).
- ``ToolNotificationHandle`` bus (Created/Fired/Removed): **X** (not implemented).
- Durable ResourcesPersistence re-announce: **X** (host).

Host inject contract (critical)
-------------------------------
This helper **only advances schedule state** (``last_fired_at``, one-shot
removal, expiry prune). It does **not** run the agent.

The host **must** take each fired prompt and inject it into the agent path,
for example:

.. code-block:: python

    from codedoggy.host.scheduler_tick import fire_due, run_tick_loop
    from codedoggy.orchestration.prompt_queue import PromptQueueItem

    def on_fire(results):
        for r in results:
            # Prefer interjection drain at safe points when a turn is mid-flight:
            if kernel.interjection_buffer is not None:
                kernel.interjection_buffer.push(r.prompt, prompt_id=r.id)
            elif kernel.prompt_queue is not None:
                kernel.prompt_queue.push(
                    PromptQueueItem(text=r.prompt, prompt_id=r.id)
                )
            else:
                # Or: session.submit_user_turn(r.prompt) / your host channel
                ...

    # One-shot poll (e.g. idle loop head):
    on_fire(fire_due(kernel.scheduler))

    # Or background tick (daemon-friendly; does not inject by itself):
    # run_tick_loop(kernel.scheduler, on_fire, stop_event, interval_s=1.0)

Main agent / kernel wiring remains the caller's job (do not expect bootstrap
to start this loop automatically).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional, Protocol

from codedoggy.tools.grok_build.scheduler_types import ScheduledTask

logger = logging.getLogger(__name__)


class SupportsDueFire(Protocol):
    """Minimal surface used by the tick helper (``Scheduler`` satisfies this)."""

    def due_tasks(self, now: Optional[datetime] = None) -> list[ScheduledTask]: ...

    def fire_next_task(
        self, now: Optional[datetime] = None
    ) -> Optional[ScheduledTask]: ...


@dataclass(frozen=True)
class FireResult:
    """Light fire record for host injection (id + prompt + task snapshot)."""

    id: str
    prompt: str
    task: ScheduledTask

    @classmethod
    def from_task(cls, task: ScheduledTask) -> "FireResult":
        return cls(id=task.id, prompt=task.prompt, task=task)


def poll_due(
    scheduler: SupportsDueFire,
    now: Optional[datetime] = None,
) -> list[ScheduledTask]:
    """Poll and fire all currently due tasks; return fired task snapshots.

    Delegates to ``Scheduler.due_tasks`` (repeated ``fire_next_task``). Schedule
    state is advanced; prompts are **not** injected into the agent.
    """
    return list(scheduler.due_tasks(now))


def fire_due(
    scheduler: SupportsDueFire,
    now: Optional[datetime] = None,
) -> list[FireResult]:
    """Fire all due tasks; return ``FireResult`` list (id + prompt) for inject.

    Uses existing ``Scheduler.due_tasks`` / ``fire_next_task`` APIs only.
    Host must feed ``result.prompt`` into the agent (queue / interjection /
    user turn). This function does not send notifications (**X**).
    """
    return [FireResult.from_task(t) for t in poll_due(scheduler, now)]


OnFire = Callable[[list[FireResult]], Any]


def run_tick_loop(
    scheduler: SupportsDueFire,
    on_fire: OnFire,
    stop_event: threading.Event,
    interval_s: float = 1.0,
    *,
    now_fn: Optional[Callable[[], datetime]] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
) -> None:
    """Daemon-friendly poll loop: fire_due → on_fire(results) → wait.

    Stops when ``stop_event`` is set. Exceptions from ``on_fire`` are logged
    and do not kill the loop (host decides policy). Does **not** start a
    thread itself — call from a host-owned thread if needed::

        stop = threading.Event()
        t = threading.Thread(
            target=run_tick_loop,
            args=(kernel.scheduler, on_fire, stop),
            kwargs={"interval_s": 1.0},
            daemon=True,
            name="codedoggy-scheduler-tick",
        )
        t.start()
        # … later: stop.set(); t.join(timeout=…)

    Parameters
    ----------
    now_fn:
        Optional clock for tests (passed as ``now=`` into ``fire_due``).
        When omitted, the scheduler uses wall clock UTC.
    sleep_fn:
        Optional delay hook for tests. Default is interruptible
        ``stop_event.wait(timeout=interval_s)`` so stop wakes the loop early.
        If provided, called as ``sleep_fn(interval_s)`` between ticks (tests
        may no-op or advance a fake clock); still checks ``stop_event`` after.
    """
    if interval_s <= 0:
        raise ValueError("interval_s must be > 0")

    while not stop_event.is_set():
        try:
            now = now_fn() if now_fn is not None else None
            results = fire_due(scheduler, now)
            if results:
                try:
                    on_fire(results)
                except Exception:  # noqa: BLE001 — host callback isolation
                    logger.exception(
                        "scheduler tick on_fire failed (%d result(s))",
                        len(results),
                    )
        except Exception:  # noqa: BLE001 — keep loop alive on store errors
            logger.exception("scheduler tick poll failed")

        if stop_event.is_set():
            break
        if sleep_fn is not None:
            sleep_fn(interval_s)
        else:
            # Interruptible wait — returns True if stop was set during sleep.
            if stop_event.wait(timeout=interval_s):
                break
