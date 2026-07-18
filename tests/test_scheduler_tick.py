"""Unit tests for host light scheduler tick (poll_due / fire_due / run_tick_loop)."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from codedoggy.host.scheduler_tick import (
    FireResult,
    fire_due,
    poll_due,
    run_tick_loop,
)
from codedoggy.tools.grok_build.scheduler_types import ScheduledTask
from codedoggy.tools.scheduler import Scheduler


def _utc(ts: str = "2026-01-01T00:00:00+00:00") -> datetime:
    return datetime.fromisoformat(ts)


def _push_due(
    sched: Scheduler,
    *,
    prompt: str,
    recurring: bool = False,
    created_at: Optional[datetime] = None,
    interval_secs: int = 60,
) -> ScheduledTask:
    """Push a task that is due at ``created_at + interval`` (manual next_fire_at)."""
    created = created_at if created_at is not None else _utc() - timedelta(seconds=interval_secs)
    task = ScheduledTask(
        id=f"t{abs(hash(prompt)) % 10**10:012d}"[:12],
        interval_secs=interval_secs,
        prompt=prompt,
        recurring=recurring,
        durable=False,
        created_at=created,
        last_fired_at=None,
        expires_at=(_utc() + timedelta(days=7)) if recurring else None,
    )
    return sched.push_task(task)


def test_poll_due_empty() -> None:
    sched = Scheduler()
    assert poll_due(sched, now=_utc()) == []


def test_poll_due_returns_fired_snapshots_and_advances() -> None:
    sched = Scheduler()
    past = _utc() - timedelta(seconds=120)
    t = _push_due(sched, prompt="check logs", created_at=past, interval_secs=60)
    now = _utc()
    due = poll_due(sched, now=now)
    assert len(due) == 1
    assert due[0].id == t.id
    assert due[0].prompt == "check logs"
    assert due[0].last_fired_at == now
    # one-shot removed
    assert sched.list() == []


def test_fire_due_returns_ids_and_prompts() -> None:
    sched = Scheduler()
    past = _utc() - timedelta(seconds=120)
    a = _push_due(sched, prompt="alpha", created_at=past)
    b = _push_due(sched, prompt="beta", created_at=past)
    results = fire_due(sched, now=_utc())
    assert len(results) == 2
    assert all(isinstance(r, FireResult) for r in results)
    by_id = {r.id: r.prompt for r in results}
    assert by_id[a.id] == "alpha"
    assert by_id[b.id] == "beta"
    assert all(r.task.last_fired_at is not None for r in results)
    assert sched.list() == []


def test_fire_due_not_yet_due() -> None:
    sched = Scheduler()
    future_anchor = _utc()
    task = ScheduledTask(
        id="notdue000001",
        interval_secs=3600,
        prompt="later",
        recurring=True,
        durable=False,
        created_at=future_anchor,
        last_fired_at=None,
        expires_at=future_anchor + timedelta(days=7),
    )
    sched.push_task(task)
    # now == created_at → next_fire is +3600s, not due
    assert fire_due(sched, now=future_anchor) == []
    assert len(sched.list()) == 1


def test_fire_due_recurring_stays_with_advanced_last_fired() -> None:
    sched = Scheduler()
    past = _utc() - timedelta(seconds=90)
    t = _push_due(
        sched,
        prompt="loop",
        recurring=True,
        created_at=past,
        interval_secs=60,
    )
    now = _utc()
    results = fire_due(sched, now=now)
    assert len(results) == 1
    remaining = sched.list()
    assert len(remaining) == 1
    assert remaining[0].id == t.id
    assert remaining[0].last_fired_at == now
    assert remaining[0].next_fire_at() == now + timedelta(seconds=60)
    # next tick at same now: not due again
    assert fire_due(sched, now=now) == []


def test_fire_due_uses_scheduler_due_tasks_api() -> None:
    """Ensure we only exercise public store APIs (no private state)."""
    calls: list[str] = []

    class Fake:
        def due_tasks(self, now=None):
            calls.append("due_tasks")
            task = ScheduledTask(
                id="fake00000001",
                interval_secs=60,
                prompt="p",
                recurring=False,
                durable=False,
                created_at=_utc(),
                last_fired_at=now or _utc(),
            )
            return [task]

        def fire_next_task(self, now=None):
            calls.append("fire_next_task")
            return None

    out = fire_due(Fake())  # type: ignore[arg-type]
    assert calls == ["due_tasks"]
    assert out[0].id == "fake00000001"
    assert out[0].prompt == "p"


def test_run_tick_loop_fires_and_stops() -> None:
    sched = Scheduler()
    past = _utc() - timedelta(seconds=120)
    _push_due(sched, prompt="tick-me", created_at=past)

    collected: list[list[FireResult]] = []
    stop = threading.Event()
    clock = {"n": 0}
    base = _utc()

    def now_fn() -> datetime:
        # first iteration: due; later iterations: already fired one-shot gone
        clock["n"] += 1
        return base

    def on_fire(results: list[FireResult]) -> None:
        collected.append(results)
        stop.set()

    run_tick_loop(
        sched,
        on_fire,
        stop,
        interval_s=0.01,
        now_fn=now_fn,
    )
    assert len(collected) == 1
    assert collected[0][0].prompt == "tick-me"
    assert sched.list() == []


def test_run_tick_loop_interval_must_be_positive() -> None:
    stop = threading.Event()
    with pytest.raises(ValueError, match="interval_s"):
        run_tick_loop(Scheduler(), lambda _r: None, stop, interval_s=0)


def test_run_tick_loop_on_fire_error_does_not_kill_loop() -> None:
    sched = Scheduler()
    past = _utc() - timedelta(seconds=120)
    # two one-shots; first on_fire raises, second iteration still runs
    _push_due(sched, prompt="a", created_at=past)
    _push_due(sched, prompt="b", created_at=past)

    stop = threading.Event()
    hits = {"n": 0}

    def on_fire(results: list[FireResult]) -> None:
        hits["n"] += 1
        if hits["n"] == 1:
            # fire both in first poll; then force second empty poll path
            raise RuntimeError("inject failed")
        stop.set()

    # First poll fires both; on_fire raises; loop continues until we stop
    # on a subsequent empty fire — stop after N waits via side thread.
    def arm_stop() -> None:
        # allow a couple of intervals then stop
        stop.wait(timeout=0.05)
        stop.set()

    threading.Thread(target=arm_stop, daemon=True).start()
    run_tick_loop(sched, on_fire, stop, interval_s=0.01, now_fn=lambda: _utc())
    assert hits["n"] >= 1


def test_run_tick_loop_daemon_thread_recipe() -> None:
    """Smoke: host-style daemon thread + stop_event."""
    sched = Scheduler()
    past = _utc() - timedelta(seconds=120)
    _push_due(sched, prompt="bg", created_at=past)

    got: list[str] = []
    stop = threading.Event()

    def on_fire(results: list[FireResult]) -> None:
        got.extend(r.prompt for r in results)
        stop.set()

    t = threading.Thread(
        target=run_tick_loop,
        args=(sched, on_fire, stop),
        kwargs={"interval_s": 0.02, "now_fn": lambda: _utc()},
        daemon=True,
        name="test-scheduler-tick",
    )
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert got == ["bg"]
