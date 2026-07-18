"""Focused tests for Grok scheduler port (interval / types / store / tools)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from codedoggy.tools import ToolRegistryBuilder
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
from codedoggy.tools.runtime import ToolCallContext, ToolError
from codedoggy.tools.scheduler import Scheduler


# ── interval.rs ───────────────────────────────────────────────────────


def test_parse_minutes() -> None:
    assert parse_interval("5m") == 300
    assert parse_interval("10m") == 600
    assert parse_interval("1m") == 60


def test_parse_hours() -> None:
    assert parse_interval("2h") == 7200
    assert parse_interval("1h") == 3600


def test_parse_days() -> None:
    assert parse_interval("1d") == 86400
    assert parse_interval("7d") == 604800


def test_parse_seconds_clamped_to_minimum() -> None:
    assert parse_interval("30s") == 60
    assert parse_interval("1s") == 60
    assert parse_interval("60s") == 60
    assert parse_interval("120s") == 120


def test_parse_empty_returns_error() -> None:
    with pytest.raises(SchedulerError, match="interval cannot be empty"):
        parse_interval("")


def test_parse_invalid_format_returns_error() -> None:
    with pytest.raises(SchedulerError, match="invalid interval format"):
        parse_interval("abc")
    with pytest.raises(SchedulerError, match="invalid interval suffix"):
        parse_interval("5x")
    with pytest.raises(SchedulerError, match="invalid interval format"):
        parse_interval("m")


def test_parse_zero_returns_error() -> None:
    with pytest.raises(SchedulerError, match="greater than 0"):
        parse_interval("0m")
    with pytest.raises(SchedulerError, match="greater than 0"):
        parse_interval("0s")


def test_parse_overflow_returns_error() -> None:
    with pytest.raises(SchedulerError, match="interval too large"):
        parse_interval("1000000000000000000d")


def test_parse_with_whitespace() -> None:
    assert parse_interval("  5m  ") == 300


def test_human_readable_minutes() -> None:
    assert interval_to_human(300) == "every 5 minutes"
    assert interval_to_human(60) == "every 1 minute"
    assert interval_to_human(600) == "every 10 minutes"


def test_human_readable_hours() -> None:
    assert interval_to_human(3600) == "every 1 hour"
    assert interval_to_human(7200) == "every 2 hours"


def test_human_readable_days() -> None:
    assert interval_to_human(86400) == "every 1 day"
    assert interval_to_human(172800) == "every 2 days"


def test_human_readable_seconds() -> None:
    assert interval_to_human(45) == "every 45 seconds"
    assert interval_to_human(1) == "every 1 second"


# ── types.rs ──────────────────────────────────────────────────────────


def test_new_recurring_task_has_7_day_expiry() -> None:
    task = ScheduledTask.new(300, "check deploy", True, False)
    assert task.expires_at is not None
    diff = task.expires_at - datetime.now(timezone.utc)
    # ~7 days (allow a few seconds of clock skew in the test window)
    assert timedelta(days=6, hours=23) < diff <= timedelta(days=7, minutes=1)


def test_new_one_shot_task_has_no_expiry() -> None:
    task = ScheduledTask.new(300, "check deploy", False, False)
    assert task.expires_at is None


def test_next_fire_at_uses_created_at_when_never_fired() -> None:
    task = ScheduledTask.new(300, "test", True, False)
    expected = task.created_at + timedelta(seconds=300)
    assert task.next_fire_at() == expected


def test_next_fire_at_uses_last_fired_at_when_present() -> None:
    task = ScheduledTask.new(300, "test", True, False)
    fired = datetime.now(timezone.utc)
    task.last_fired_at = fired
    assert task.next_fire_at() == fired + timedelta(seconds=300)


def test_is_expired_returns_true_when_past_expiry() -> None:
    task = ScheduledTask.new(300, "test", True, False)
    task.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    assert task.is_expired()


def test_is_expired_returns_false_when_before_expiry() -> None:
    task = ScheduledTask.new(300, "test", True, False)
    assert not task.is_expired()


def test_is_expired_returns_false_for_one_shot() -> None:
    task = ScheduledTask.new(300, "test", False, False)
    assert not task.is_expired()


def test_is_missed_returns_true_for_unfired_one_shot_past_due() -> None:
    task = ScheduledTask.new(1, "test", False, False)
    task.created_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    assert task.is_missed()


def test_is_missed_returns_false_for_recurring() -> None:
    task = ScheduledTask.new(1, "test", True, False)
    task.created_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    assert not task.is_missed()


def test_is_missed_returns_false_if_already_fired() -> None:
    task = ScheduledTask.new(1, "test", False, False)
    task.created_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    task.last_fired_at = datetime.now(timezone.utc)
    assert not task.is_missed()


def test_task_id_is_12_chars() -> None:
    task = ScheduledTask.new(300, "test", True, False)
    assert len(task.id) == 12
    assert re.fullmatch(r"[0-9a-f]{12}", task.id)


def test_scheduler_state_default_is_empty() -> None:
    state = SchedulerState()
    assert state.tasks == []


def test_fire_immediately_anchors_created_at() -> None:
    task = ScheduledTask.with_fire_immediately(300, "now", True, False, True)
    # next_fire_at ≈ now
    delta = abs((task.next_fire_at() - datetime.now(timezone.utc)).total_seconds())
    assert delta < 2.0


# ── actor store subset ────────────────────────────────────────────────


def test_create_and_list_task() -> None:
    sched = Scheduler()
    task = sched.create(interval="5m", prompt="check deploy", recurring=True)
    tasks = sched.list()
    assert len(tasks) == 1
    assert tasks[0].prompt == "check deploy"
    assert tasks[0].id == task.id


def test_delete_task() -> None:
    sched = Scheduler()
    task = sched.create(interval="5m", prompt="test", recurring=True)
    assert sched.delete(task.id) is True
    assert sched.list() == []


def test_delete_nonexistent_returns_false() -> None:
    sched = Scheduler()
    assert sched.delete("nonexistent") is False


def test_max_task_limit_enforced() -> None:
    sched = Scheduler()
    for i in range(MAX_SCHEDULED_TASKS):
        sched.create(interval="5m", prompt=f"task {i}", recurring=True)
    with pytest.raises(SchedulerError, match="maximum of 50 scheduled tasks reached"):
        sched.create(interval="5m", prompt="one too many", recurring=True)


def test_fire_one_shot_removes() -> None:
    sched = Scheduler()
    task = sched.create(
        interval="60s",
        prompt="once",
        recurring=False,
        fire_immediately=True,
    )
    due = sched.due_tasks()
    assert len(due) == 1
    assert due[0].id == task.id
    assert sched.list() == []


def test_fire_recurring_advances_last_fired() -> None:
    sched = Scheduler()
    task = sched.create(
        interval="60s",
        prompt="loop",
        recurring=True,
        fire_immediately=True,
    )
    due = sched.due_tasks()
    assert len(due) == 1
    remaining = sched.list()
    assert len(remaining) == 1
    assert remaining[0].last_fired_at is not None
    # next fire is in the future
    assert remaining[0].next_fire_at() > datetime.now(timezone.utc)
    assert remaining[0].id == task.id


def test_handle_missed_one_shots() -> None:
    sched = Scheduler()
    past = datetime.now(timezone.utc) - timedelta(seconds=60)
    for i in range(3):
        t = ScheduledTask.new(1, f"missed-{i}", False, False)
        t.id = f"missed-{i:06d}"  # 12-ish
        t.created_at = past
        sched.push_task(t)
    fired = sched.handle_missed_tasks()
    assert len(fired) == 3
    assert sched.list() == []


def test_durable_snapshot_filters() -> None:
    sched = Scheduler()
    sched.create(interval="5m", prompt="ephemeral", durable=False)
    sched.create(interval="5m", prompt="keep", durable=True)
    snap = sched.durable_snapshot()
    assert len(snap.tasks) == 1
    assert snap.tasks[0].prompt == "keep"


# ── tool wire ─────────────────────────────────────────────────────────


def _tools():
    return ToolRegistryBuilder.new().finalize()


def test_tool_create_list_delete(tmp_path: Path) -> None:
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    out = tools.call(
        "scheduler_create",
        {"interval": "5m", "prompt": "check status", "recurring": True},
        ctx,
    )
    assert out.startswith("Scheduled task created (ID: ")
    assert "every 5 minutes" in out
    assert "recurring: true" in out

    m = re.search(r"ID: ([0-9a-f]{12}),", out)
    assert m, out
    tid = m.group(1)

    listed = tools.call("scheduler_list", {}, ctx)
    data = json.loads(listed)
    assert isinstance(data, list) and len(data) == 1
    assert data[0]["id"] == tid
    assert data[0]["prompt"] == "check status"
    assert data[0]["intervalHuman"] == "every 5 minutes"
    assert data[0]["recurring"] is True
    assert "nextFireAt" in data[0]
    assert "createdAt" in data[0]

    deleted = tools.call("scheduler_delete", {"id": tid}, ctx)
    assert deleted == f"Scheduled task {tid} cancelled."

    listed2 = tools.call("scheduler_list", {}, ctx)
    assert listed2 == "No scheduled tasks."


def test_tool_delete_not_found(tmp_path: Path) -> None:
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    out = tools.call("scheduler_delete", {"id": "deadbeefdead"}, ctx)
    assert out == (
        "No scheduled task with ID deadbeefdead found. "
        "Use scheduler_list to see active tasks."
    )


def test_tool_invalid_interval_message(tmp_path: Path) -> None:
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    with pytest.raises(ToolError) as ei:
        tools.call(
            "scheduler_create",
            {"interval": "5x", "prompt": "x"},
            ctx,
        )
    assert ei.value.code == "invalid_arguments"
    assert str(ei.value).startswith("invalid interval:")
    assert "invalid interval suffix" in str(ei.value)


def test_tool_task_limit_message(tmp_path: Path) -> None:
    tools = _tools()
    sched = Scheduler()
    ctx = ToolCallContext(cwd=tmp_path, extra={"scheduler": sched})
    for i in range(MAX_SCHEDULED_TASKS):
        tools.call(
            "scheduler_create",
            {"interval": "5m", "prompt": f"t{i}"},
            ctx,
        )
    with pytest.raises(ToolError) as ei:
        tools.call(
            "scheduler_create",
            {"interval": "5m", "prompt": "overflow"},
            ctx,
        )
    assert ei.value.code == "invalid_arguments"
    assert str(ei.value) == "maximum of 50 scheduled tasks reached"


def test_tool_clamps_sub_minute(tmp_path: Path) -> None:
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    out = tools.call(
        "scheduler_create",
        {"interval": "30s", "prompt": "fast"},
        ctx,
    )
    # clamped to 60s → every 1 minute
    assert "every 1 minute" in out
