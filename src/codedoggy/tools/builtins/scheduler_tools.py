"""scheduler_create / delete / list — Grok scheduler tools (thin shells).

Ported from grok-build/crates/codegen/xai-grok-tools/src/implementations/grok_build/scheduler/
  create.rs  — SchedulerCreateTool schema, description, run → Create output text
  delete.rs  — SchedulerDeleteTool schema, description, success/fail messages
  list.rs    — SchedulerListTool schema, description, empty / JSON summaries
  interval.rs / types.rs / actor.rs — via tools/scheduler.py + grok_build.scheduler_*

Pure logic: ``codedoggy.tools.grok_build.scheduler_interval``,
``codedoggy.tools.grok_build.scheduler_types``, ``codedoggy.tools.scheduler``.

Storage / timer divergence:
  Grok: SchedulerHandle → SchedulerActor (tokio timer + notifications + Resources).
  CodeDoggy: in-process Scheduler on ctx.extra / kernel; host must poll due_tasks.
  Model-facing create/delete/list strings and interval errors match Grok.
"""

from __future__ import annotations

import json
from typing import Any

from codedoggy.tools.grok_build.scheduler_interval import interval_to_human
from codedoggy.tools.grok_build.scheduler_types import (
    ScheduledTask,
    SchedulerError,
    to_rfc3339,
    truncate_prompt,
)
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolError,
    ToolId,
)
from codedoggy.tools.scheduler import ensure_scheduler

# ── Grok description_template strings ────────────────────────────────

_CREATE_DESC = """\
Create a scheduled task that runs a prompt on a recurring interval.

Set fire_immediately: true to also fire once on creation; by default the first run waits for the interval.

Usage notes:
- Interval format: "5m" (minutes), "2h" (hours), "1d" (days), "60s" (seconds, min 60)
- Maximum 50 scheduled tasks at once
- Recurring tasks auto-expire after 7 days"""

_DELETE_DESC = """\
Cancel a scheduled task by ID.

Returns success: true if the task was found and removed, false if no task with that ID exists."""

_LIST_DESC = (
    "List all active scheduled tasks with their IDs, prompts, intervals, and next fire times."
)


def _lenient_bool(raw: Any, *, default: bool) -> bool:
    """Grok deserialize_lenient_bool subset for optional bool fields."""
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no", ""}:
            return False
    return bool(raw)


def _format_create_output(task_id: str, human_schedule: str, recurring: bool) -> str:
    # types/output.rs ToolOutput::SchedulerCreate
    return (
        f"Scheduled task created (ID: {task_id}, {human_schedule}, "
        f"recurring: {str(recurring).lower()})."
    )


def _format_delete_success(task_id: str) -> str:
    return f"Scheduled task {task_id} cancelled."


def _format_delete_not_found(task_id: str) -> str:
    return (
        f"No scheduled task with ID {task_id} found. "
        "Use scheduler_list to see active tasks."
    )


def _task_summary(t: ScheduledTask) -> dict[str, Any]:
    """list.rs ScheduledTaskSummary (serde camelCase)."""
    return {
        "id": t.id,
        "prompt": truncate_prompt(t.prompt, 80),
        "intervalHuman": interval_to_human(t.interval_secs),
        "nextFireAt": to_rfc3339(t.next_fire_at()),
        "createdAt": to_rfc3339(t.created_at),
        "recurring": t.recurring,
    }


def _format_list_output(tasks: list[ScheduledTask]) -> str:
    # types/output.rs ToolOutput::SchedulerList
    if not tasks:
        return "No scheduled tasks."
    summaries = [_task_summary(t) for t in tasks]
    return json.dumps(summaries, indent=2, ensure_ascii=False)


class SchedulerCreateTool(Tool):
    def id(self) -> ToolId:
        return ToolId("scheduler_create")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        # Mutating schedule registration — EXECUTE ladder, not opaque Other.
        return ToolKind.Execute

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(
            name="scheduler_create",
            description=_CREATE_DESC,
        )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "interval": {
                    "type": "string",
                    "description": 'Interval between executions, e.g. "5m", "2h", "1d"',
                },
                "prompt": {
                    "type": "string",
                    "description": "The prompt text to execute on each scheduled fire",
                },
                "recurring": {
                    "type": "boolean",
                    "description": (
                        "Whether the task repeats (true) or fires once (false). "
                        "Default: true"
                    ),
                },
                "durable": {
                    "type": "boolean",
                    "description": (
                        "Whether the task persists across sessions. Default: false"
                    ),
                },
                "fire_immediately": {
                    "type": "boolean",
                    "description": (
                        "Whether to fire immediately on creation (true) or wait for "
                        "the first interval (false). Default: false"
                    ),
                },
            },
            "required": ["interval", "prompt"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        interval = str(args.get("interval") if args.get("interval") is not None else "")
        # Grok passes prompt through as-is (may be empty)
        prompt = args.get("prompt")
        if prompt is None:
            prompt = ""
        else:
            prompt = str(prompt)

        recurring = _lenient_bool(args.get("recurring"), default=True)
        # durable: Option<bool>, default None → false
        durable_raw = args.get("durable")
        durable = False if durable_raw is None else _lenient_bool(durable_raw, default=False)
        fire_immediately = _lenient_bool(args.get("fire_immediately"), default=False)

        sched = ensure_scheduler(ctx.extra)
        try:
            task = sched.create(
                interval=interval,
                prompt=prompt,
                recurring=recurring,
                durable=durable,
                fire_immediately=fire_immediately,
            )
        except SchedulerError as e:
            # create.rs maps both parse + limit to invalid_arguments
            raise ToolError.invalid_arguments(str(e)) from e

        return _format_create_output(
            task.id,
            interval_to_human(task.interval_secs),
            task.recurring,
        )


class SchedulerDeleteTool(Tool):
    def id(self) -> ToolId:
        return ToolId("scheduler_delete")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Execute

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(
            name="scheduler_delete",
            description=_DELETE_DESC,
        )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The task ID to cancel (from scheduler_create output)",
                },
            },
            "required": ["id"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        tid = str(args.get("id") if args.get("id") is not None else "")
        sched = ensure_scheduler(ctx.extra)
        removed = sched.delete(tid)
        if removed:
            return _format_delete_success(tid)
        return _format_delete_not_found(tid)


class SchedulerListTool(Tool):
    def id(self) -> ToolId:
        return ToolId("scheduler_list")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        # Read-only inventory — must stay allowed under READ_ONLY capability.
        return ToolKind.Read

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(
            name="scheduler_list",
            description=_LIST_DESC,
        )

    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        sched = ensure_scheduler(ctx.extra)
        return _format_list_output(sched.list())
