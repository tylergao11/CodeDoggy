"""wait_tasks — Grok wait_tasks / wait_commands_or_subagents.

Ported from:
  grok-build/.../grok_build/task_output/wait_tasks.rs
  grok-build/.../common/xai-tool-types/src/task.rs (WaitTasksToolInput, description)

Prefer get_task_output with positive timeout_ms (wait-all). This tool remains
for older prompts; mode wait_any is still honored here only.
"""

from __future__ import annotations

from typing import Any, Optional

from codedoggy.tools.builtins.get_task_output import run_multi_wait
from codedoggy.tools.grok_build.task_output_logic import (
    DEFAULT_WAIT_TIMEOUT_MS,
    MAX_MULTI_WAIT_IDS,
    build_wait_tasks_description,
    capped_wait_timeout_ms,
    resolve_task_ids,
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

_DESC = build_wait_tasks_description()


class WaitTasksTool(Tool):
    def id(self) -> ToolId:
        return ToolId("wait_tasks")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.WaitTasksAction

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="wait_tasks", description=_DESC)

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Task IDs to wait for",
                },
                "mode": {
                    "type": "string",
                    "enum": ["wait_all", "wait_any"],
                    "description": (
                        "Wait mode: 'wait_any' (return when first completes) or "
                        "'wait_all' (wait for all)"
                    ),
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Max wait time in milliseconds",
                    "minimum": 0,
                },
            },
            "required": ["task_ids", "mode"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        raw_ids = args.get("task_ids")
        if not isinstance(raw_ids, list):
            raise ToolError.invalid_arguments("task_ids must not be empty.")
        task_ids = resolve_task_ids([str(x) for x in raw_ids])
        if not task_ids:
            raise ToolError.invalid_arguments("task_ids must not be empty.")
        if len(task_ids) > MAX_MULTI_WAIT_IDS:
            raise ToolError.invalid_arguments(
                f"task_ids exceeds maximum of {MAX_MULTI_WAIT_IDS} entries."
            )

        mode = str(args.get("mode") or "").strip().lower()
        if mode not in {"wait_all", "wait_any"}:
            raise ToolError.invalid_arguments(
                "mode must be 'wait_all' or 'wait_any'"
            )

        timeout_raw = args.get("timeout_ms")
        timeout_ms: Optional[int]
        if timeout_raw is None:
            timeout_ms = None
        else:
            try:
                timeout_ms = int(timeout_raw)
            except (TypeError, ValueError) as e:
                raise ToolError.invalid_arguments(
                    f"invalid timeout_ms: {timeout_raw}"
                ) from e

        if mode == "wait_all":
            # Legacy wait always blocks: omit or 0 => default budget (not a snapshot).
            ms = (
                timeout_ms
                if timeout_ms is not None and timeout_ms > 0
                else DEFAULT_WAIT_TIMEOUT_MS
            )
            ms = capped_wait_timeout_ms(ms)
            return run_multi_wait(ctx, task_ids, timeout_ms=ms, mode="wait_all")

        # wait_any: capped_wait_timeout (None → default 30s)
        ms = capped_wait_timeout_ms(timeout_ms)
        return run_multi_wait(ctx, task_ids, timeout_ms=ms, mode="wait_any")
