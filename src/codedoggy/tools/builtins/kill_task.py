"""kill_task — terminate a background shell task or subagent.

Ported from:
  grok-build/.../grok_build/kill_task/mod.rs (KillTaskTool)
  grok-build/.../types/output.rs (KillTask to_prompt_format)
  grok-build/.../common/xai-tool-types/src/task.rs (build_kill_task_description)

Grok product name: kill_command_or_subagent.

Description: Grok ``build_kill_task_description`` (Job Object verb on Windows).
Kill path: shared ``job_object.kill_process_tree`` — Windows TerminateJobObject
+ child kill (Grok start_kill); POSIX killpg. **No taskkill** (Grok has none).
"""

from __future__ import annotations

from typing import Any

from codedoggy.tools.grok_build.task_output_logic import (
    KILL_MSG_ALREADY_EXITED,
    KILL_MSG_SUBAGENT_CANCEL,
    KILL_MSG_TERMINATED,
    build_kill_task_description,
    format_kill_result,
    kill_not_found_message,
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
from codedoggy.tools.task_manager import ensure_task_manager

_DESC = build_kill_task_description()


class KillTaskTool(Tool):
    def id(self) -> ToolId:
        return ToolId("kill_task")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.KillTaskAction

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="kill_task", description=_DESC)

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task ID to terminate",
                },
            },
            "required": ["task_id"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        task_id = str(args.get("task_id") or "").strip()
        if not task_id:
            raise ToolError.invalid_arguments("task_id is required")

        tm = ensure_task_manager(ctx.extra)
        outcome, message = tm.kill(task_id)
        if outcome == "killed":
            return format_kill_result(outcome="killed", message=KILL_MSG_TERMINATED)
        if outcome == "already_exited":
            return format_kill_result(
                outcome="already_exited", message=KILL_MSG_ALREADY_EXITED
            )
        # Backward-compat if task_manager still returns already_completed
        if outcome == "already_completed":
            return format_kill_result(
                outcome="already_exited", message=KILL_MSG_ALREADY_EXITED
            )
        if outcome != "not_found":
            return format_kill_result(outcome=outcome, message=message)

        # Subagent path (Grok: SubagentBackend cancel)
        coord = (ctx.extra or {}).get("subagent_coordinator")
        if coord is not None:
            cancel = getattr(coord, "cancel", None)
            if callable(cancel):
                try:
                    ok = cancel(task_id)
                except Exception:  # noqa: BLE001
                    ok = False
                if ok:
                    return format_kill_result(
                        outcome="killed", message=KILL_MSG_SUBAGENT_CANCEL
                    )
            lookup = getattr(coord, "lookup", None)
            if callable(lookup):
                snap = lookup(task_id)
                if snap is not None:
                    status = getattr(snap, "status", "")
                    if status in {"completed", "failed", "cancelled"}:
                        return format_kill_result(
                            outcome="already_exited",
                            message=f"Subagent already {status}",
                        )
                    # Found but cancel failed / still running — report kill attempt
                    return format_kill_result(
                        outcome="killed", message=KILL_MSG_SUBAGENT_CANCEL
                    )

        return kill_not_found_message(task_id, tm.known_ids())
