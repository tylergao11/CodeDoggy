"""monitor — long-running command with line-event stream.

Ported from:
  grok-build/.../grok_build/monitor/tool.rs (MonitorTool::run start path)
  grok-build/.../grok_build/monitor/types.rs (timeouts / validate)
  grok-build/.../types/output.rs (Monitor to_prompt_format)

Grok product-facing start message is plain text (not <task-id> XML).

Honest gaps (fidelity A/X):
  - No ToolNotificationHandle pipeline (MonitorEvent auto-wake) — events are
    retained on the task output file; model uses get_task_output.
  - Rate limiter / LineProcessor ported as pure modules under grok_build/ but
    not driven by a background notification actor.
  - PYTHONUNBUFFERED=1 is set on spawn (Grok).
"""

from __future__ import annotations

import os
from typing import Any, Optional

from codedoggy.tools.builtins.run_terminal_cmd import (
    _popen_kwargs,
    _policy_check_shell,
    scrub_shell_env,
)
from codedoggy.tools.defaults import BASH_BACKGROUND_MAX_RUNTIME_S
from codedoggy.tools.grok_build.monitor_types import (
    DEFAULT_KILL_TOOL_NAME,
    MONITOR_DESC,
    format_monitor_started,
    resolved_timeout_ms,
    validate_monitor_input,
    MonitorError,
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
from codedoggy.tools.util.shell import shell_command_argv


class MonitorTool(Tool):
    def id(self) -> ToolId:
        return ToolId("monitor")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Monitor

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="monitor", description=MONITOR_DESC)

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Shell command or script. Each stdout line is an event; "
                        "exit ends the watch."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Short human-readable description of what you are monitoring "
                        "(shown in every notification)."
                    ),
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": (
                        "Kill the monitor after this deadline (ms). "
                        "Default: 36000000 (10 hr)."
                    ),
                    "minimum": 0,
                },
                "persistent": {
                    "type": "boolean",
                    "description": (
                        "Run for the lifetime of the session (no timeout). "
                        f"Stop with {DEFAULT_KILL_TOOL_NAME}."
                    ),
                },
            },
            "required": ["command", "description"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolError.invalid_arguments("command is required")
        desc = args.get("description")
        if not isinstance(desc, str) or not desc.strip():
            raise ToolError.invalid_arguments("description is required")

        persistent = bool(args.get("persistent"))
        timeout_raw = args.get("timeout_ms")
        timeout_opt: Optional[int]
        if timeout_raw is None:
            timeout_opt = None
        else:
            try:
                timeout_opt = int(timeout_raw)
            except (TypeError, ValueError) as e:
                raise ToolError.invalid_arguments(
                    f"invalid timeout_ms: {timeout_raw}"
                ) from e
            if timeout_opt < 0:
                raise ToolError.invalid_arguments("timeout_ms must be non-negative")

        try:
            validate_monitor_input(timeout_ms=timeout_opt, persistent=persistent)
        except MonitorError as e:
            raise ToolError.invalid_arguments(str(e)) from e

        resolved = resolved_timeout_ms(timeout_ms=timeout_opt, persistent=persistent)
        # task_manager max_runtime_s: 0 persistent → session lifetime cap
        if resolved == 0:
            max_runtime = BASH_BACKGROUND_MAX_RUNTIME_S
        else:
            max_runtime = resolved / 1000.0

        _policy_check_shell(ctx, command)

        inv = shell_command_argv(command)
        env = scrub_shell_env({**os.environ, **inv.env})
        # Grok monitor always sets PYTHONUNBUFFERED=1
        env["PYTHONUNBUFFERED"] = "1"

        tm = ensure_task_manager(ctx.extra)
        handle = tm.spawn(
            [inv.program, *inv.args],
            command=command,
            cwd=str(ctx.cwd),
            env=env,
            description=desc.strip(),
            owner_session_id=ctx.session_id,
            kind="monitor",
            max_runtime_s=max_runtime,
            display_command=f"[monitor] {desc.strip()}",
            popen_kwargs=_popen_kwargs(),
        )

        return format_monitor_started(
            handle.task_id,
            timeout_ms=resolved,
            persistent=(resolved == 0),
            kill_tool_name=DEFAULT_KILL_TOOL_NAME,
        )
