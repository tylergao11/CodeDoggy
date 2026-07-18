"""parallel_tasks — tool MAIN may call to fan out (not auto-orchestration).

Product rule: the harness never decides to parallelize for MAIN. This tool
runs only when MAIN invokes it. Optional ``wait=false`` lets MAIN keep doing
serial work after dispatch; that is still MAIN's choice, not a runtime policy.
"""

from __future__ import annotations

from typing import Any

from codedoggy.orchestration.subagent import (
    SubagentRequest,
    format_parallel_aggregate,
    format_parallel_dispatched,
)
from codedoggy.orchestration.types import CapabilityMode, IsolationMode
from codedoggy.tools.grok_build.task_format import (
    DEFAULT_SUBAGENT_TYPE,
    MAX_SUBAGENT_DEPTH,
    depth_limit_error_message,
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

# Cap fan-out size so one call cannot starve the pool forever.
MAX_PARALLEL_TASKS = 12
DEFAULT_PARALLEL_WAIT_MS = 180_000  # 3 minutes shared budget
MAX_PARALLEL_WAIT_MS = 600_000


_DESC = (
    "Optional tool for MAIN to fan out independent sub-tasks in one call. "
    "Nothing is auto-dispatched — only runs when you invoke it. "
    "wait=true (default): block until all finish and return a structured aggregate. "
    "wait=false: start children in background, return task_ids immediately so you "
    "can keep doing your own serial work, then join later with "
    "wait_commands_or_subagents / get_command_or_subagent_output."
)


class ParallelTasksTool(Tool):
    """MAIN-invoked multi-spawn helper; not an automatic orchestrator."""

    def id(self) -> ToolId:
        return ToolId("parallel_tasks")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Task

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="parallel_tasks", description=_DESC)

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": (
                        "Independent sub-tasks to run in parallel. "
                        f"Max {MAX_PARALLEL_TASKS}."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Full task prompt for this child.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Short label (3–5 words).",
                            },
                            "subagent_type": {
                                "type": "string",
                                "description": (
                                    'Built-ins: "general-purpose", "explore", "plan". '
                                    f'Default "{DEFAULT_SUBAGENT_TYPE}".'
                                ),
                            },
                            "capability_mode": {
                                "type": "string",
                                "description": (
                                    '"read-only", "read-write", "execute", or "all".'
                                ),
                            },
                            "isolation": {
                                "type": "string",
                                "description": '"none" (default) or "worktree".',
                            },
                        },
                        "required": ["prompt", "description"],
                    },
                    "minItems": 1,
                },
                "wait": {
                    "type": "boolean",
                    "description": (
                        "Your choice. true (default): wait for all and return the "
                        "aggregate. false: return task_ids immediately so you can "
                        "continue other work, then join later."
                    ),
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": (
                        "Shared wait budget when wait=true (ms). "
                        f"Default {DEFAULT_PARALLEL_WAIT_MS}; "
                        f"capped at {MAX_PARALLEL_WAIT_MS}. "
                        "Ignored when wait=false (except 0 still snapshots after spawn)."
                    ),
                    "minimum": 0,
                },
            },
            "required": ["tasks"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        bag = ctx.extra or {}
        coord = bag.get("subagent_coordinator")
        run_fn = bag.get("subagent_run_fn")
        if coord is None or run_fn is None:
            raise ToolError(
                "parallel_tasks requires subagent coordinator (missing backend).",
                code="missing_resource",
            )

        depth = _read_depth(bag)
        if depth >= MAX_SUBAGENT_DEPTH:
            raise ToolError.invalid_arguments(
                depth_limit_error_message(depth, MAX_SUBAGENT_DEPTH)
            )

        raw_tasks = args.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ToolError.invalid_arguments("tasks must be a non-empty list")
        if len(raw_tasks) > MAX_PARALLEL_TASKS:
            raise ToolError.invalid_arguments(
                f"tasks exceeds maximum of {MAX_PARALLEL_TASKS} entries."
            )

        wait = _parse_wait(args.get("wait"), default=True)

        requests: list[SubagentRequest] = []
        parent_sid = ctx.session_id or ""
        for i, item in enumerate(raw_tasks):
            if not isinstance(item, dict):
                raise ToolError.invalid_arguments(
                    f"tasks[{i}] must be an object with prompt + description"
                )
            prompt = str(item.get("prompt") or "").strip()
            if not prompt:
                raise ToolError.invalid_arguments(f"tasks[{i}].prompt is required")
            description = str(item.get("description") or "").strip() or f"task-{i + 1}"
            st = str(item.get("subagent_type") or "").strip() or DEFAULT_SUBAGENT_TYPE
            cap = None
            raw_cap = item.get("capability_mode")
            if isinstance(raw_cap, str) and raw_cap.strip():
                cap = CapabilityMode.parse(raw_cap)
            isolation = IsolationMode.parse(
                str(item.get("isolation")) if item.get("isolation") is not None else None
            )
            requests.append(
                SubagentRequest(
                    subagent_type=st,
                    prompt=prompt,
                    description=description,
                    parent_session_id=parent_sid,
                    run_in_background=True,
                    capability_mode=cap,
                    isolation=isolation,
                )
            )

        # Optional host allowlist for types (same as spawn_subagent)
        available = bag.get("subagent_available_types")
        if isinstance(available, (list, tuple, set)) and available:
            allow = {str(x).strip().lower() for x in available if str(x).strip()}
            for i, req in enumerate(requests):
                if req.subagent_type.strip().lower() not in allow:
                    raise ToolError.invalid_arguments(
                        f"tasks[{i}]: unknown subagent_type {req.subagent_type!r}"
                    )

        spawn_many = getattr(coord, "spawn_many", None)
        if callable(spawn_many):
            snaps = spawn_many(requests, run_fn=run_fn)
        else:
            snaps = []
            for req in requests:
                req.run_in_background = True
                snaps.append(coord.spawn(req, run_fn=run_fn))

        if not wait:
            return format_parallel_dispatched(snaps)

        timeout_raw = args.get("timeout_ms")
        if timeout_raw is None:
            timeout_ms = DEFAULT_PARALLEL_WAIT_MS
        else:
            try:
                timeout_ms = int(timeout_raw)
            except (TypeError, ValueError) as e:
                raise ToolError.invalid_arguments(
                    f"invalid timeout_ms: {timeout_raw}"
                ) from e
            if timeout_ms < 0:
                raise ToolError.invalid_arguments("timeout_ms must be >= 0")
            if timeout_ms == 0:
                timeout_ms = 0
            else:
                timeout_ms = min(timeout_ms, MAX_PARALLEL_WAIT_MS)

        ids = [s.subagent_id for s in snaps]
        wait_all = getattr(coord, "wait_all", None)
        if timeout_ms == 0:
            final = [coord.lookup(i) or s for i, s in zip(ids, snaps)]
        elif callable(wait_all):
            final = wait_all(ids, timeout_ms=timeout_ms)
        else:
            final = []
            for sid in ids:
                final.append(coord.wait(sid, timeout_ms=timeout_ms) or snaps[0])

        return format_parallel_aggregate(final)


def _parse_wait(raw: Any, *, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "on", "wait"}:
        return True
    if s in {"0", "false", "no", "off", "nowait", "background"}:
        return False
    return default


def _read_depth(bag: dict[str, Any]) -> int:
    raw = bag.get("subagent_depth")
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0
