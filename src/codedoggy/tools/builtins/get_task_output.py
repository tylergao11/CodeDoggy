"""get_task_output — status/output for background shell tasks (and subagents).

Ported from:
  grok-build/.../grok_build/task_output/mod.rs (TaskOutputTool)
  grok-build/.../common/xai-tool-types/src/task.rs (schema + waits)
  grok-build/.../types/output.rs (to_prompt_format cards)

Grok product name: get_command_or_subagent_output.
- Omit timeout_ms → non-blocking snapshot
- Positive timeout_ms → wait (multi-id = wait-all), capped at MAX_WAIT_BLOCK
"""

from __future__ import annotations

import time
from typing import Any, Optional

from codedoggy.tools.defaults import DEFAULT_TOOL_OUTPUT_BYTES
from codedoggy.tools.grok_build.task_output_logic import (
    MAX_MULTI_WAIT_IDS,
    TaskOutputResult,
    build_task_output_description,
    capped_wait_timeout_ms,
    format_multi_result,
    format_subagent_card,
    format_task_result_card,
    not_found_result,
    resolve_task_ids,
    snapshot_to_result,
    task_output_not_found_message,
    task_output_waits,
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

_DESC = build_task_output_description()


class GetTaskOutputTool(Tool):
    def id(self) -> ToolId:
        return ToolId("get_task_output")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.BackgroundTaskAction

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="get_task_output", description=_DESC)

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Task IDs to get output from. Pass one or more; for a single "
                        "task use a one-element array. With a positive timeout_ms, "
                        "multiple ids wait until all complete. Omit timeout_ms or "
                        "pass 0 for a non-blocking status snapshot."
                    ),
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": (
                        "Max wait time in milliseconds. A positive value waits for "
                        "completion; omit or pass 0 for a non-blocking status poll."
                    ),
                    "minimum": 0,
                },
            },
            "required": [],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        raw_ids = args.get("task_ids")
        if raw_ids is None:
            raw_ids = []
        if not isinstance(raw_ids, list):
            raise ToolError.invalid_arguments("task_ids must be an array of strings")
        task_ids = resolve_task_ids([str(x) for x in raw_ids])
        if not task_ids:
            raise ToolError.invalid_arguments(
                "Provide a non-empty task_ids list."
            )
        if len(task_ids) > MAX_MULTI_WAIT_IDS:
            raise ToolError.invalid_arguments(
                f"task_ids exceeds maximum of {MAX_MULTI_WAIT_IDS} entries."
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

        waits = task_output_waits(timeout_ms)
        wait_budget_ms = capped_wait_timeout_ms(timeout_ms) if waits else 0

        tm = ensure_task_manager(ctx.extra)
        coord = (ctx.extra or {}).get("subagent_coordinator")

        if len(task_ids) == 1:
            return self._run_single(
                ctx, tm, coord, task_ids[0], waits=waits, wait_ms=wait_budget_ms
            )

        return self._run_multi(
            ctx,
            tm,
            coord,
            task_ids,
            waits=waits,
            wait_ms=wait_budget_ms,
            mode="wait_all" if waits else "poll",
        )

    def _run_single(
        self,
        ctx: ToolCallContext,
        tm: Any,
        coord: Any,
        tid: str,
        *,
        waits: bool,
        wait_ms: int,
    ) -> str:
        result = self._resolve_one(tm, coord, tid, waits=waits, wait_ms=wait_ms)
        if result is None or result.status == "not_found":
            return self._not_found(ctx, tm, coord, tid)
        return format_task_result_card(result)

    def _run_multi(
        self,
        ctx: ToolCallContext,
        tm: Any,
        coord: Any,
        task_ids: list[str],
        *,
        waits: bool,
        wait_ms: int,
        mode: str,
        wait_any: bool = False,
    ) -> str:
        """Shared multi path for get_task_output (wait-all) and wait_tasks."""
        results = self._resolve_all(tm, coord, task_ids)
        pending = [r for r in results if not r.is_terminal() and r.status != "not_found"]

        if waits and pending:
            deadline = time.monotonic() + (wait_ms / 1000.0)
            self._wait_pending(
                tm,
                coord,
                [r.task_id for r in pending],
                deadline=deadline,
                wait_any=wait_any,
            )
            results = self._resolve_all(tm, coord, task_ids)

        return format_multi_result(results, mode=mode)

    def _wait_pending(
        self,
        tm: Any,
        coord: Any,
        pending_ids: list[str],
        *,
        deadline: float,
        wait_any: bool,
    ) -> None:
        """Poll until pending complete or deadline.

        CodeDoggy glue: Grok uses event-driven Notify / join_all; we poll
        task_manager Event + coordinator.wait with shared deadline.
        """
        remaining = set(pending_ids)
        while remaining and time.monotonic() < deadline:
            done_now: list[str] = []
            slice_ms = min(
                200.0,
                max(1.0, (deadline - time.monotonic()) * 1000.0),
            )
            for tid in list(remaining):
                # Prefer bash wait (blocks up to slice)
                snap = tm.get(tid)
                if snap is not None:
                    if snap.completed:
                        done_now.append(tid)
                        continue
                    # Short wait on this id
                    waited = tm.wait(tid, timeout_ms=slice_ms / max(1, len(remaining)))
                    if waited is not None and waited.completed:
                        done_now.append(tid)
                        continue
                elif coord is not None:
                    s = self._lookup_sub(coord, tid)
                    if s is None:
                        done_now.append(tid)  # vanished → treat as resolved
                        continue
                    if not getattr(s, "is_running", False) and getattr(
                        s, "status", ""
                    ) not in {"pending", "running", ""}:
                        done_now.append(tid)
                        continue
                    wait_fn = getattr(coord, "wait", None)
                    if callable(wait_fn):
                        try:
                            s2 = wait_fn(
                                tid,
                                timeout_ms=int(slice_ms / max(1, len(remaining))),
                            )
                        except Exception:  # noqa: BLE001
                            s2 = None
                        if s2 is not None and not getattr(s2, "is_running", True):
                            done_now.append(tid)
            for tid in done_now:
                remaining.discard(tid)
            if wait_any and done_now:
                return
            if remaining:
                time.sleep(0.02)

    def _resolve_all(
        self, tm: Any, coord: Any, task_ids: list[str]
    ) -> list[TaskOutputResult]:
        out: list[TaskOutputResult] = []
        for tid in task_ids:
            r = self._resolve_one(tm, coord, tid, waits=False, wait_ms=0)
            out.append(r if r is not None else not_found_result(tid))
        return out

    def _resolve_one(
        self,
        tm: Any,
        coord: Any,
        tid: str,
        *,
        waits: bool,
        wait_ms: int,
    ) -> TaskOutputResult | None:
        if waits:
            snap = tm.wait(tid, timeout_ms=float(wait_ms))
        else:
            snap = tm.get(tid)
        if snap is not None:
            return snapshot_to_result(
                snap,
                read_file_name="read_file",
                max_output_chars=DEFAULT_TOOL_OUTPUT_BYTES,
            )

        if coord is None:
            return None
        if waits:
            wait_fn = getattr(coord, "wait", None)
            if callable(wait_fn):
                try:
                    s = wait_fn(tid, timeout_ms=wait_ms)
                except Exception:  # noqa: BLE001
                    s = self._lookup_sub(coord, tid)
            else:
                s = self._lookup_sub(coord, tid)
        else:
            s = self._lookup_sub(coord, tid)
        if s is None:
            return None
        return format_subagent_card(s)

    @staticmethod
    def _lookup_sub(coord: Any, tid: str) -> Any | None:
        lookup = getattr(coord, "lookup", None)
        if not callable(lookup):
            return None
        try:
            return lookup(tid)
        except Exception:  # noqa: BLE001
            return None

    def _not_found(
        self, ctx: ToolCallContext, tm: Any, coord: Any, tid: str
    ) -> str:
        known = list(tm.known_ids())
        if coord is not None:
            list_fn = getattr(coord, "list_for_parent", None)
            if callable(list_fn):
                try:
                    for s in list_fn(ctx.session_id or "") or []:
                        sid = getattr(s, "subagent_id", None)
                        if sid:
                            known.append(str(sid))
                except Exception:  # noqa: BLE001
                    pass
        return task_output_not_found_message(tid, known)


# Public helper used by wait_tasks
def run_multi_wait(
    ctx: ToolCallContext,
    task_ids: list[str],
    *,
    timeout_ms: int,
    mode: str,
) -> str:
    """Shared entry for wait_tasks (wait_all / wait_any)."""
    tool = GetTaskOutputTool()
    tm = ensure_task_manager(ctx.extra)
    coord = (ctx.extra or {}).get("subagent_coordinator")
    return tool._run_multi(
        ctx,
        tm,
        coord,
        task_ids,
        waits=True,
        wait_ms=timeout_ms,
        mode=mode,
        wait_any=(mode == "wait_any"),
    )
