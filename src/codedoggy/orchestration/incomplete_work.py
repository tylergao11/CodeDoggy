"""Incomplete-work gate — block premature turn completion.

Sample with no tool_calls is *sample-done*, not *task-done*. Before finishing
successfully, refuse when open todos / running children / bg shell tasks remain.

Plan-first (RequirePlanArtifact) is enforced only at tool prepare — not here.
Unmet plan must not block prose COMPLETED (go-steer: mutate gate, not turn end).

Single source for the loop; no per-call hardcoded lists elsewhere.
"""

from __future__ import annotations

from typing import Any


def open_todo_ids(todo_state: Any) -> list[str]:
    """Ids still pending or in_progress (TodoState.todo_items_with_ids)."""
    if todo_state is None:
        return []
    paired = getattr(todo_state, "todo_items_with_ids", None)
    if not callable(paired):
        return []
    out: list[str] = []
    try:
        for tid, item in paired():
            st = getattr(item, "status", None)
            if st in {"pending", "in_progress"}:
                out.append(str(tid))
    except Exception:  # noqa: BLE001
        return []
    return out


def running_subagent_ids(extra: dict[str, Any] | None) -> list[str]:
    bag = extra or {}
    coord = bag.get("subagent_coordinator")
    if coord is None:
        return []
    session_id = bag.get("session_id") or bag.get("parent_session_id")
    kernel = bag.get("kernel")
    if not session_id and kernel is not None:
        session_id = getattr(kernel, "session_id", None)
    if not session_id:
        return []
    list_fn = getattr(coord, "list_for_parent", None)
    if not callable(list_fn):
        return []
    try:
        snaps = list_fn(str(session_id))
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    for snap in snaps:
        status = getattr(snap, "status", None)
        if status in {"pending", "running"}:
            out.append(str(getattr(snap, "id", "") or ""))
    return [x for x in out if x]


def incomplete_work_reasons(extra: dict[str, Any] | None) -> list[str]:
    """Human/model-facing reasons the turn must not complete yet."""
    bag = dict(extra or {})
    reasons: list[str] = []

    todos = open_todo_ids(bag.get("todo_state"))
    if not todos:
        kernel = bag.get("kernel")
        if kernel is not None:
            todos = open_todo_ids(getattr(kernel, "todo_state", None))
    if todos:
        preview = ", ".join(todos[:5])
        more = f" (+{len(todos) - 5} more)" if len(todos) > 5 else ""
        reasons.append(
            f"open todos still pending/in_progress: {preview}{more}"
        )

    runners = running_subagent_ids(bag)
    if runners:
        preview = ", ".join(runners[:5])
        more = f" (+{len(runners) - 5} more)" if len(runners) > 5 else ""
        reasons.append(f"subagents still running: {preview}{more}")

    tm = bag.get("task_manager")
    if tm is None:
        kernel = bag.get("kernel")
        if kernel is not None:
            tm = getattr(kernel, "task_manager", None)
    list_fn = getattr(tm, "list_tasks", None) if tm is not None else None
    if callable(list_fn):
        try:
            bg = [
                str(getattr(s, "task_id", "") or "")
                for s in list_fn()
                if not bool(getattr(s, "completed", True))
            ]
            bg = [x for x in bg if x]
        except Exception:  # noqa: BLE001
            bg = []
        if bg:
            preview = ", ".join(bg[:5])
            more = f" (+{len(bg) - 5} more)" if len(bg) > 5 else ""
            reasons.append(f"background shell tasks still running: {preview}{more}")

    # Goal mode is often a session constraint ("only touch auth"), not a
    # completion checklist — do not block prose-stop solely on goal flags.
    # Completion of checklist-style goals is enforced via todos / update_goal.
    # Plan-first stays at tool prepare only (see tool_pipeline plan_first_denial).

    return reasons


def format_incomplete_work_nudge(reasons: list[str]) -> str:
    bullets = "\n".join(f"- {r}" for r in reasons)
    return (
        "[incomplete_work] You stopped without tool calls, but open work remains:\n"
        f"{bullets}\n"
        "Do not claim the task is done. Continue with tools (or update todos / "
        "wait for subagents) until the work is actually finished."
    )
