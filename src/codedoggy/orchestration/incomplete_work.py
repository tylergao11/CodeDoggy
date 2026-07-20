"""Incomplete-work gate — block premature turn completion.

Sample with no tool_calls is *sample-done*, not *task-done*. Before finishing
successfully, refuse when open todos / running children / bg shell tasks remain.

Plan mode is a session edit gate (enter/exit), not a completion gate.
Unmet plans must not block prose COMPLETED.

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
            # SubagentSnapshot field is subagent_id (not id). Accept id as
            # fallback for duck-typed test doubles.
            sid = getattr(snap, "subagent_id", None) or getattr(snap, "id", None)
            out.append(str(sid or ""))
    return [x for x in out if x]


def incomplete_work_reasons(extra: dict[str, Any] | None) -> list[str]:
    """Human/model-facing reasons the turn must not complete yet."""
    bag = dict(extra or {})
    reasons: list[str] = []
    is_child = bool(bag.get("is_subagent"))

    # Todos are session-scoped: MAIN only its list; child only its own.
    # Never let a child's checklist block MAIN completion (or vice versa).
    if is_child:
        todos = open_todo_ids(bag.get("todo_state"))
    else:
        kernel = bag.get("kernel")
        if kernel is not None and hasattr(kernel, "todo_state"):
            # Prefer kernel even when empty (completed-only must not fall back
            # to a polluted bag.todo_state from child tooling).
            todos = open_todo_ids(getattr(kernel, "todo_state", None))
        else:
            todos = open_todo_ids(bag.get("todo_state"))
    if todos:
        preview = ", ".join(todos[:5])
        more = f"（另{len(todos) - 5}项）" if len(todos) > 5 else ""
        scope = "子 agent" if is_child else "MAIN"
        reasons.append(
            f"未完成 todo（{scope} · pending/in_progress）: {preview}{more}"
        )

    # Only MAIN waits on children; a child does not gate on siblings.
    runners = [] if is_child else running_subagent_ids(bag)
    if runners:
        preview = ", ".join(runners[:5])
        more = f"（另{len(runners) - 5}个）" if len(runners) > 5 else ""
        reasons.append(f"子 agent 仍在运行: {preview}{more}")

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
            more = f"（另{len(bg) - 5}个）" if len(bg) > 5 else ""
            reasons.append(f"后台 shell 任务仍在运行: {preview}{more}")

    # Goal mode is often a session constraint ("only touch auth"), not a
    # completion checklist — do not block prose-stop solely on goal flags.
    # Completion of checklist-style goals is enforced via todos / update_goal.

    return reasons


def format_incomplete_work_nudge(reasons: list[str]) -> str:
    """Model-facing steer after prose-stop with open work (Chinese primary)."""
    bullets = "\n".join(f"- {r}" for r in reasons)
    return (
        "[incomplete_work] 你已停止发起工具调用，但仍有未完成工作：\n"
        f"{bullets}\n"
        "不要声称任务已完成。请继续使用工具（更新 todo / 等待或汇总子 agent），"
        "直到工作真正做完。\n"
        "(Do not claim done — continue with tools until open work is finished.)"
    )
