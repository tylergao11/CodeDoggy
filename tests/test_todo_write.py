"""Focused tests for Grok TodoWrite pure logic + tool wire."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.grok_build.todo_logic import (
    TodoState,
    TodoUpdate,
    apply_merge,
    apply_replace,
    apply_todo_write,
    duplicate_todo_id_message,
    effective_merge,
    summarize_todo_state,
    validate_no_duplicate_ids,
)
from codedoggy.tools.runtime import ToolCallContext, ToolError


def _u(
    tid: str,
    content: str | None = None,
    status: str | None = None,
) -> TodoUpdate:
    return TodoUpdate(id=tid, content=content, status=status)


def _seed(items: list[tuple[str, str, str]]) -> TodoState:
    state = TodoState()
    for tid, content, status in items:
        apply_replace(state, [_u(tid, content, status)])
        # re-apply as cumulative for multi — better push via replace once
    state = TodoState()
    apply_replace(
        state,
        [_u(tid, content, status) for tid, content, status in items],
    )
    return state


# ── pure logic: replace ──────────────────────────────────────────────


def test_replace_creates_items() -> None:
    state = TodoState()
    apply_replace(
        state,
        [
            _u("1", "Task A", "pending"),
            _u("2", "Task B", "in_progress"),
        ],
    )
    items = list(state.todo_items_with_ids())
    assert len(items) == 2
    assert items[0][1].content == "Task A"
    assert items[1][1].status == "in_progress"


def test_replace_without_content_falls_back_to_id() -> None:
    state = TodoState()
    apply_replace(state, [_u("build_project", None, "pending")])
    item = state.get("build_project")
    assert item is not None
    assert item.content == "build_project"
    assert item.status == "pending"


def test_replace_empty_string_content_falls_back_to_id() -> None:
    state = TodoState()
    apply_replace(state, [_u("task_1", "", "pending")])
    assert state.get("task_1") is not None
    assert state.get("task_1").content == "task_1"  # type: ignore[union-attr]


def test_replace_clears_previous() -> None:
    state = _seed([("old", "Old task", "completed")])
    apply_replace(state, [_u("new", "New task", "pending")])
    assert not state.has_id("old")
    assert state.get("new") is not None
    assert state.get("new").content == "New task"  # type: ignore[union-attr]


# ── pure logic: merge ────────────────────────────────────────────────


def test_merge_status_only_preserves_content() -> None:
    state = _seed([("1", "Build the project", "in_progress")])
    apply_merge(state, [_u("1", None, "completed")])
    item = state.get("1")
    assert item is not None
    assert item.status == "completed"
    assert item.content == "Build the project"


def test_merge_empty_string_content_preserves_original() -> None:
    state = _seed([("1", "Build the project", "in_progress")])
    apply_merge(state, [_u("1", "", "completed")])
    item = state.get("1")
    assert item is not None
    assert item.content == "Build the project"
    assert item.status == "completed"


def test_merge_new_item_without_content_uses_id_fallback() -> None:
    state = TodoState()
    apply_merge(state, [_u("explore_codebase", None, "completed")])
    item = state.get("explore_codebase")
    assert item is not None
    assert item.content == "explore_codebase"
    assert item.status == "completed"


def test_merge_mixed_existing_and_new() -> None:
    state = _seed([("exist", "Existing task", "in_progress")])
    apply_merge(
        state,
        [
            _u("exist", None, "completed"),
            _u("fresh", "New task", "pending"),
        ],
    )
    assert state.get("exist").status == "completed"  # type: ignore[union-attr]
    assert state.get("exist").content == "Existing task"  # type: ignore[union-attr]
    assert state.get("fresh").content == "New task"  # type: ignore[union-attr]


# ── duplicates + summary ─────────────────────────────────────────────


def test_duplicate_ids_rejected() -> None:
    updates = [_u("dup", "A", "pending"), _u("dup", "B", "pending")]
    assert validate_no_duplicate_ids(updates) == "dup"
    msg = apply_todo_write(TodoState(), merge=False, updates=updates)
    assert msg == duplicate_todo_id_message("dup")
    assert "dup" in msg
    assert "unique ID" in msg


def test_summarize_empty() -> None:
    assert summarize_todo_state(TodoState()) == "No tasks currently tracked."


def test_summarize_uses_full_status_tags() -> None:
    state = _seed(
        [
            ("1", "Task A", "pending"),
            ("2", "Task B", "in_progress"),
            ("3", "Task C", "completed"),
            ("4", "Task D", "cancelled"),
        ]
    )
    s = summarize_todo_state(state)
    assert "[pending] 1: Task A" in s
    assert "[in_progress] 2: Task B" in s
    assert "[completed] 3: Task C" in s
    assert "[cancelled] 4: Task D" in s
    # Grok writeln trailing newline
    assert s.endswith("\n")


# ── auto-upgrade merge ───────────────────────────────────────────────


def test_missing_merge_flag_auto_upgrades_status_only() -> None:
    state = _seed(
        [
            ("1", "Explore codebase", "in_progress"),
            ("2", "Review tools", "pending"),
            ("3", "Write tests", "pending"),
        ]
    )
    updates = [
        _u("1", None, "completed"),
        _u("2", None, "completed"),
        _u("3", None, "in_progress"),
    ]
    assert effective_merge(False, state, updates) is True
    summary = apply_todo_write(state, merge=False, updates=updates)
    assert "Explore codebase" in summary
    assert "Review tools" in summary
    assert "Write tests" in summary
    assert state.get("1").status == "completed"  # type: ignore[union-attr]
    assert state.get("3").status == "in_progress"  # type: ignore[union-attr]


def test_replace_still_works_when_content_present() -> None:
    state = _seed([("1", "Old", "pending")])
    updates = [_u("1", "New content", "completed")]
    # content present → no auto-upgrade
    assert effective_merge(False, state, updates) is False
    apply_todo_write(state, merge=False, updates=updates)
    assert list(state.todo_items())[0].content == "New content"
    assert len(list(state.todo_items())) == 1


# ── tool wire ────────────────────────────────────────────────────────


def test_tool_replace_and_merge(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    out = tools.call(
        "todo_write",
        {
            "merge": False,
            "todos": [
                {"id": "1", "content": "first", "status": "pending"},
                {"id": "2", "content": "second", "status": "in_progress"},
            ],
        },
        ctx,
    )
    assert "first" in out and "second" in out
    assert "[pending]" in out and "[in_progress]" in out
    # Grok: no "Todos updated." prefix — summary only
    assert not out.startswith("Todos updated")

    out2 = tools.call(
        "todo_write",
        {"merge": True, "todos": [{"id": "2", "status": "completed"}]},
        ctx,
    )
    assert "[completed]" in out2
    assert "second" in out2  # content preserved
    assert "first" in out2


def test_tool_empty_todos_replace(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    tools.call(
        "todo_write",
        {"merge": False, "todos": [{"id": "1", "content": "x", "status": "pending"}]},
        ctx,
    )
    out = tools.call("todo_write", {"merge": False, "todos": []}, ctx)
    assert "No tasks currently tracked." in out


def test_tool_duplicate_id_returns_message(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    out = tools.call(
        "todo_write",
        {
            "merge": False,
            "todos": [
                {"id": "dup", "content": "A", "status": "pending"},
                {"id": "dup", "content": "B", "status": "pending"},
            ],
        },
        ctx,
    )
    assert out == duplicate_todo_id_message("dup")


def test_tool_schema_and_description() -> None:
    tools = ToolRegistryBuilder.new().finalize()
    defs = {d.name: d for d in tools.tool_definitions()}
    td = defs["todo_write"]
    assert "task list" in (td.description or "")
    props = (td.parameters or {}).get("properties") or {}
    assert "todos" in props and "merge" in props
    merge_desc = props["merge"].get("description") or ""
    assert "default" in merge_desc.lower()
    assert "id + status" in merge_desc or "id + status" in merge_desc.replace(" ", "")


def test_tool_invalid_status(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    with pytest.raises(ToolError):
        tools.call(
            "todo_write",
            {"todos": [{"id": "1", "content": "x", "status": "nope"}]},
            ctx,
        )
