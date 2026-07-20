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


def test_todo_counts_badge_excludes_cancelled() -> None:
    from codedoggy.tools.grok_build.todo_logic import TodoItem, count_todos

    state = TodoState()
    state.push("a", TodoItem("one", status="completed"))
    state.push("b", TodoItem("two", status="completed"))
    state.push("c", TodoItem("three", status="pending"))
    state.push("d", TodoItem("skip", status="cancelled"))
    counts = count_todos(state)
    assert counts.completed == 2
    assert counts.pending == 1
    assert counts.cancelled == 1
    # Grok: cancelled not in denominator → 2/3
    assert counts.badge_text() == "2/3"
    assert count_todos(TodoState()).badge_text() is None


def test_todo_write_notifies_host_callback(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    seen: list[int] = []
    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={"todo_changed_fn": lambda: seen.append(1)},
    )
    tools.call(
        "todo_write",
        {
            "merge": False,
            "todos": [{"id": "1", "content": "A", "status": "pending"}],
        },
        ctx,
    )
    assert seen == [1]


def test_todo_state_json_round_trip(tmp_path: Path) -> None:
    from codedoggy.tools.grok_build.todo_logic import (
        TodoItem,
        load_todo_state,
        save_todo_state,
        todo_state_json_path,
    )

    state = TodoState()
    state.push("1", TodoItem("A", status="completed"))
    state.push("2", TodoItem("B", status="pending"))
    state.push("3", TodoItem("C", status="cancelled"))
    path = save_todo_state(state, cwd=tmp_path, session_id="s1")
    assert path == todo_state_json_path(tmp_path, "s1")
    assert path is not None and path.is_file()
    restored = load_todo_state(cwd=tmp_path, session_id="s1")
    assert restored is not None
    items = list(restored.todo_items_with_ids())
    assert [t for t, _ in items] == ["1", "2", "3"]
    assert items[0][1].status == "completed"
    assert items[2][1].status == "cancelled"
    # bad json soft-fails
    path.write_text("{not json", encoding="utf-8")
    assert load_todo_state(cwd=tmp_path, session_id="s1") is None


def test_child_session_todo_path_isolated(tmp_path: Path) -> None:
    """Subagent todos land under parent:child session id, not MAIN."""
    from codedoggy.tools.grok_build.todo_logic import (
        load_todo_state,
        todo_state_json_path,
    )

    tools = ToolRegistryBuilder.new().finalize()
    main_ctx = ToolCallContext(
        cwd=tmp_path,
        session_id="sess-main",
        extra={},
    )
    tools.call(
        "todo_write",
        {
            "merge": False,
            "todos": [{"id": "m", "content": "MAIN item", "status": "pending"}],
        },
        main_ctx,
    )
    child_ctx = ToolCallContext(
        cwd=tmp_path,
        session_id="sess-main:child-1",
        extra={
            "is_subagent": True,
            "parent_session_id": "sess-main",
            "subagent_id": "child-1",
        },
    )
    tools.call(
        "todo_write",
        {
            "merge": False,
            "todos": [{"id": "c", "content": "CHILD item", "status": "completed"}],
        },
        child_ctx,
    )
    main = load_todo_state(cwd=tmp_path, session_id="sess-main")
    child = load_todo_state(cwd=tmp_path, session_id="sess-main:child-1")
    assert main is not None and main.get("m") is not None
    assert child is not None and child.get("c") is not None
    assert main.get("c") is None
    assert child.get("m") is None
    assert todo_state_json_path(tmp_path, "sess-main:child-1").is_file()


def test_kernel_persist_load_todo(tmp_path: Path) -> None:
    from codedoggy.session.kernel import RuntimeKernel
    from codedoggy.tools.grok_build.todo_logic import TodoItem

    k = RuntimeKernel(cwd=tmp_path, session_id="k-todo")
    from codedoggy.tools.grok_build.todo_logic import TodoState

    st = TodoState()
    st.push("x", TodoItem("do it", status="in_progress"))
    k.todo_state = st
    k.persist_todo_state()
    k2 = RuntimeKernel(cwd=tmp_path, session_id="k-todo")
    assert k2.load_todo_state() is True
    assert k2.todo_state is not None
    item = k2.todo_state.get("x")
    assert item is not None and item.status == "in_progress"


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
