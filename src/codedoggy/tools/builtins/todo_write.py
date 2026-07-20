"""todo_write — session todo list (Grok TodoWrite).

Ported from grok-build/crates/codegen/xai-grok-tools/src/implementations/grok_build/todo/mod.rs

Pure logic: ``codedoggy.tools.grok_build.todo_logic``

Storage divergence (documented):
  Grok persists ``State<TodoState>`` in Resources (``grok_build.Todo``).
  CodeDoggy keeps session state on ``ctx.extra["todo_state"]`` / kernel.todo_state
  (same bag pattern as other Doggy orchestration tools). Merge/replace/summary
  semantics and model-facing strings match Grok.
"""

from __future__ import annotations

from typing import Any

from codedoggy.tools.grok_build.todo_logic import (
    TodoItem,
    TodoState,
    apply_todo_write,
    parse_todo_updates,
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

# Re-export state types for callers / kernel isinstance checks
__all__ = ["TodoWriteTool", "TodoState", "TodoItem"]

# Grok description_template
_DESC = """\
Create and manage a structured task list. The user sees this list live — it is your primary way to show progress.

Use for any task with 3+ steps. Skip for trivial single-step work.
"""


def _get_state(ctx: ToolCallContext) -> TodoState:
    """Resolve session TodoState from extra / kernel (CodeDoggy storage)."""
    bag = ctx.extra if ctx.extra is not None else {}
    st = bag.get("todo_state")
    if isinstance(st, TodoState):
        return st
    # Prefer kernel-owned state
    kernel = bag.get("kernel")
    if kernel is not None:
        existing = getattr(kernel, "todo_state", None)
        if isinstance(existing, TodoState):
            bag["todo_state"] = existing
            return existing
        st = TodoState()
        try:
            kernel.todo_state = st
        except Exception:  # noqa: BLE001
            pass
        bag["todo_state"] = st
        return st
    st = TodoState()
    bag["todo_state"] = st
    return st


def _parse_merge(raw: Any) -> bool:
    """Default true (Grok default_merge). Lenient-ish for common model slips."""
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    return bool(raw)


class TodoWriteTool(Tool):
    def id(self) -> ToolId:
        return ToolId("todo_write")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        # Grok marks this as ToolKind::Plan; CodeDoggy surface keeps Todo.
        return ToolKind.Todo

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="todo_write", description=_DESC.strip())

    def parameters_schema(self) -> dict[str, Any]:
        # Schemars descriptions from TodoWriteInput / TodoUpdate
        return {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "Array of todo items to write to the workspace",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique identifier for the todo item",
                            },
                            "content": {
                                "type": "string",
                                "description": "The description/content of the todo item",
                            },
                            "status": {
                                "type": "string",
                                "enum": [
                                    "pending",
                                    "in_progress",
                                    "completed",
                                    "cancelled",
                                ],
                                "description": (
                                    "The status of the todo item: pending, "
                                    "in_progress, completed, or cancelled"
                                ),
                            },
                        },
                        "required": ["id"],
                    },
                },
                "merge": {
                    "type": "boolean",
                    "description": (
                        "Optional. When true (default), merges the provided todos "
                        "into the existing list by id — send only the items you are "
                        "changing, and to flip status without changing content send "
                        "just id + status. When false, the provided todos replace "
                        "the existing list."
                    ),
                },
            },
            "required": ["todos"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        # Grok: TodoWriteInput { merge (default true), todos: Vec }
        todos_raw = args.get("todos")
        if not isinstance(todos_raw, list):
            raise ToolError.invalid_arguments("todos must be an array")
        # Empty list is valid (clears on replace / no-op on empty merge)
        try:
            updates = parse_todo_updates(todos_raw)
        except ValueError as e:
            raise ToolError.invalid_arguments(str(e)) from e

        merge = _parse_merge(args.get("merge"))
        state = _get_state(ctx)

        # Grok prompt_text = summary_for_prompt (or DuplicateId string).
        # No "Todos updated." prefix.
        out = apply_todo_write(state, merge=merge, updates=updates)
        bag = ctx.extra if ctx.extra is not None else {}
        # Persist then notify host (Grok write_plan_state after mutation).
        # Child agents use session_id ``parent:sub_id`` (isolated todo file).
        kernel = bag.get("kernel")
        if kernel is not None and hasattr(kernel, "persist_todo_state"):
            try:
                kernel.persist_todo_state()
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                from codedoggy.tools.grok_build.todo_logic import save_todo_state

                sid = str(
                    getattr(ctx, "session_id", None)
                    or bag.get("session_id")
                    or (
                        f"{bag.get('parent_session_id')}:{bag.get('subagent_id')}"
                        if bag.get("parent_session_id") and bag.get("subagent_id")
                        else ""
                    )
                    or getattr(kernel, "session_id", None)
                    or ""
                )
                if sid and ctx.cwd is not None:
                    save_todo_state(state, cwd=ctx.cwd, session_id=sid)
            except Exception:  # noqa: BLE001
                pass
        notify = bag.get("todo_changed_fn")
        if callable(notify):
            try:
                notify()
            except Exception:  # noqa: BLE001
                pass
        return out
