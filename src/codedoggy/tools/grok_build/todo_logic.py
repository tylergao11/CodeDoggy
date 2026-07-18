"""TodoWrite pure logic — source port from Grok.

Ported from grok-build/crates/codegen/xai-grok-tools/src/implementations/grok_build/todo/mod.rs

Maps 1:1 where practical:
  TodoStatus / tag, TodoPriority, TodoItem, TodoState, TodoUpdate
  validate_no_duplicate_ids, apply_replace, apply_merge
  summarize_todo_state, effective_merge (auto-upgrade)
  DUPLICATE_ID_MSG formatting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional


# ── statuses / priority (mod.rs) ─────────────────────────────────────

VALID_STATUSES = frozenset({"pending", "in_progress", "completed", "cancelled"})

_STATUS_TAGS: dict[str, str] = {
    "pending": "[pending]",
    "in_progress": "[in_progress]",
    "completed": "[completed]",
    "cancelled": "[cancelled]",
}


def status_tag(status: str) -> str:
    """Grok TodoStatus::tag."""
    return _STATUS_TAGS.get(status, f"[{status}]")


# ── data ─────────────────────────────────────────────────────────────


@dataclass
class TodoItem:
    """Grok TodoItem (id is map key on TodoState, not a field)."""

    content: str
    status: str = "pending"
    priority: str = "medium"
    meta: Any | None = None


@dataclass
class TodoUpdate:
    """Grok TodoUpdate — partial fields allowed on merge."""

    id: str
    content: Optional[str] = None
    status: Optional[str] = None

    def has_no_content(self) -> bool:
        """True when update carries no meaningful content (None or empty)."""
        return self.content is None or self.content == ""


@dataclass
class TodoState:
    """Grok TodoState — insertion-ordered id → item map."""

    # list of (id, item) preserves IndexMap order
    _items: dict[str, TodoItem] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)

    def push(self, todo_id: str, item: TodoItem) -> None:
        if todo_id not in self._items:
            self._order.append(todo_id)
        self._items[todo_id] = item

    def clear(self) -> None:
        self._items.clear()
        self._order.clear()

    def update(
        self,
        todo_id: str,
        content: Optional[str],
        status: Optional[str],
    ) -> bool:
        """Partial update. Returns False if id missing.

        Empty-string content is ignored (must not wipe existing content).
        """
        item = self._items.get(todo_id)
        if item is None:
            return False
        if content is not None and content != "":
            item.content = content
        if status is not None:
            item.status = status
        return True

    def is_empty(self) -> bool:
        return not self._order

    def has_id(self, todo_id: str) -> bool:
        return todo_id in self._items

    def todo_items(self) -> Iterator[TodoItem]:
        for tid in self._order:
            yield self._items[tid]

    def todo_items_with_ids(self) -> Iterator[tuple[str, TodoItem]]:
        for tid in self._order:
            yield tid, self._items[tid]

    def get(self, todo_id: str) -> TodoItem | None:
        return self._items.get(todo_id)


# ── errors (Display strings from TodoError) ──────────────────────────


class TodoLogicError(Exception):
    """Logic-layer error (Grok TodoError)."""

    pass


def duplicate_todo_id_message(todo_id: str) -> str:
    """Model-facing DuplicateId output string (run path, not enum Display)."""
    return (
        f'Duplicate todo ID in request: "{todo_id}". '
        "Each todo item must have a unique ID."
    )


# Enum Display (kept for fidelity; run path uses duplicate_todo_id_message):
#   "Duplicate Todo ID in response: {id}"
#   "Missing Todo content in mode: {mode}"
#   "Missing Todo ID in mode: {mode}"


# ── core ops ─────────────────────────────────────────────────────────


def validate_no_duplicate_ids(updates: Iterable[TodoUpdate]) -> str | None:
    """Return first duplicate id, or None if unique.

    Grok: Err(TodoError::DuplicateTodoID(id)).
    """
    seen: set[str] = set()
    for u in updates:
        if u.id in seen:
            return u.id
        seen.add(u.id)
    return None


def apply_replace(state: TodoState, updates: list[TodoUpdate]) -> None:
    """merge=false: incoming list fully replaces existing state.

    Missing/empty content → id fallback. Missing status → pending.
    """
    state.clear()
    for u in updates:
        content = u.id if u.has_no_content() else (u.content or u.id)
        status = u.status if u.status is not None else "pending"
        state.push(u.id, TodoItem(content=content, status=status))


def apply_merge(state: TodoState, updates: list[TodoUpdate]) -> None:
    """merge=true: partial updates by id.

    Existing: content optional (omit / empty keeps prior).
    New: content falls back to id; status defaults to pending.
    """
    for u in updates:
        if state.update(u.id, u.content, u.status):
            continue
        content = u.id if u.has_no_content() else (u.content or u.id)
        status = u.status if u.status is not None else "pending"
        state.push(u.id, TodoItem(content=content, status=status))


def summarize_todo_state(state: TodoState) -> str:
    """Grok summarize_todo_state — tags are full status names."""
    if state.is_empty():
        return "No tasks currently tracked."
    lines: list[str] = []
    for tid, item in state.todo_items_with_ids():
        lines.append(f"- {status_tag(item.status)} {tid}: {item.content}")
    # writeln! leaves a trailing newline on the last line
    return "\n".join(lines) + "\n"


def effective_merge(merge: bool, state: TodoState, updates: list[TodoUpdate]) -> bool:
    """Auto-upgrade to merge when model forgot merge=true on status-only updates.

    Grok condition:
      input.merge
      || (!state.empty
          && !updates.empty
          && every update has_no_content and has_id in state)
    """
    if merge:
        return True
    if state.is_empty() or not updates:
        return False
    return all(u.has_no_content() and state.has_id(u.id) for u in updates)


def parse_todo_updates(raw_todos: list[Any]) -> list[TodoUpdate]:
    """Parse tool args into TodoUpdate list; raises ValueError with message."""
    if not isinstance(raw_todos, list):
        raise ValueError("todos must be an array")
    updates: list[TodoUpdate] = []
    for raw in raw_todos:
        if not isinstance(raw, dict):
            raise ValueError("each todo must be an object")
        tid = raw.get("id")
        if tid is None:
            raise ValueError("todo id is required")
        tid_s = str(tid)
        # Grok schema requires id: String; empty string is still a key
        content = raw.get("content")
        if content is not None:
            content = str(content)
        status = raw.get("status")
        if status is not None:
            status_s = str(status).strip()
            # accept snake_case as Grok serde rename_all
            if status_s not in VALID_STATUSES:
                # try lower
                status_s = status_s.lower()
            if status_s not in VALID_STATUSES:
                raise ValueError(
                    f"invalid status {status!r} for todo {tid_s}"
                )
            status = status_s
        updates.append(TodoUpdate(id=tid_s, content=content, status=status))
    return updates


def apply_todo_write(
    state: TodoState,
    *,
    merge: bool,
    updates: list[TodoUpdate],
) -> str:
    """Full apply path after validation: returns summary_for_prompt or duplicate msg.

    Duplicate ids → model-facing string (Grok DuplicateId output variant).
    """
    dup = validate_no_duplicate_ids(updates)
    if dup is not None:
        return duplicate_todo_id_message(dup)

    if effective_merge(merge, state, updates):
        apply_merge(state, updates)
    else:
        apply_replace(state, updates)
    return summarize_todo_state(state)
