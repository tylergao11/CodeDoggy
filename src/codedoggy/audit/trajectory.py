"""In-session mutation trajectory (audit memory of writes — not MEMORY.md)."""

from __future__ import annotations

from threading import Lock

from codedoggy.audit.types import MutationEvent


class MutationTrajectory:
    """Ordered log of workspace mutations for one session (or task scope).

    This is the auditor's working memory of *what changed*, distinct from
    curated MEMORY.md / USER.md. Hermes session_search is a different store;
    selection across stores happens in MemorySelector.
    """

    def __init__(self) -> None:
        self._events: list[MutationEvent] = []
        self._lock = Lock()

    def append(self, event: MutationEvent) -> None:
        with self._lock:
            self._events.append(event)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def events(self) -> list[MutationEvent]:
        with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def __bool__(self) -> bool:
        # Always true as a handle so ``traj or default`` never drops an empty log.
        return True

    def for_path(self, path: str) -> list[MutationEvent]:
        with self._lock:
            return [e for e in self._events if e.path == path]

    def by_agent(self, agent_id: str) -> list[MutationEvent]:
        with self._lock:
            return [e for e in self._events if e.agent_id == agent_id]

    def summary(self, *, max_events: int = 24, path_only: bool = False) -> str:
        """Compact text for model-brain auditors (token-bounded)."""
        with self._lock:
            items = list(self._events)
        if not items:
            return "(no mutations yet)"
        start = max(0, len(items) - max_events)
        window = items[start:]
        omitted = start
        lines: list[str] = []
        if omitted:
            lines.append(f"(… {omitted} earlier mutation(s) omitted)")
        for e in window:
            kind = "create" if e.is_create else "edit"
            if path_only:
                lines.append(f"- [{e.agent_id}] {kind} {e.path} via {e.tool_name}")
            else:
                before_n = len(e.before or "")
                after_n = len(e.after or "")
                lines.append(
                    f"- [{e.agent_id}] {kind} {e.path} via {e.tool_name} "
                    f"(before={before_n}c after={after_n}c call={e.call_id})"
                )
        return "\n".join(lines)
