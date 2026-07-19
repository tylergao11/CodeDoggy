"""Live execution activity for the boss cockpit (tool calls / short status).

Fed only from turn ``on_live_message`` (and optional text stream hints).
Does not replace the transcript; detail view stays full-fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any


def _role_value(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "").lower()
    role = getattr(message, "role", None)
    return str(getattr(role, "value", role) or "").lower()


def _msg_get(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _tool_names(message: Any) -> list[tuple[str, str]]:
    """Return (call_id, name) pairs from an assistant message."""
    out: list[tuple[str, str]] = []
    raw_tcs = _msg_get(message, "tool_calls", None) or []
    for tc in raw_tcs:
        if isinstance(tc, dict):
            name = str(
                (tc.get("function") or {}).get("name")
                or tc.get("name")
                or "tool"
            )
            cid = str(tc.get("id") or name)
        else:
            name = str(getattr(tc, "name", None) or "tool")
            cid = str(getattr(tc, "id", None) or name)
        name = name.strip() or "tool"
        out.append((cid, name))
    return out


def format_tools_running(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return f"→ {names[0]} · 调用中"
    head = ", ".join(names[:3])
    more = "…" if len(names) > 3 else ""
    return f"→ {head}{more} · 调用中"


def format_tool_done(name: str, *, failed: bool, still_open: list[str]) -> str:
    mark = "✗" if failed else "✓"
    tail = "失败" if failed else "完成"
    if still_open:
        rest = ", ".join(still_open[:2])
        extra = "…" if len(still_open) > 2 else ""
        return f"{mark} {name} · 仍在 {rest}{extra}"
    return f"{mark} {name} · {tail}"


def looks_like_tool_failure(content: str | None) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    head = text[:400].lower()
    return (
        head.startswith("error")
        or "traceback (most recent call last)" in head
        or "exception:" in head
        or head.startswith("failed")
    )


@dataclass
class AgentActivity:
    line: str = ""
    open_tools: dict[str, str] = field(default_factory=dict)
    last_tool: str = ""


class LiveActivityBoard:
    """Per-(task, agent) short activity line for list + turn status."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._agents: dict[tuple[str, str], AgentActivity] = {}

    def clear(self) -> None:
        with self._lock:
            self._agents.clear()

    def clear_task(self, task_id: str) -> None:
        with self._lock:
            for key in [k for k in self._agents if k[0] == task_id]:
                del self._agents[key]

    def clear_agent(self, task_id: str, agent_id: str) -> None:
        with self._lock:
            self._agents.pop((task_id, agent_id), None)

    def rebuild(self, task_id: str, agent_id: str, messages: list[Any]) -> str:
        """Recompute activity from a transcript snapshot (subagent poll path)."""
        with self._lock:
            self.clear_agent(task_id, agent_id)
            line = ""
            for message in messages:
                line = self.observe(task_id, agent_id, message)
            return line

    def line(self, task_id: str, agent_id: str) -> str:
        with self._lock:
            state = self._agents.get((task_id, agent_id))
            return state.line if state is not None else ""

    def main_line(self, task_id: str) -> str:
        with self._lock:
            main_key = (task_id, f"{task_id}:main")
            state = self._agents.get(main_key)
            if state is not None and state.line:
                return state.line
            for (tid, _aid), st in self._agents.items():
                if tid == task_id and st.line:
                    return st.line
            return ""

    def observe(self, task_id: str, agent_id: str, message: Any) -> str:
        """Ingest one archived live message; return updated display line."""
        with self._lock:
            return self._observe_locked(task_id, agent_id, message)

    def _observe_locked(self, task_id: str, agent_id: str, message: Any) -> str:
        key = (task_id, agent_id)
        state = self._agents.setdefault(key, AgentActivity())
        role = _role_value(message)

        if role == "assistant":
            pairs = _tool_names(message)
            if pairs:
                for cid, name in pairs:
                    state.open_tools[cid] = name
                state.line = format_tools_running([n for _c, n in pairs])
                return state.line
            content = str(_msg_get(message, "content", None) or "").strip()
            if content and not state.open_tools:
                first = content.splitlines()[0].strip()
                if first:
                    state.line = f"… {first[:48]}"
            return state.line

        if role == "tool":
            cid = str(_msg_get(message, "tool_call_id", None) or "")
            name = state.open_tools.pop(cid, None) if cid else None
            if not name:
                name = str(_msg_get(message, "name", None) or state.last_tool or "tool")
            name = name.strip() or "tool"
            state.last_tool = name
            failed = looks_like_tool_failure(_msg_get(message, "content", None))
            still = list(state.open_tools.values())
            state.line = format_tool_done(name, failed=failed, still_open=still)
            return state.line

        return state.line
