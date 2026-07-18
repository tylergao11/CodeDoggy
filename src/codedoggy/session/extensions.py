"""Optional dependencies plugged into a Session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from codedoggy.session.types import TurnRequest, TurnResult


@runtime_checkable
class TurnRunner(Protocol):
    """Runs one agentic turn for a session."""

    def run(self, request: TurnRequest, *, session: Any) -> TurnResult:
        ...


@dataclass(slots=True)
class SessionExtensions:
    """Swappable backends. All fields optional."""

    turn_runner: TurnRunner | None = None
    tools: Any | None = None
    context: Any | None = None  # ContextCompactor (Grok live window)
    memory: Any | None = None  # MemoryStore (curated)
    memory_manager: Any | None = None  # MemoryManager (orchestration)
    session_store: Any | None = None  # SessionStore (FTS archive)
    audit: Any | None = None  # AuditServices
    graph: Any | None = None
    policy: Any | None = None  # WorkspacePolicy

    def _copy(self, **kwargs: Any) -> SessionExtensions:
        base = {
            "turn_runner": self.turn_runner,
            "tools": self.tools,
            "context": self.context,
            "memory": self.memory,
            "memory_manager": self.memory_manager,
            "session_store": self.session_store,
            "audit": self.audit,
            "graph": self.graph,
            "policy": self.policy,
        }
        base.update(kwargs)
        return SessionExtensions(**base)

    def with_turn_runner(self, runner: TurnRunner) -> SessionExtensions:
        return self._copy(turn_runner=runner)

    def with_tools(self, tools: Any) -> SessionExtensions:
        return self._copy(tools=tools)

    def with_memory(self, memory: Any) -> SessionExtensions:
        return self._copy(memory=memory)

    def with_audit(self, audit: Any) -> SessionExtensions:
        return self._copy(audit=audit)

    def with_session_store(self, session_store: Any) -> SessionExtensions:
        return self._copy(session_store=session_store)
