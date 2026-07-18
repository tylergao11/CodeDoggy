"""Tool trait, call context, and dispatch surface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ToolId(str):
    """Short tool id, e.g. ``read_file`` (no namespace)."""

    def __new__(cls, value: str) -> ToolId:
        if not value or ":" in value:
            raise ValueError(f"ToolId must be a non-empty short name, got {value!r}")
        return str.__new__(cls, value)


@dataclass(slots=True)
class ToolDescription:
    """Name and prose shown to the model."""

    name: str
    description: str


@dataclass(slots=True)
class ListToolsContext:
    """Per-turn context for listing/description (usually unused)."""

    pass


@dataclass(slots=True)
class ToolCallContext:
    """Host context for one tool invocation."""

    cwd: Path
    session_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    """Optional host handles (e.g. memory_store: MemoryStore)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "cwd", Path(self.cwd).resolve())

    def set_mutation(
        self,
        *,
        path: str,
        before: str | None,
        after: str | None,
        is_create: bool = False,
        tool_name: str = "",
        call_id: str = "",
        args: dict[str, Any] | None = None,
    ) -> None:
        """Record a first-hand file mutation for resident audit (no second read)."""
        from codedoggy.turn.types import FileMutation

        mut_args = dict(args or {})
        # Stamp policy allow decision for audit (writes only reach here if allowed)
        pol = self.extra.get("policy")
        if pol is not None:
            check = getattr(pol, "check_write", None)
            if callable(check):
                try:
                    d = check(path)
                    mut_args["_policy"] = {
                        "allowed": bool(getattr(d, "allowed", True)),
                        "code": getattr(d, "code", "ok"),
                        "reason": getattr(d, "reason", "") or "",
                    }
                except Exception:  # noqa: BLE001
                    pass
        self.extra["mutation"] = FileMutation(
            path=path,
            tool_name=tool_name,
            call_id=call_id,
            args=mut_args,
            before=before,
            after=after,
            is_create=is_create,
        )
        # Codebase graph: file-event spirit — mark index dirty for next query
        graph = self.extra.get("graph")
        mark = getattr(graph, "mark_dirty", None)
        if callable(mark):
            try:
                mark(path)
            except Exception:  # noqa: BLE001
                pass


class ToolError(Exception):
    """Tool failure with a stable machine-readable code."""

    def __init__(self, message: str, *, code: str = "tool_error") -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    @classmethod
    def invalid_arguments(cls, message: str) -> ToolError:
        return cls(message, code="invalid_arguments")

    @classmethod
    def not_found(cls, tool_id: str) -> ToolError:
        return cls(f"tool not found: {tool_id}", code="not_found")

    @classmethod
    def not_implemented(cls, message: str = "not implemented") -> ToolError:
        return cls(message, code="not_implemented")


@dataclass(slots=True)
class ToolSpec:
    """Model-facing tool definition for sampling."""

    name: str
    description: str | None
    parameters: dict[str, Any]


class Tool(ABC):
    """Implement `run` (and metadata) for a single tool."""

    @abstractmethod
    def id(self) -> ToolId:
        ...

    @abstractmethod
    def tool_namespace(self) -> "ToolNamespace":  # noqa: F821
        ...

    @abstractmethod
    def kind(self) -> "ToolKind":  # noqa: F821
        ...

    @abstractmethod
    def description(self, ctx: ListToolsContext | None = None) -> ToolDescription:
        ...

    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema for arguments."""
        ...

    def capabilities(self) -> dict[str, Any]:
        return {}

    def should_list(self, ctx: ListToolsContext | None = None) -> bool:
        return True

    def has_dynamic_description(self) -> bool:
        return False

    @abstractmethod
    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        """Execute and return model-facing text."""
        ...

    def qualified_id(self) -> str:
        """Registry key: ``Namespace:short_id``."""
        from codedoggy.tools.kinds import ToolNamespace

        ns = self.tool_namespace()
        name = ns.value if isinstance(ns, ToolNamespace) else str(ns)
        return f"{name}:{self.id()}"


class ToolDispatch(ABC):
    """Route a client-facing tool name to an implementation."""

    @abstractmethod
    def call(
        self,
        tool_name: str,
        args: dict[str, Any],
        ctx: ToolCallContext,
    ) -> str:
        ...
