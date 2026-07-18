"""Types for the agentic turn loop (sample → tools → writeback)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from codedoggy.tools.kinds import ToolKind


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class ToolCall:
    """One model-requested tool invocation.

    ``arguments`` may be a dict or a JSON string from the model; the loop
    normalizes to a dict before dispatch.
    """

    id: str
    name: str
    arguments: Any = field(default_factory=dict)


@dataclass(slots=True)
class Message:
    """One transcript message (OpenAI-style roles)."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(slots=True)
class SampleResult:
    """Model output for one sampling step."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass(slots=True)
class FileMutation:
    """Workspace mutation — Grok hunk unit for Shadow (multi per tool call)."""

    path: str
    tool_name: str
    call_id: str
    args: dict[str, Any] = field(default_factory=dict)
    before: str | None = None
    after: str | None = None
    is_create: bool = False
    is_delete: bool = False


@dataclass(slots=True)
class ToolResultRecord:
    """Outcome of executing one tool call."""

    call: ToolCall
    content: str
    ok: bool
    error_code: str | None = None
    kind: ToolKind | None = None
    mutation: FileMutation | None = None
    mutations: list[FileMutation] = field(default_factory=list)


@dataclass(slots=True)
class HookDecision:
    """Optional hook output after a tool (or mutation)."""

    # Extra text appended to the tool observation the model sees.
    append_observation: str | None = None
    # Stop the loop early (e.g. hard quality gate / shadow P0).
    abort: bool = False
    abort_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoopResult:
    """Result of one agentic run (one user prompt).

    Exit flags mirror Grok TurnOutcome: completed / max_turns / cancelled
    (user or permission) / aborted (hook hard) / error.
    """

    final_text: str | None
    messages: list[Message]
    tools_called: list[str]
    rounds: int
    completed: bool
    max_turns_reached: bool = False
    cancelled: bool = False
    aborted: bool = False
    error: str | None = None
    # Grok-aligned exit label: completed|max_turns|cancelled|permission_reject|…
    exit_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
