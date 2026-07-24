"""Types for the agentic turn loop (sample → tools → writeback)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from codedoggy.tools.kinds import ToolKind
from codedoggy.attachments import ImageAttachment


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
    # Provider-required call metadata (Gemini thought signatures,
    # Responses call_id, OpenRouter extra_content).  This must survive replay.
    provider_data: dict[str, Any] | None = None


@dataclass(slots=True)
class Message:
    """One transcript message (OpenAI-style roles)."""

    role: Role
    content: str | list[dict[str, Any]] | None = None
    attachments: tuple[ImageAttachment, ...] = ()
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    # DeepSeek / thinking-mode CoT; replayed by model profile when required
    reasoning_content: str | None = None
    # e.g. anthropic_content_blocks for signed thinking + tool_use order
    provider_data: dict[str, Any] | None = None


@dataclass(slots=True)
class SampleResult:
    """Model output for one sampling step."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    reasoning_content: str | None = None
    provider_data: dict[str, Any] | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass(slots=True)
class FileMutation:
    """Workspace mutation — one path hunk unit (multi per tool call)."""

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
    # Stop the loop early (hard gate from hooks).
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
