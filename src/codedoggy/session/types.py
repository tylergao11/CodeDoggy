"""Session public types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class SessionId(str):
    """Opaque session identifier."""

    @classmethod
    def new(cls) -> SessionId:
        return cls(str(uuid4()))


class SessionPhase(str, Enum):
    """Coarse session phase for observability."""

    IDLE = "idle"
    TURN_RUNNING = "turn_running"
    CLOSED = "closed"


class TurnStatus(str, Enum):
    """Result status of one `handle_prompt` call."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"
    MAX_TURNS_REACHED = "max_turns_reached"
    NOT_IMPLEMENTED = "not_implemented"
    # Mid-turn concurrent prompt was queued (interjection or PromptQueue) — not a finished turn
    QUEUED = "queued"


@dataclass(slots=True)
class TurnRequest:
    """Input for one agent turn."""

    text: str
    prompt_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TurnResult:
    """Output of one agent turn."""

    status: TurnStatus
    final_text: str | None = None
    tools_called: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
