"""Grok-aligned orchestration types (ToolLoop / Turn exits / capability).

Ported from xai-grok-shell ``acp_session_impl/types.rs`` ToolLoop + TurnOutcome
semantics — not a re-invention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from codedoggy.turn.types import ToolCall, ToolResultRecord


class SessionMode(str, Enum):
    """Session-level mode (Grok: Normal / Plan / Goal)."""

    NORMAL = "normal"
    PLAN = "plan"
    GOAL = "goal"


class CapabilityMode(str, Enum):
    """Subagent / agent capability mode — wire kebab-case like Grok.

    Source: ``xai_tool_types::SubagentCapabilityMode``.
    """

    READ_ONLY = "read-only"
    READ_WRITE = "read-write"
    EXECUTE = "execute"
    ALL = "all"

    @classmethod
    def parse(cls, raw: str | None) -> CapabilityMode:
        if raw is None or not str(raw).strip():
            return cls.ALL
        s = str(raw).strip().lower().replace("_", "-")
        aliases = {
            "readonly": cls.READ_ONLY,
            "read-only": cls.READ_ONLY,
            "readwrite": cls.READ_WRITE,
            "read-write": cls.READ_WRITE,
            "execute": cls.EXECUTE,
            "all": cls.ALL,
        }
        return aliases.get(s, cls.ALL)


class IsolationMode(str, Enum):
    """Grok ``SubagentIsolationMode``."""

    NONE = "none"
    WORKTREE = "worktree"

    @classmethod
    def parse(cls, raw: str | None) -> IsolationMode:
        if raw is None or not str(raw).strip():
            return cls.NONE
        s = str(raw).strip().lower().replace("_", "-")
        if s in {"worktree", "wt", "git-worktree"}:
            return cls.WORKTREE
        if s in {"none", "off", "shared", "parent"}:
            return cls.NONE
        return cls.NONE


class TurnExit(str, Enum):
    """Why a Turn stopped (Grok TurnOutcome categories)."""

    COMPLETED = "completed"
    MAX_TURNS = "max_turns"
    CANCELLED = "cancelled"
    PERMISSION_REJECT = "permission_reject"
    HOOK_DENIED = "hook_denied"  # non-terminal in tool batch; may appear in meta
    FOLLOWUP = "followup"  # drained interjection → continue sample
    ABORTED = "aborted"
    ERROR = "error"
    STRUCTURED_OUTPUT = "structured_output"


class ToolLoopOutcome(str, Enum):
    """Grok ``ToolLoop`` enum — batch-level result after tool phase."""

    CONTINUE = "continue"
    PERMISSION_REJECT = "permission_reject"
    CANCELLED = "cancelled"
    FOLLOWUP_MESSAGE = "followup_message"
    HOOK_DENIED = "hook_denied"  # non-terminal: reason fed back, turn continues
    NON_EXISTING_TOOL = "non_existing_tool"
    TOOL_PARSING_ERROR = "tool_parsing_error"


class PrecheckVerdict(str, Enum):
    """Phase-1 prepare result for one tool call."""

    APPROVE = "approve"
    # Soft: observation returned to model; batch continues (Grok HookDenied).
    HOOK_DENY = "hook_deny"
    # Hard: stop remaining prepares/executes (Grok PermissionReject).
    PERMISSION_REJECT = "permission_reject"
    # Hard: user cancel mid-batch.
    CANCELLED = "cancelled"
    # Soft-ish: unknown tool — observation only, continue others.
    NON_EXISTING = "non_existing"
    # Soft: bad args — observation only.
    PARSE_ERROR = "parse_error"
    # Plan mode rejected non-plan edit (hard for that call; batch stops like permission).
    PLAN_REJECT = "plan_reject"


@dataclass(slots=True)
class PreparedToolCall:
    """Grok PreparedToolCall — survived phase-1 preflight."""

    call: ToolCall
    parsed_args: dict[str, Any]
    tool_name: str
    is_read_only: bool = True
    lock_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PrecheckResult:
    """Outcome of preparing one tool call."""

    verdict: PrecheckVerdict
    prepared: PreparedToolCall | None = None
    # Model-facing observation when not approved (deny / error / cancel).
    observation: str | None = None
    tool_name: str = ""
    reason: str = ""
    hook_name: str | None = None


@dataclass(slots=True)
class ToolBatchResult:
    """Result of two-phase tool execution for one sample's tool_calls."""

    outcome: ToolLoopOutcome
    # Ordered results for every call in the batch (incl. cancelled synthetics).
    records: list[ToolResultRecord] = field(default_factory=list)
    # Observations already applied into records; convenience parallel list.
    tool_name: str | None = None
    reason: str | None = None
    followup_message: str | None = None
    hook_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Interjection:
    """User mid-turn message (Grok interjection / followup)."""

    text: str
    images: list[Any] = field(default_factory=list)
    prompt_id: str | None = None
