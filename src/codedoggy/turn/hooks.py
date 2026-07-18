"""Hooks into the turn loop (quality gates plug in here later)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from codedoggy.turn.types import HookDecision, SampleResult, ToolResultRecord


@dataclass(slots=True)
class HookContext:
    """Per-invocation context for loop hooks."""

    cwd: Path
    round_index: int
    session: Any = None
    prompt_id: str | None = None
    goal: str | None = None


@runtime_checkable
class LoopHooks(Protocol):
    """Optional callbacks around sample / tool execution.

    Default implementations may omit methods; the loop uses getattr.

    Grok alignment: ``pre_tool_use`` is phase-1 (before run). Soft deny
    (abort without metadata hard) → HookDenied non-terminal; hard →
    PermissionReject stops the batch.
    """

    def after_sample(
        self, sample: SampleResult, ctx: HookContext
    ) -> HookDecision | None:
        """After a successful sample, before tools run."""
        ...

    def pre_tool_use(
        self, call: Any, ctx: HookContext
    ) -> HookDecision | None:
        """Grok PreToolUse — before execute. Soft deny feeds model; hard stops batch."""
        ...

    def after_tool(
        self, record: ToolResultRecord, ctx: HookContext
    ) -> HookDecision | None:
        """After every tool execution (success or ToolError observation)."""
        ...

    def after_mutation(
        self, record: ToolResultRecord, ctx: HookContext
    ) -> HookDecision | None:
        """After a mutating tool (edit/write/delete/move) succeeds.

        Resident audit: P0 red cards here; non-P0 buffered until on_turn_end.
        """
        ...

    def on_turn_end(self, ctx: HookContext) -> str | None:
        """Flush deferred (non-P0) audit notes at end of the agentic turn."""
        ...


class NoopHooks:
    """Default: no extra observation text, never abort."""

    def after_sample(
        self, sample: SampleResult, ctx: HookContext
    ) -> HookDecision | None:
        return None

    def pre_tool_use(self, call: Any, ctx: HookContext) -> HookDecision | None:
        return None

    def after_tool(
        self, record: ToolResultRecord, ctx: HookContext
    ) -> HookDecision | None:
        return None

    def after_mutation(
        self, record: ToolResultRecord, ctx: HookContext
    ) -> HookDecision | None:
        return None

    def on_turn_end(self, ctx: HookContext) -> str | None:
        return None
