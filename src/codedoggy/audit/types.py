"""Resident audit types: mutations, verdicts, memory-select requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any
from uuid import uuid4


class FindingSeverity(str, Enum):
    CRITICAL = "critical"
    IMPORTANT = "important"
    SUGGESTION = "suggestion"


@dataclass(slots=True)
class MutationEvent:
    """One workspace write unit — first-hand input for resident audit.

    Shared bus unit: main agent and (later) subagents all emit these.
    """

    path: str
    tool_name: str
    call_id: str
    before: str | None = None
    after: str | None = None
    is_create: bool = False
    is_delete: bool = False
    agent_id: str = "main"
    session_id: str | None = None
    goal_snapshot: str | None = None
    prompt_id: str | None = None
    round_index: int = 0
    args: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex[:16])
    timestamp: float = field(default_factory=time)

    def unified_diff_hint(self, *, max_chars: int = 8_000) -> str:
        """Real unified diff (difflib) for Shadow — middle edits stay visible.

        Truncates with balanced head/tail of the unified diff if over budget.
        """
        import difflib

        if self.is_create:
            after = self.after or ""
            lines = list(
                difflib.unified_diff(
                    [],
                    after.splitlines(keepends=True),
                    fromfile="/dev/null",
                    tofile=self.path,
                    lineterm="",
                )
            )
            text = "\n".join(lines) if lines else f"+++ create {self.path}\n{after}"
        elif self.is_delete:
            before = self.before or ""
            lines = list(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    [],
                    fromfile=self.path,
                    tofile="/dev/null",
                    lineterm="",
                )
            )
            text = "\n".join(lines) if lines else f"--- delete {self.path}\n{before}"
        else:
            before = self.before or ""
            after = self.after or ""
            lines = list(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=f"a/{self.path}",
                    tofile=f"b/{self.path}",
                    lineterm="",
                    n=3,
                )
            )
            text = "\n".join(lines) if lines else f"(no textual diff) {self.path}"
        if len(text) <= max_chars:
            return text
        # Keep head + tail of the unified diff so middle hunks can still appear
        keep = max_chars // 2 - 20
        return text[:keep] + "\n…[diff truncated]…\n" + text[-keep:]


@dataclass(slots=True)
class AuditFinding:
    """One issue the auditor wants the coding agent to rethink."""

    message: str
    severity: FindingSeverity = FindingSeverity.IMPORTANT
    path: str | None = None
    code: str | None = None


@dataclass(slots=True)
class AuditVerdict:
    """Auditor output. Product default: pass → silent; fail → soft feedback only."""

    ok: bool
    findings: list[AuditFinding] = field(default_factory=list)
    # Soft path only by default — hard abort is opt-in extreme guard.
    abort: bool = False
    abort_reason: str | None = None
    # Opaque notes for logs / future UI (not shown to the coding model).
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def pass_silent(cls) -> AuditVerdict:
        return cls(ok=True)

    @classmethod
    def fail(
        cls,
        findings: list[AuditFinding],
        *,
        abort: bool = False,
        abort_reason: str | None = None,
    ) -> AuditVerdict:
        return cls(
            ok=False,
            findings=list(findings),
            abort=abort,
            abort_reason=abort_reason,
        )


@dataclass(slots=True)
class MemorySelectRequest:
    """What an auditor needs when pulling memory for a review unit.

    Hermes integration will fill session_search / provider hits via a
    :class:`~codedoggy.audit.memory_select.MemorySelector` implementation.
    """

    goal: str | None
    mutation: MutationEvent
    # Compact trajectory summary (not full file bodies of every past edit).
    trajectory_summary: str
    session_id: str | None = None
    agent_id: str = "main"
    # Selection knobs for future Hermes-style backends.
    max_curated_chars: int = 2_000
    max_session_hits: int = 5
    query_hint: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemorySelectResult:
    """Selected memory slices for one audit call.

    Keep sources separate so Hermes can rank curated vs session_search later.
    """

    # MEMORY.md / USER.md style blocks (already formatted prose).
    curated_blocks: list[str] = field(default_factory=list)
    # Future: session_search / provider snippets.
    session_hits: list[str] = field(default_factory=list)
    # Future: external provider (Honcho, etc.) blobs.
    provider_hits: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def combined_text(self, *, max_chars: int = 6_000) -> str:
        parts: list[str] = []
        if self.curated_blocks:
            parts.append("## Curated memory\n" + "\n\n".join(self.curated_blocks))
        if self.session_hits:
            parts.append("## Session recall\n" + "\n\n".join(self.session_hits))
        if self.provider_hits:
            parts.append("## Provider memory\n" + "\n\n".join(self.provider_hits))
        text = "\n\n".join(parts)
        if len(text) > max_chars:
            return text[: max_chars - 20] + "\n… (truncated)"
        return text


@dataclass(slots=True)
class AuditContext:
    """Full package passed to :class:`~codedoggy.audit.auditor.ResidentAuditor`."""

    goal: str | None
    mutation: MutationEvent
    trajectory_summary: str
    memory: MemorySelectResult
    cwd: str
    session_id: str | None = None
    agent_id: str = "main"
    round_index: int = 0
    # Host handles (session object, etc.) — optional, for advanced auditors.
    session: Any = None
    # Workspace policy snapshot from tool layer (four-pillar fuse).
    policy: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
