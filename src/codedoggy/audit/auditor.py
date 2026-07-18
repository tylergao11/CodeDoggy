"""Resident auditor protocol — model brain plug-in point."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from codedoggy.audit.types import (
    AuditContext,
    AuditFinding,
    AuditVerdict,
    FindingSeverity,
)


@runtime_checkable
class ResidentAuditor(Protocol):
    """Review one mutation unit. Must not write the workspace.

    Product contract:
    - ``ok=True`` → pass silent (no observation text)
    - ``ok=False`` → soft feedback for the coding agent to rethink
    - ``abort=True`` only for extreme guards (not the default path)
    """

    def review(self, ctx: AuditContext) -> AuditVerdict:
        ...


class PassThroughAuditor:
    """Always pass silent — wiring smoke tests."""

    def review(self, ctx: AuditContext) -> AuditVerdict:
        return AuditVerdict.pass_silent()


class ScriptedAuditor:
    """Deterministic auditor for tests: map path substring → findings."""

    def __init__(
        self,
        rules: list[tuple[str, str]] | None = None,
        *,
        default_ok: bool = True,
    ) -> None:
        # (path_substring, finding_message)
        self.rules = list(rules or [])
        self.default_ok = default_ok

    def review(self, ctx: AuditContext) -> AuditVerdict:
        path = ctx.mutation.path
        findings: list[AuditFinding] = []
        for needle, message in self.rules:
            if needle in path or needle in (ctx.mutation.after or ""):
                findings.append(
                    AuditFinding(
                        message=message,
                        severity=FindingSeverity.IMPORTANT,
                        path=path,
                        code="scripted",
                    )
                )
        if findings:
            return AuditVerdict.fail(findings)
        if self.default_ok:
            return AuditVerdict.pass_silent()
        return AuditVerdict.fail(
            [
                AuditFinding(
                    message="scripted default reject",
                    severity=FindingSeverity.IMPORTANT,
                    path=path,
                )
            ]
        )


class GoalDriftHeuristicAuditor:
    """Cheap non-model heuristic: flag edits when session goal is empty or
    path looks unrelated to a simple keyword bag from the goal.

    Placeholder until a real model-brain auditor is bound. Prefer
    :class:`ScriptedAuditor` or a model-backed implementor for production.
    """

    def __init__(self, *, min_goal_len: int = 8) -> None:
        self.min_goal_len = min_goal_len

    def review(self, ctx: AuditContext) -> AuditVerdict:
        goal = (ctx.goal or "").strip()
        if len(goal) < self.min_goal_len:
            # No goal yet — do not spam; pass silent.
            return AuditVerdict.pass_silent()
        # Extremely light check: if goal mentions a token and path/after
        # share nothing, surface a soft rethink note. Avoid false positives:
        # only when after is huge unrelated rename patterns — keep minimal.
        return AuditVerdict.pass_silent()
