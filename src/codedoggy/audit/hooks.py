"""LoopHooks: P0 red cards immediate; other findings deferred to turn end."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from codedoggy.audit.format import (
    format_deferred_summary,
    format_p0_red_card,
    partition_findings,
)
from codedoggy.audit.services import AuditServices
from codedoggy.audit.types import (
    AuditContext,
    AuditFinding,
    MemorySelectRequest,
    MutationEvent,
)
from codedoggy.turn.hooks import HookContext, NoopHooks
from codedoggy.turn.types import HookDecision, SampleResult, ToolResultRecord


@dataclass
class _BufferedFinding:
    finding: AuditFinding
    path: str
    round_index: int


class ResidentAuditHooks(NoopHooks):
    """Shadow (影子) hooks: after_mutation → trajectory → auditor → P0 / deferred.

    Product name **shadow** — in-loop write-time soft review, not a normal
    offline code audit.

    - ``ok`` / no findings → silent
    - ``critical`` → immediate observation red card
    - ``important`` / ``suggestion`` → buffer → ``on_turn_end`` summary
    - Never writes the workspace
    """

    def __init__(self, services: AuditServices) -> None:
        self.services = services
        self._deferred: list[_BufferedFinding] = []

    def reset_turn_buffer(self) -> None:
        self._deferred.clear()

    def after_sample(
        self, sample: SampleResult, ctx: HookContext
    ) -> HookDecision | None:
        return None

    def after_tool(
        self, record: ToolResultRecord, ctx: HookContext
    ) -> HookDecision | None:
        return None

    def after_mutation(
        self, record: ToolResultRecord, ctx: HookContext
    ) -> HookDecision | None:
        mut = record.mutation
        if mut is None or not record.ok:
            return None

        session = ctx.session
        goal = getattr(session, "goal", None) if session is not None else None
        session_id = None
        if session is not None:
            sid = getattr(session, "id", None)
            session_id = str(sid) if sid is not None else None

        event = MutationEvent(
            path=mut.path,
            tool_name=mut.tool_name or record.call.name,
            call_id=mut.call_id or record.call.id,
            before=mut.before,
            after=mut.after,
            is_create=bool(mut.is_create),
            agent_id=self.services.agent_id,
            session_id=session_id,
            goal_snapshot=goal if isinstance(goal, str) else None,
            prompt_id=getattr(ctx, "prompt_id", None),
            round_index=ctx.round_index,
            args=dict(mut.args or {}),
        )
        self.services.trajectory.append(event)

        traj_summary = self.services.trajectory.summary()
        mem_req = MemorySelectRequest(
            goal=event.goal_snapshot,
            mutation=event,
            trajectory_summary=traj_summary,
            session_id=session_id,
            agent_id=self.services.agent_id,
            query_hint=event.path,
        )
        try:
            memory = self.services.memory_selector.select(mem_req)
        except Exception:  # noqa: BLE001
            from codedoggy.audit.types import MemorySelectResult

            memory = MemorySelectResult(raw={"select_error": True})

        # Fuse tool policy snapshot into audit context (when present)
        policy_note = None
        if session is not None:
            pol = getattr(getattr(session, "extensions", None), "policy", None)
            snap = getattr(pol, "snapshot", None)
            if callable(snap):
                try:
                    policy_note = snap()
                except Exception:  # noqa: BLE001
                    policy_note = None
        if policy_note and hasattr(memory, "raw") and isinstance(memory.raw, dict):
            memory.raw["policy"] = policy_note

        audit_ctx = AuditContext(
            goal=event.goal_snapshot,
            mutation=event,
            trajectory_summary=traj_summary,
            memory=memory,
            cwd=str(ctx.cwd),
            session_id=session_id,
            agent_id=self.services.agent_id,
            round_index=ctx.round_index,
            session=session,
            policy=policy_note if isinstance(policy_note, dict) else None,
        )
        try:
            verdict = self.services.auditor.review(audit_ctx)
        except Exception as e:  # noqa: BLE001
            return HookDecision(
                append_observation=(
                    "\n── shadow ──\n"
                    f"Shadow error (write not rolled back): {type(e).__name__}: {e}\n"
                    "── end shadow ──"
                ),
                abort=False,
            )

        if verdict.ok or not verdict.findings:
            return HookDecision(abort=bool(verdict.abort), abort_reason=verdict.abort_reason)

        p0, rest = partition_findings(list(verdict.findings))
        for f in rest:
            self._deferred.append(
                _BufferedFinding(
                    finding=f,
                    path=event.path,
                    round_index=ctx.round_index,
                )
            )

        card = format_p0_red_card(p0)
        if card is None and not verdict.abort:
            return None
        return HookDecision(
            append_observation=card,
            abort=bool(verdict.abort),
            abort_reason=verdict.abort_reason,
        )

    def on_turn_end(self, ctx: HookContext) -> str | None:
        """Flush non-P0 findings once at end of the agentic turn."""
        if not self._deferred:
            return None
        findings = [b.finding for b in self._deferred]
        # Stamp path onto findings that lack one
        for b in self._deferred:
            if b.finding.path is None:
                b.finding.path = b.path
        text = format_deferred_summary(findings)
        self._deferred.clear()
        return text

    def deferred_count(self) -> int:
        return len(self._deferred)


def resolve_audit_hooks(
    session: Any,
    *,
    explicit_hooks: Any | None = None,
) -> Any | None:
    """Prefer explicit hooks; else build ResidentAuditHooks from session.extensions.audit."""
    if explicit_hooks is not None:
        return explicit_hooks
    ext = getattr(session, "extensions", None)
    if ext is None:
        return None
    audit = getattr(ext, "audit", None)
    if audit is None:
        return None
    if isinstance(audit, AuditServices):
        return ResidentAuditHooks(audit)
    return None
