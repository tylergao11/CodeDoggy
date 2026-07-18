"""LoopHooks: P0 red cards immediate; other findings deferred to turn end."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codedoggy.audit.format import (
    format_deferred_summary,
    format_p0_red_card,
    partition_findings,
)
from codedoggy.audit.restore import restore_mutation_before, shadow_restore_enabled
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
    - ``critical`` → immediate observation red card + stop further writes;
      optional soft restore of ``before`` (``CODEDOGGY_SHADOW_RESTORE``, default ON)
    - ``important`` / ``suggestion`` → buffer → ``on_turn_end`` summary
    - Soft restore is best-effort only (not a full OS transaction)
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
        # Grok multi-hunk: review every mutation; also review partial shell writes
        # when tool returned non-ok but still mutated the workspace.
        muts = list(getattr(record, "mutations", None) or [])
        if not muts and record.mutation is not None:
            muts = [record.mutation]
        if not muts:
            return None

        session = ctx.session
        goal = getattr(session, "goal", None) if session is not None else None
        session_id = None
        if session is not None:
            sid = getattr(session, "id", None)
            session_id = str(sid) if sid is not None else None

        policy_note = None
        if session is not None:
            pol = getattr(getattr(session, "extensions", None), "policy", None)
            snap = getattr(pol, "snapshot", None)
            if callable(snap):
                try:
                    policy_note = snap()
                except Exception:  # noqa: BLE001
                    policy_note = None

        all_p0: list = []
        all_rest: list = []
        paths: list[str] = []
        hard_abort = False
        abort_reason = None
        obs_parts: list[str] = []

        for mut in muts:
            event = MutationEvent(
                path=mut.path,
                tool_name=mut.tool_name or record.call.name,
                call_id=mut.call_id or record.call.id,
                before=mut.before,
                after=mut.after,
                is_create=bool(mut.is_create),
                is_delete=bool(getattr(mut, "is_delete", False)),
                agent_id=self.services.agent_id,
                session_id=session_id,
                goal_snapshot=goal if isinstance(goal, str) else None,
                prompt_id=getattr(ctx, "prompt_id", None),
                round_index=ctx.round_index,
                args=dict(mut.args or {}),
            )
            self.services.trajectory.append(event)
            paths.append(event.path)

            traj_summary = self.services.trajectory.summary()
            mem_req = MemorySelectRequest(
                goal=event.goal_snapshot,
                mutation=event,
                trajectory_summary=traj_summary,
                session_id=session_id,
                agent_id=self.services.agent_id,
                query_hint=event.path,
                extra={
                    "cwd": str(ctx.cwd) if ctx.cwd is not None else None,
                    "roles": ["user", "assistant"],
                },
            )
            try:
                memory = self.services.memory_selector.select(mem_req)
            except Exception:  # noqa: BLE001
                from codedoggy.audit.types import MemorySelectResult

                memory = MemorySelectResult(raw={"select_error": True})
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
                obs_parts.append(
                    f"Shadow error on {event.path} (write not rolled back): "
                    f"{type(e).__name__}: {e}"
                )
                continue

            if verdict.abort:
                hard_abort = True
                abort_reason = verdict.abort_reason
            if verdict.ok or not verdict.findings:
                continue
            p0, rest = partition_findings(list(verdict.findings))
            all_p0.extend(p0)
            for f in rest:
                all_rest.append(
                    _BufferedFinding(
                        finding=f,
                        path=event.path,
                        round_index=ctx.round_index,
                    )
                )

        for f in all_rest:
            self._deferred.append(f)

        card = format_p0_red_card(all_p0)
        pause = bool(all_p0)
        abort = hard_abort or pause

        # P0 / hard abort: best-effort soft restore of reviewed mutations.
        restored: list[dict[str, Any]] = []
        restore_failed: list[dict[str, Any]] = []
        if abort and shadow_restore_enabled() and ctx.cwd is not None:
            cwd = Path(ctx.cwd)
            # Reverse order: A→B→C must restore C then B then A (not leave B)
            for mut in reversed(list(muts)):
                has_before = getattr(mut, "before", None) is not None
                is_create = bool(getattr(mut, "is_create", False))
                if not has_before and not is_create:
                    continue
                try:
                    r = restore_mutation_before(cwd, mut)
                except Exception as e:  # noqa: BLE001 — never raise into loop
                    r = {
                        "ok": False,
                        "path": str(getattr(mut, "path", "") or ""),
                        "reason": f"unexpected: {type(e).__name__}: {e}",
                    }
                if r.get("ok"):
                    restored.append(r)
                else:
                    restore_failed.append(r)

        if obs_parts:
            err_obs = "\n── shadow ──\n" + "\n".join(obs_parts) + "\n── end shadow ──"
            if card:
                card = err_obs + "\n" + card
            else:
                card = err_obs
        if pause and not abort_reason:
            abort_reason = "shadow P0 (critical) — remaining writes cancelled"
        if card is None and not abort:
            return None
        meta: dict[str, Any] = {
            "pause_writes": pause or abort,
            "p0_count": len(all_p0),
            "mutation_paths": paths,
        }
        if restored:
            meta["restored"] = restored
        if restore_failed:
            meta["restore_failed"] = restore_failed
        return HookDecision(
            append_observation=card,
            abort=abort,
            abort_reason=abort_reason,
            metadata=meta,
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
