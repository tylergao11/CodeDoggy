"""Two-phase tool pipeline (Grok execute_tool_calls + prepare_tool_call).

Phase 1 — preflight **all** calls: parse, PreToolUse hooks, plan gate, policy.
Phase 2 — execute approved: path locks serialise same-file writes; others may
parallelise (ThreadPoolExecutor). PermissionReject / Cancel stop the batch;
HookDenied is non-terminal (reason → observation, continue).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from codedoggy.orchestration.capability import is_mutating_action, is_read_only_kind
from codedoggy.orchestration.path_lock import PathLockTable, lock_path_for_args
from codedoggy.orchestration.session_mode import (
    PLAN_REJECT_MESSAGE,
    PlanEditGate,
    SessionModeState,
    plan_mode_edit_gate,
)
from dataclasses import dataclass

from codedoggy.orchestration.types import (
    PrecheckResult,
    PrecheckVerdict,
    PreparedToolCall,
    ToolBatchResult,
    ToolLoopOutcome,
)
from codedoggy.tools.gate import enforce_policy, validate_args_against_schema
from codedoggy.tools.registry import FinalizedToolset
from codedoggy.tools.runtime import ToolCallContext, ToolError
from codedoggy.turn.executor import execute_tool_call, parse_tool_arguments
from codedoggy.turn.types import ToolCall, ToolResultRecord

logger = logging.getLogger(__name__)

# Hooks protocol duck-type: pre_tool_use(call, ctx) -> HookDecision | None
# deny via decision.abort → soft HookDenied (Grok); metadata hard=True → permission


def prepare_tool_call(
    tools: FinalizedToolset,
    call: ToolCall,
    *,
    cwd: Path,
    session_id: str | None = None,
    extra: dict[str, Any] | None = None,
    mode_state: SessionModeState | None = None,
    pre_tool_hook: Callable[..., Any] | None = None,
    hook_ctx: Any = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> PrecheckResult:
    """Phase 1: Grok prepare_tool_call (single call)."""
    name = call.name
    if is_cancelled is not None and is_cancelled():
        return PrecheckResult(
            verdict=PrecheckVerdict.CANCELLED,
            observation=f"Tool execution cancelled for tool `{name}`",
            tool_name=name,
            reason="cancelled",
        )

    args = parse_tool_arguments(call.arguments)
    call = ToolCall(id=call.id, name=call.name, arguments=args)

    ft = tools.by_client_name.get(name)
    if ft is None:
        return PrecheckResult(
            verdict=PrecheckVerdict.NON_EXISTING,
            observation=f"Error (not_found): unknown tool {name!r}",
            tool_name=name,
            reason="not_found",
        )

    kind = ft.kind
    # Schema
    try:
        validate_args_against_schema(args, ft.parameters or {})
    except ToolError as e:
        return PrecheckResult(
            verdict=PrecheckVerdict.PARSE_ERROR,
            observation=f"Error ({e.code}): {e.message}",
            tool_name=name,
            reason=e.message,
        )

    # PreToolUse hook (soft deny by default)
    if pre_tool_hook is not None:
        try:
            decision = pre_tool_hook(call, hook_ctx)
        except Exception:  # noqa: BLE001
            logger.exception("pre_tool_use hook failed")
            decision = None
        if decision is not None and getattr(decision, "abort", False):
            reason = getattr(decision, "abort_reason", None) or "denied by pre_tool_use hook"
            meta = getattr(decision, "metadata", None) or {}
            hard = bool(meta.get("hard") or meta.get("permission_reject"))
            obs = getattr(decision, "append_observation", None) or f"Error (hook_denied): {reason}"
            if hard:
                return PrecheckResult(
                    verdict=PrecheckVerdict.PERMISSION_REJECT,
                    observation=obs,
                    tool_name=name,
                    reason=reason,
                    hook_name=str(meta.get("hook_name") or "pre_tool_use"),
                )
            return PrecheckResult(
                verdict=PrecheckVerdict.HOOK_DENY,
                observation=obs,
                tool_name=name,
                reason=reason,
                hook_name=str(meta.get("hook_name") or "pre_tool_use"),
            )

    # Plan mode hard gate (Grok plan_mode_edit_gate — independent of yolo)
    if mode_state is not None:
        gate = plan_mode_edit_gate(
            mode_state, cwd=cwd, kind=kind, tool_name=name, args=args
        )
        if gate == PlanEditGate.REJECT_NON_PLAN_FILE:
            msg = PLAN_REJECT_MESSAGE.format(plan_file=mode_state.plan_file)
            return PrecheckResult(
                verdict=PrecheckVerdict.PLAN_REJECT,
                observation=f"Error (plan_mode): {msg}",
                tool_name=name,
                reason=msg,
            )

    # Workspace policy (hard — PermissionReject). Prefer wire short_id so
    # HARD_* name lists match registry.call (no client/wire drift).
    short_id = getattr(ft, "short_id", None) or name
    ctx = ToolCallContext(cwd=cwd, session_id=session_id, extra=dict(extra or {}))
    try:
        enforce_policy(
            tool_name=str(short_id),
            kind=kind,
            args=args,
            ctx=ctx,
            registered_kind=kind,
        )
    except ToolError as e:
        return PrecheckResult(
            verdict=PrecheckVerdict.PERMISSION_REJECT,
            observation=f"Error ({e.code}): {e.message}",
            tool_name=name,
            reason=e.message,
        )

    # writes_paused — single mutating classifier (capability.is_mutating_action)
    if (extra or {}).get("writes_paused") and is_mutating_action(kind, name):
        return PrecheckResult(
            verdict=PrecheckVerdict.PERMISSION_REJECT,
            observation=(
                "Error (writes_paused): writes paused — "
                "fix the blocking issue before more writes"
            ),
            tool_name=name,
            reason="writes_paused",
        )

    lock_path = None
    read_only = is_read_only_kind(kind)
    if not read_only:
        lock_path = lock_path_for_args(args)

    prepared = PreparedToolCall(
        call=call,
        parsed_args=args,
        tool_name=name,
        is_read_only=read_only,
        lock_path=lock_path,
    )
    return PrecheckResult(verdict=PrecheckVerdict.APPROVE, prepared=prepared, tool_name=name)


@dataclass(slots=True)
class Phase1Batch:
    """Result of prepare-all (Grok phase 1 only)."""

    outcome: ToolLoopOutcome
    # Parallel to original calls: PrecheckResult or None if skipped after hard stop
    prechecks: list[PrecheckResult]
    approved: list[tuple[int, PreparedToolCall]]
    tool_name: str | None = None
    reason: str | None = None
    followup_message: str | None = None
    hook_name: str | None = None


def prepare_tool_batch(
    tools: FinalizedToolset,
    calls: list[ToolCall],
    *,
    cwd: Path | str,
    session_id: str | None = None,
    extra: dict[str, Any] | None = None,
    mode_state: SessionModeState | None = None,
    pre_tool_hook: Callable[..., Any] | None = None,
    hook_ctx: Any = None,
    is_cancelled: Callable[[], bool] | None = None,
    interjection_pending: Callable[[], bool] | None = None,
) -> Phase1Batch:
    """Grok phase 1 only: preflight every call before any execute."""
    cwd_path = Path(cwd).resolve()
    extra = dict(extra or {})
    prechecks: list[PrecheckResult] = []
    approved: list[tuple[int, PreparedToolCall]] = []
    final_outcome = ToolLoopOutcome.CONTINUE
    final_tool: str | None = None
    final_reason: str | None = None
    final_hook: str | None = None
    followup: str | None = None

    for idx, call in enumerate(calls):
        if final_outcome in {
            ToolLoopOutcome.PERMISSION_REJECT,
            ToolLoopOutcome.CANCELLED,
            ToolLoopOutcome.FOLLOWUP_MESSAGE,
        }:
            prechecks.append(
                PrecheckResult(
                    verdict=PrecheckVerdict.CANCELLED,
                    observation=_cancel_msg(final_outcome, final_tool, call.name),
                    tool_name=call.name,
                    reason="batch_cancelled",
                )
            )
            continue

        if interjection_pending is not None and interjection_pending():
            final_outcome = ToolLoopOutcome.FOLLOWUP_MESSAGE
            followup = "(interjection pending)"
            prechecks.append(
                PrecheckResult(
                    verdict=PrecheckVerdict.CANCELLED,
                    observation=(
                        f"Tool execution cancelled due to earlier user followup "
                        f"message for tool `{call.name}`"
                    ),
                    tool_name=call.name,
                    reason="followup",
                )
            )
            continue

        pre = prepare_tool_call(
            tools,
            call,
            cwd=cwd_path,
            session_id=session_id,
            extra=extra,
            mode_state=mode_state,
            pre_tool_hook=pre_tool_hook,
            hook_ctx=hook_ctx,
            is_cancelled=is_cancelled,
        )
        prechecks.append(pre)

        if pre.verdict is PrecheckVerdict.APPROVE and pre.prepared is not None:
            approved.append((idx, pre.prepared))
            continue

        if pre.verdict is PrecheckVerdict.HOOK_DENY:
            final_hook = pre.hook_name
            continue
        if pre.verdict is PrecheckVerdict.PERMISSION_REJECT:
            final_outcome = ToolLoopOutcome.PERMISSION_REJECT
            final_tool = pre.tool_name
            final_reason = pre.reason
            continue
        if pre.verdict is PrecheckVerdict.PLAN_REJECT:
            final_outcome = ToolLoopOutcome.PERMISSION_REJECT
            final_tool = pre.tool_name
            final_reason = pre.reason
            continue
        if pre.verdict is PrecheckVerdict.CANCELLED:
            final_outcome = ToolLoopOutcome.CANCELLED
            final_tool = pre.tool_name
            final_reason = "cancelled"
            continue

    return Phase1Batch(
        outcome=final_outcome,
        prechecks=prechecks,
        approved=approved,
        tool_name=final_tool,
        reason=final_reason,
        followup_message=followup,
        hook_name=final_hook,
    )


def execute_prepared(
    tools: FinalizedToolset,
    prepared: PreparedToolCall,
    *,
    cwd: Path | str,
    session_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> ToolResultRecord:
    """Phase 2 single execute (caller owns after_tool / after_mutation order)."""
    return execute_tool_call(
        tools,
        prepared.call,
        cwd=Path(cwd).resolve(),
        session_id=session_id,
        extra=extra,
    )


def execute_approved_batch(
    tools: FinalizedToolset,
    approved: list[tuple[int, PreparedToolCall]],
    *,
    cwd: Path | str,
    session_id: str | None = None,
    extra: dict[str, Any] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    max_workers: int = 8,
    parallel: bool = True,
) -> dict[int, ToolResultRecord]:
    """Phase 2 — Grok ``execute_tool_calls`` dispatch with path locks.

    Same write path shares a mutex (serialised under lock); distinct paths and
    read-only tools run concurrently via a thread pool. Each tool gets a shallow
    copy of ``extra`` (see ``execute_tool_call``).
    """
    if not approved:
        return {}
    cwd_path = Path(cwd).resolve()
    base_extra = dict(extra or {})
    locks = PathLockTable()
    write_paths = {
        p.lock_path for _, p in approved if p.lock_path and not p.is_read_only
    }

    def _run_one(item: tuple[int, PreparedToolCall]) -> tuple[int, ToolResultRecord]:
        idx, prepared = item
        lock = (
            locks.lock_for(prepared.lock_path)
            if prepared.lock_path and prepared.lock_path in write_paths
            else None
        )
        if lock is not None:
            lock.acquire()
        try:
            if is_cancelled is not None and is_cancelled():
                return idx, ToolResultRecord(
                    call=prepared.call,
                    content=f"Tool execution cancelled for tool `{prepared.tool_name}`",
                    ok=False,
                    error_code="cancelled",
                    kind=tools.kind_of(prepared.tool_name),
                )
            rec = execute_tool_call(
                tools,
                prepared.call,
                cwd=cwd_path,
                session_id=session_id,
                extra=base_extra,
            )
            return idx, rec
        finally:
            if lock is not None:
                lock.release()

    out: dict[int, ToolResultRecord] = {}
    if parallel and len(approved) > 1:
        workers = min(max_workers, len(approved))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_run_one, item) for item in approved]
            for fut in as_completed(futs):
                idx, rec = fut.result()
                out[idx] = rec
    else:
        for item in approved:
            idx, rec = _run_one(item)
            out[idx] = rec
    return out


def execute_tool_calls_two_phase(
    tools: FinalizedToolset,
    calls: list[ToolCall],
    *,
    cwd: Path | str,
    session_id: str | None = None,
    extra: dict[str, Any] | None = None,
    mode_state: SessionModeState | None = None,
    pre_tool_hook: Callable[..., Any] | None = None,
    hook_ctx: Any = None,
    is_cancelled: Callable[[], bool] | None = None,
    interjection_pending: Callable[[], bool] | None = None,
    max_workers: int = 8,
    parallel: bool = True,
) -> ToolBatchResult:
    """Grok ``execute_tool_calls``: phase-1 all, then phase-2 path-lock dispatch."""
    cwd_path = Path(cwd).resolve()
    extra = dict(extra or {})
    phase1 = prepare_tool_batch(
        tools,
        calls,
        cwd=cwd_path,
        session_id=session_id,
        extra=extra,
        mode_state=mode_state,
        pre_tool_hook=pre_tool_hook,
        hook_ctx=hook_ctx,
        is_cancelled=is_cancelled,
        interjection_pending=interjection_pending,
    )
    records: list[ToolResultRecord | None] = [None] * len(calls)

    for idx, pre in enumerate(phase1.prechecks):
        if pre.verdict is PrecheckVerdict.APPROVE:
            continue
        call = calls[idx]
        records[idx] = ToolResultRecord(
            call=ToolCall(
                id=call.id,
                name=call.name,
                arguments=parse_tool_arguments(call.arguments),
            ),
            content=pre.observation or f"Error: {pre.reason}",
            ok=False,
            error_code=pre.verdict.value,
            kind=tools.kind_of(call.name),
        )

    approved = phase1.approved
    if approved:
        executed = execute_approved_batch(
            tools,
            approved,
            cwd=cwd_path,
            session_id=session_id,
            extra=extra,
            is_cancelled=is_cancelled,
            max_workers=max_workers,
            parallel=parallel,
        )
        for idx, rec in executed.items():
            records[idx] = rec

    for i, call in enumerate(calls):
        if records[i] is None:
            records[i] = ToolResultRecord(
                call=call,
                content=f"Error (internal): missing result for {call.name}",
                ok=False,
                error_code="internal",
            )

    return ToolBatchResult(
        outcome=phase1.outcome,
        records=list(records),  # type: ignore[arg-type]
        tool_name=phase1.tool_name,
        reason=phase1.reason,
        followup_message=phase1.followup_message,
        hook_name=phase1.hook_name,
        metadata={"approved": len(approved), "parallel": parallel and len(approved) > 1},
    )


def _cancel_msg(outcome: ToolLoopOutcome, first_tool: str | None, name: str) -> str:
    if outcome is ToolLoopOutcome.PERMISSION_REJECT:
        return (
            f"Tool execution cancelled due to earlier permission rejection "
            f"for tool `{first_tool or '?'}`"
        )
    if outcome is ToolLoopOutcome.CANCELLED:
        return f"Tool execution cancelled due to earlier user cancellation for tool `{name}`"
    if outcome is ToolLoopOutcome.FOLLOWUP_MESSAGE:
        return (
            f"Tool execution cancelled due to earlier user followup message "
            f"for tool `{name}`"
        )
    return f"Tool execution cancelled for tool `{name}`"
