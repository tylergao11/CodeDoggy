"""Agentic turn loop: sample → tool calls → observations → repeat."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from codedoggy.context.budget import ContextBudget
from codedoggy.context.compactor import ContextCompactor
from codedoggy.tools.registry import FinalizedToolset
from codedoggy.orchestration.tool_pipeline import (
    execute_approved_batch,
    prepare_tool_batch,
)
from codedoggy.orchestration.types import PrecheckVerdict, ToolLoopOutcome
from codedoggy.turn.executor import parse_tool_arguments
from codedoggy.turn.hooks import HookContext, LoopHooks, NoopHooks
from codedoggy.turn.sampler import Sampler
from codedoggy.turn.types import (
    HookDecision,
    LoopResult,
    Message,
    Role,
    SampleResult,
    ToolCall,
    ToolResultRecord,
)

logger = logging.getLogger(__name__)


def run_agent_loop(
    *,
    user_text: str,
    sampler: Sampler,
    tools: FinalizedToolset,
    cwd: Path | str,
    max_turns: int | None = None,
    system_prompt: str | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    hooks: LoopHooks | None = None,
    session: Any = None,
    session_id: str | None = None,
    tool_extra: dict[str, Any] | None = None,
    prompt_id: str | None = None,
    context_budget: ContextBudget | None = None,
    context_compactor: ContextCompactor | None = None,
    prior_messages: list[Message] | None = None,
    on_archive_message: Callable[[Message], None] | None = None,
) -> LoopResult:
    """Run one user prompt through the ReAct-style loop.

    Parameters
    ----------
    max_turns:
        Max *sampling* rounds (each may include a tool batch). ``None`` = unlimited.
    is_cancelled:
        Polled between steps; when true, returns ``cancelled=True``.
    hooks:
        Optional ``after_sample`` / ``after_tool`` / ``after_mutation`` /
        ``on_turn_end``.
    context_budget / context_compactor:
        Grok-style in-session window control (prune tool noise, fold middle).
        Hermes MEMORY in system messages is never dropped.
    prior_messages:
        Cross-prompt live window (non-system history from previous prompts).
    on_archive_message:
        Called with a *copy* of each new non-system message as it is created
        (before later prune/fold). SessionStore uses this for full-fidelity FTS.
    """
    cwd_path = Path(cwd).resolve()
    hook_impl: LoopHooks = hooks if hooks is not None else NoopHooks()
    reset_buf = getattr(hook_impl, "reset_turn_buffer", None)
    if callable(reset_buf):
        reset_buf()

    if context_compactor is None and context_budget is not None:
        context_compactor = ContextCompactor(budget=context_budget)
    elif context_compactor is None:
        # Default on: portable char budget from env / defaults.
        context_compactor = ContextCompactor(budget=ContextBudget.from_env())
    compact_meta: dict[str, Any] = {"compactions": 0}

    extra: dict[str, Any] = dict(tool_extra or {})
    if session is not None:
        ext = getattr(session, "extensions", None)
        if "memory_store" not in extra and ext is not None:
            mem = getattr(ext, "memory", None)
            if mem is not None:
                extra["memory_store"] = mem
                reset = getattr(mem, "reset_consolidation_failures", None)
                if callable(reset):
                    reset()
        if "session_store" not in extra and ext is not None:
            ss = getattr(ext, "session_store", None)
            if ss is not None:
                extra["session_store"] = ss
        if "policy" not in extra and ext is not None:
            pol = getattr(ext, "policy", None)
            if pol is not None:
                extra["policy"] = pol
        if "memory_manager" not in extra and ext is not None:
            mm = getattr(ext, "memory_manager", None)
            if mm is not None:
                extra["memory_manager"] = mm
        if "graph" not in extra and ext is not None:
            gr = getattr(ext, "graph", None)
            if gr is not None:
                extra["graph"] = gr

    from codedoggy.context.live_history import (
        copy_message,
        model_sample_messages,
        seed_messages,
    )
    from codedoggy.prompt.user_message import construct_user_message

    messages = seed_messages(
        system_prompt=system_prompt,
        user_text=user_text,
        prior_messages=prior_messages,
    )
    # Hermes seam: strip leaked fences from prior live; inject only at sample time
    from codedoggy.memory.context_fence import strip_memory_context_from_messages
    from codedoggy.memory.hermes_seam import sample_messages_with_memory

    messages = strip_memory_context_from_messages(messages)
    # Grok MAIN prefix is model-facing only.  Keep the canonical live/archive
    # transcript free of user_info, git snapshots, and <user_query> wrappers.
    prefix_owner = extra.get("kernel") or session
    cached_prefix = (
        getattr(prefix_owner, "_grok_main_user_message_prefix", None)
        if prefix_owner is not None
        else None
    )
    if isinstance(cached_prefix, str) and cached_prefix.strip():
        user_message_prefix = cached_prefix
    else:
        user_message_prefix = construct_user_message(cwd_path)
        if prefix_owner is not None:
            try:
                setattr(
                    prefix_owner,
                    "_grok_main_user_message_prefix",
                    user_message_prefix,
                )
            except Exception:  # noqa: BLE001 - cache is optional
                pass
    # Ephemeral fence only at sample time (conversation_loop inject into user).
    # Not archived, not SYSTEM, not a separate USER turn.
    prefetch_block = extra.get("prefetch_user_block")
    if not isinstance(prefetch_block, str) or not prefetch_block.strip():
        prefetch_block = None
    # Archive only *this prompt's* clean user line (priors already archived).
    if on_archive_message is not None:
        try:
            on_archive_message(copy_message(messages[-1]))
        except Exception:  # noqa: BLE001
            logger.exception("on_archive_message failed for user prompt")

    tools_called: list[str] = []
    rounds = 0
    # Successful model samples count toward max_turns. Context-overflow retries
    # have their own bounded budget and never consume a normal Grok turn.
    sample_attempts = 0
    overflow_resubmits = 0
    MAX_OVERFLOW_RESUBMITS = 2
    final_text: str | None = None

    def _hctx(round_index: int) -> HookContext:
        goal = getattr(session, "goal", None) if session is not None else None
        return HookContext(
            cwd=cwd_path,
            round_index=round_index,
            session=session,
            prompt_id=prompt_id,
            goal=goal if isinstance(goal, str) else None,
        )

    def _finish(
        *,
        completed: bool,
        cancelled: bool = False,
        aborted: bool = False,
        max_turns_reached: bool = False,
        error: str | None = None,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
        exit_reason: str | None = None,
    ) -> LoopResult:
        meta = dict(metadata or {})
        if compact_meta.get("compactions"):
            meta["context_compactions"] = compact_meta["compactions"]
            meta["context_last"] = compact_meta.get("last")
        # Grok: drain mid-turn interjections into metadata for host visibility
        inj = extra.get("interjection_buffer")
        if inj is not None and hasattr(inj, "__len__") and len(inj) > 0:
            meta["interjections_pending"] = len(inj)
        body = text if text is not None else final_text
        deferred = _call_on_turn_end(hook_impl, _hctx(rounds))
        if deferred:
            meta["turn_end_notes"] = deferred
            body = f"{body}\n\n{deferred}" if body else deferred
            note_msg = Message(
                role=Role.USER,
                content=("[end-of-turn notes]\n" + deferred.lstrip()),
            )
            messages.append(note_msg)
            _archive(on_archive_message, note_msg)
        reason = exit_reason
        if reason is None:
            if completed:
                reason = "completed"
            elif max_turns_reached:
                reason = "max_turns"
            elif cancelled:
                reason = "cancelled"
            elif aborted:
                reason = "aborted"
            elif error:
                reason = "error"
        return LoopResult(
            final_text=body,
            messages=messages,
            tools_called=tools_called,
            rounds=rounds,
            completed=completed,
            cancelled=cancelled,
            aborted=aborted,
            max_turns_reached=max_turns_reached,
            error=error,
            exit_reason=reason,
            metadata=meta,
        )

    while True:
        if is_cancelled is not None and is_cancelled():
            return _finish(completed=False, cancelled=True, exit_reason="cancelled")

        if max_turns is not None and rounds >= max_turns:
            partial = final_text or _last_assistant_text(messages)
            return _finish(
                completed=False,
                max_turns_reached=True,
                text=partial,
                exit_reason="max_turns",
                metadata={"hint": f"stopped after {max_turns} sampling round(s)"},
            )
        # Grok: drain pending interjections before each sample (turn.rs safe point)
        # Framing: xai-interjection-core format_interjection — not [interjection]
        _drain_interjections_into_messages(
            messages, extra, on_archive_message=on_archive_message
        )

        # Grok foundation: enforce live context budget before every sample.
        if context_compactor is not None:
            cres = context_compactor.ensure(messages)
            messages = cres.messages
            if cres.did_compact:
                compact_meta["compactions"] = int(compact_meta.get("compactions", 0)) + 1
                compact_meta["last"] = {
                    "mode": cres.mode,
                    "pruned_tools": cres.pruned_tools,
                    "folded_messages": cres.folded_messages,
                    "chars_before": cres.chars_before,
                    "chars_after": cres.chars_after,
                }
                logger.info(
                    "context compacted mode=%s before=%s after=%s folded=%s",
                    cres.mode,
                    cres.chars_before,
                    cres.chars_after,
                    cres.folded_messages,
                )

        # Dynamic tool descriptions (skill <available_skills>) re-render each turn.
        bind = getattr(tools, "bind_list_context", None)
        if callable(bind):
            try:
                bind(cwd=cwd_path, extra=extra)
            except Exception:  # noqa: BLE001
                logger.debug("bind_list_context failed", exc_info=True)
        tool_specs = tools.tool_definitions()
        # Grok: tools_reserve counts against the window every sample
        if context_compactor is not None:
            bind = getattr(context_compactor, "bind_sample_tools", None)
            if callable(bind):
                try:
                    bind(tool_specs)
                except Exception:  # noqa: BLE001
                    logger.debug("bind_sample_tools failed", exc_info=True)
        # Hermes seam: API-only inject into current user (no transcript mutate)
        sample_messages = sample_messages_with_memory(messages, prefetch_block)
        sample_messages = model_sample_messages(
            sample_messages,
            user_message_prefix=user_message_prefix,
        )
        sample_attempts += 1
        try:
            sample = _sample_with_host_stream(
                sampler, sample_messages, tool_specs, extra
            )
        except Exception as e:
            # Grok compact-and-resubmit on context overflow (not silent end)
            err_s = str(e).lower()
            overflow = any(
                k in err_s
                for k in (
                    "context_length",
                    "context window",
                    "maximum context",
                    "too many tokens",
                    "prompt is too long",
                    "model_context_window",
                    "context_window_exceeded",
                )
            )
            if (
                overflow
                and context_compactor is not None
                and overflow_resubmits < MAX_OVERFLOW_RESUBMITS
            ):
                logger.warning("context overflow — compact and resubmit: %s", e)
                try:
                    from codedoggy.context.budget import estimate_chars

                    before_chars = estimate_chars(messages)
                    bud = getattr(context_compactor, "budget", None)
                    old_pct = None
                    if bud is not None:
                        old_pct = bud.threshold_percent
                        bud.threshold_percent = min(40, int(bud.threshold_percent))
                    cres = context_compactor.ensure(messages)
                    messages = cres.messages
                    if old_pct is not None and bud is not None:
                        bud.threshold_percent = old_pct
                    after_chars = estimate_chars(messages)
                    overflow_resubmits += 1
                    compact_meta["compactions"] = int(
                        compact_meta.get("compactions", 0)
                    ) + 1
                    compact_meta["last"] = {
                        "mode": "compact_resubmit",
                        "error": str(e)[:200],
                        "chars_before": before_chars,
                        "chars_after": after_chars,
                        "overflow_resubmits": overflow_resubmits,
                    }
                    # No progress → stop (do not infinite resubmit same window)
                    if after_chars >= before_chars and not getattr(
                        cres, "did_compact", False
                    ):
                        return _finish(
                            completed=False,
                            error=(
                                f"context overflow and compaction made no progress "
                                f"({before_chars}→{after_chars}): {e}"
                            ),
                            exit_reason="error",
                            metadata=compact_meta,
                        )
                    continue
                except Exception:  # noqa: BLE001
                    logger.exception("compact-and-resubmit failed")
            logger.exception("sampler failed")
            return _finish(
                completed=False,
                error=f"sampler error: {e}",
                exit_reason="error",
                metadata={
                    "sample_attempts": sample_attempts,
                    "overflow_resubmits": overflow_resubmits,
                },
            )

        sample = _normalize_sample(sample)
        rounds += 1
        overflow_resubmits = 0  # reset after a successful sample
        # Real usage → budget (single path; clears awaiting_real_usage after fold)
        _note_sample_usage(context_compactor, sample)

        _pdata = getattr(sample, "provider_data", None)
        if not isinstance(_pdata, dict) and isinstance(sample.raw, dict):
            _pdata = sample.raw.get("provider_data")
        messages.append(
            Message(
                role=Role.ASSISTANT,
                content=sample.content,
                tool_calls=list(sample.tool_calls) if sample.tool_calls else None,
                reasoning_content=getattr(sample, "reasoning_content", None),
                provider_data=dict(_pdata) if isinstance(_pdata, dict) else None,
            )
        )

        hctx = _hctx(rounds)
        decision = _call_hook(hook_impl, "after_sample", sample, hctx)
        if decision and decision.append_observation and sample.content:
            messages[-1] = Message(
                role=Role.ASSISTANT,
                content=f"{sample.content}\n{decision.append_observation}",
                tool_calls=messages[-1].tool_calls,
                reasoning_content=messages[-1].reasoning_content,
                provider_data=messages[-1].provider_data,
            )
        elif decision and decision.append_observation and not sample.content:
            messages[-1] = Message(
                role=Role.ASSISTANT,
                content=decision.append_observation,
                tool_calls=messages[-1].tool_calls,
                reasoning_content=messages[-1].reasoning_content,
                provider_data=messages[-1].provider_data,
            )
        _archive(on_archive_message, messages[-1])
        if decision and decision.abort:
            return _finish(
                completed=False,
                aborted=True,
                text=messages[-1].content,
                error=decision.abort_reason or "aborted after_sample",
            )

        if not sample.has_tool_calls:
            final_text = (messages[-1].content or "").strip() or None
            if not final_text:
                # Model ended the turn with empty prose (common after tool-only
                # rounds with local models). Prefer last non-empty assistant,
                # else an honest harness note so callers never get silent "".
                final_text = _fallback_final_text(messages, tools_called)
                if final_text:
                    messages[-1] = Message(
                        role=Role.ASSISTANT,
                        content=final_text,
                        tool_calls=messages[-1].tool_calls,
                        reasoning_content=messages[-1].reasoning_content,
                        provider_data=messages[-1].provider_data,
                    )
                    # Re-archive filled final (first archive may have been empty).
                    _archive(on_archive_message, messages[-1])
            return _finish(completed=True, text=final_text)

        if is_cancelled is not None and is_cancelled():
            return _finish(completed=False, cancelled=True, text=sample.content)

        # True prefire overlap: start flush LLM *before* tools run so it
        # shares wall-clock with tool I/O; next ensure() joins the result.
        if context_compactor is not None and sample.tool_calls:
            sched = getattr(context_compactor, "schedule_prefire_flush", None)
            if callable(sched):
                try:
                    sched(messages)
                except Exception:  # noqa: BLE001
                    logger.exception("schedule_prefire_flush (pre-tools) failed")

        # ── Grok two-phase tools (tool_calls.rs / tool_dispatch.rs) ──
        # Phase 1: prepare ALL (permission hard-stop remaining prepares).
        # Phase 2: execute approved with path locks — same write path serial
        # under mutex; other tools concurrent (Grok FuturesUnordered spirit).
        # Phase 3: writeback observations in model emission order + hooks.
        batch_calls = list(sample.tool_calls or [])

        def _pre_hook(call: ToolCall, _ctx: Any) -> HookDecision | None:
            return _call_hook(hook_impl, "pre_tool_use", call, hctx)

        def _inj_pending() -> bool:
            buf = extra.get("interjection_buffer")
            if buf is None:
                return False
            empty = getattr(buf, "is_empty", None)
            if callable(empty):
                return not empty()
            return bool(len(buf)) if hasattr(buf, "__len__") else False

        mode_state = extra.get("session_mode_state")
        phase1 = prepare_tool_batch(
            tools,
            batch_calls,
            cwd=cwd_path,
            session_id=session_id,
            extra=extra,
            mode_state=mode_state,
            pre_tool_hook=_pre_hook,
            hook_ctx=hctx,
            is_cancelled=is_cancelled,
            interjection_pending=_inj_pending,
        )

        if is_cancelled is not None and is_cancelled():
            _append_cancelled_tool_results(
                messages,
                batch_calls,
                on_archive_message,
                reason="cancelled",
            )
            return _finish(
                completed=False,
                cancelled=True,
                text=sample.content,
                exit_reason="cancelled",
            )

        # Phase 2: path-lock parallel dispatch (Grok execute_tool_calls)
        executed = execute_approved_batch(
            tools,
            phase1.approved,
            cwd=cwd_path,
            session_id=session_id,
            extra=extra,
            is_cancelled=is_cancelled,
            parallel=True,
        )

        abort_reason: str | None = None
        abort_meta: dict[str, Any] = {}

        # Phase 3: model-order writeback + optional hooks (after all exec)
        for idx, call in enumerate(batch_calls):
            pre = phase1.prechecks[idx]

            if pre.verdict is not PrecheckVerdict.APPROVE:
                record = ToolResultRecord(
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
                tools_called.append(record.call.name)
                _append_tool_message(messages, record, record.content, on_archive_message)
                continue

            record = executed.get(idx)
            if record is None:
                record = ToolResultRecord(
                    call=call,
                    content=(
                        f"Error (internal): missing result for tool "
                        f"{call.name!r} ({call.id})"
                    ),
                    ok=False,
                    error_code="internal",
                    kind=tools.kind_of(call.name),
                )

            tools_called.append(record.call.name)
            observation = record.content

            decision = _call_hook(hook_impl, "after_tool", record, hctx)
            observation, abort = _apply_decision(observation, decision)
            if abort is not None:
                _append_tool_message(messages, record, observation, on_archive_message)
                if abort_reason is None:
                    abort_reason = abort
                continue

            # Optional after_mutation
            has_mut = bool(getattr(record, "mutations", None) or record.mutation)
            if has_mut:
                mut_decision = _call_hook(hook_impl, "after_mutation", record, hctx)
                observation, abort = _apply_decision(observation, mut_decision)
                if abort is not None:
                    _append_tool_message(messages, record, observation, on_archive_message)
                    extra["writes_paused"] = True
                    if abort_reason is None:
                        abort_reason = abort
                    path = None
                    if record.mutation is not None:
                        path = record.mutation.path
                    abort_meta = {"mutation_path": path} if path else {}
                    continue
                if mut_decision and getattr(mut_decision, "metadata", None):
                    if (mut_decision.metadata or {}).get("pause_writes"):
                        extra["writes_paused"] = True

            _append_tool_message(messages, record, observation, on_archive_message)

        if abort_reason is not None:
            return _finish(
                completed=False,
                aborted=True,
                text=sample.content,
                error=abort_reason,
                metadata=abort_meta,
                exit_reason="aborted",
            )

        # Grok ToolLoop hard outcomes from phase 1
        if phase1.outcome is ToolLoopOutcome.PERMISSION_REJECT:
            return _finish(
                completed=False,
                cancelled=True,
                text=sample.content,
                error=phase1.reason or "permission rejected",
                exit_reason="permission_reject",
                metadata={
                    "tool_name": phase1.tool_name,
                    "reason": phase1.reason,
                },
            )
        if phase1.outcome is ToolLoopOutcome.CANCELLED:
            return _finish(
                completed=False,
                cancelled=True,
                text=sample.content,
                exit_reason="cancelled",
            )
        if phase1.outcome is ToolLoopOutcome.FOLLOWUP_MESSAGE:
            drained = _drain_interjections_into_messages(
                messages, extra, on_archive_message=on_archive_message
            )
            if not drained and phase1.followup_message:
                from codedoggy.orchestration.interjection import format_interjection

                um = Message(
                    role=Role.USER,
                    content=format_interjection(phase1.followup_message),
                )
                messages.append(um)
                _archive(on_archive_message, um)

        # After tools: window prune/fold
        if context_compactor is not None and sample.tool_calls:
            cres = context_compactor.ensure(messages)
            messages = cres.messages
            if cres.did_compact:
                compact_meta["compactions"] = int(compact_meta.get("compactions", 0)) + 1
                compact_meta["last"] = {
                    "mode": cres.mode,
                    "pruned_tools": cres.pruned_tools,
                    "folded_messages": cres.folded_messages,
                    "chars_before": cres.chars_before,
                    "chars_after": cres.chars_after,
                    "when": "post_tool_batch",
                }

        # Drain interjections at next safe point when batch continued
        if phase1.outcome is ToolLoopOutcome.CONTINUE:
            _drain_interjections_into_messages(
                messages, extra, on_archive_message=on_archive_message
            )

        # Continue: model sees tool results on next sample.


def _call_on_turn_end(hooks: LoopHooks, ctx: HookContext) -> str | None:
    fn = getattr(hooks, "on_turn_end", None)
    if fn is None or not callable(fn):
        return None
    try:
        out = fn(ctx)
    except Exception:  # noqa: BLE001
        logger.exception("on_turn_end failed")
        return None
    return out if isinstance(out, str) and out.strip() else None


def _last_assistant_text(messages: list[Message]) -> str | None:
    for msg in reversed(messages):
        if msg.role is Role.ASSISTANT and msg.content:
            return msg.content
    return None


def _append_cancelled_tool_results(
    messages: list[Message],
    calls: list[ToolCall],
    on_archive: Callable[[Message], None] | None,
    *,
    reason: str,
) -> None:
    """Fill missing tool results so assistant tool_calls stay protocol-valid."""
    for call in calls:
        msg = Message(
            role=Role.TOOL,
            content=(
                f"Error ({reason}): tool {call.name!r} ({call.id}) "
                f"did not run to completion"
            ),
            tool_call_id=call.id,
            name=call.name,
        )
        messages.append(msg)
        _archive(on_archive, msg)


def _fallback_final_text(
    messages: list[Message],
    tools_called: list[str],
) -> str:
    """Honest final when the model returns empty content without tool_calls."""
    for msg in reversed(messages):
        if msg.role is Role.ASSISTANT and (msg.content or "").strip():
            return msg.content or ""
    # Summarize recent tool outcomes for the user/CLI
    tool_bits: list[str] = []
    for msg in reversed(messages):
        if msg.role is Role.TOOL and (msg.content or "").strip():
            name = msg.name or "tool"
            body = (msg.content or "").replace("\n", " ").strip()
            if len(body) > 160:
                body = body[:157] + "…"
            tool_bits.append(f"- {name}: {body}")
            if len(tool_bits) >= 3:
                break
    tool_bits.reverse()
    names = ", ".join(tools_called[-8:]) if tools_called else "(none)"
    lines = [
        "(Model returned no final prose after tools; harness summary)",
        f"Tools this turn: {names}",
    ]
    if tool_bits:
        lines.append("Latest tool results:")
        lines.extend(tool_bits)
    return "\n".join(lines)


def _normalize_sample(sample: SampleResult) -> SampleResult:
    calls: list[ToolCall] = []
    for i, tc in enumerate(sample.tool_calls or []):
        call_id = tc.id or f"call_{uuid4().hex[:12]}"
        args = parse_tool_arguments(tc.arguments)
        calls.append(ToolCall(id=call_id, name=tc.name, arguments=args))
    pdata = getattr(sample, "provider_data", None)
    if not isinstance(pdata, dict) and isinstance(sample.raw, dict):
        pdata = sample.raw.get("provider_data")
    return SampleResult(
        content=sample.content,
        tool_calls=calls,
        raw=sample.raw,
        reasoning_content=getattr(sample, "reasoning_content", None),
        provider_data=dict(pdata) if isinstance(pdata, dict) else None,
    )


def _append_tool_message(
    messages: list[Message],
    record: ToolResultRecord,
    observation: str,
    on_archive_message: Callable[[Message], None] | None = None,
) -> None:
    msg = Message(
        role=Role.TOOL,
        content=observation,
        tool_call_id=record.call.id,
        name=record.call.name,
    )
    messages.append(msg)
    # Archive *now* — full tool body including P0 footers, before any later
    # retain-prune / fold rewrites the live window.
    _archive(on_archive_message, msg)


def _archive(
    on_archive_message: Callable[[Message], None] | None,
    message: Message,
) -> None:
    if on_archive_message is None:
        return
    try:
        from codedoggy.context.live_history import copy_message

        on_archive_message(copy_message(message))
    except Exception:  # noqa: BLE001
        logger.exception("on_archive_message failed")


def _sample_with_host_stream(
    sampler: Any,
    messages: list[Message],
    tool_specs: list,
    extra: dict[str, Any],
) -> SampleResult:
    """Sample; optional host progressive deltas only.

    Glue: ``tool_extra.stream_sample`` / ``on_sample_delta`` for CLI/ACP hosts.
    Interjections are **not** coupled here — Grok drains at safe points only
    (loop head / post-tool via ``_drain_interjections_into_messages``).
    """
    on_delta_host = extra.get("on_sample_delta")
    want_stream = bool(extra.get("stream_sample")) or callable(on_delta_host)
    if not want_stream:
        return sampler.sample(messages, tool_specs)

    def _on_delta(chunk: str) -> None:
        if callable(on_delta_host) and chunk:
            try:
                on_delta_host(chunk)
            except Exception:  # noqa: BLE001
                logger.debug("on_sample_delta failed", exc_info=True)

    if hasattr(sampler, "stream") and hasattr(sampler, "sample"):
        prev_stream = getattr(sampler, "stream", False)
        prev_delta = getattr(sampler, "on_delta", None)
        try:
            sampler.stream = True
            sampler.on_delta = _on_delta
            return sampler.sample(messages, tool_specs)
        finally:
            try:
                sampler.stream = prev_stream
                sampler.on_delta = prev_delta
            except Exception:  # noqa: BLE001
                pass

    stream_fn = getattr(sampler, "sample_stream", None)
    if callable(stream_fn):
        try:
            return stream_fn(messages, tool_specs, on_delta=_on_delta)
        except TypeError:
            return stream_fn(messages, tool_specs)

    return sampler.sample(messages, tool_specs)


def _drain_interjections_into_messages(
    messages: list[Message],
    extra: dict[str, Any],
    *,
    on_archive_message: Callable[[Message], None] | None,
) -> int:
    """Grok drain_pending_interjections at a safe point.

    Uses ``drain_formatted`` → ``format_interjection`` (xai-interjection-core).
    Returns number of synthetic USER messages appended.
    """
    buf = extra.get("interjection_buffer")
    if buf is None:
        return 0
    texts: list[str] = []
    drain_fmt = getattr(buf, "drain_formatted", None)
    if callable(drain_fmt):
        try:
            texts = list(drain_fmt() or [])
        except Exception:  # noqa: BLE001
            logger.debug("drain_formatted failed", exc_info=True)
            texts = []
    if not texts:
        drain = getattr(buf, "drain", None)
        if callable(drain):
            try:
                from codedoggy.orchestration.interjection import drain_formatted

                texts = drain_formatted(list(drain() or []))
            except Exception:  # noqa: BLE001
                logger.debug("drain interjections failed", exc_info=True)
                return 0
    n = 0
    for text_i in texts:
        if not text_i or not str(text_i).strip():
            continue
        um = Message(role=Role.USER, content=str(text_i))
        messages.append(um)
        _archive(on_archive_message, um)
        n += 1
    return n


def _note_sample_usage(compactor: ContextCompactor | None, sample: SampleResult) -> None:
    """Feed model-reported usage into the budget (prefer real tokens over estimates).

    Single path: ContextCompactor.update_from_response when available, else
    budget fields directly. Clears awaiting_real_usage after fold.
    """
    if compactor is None:
        return
    raw = sample.raw or {}
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if not isinstance(usage, dict) or not usage:
        return
    upd = getattr(compactor, "update_from_response", None)
    if callable(upd):
        try:
            upd(usage)
            return
        except Exception:  # noqa: BLE001
            logger.debug("update_from_response failed", exc_info=True)
    budget = getattr(compactor, "budget", None)
    if budget is None:
        return
    for key in ("prompt_tokens", "input_tokens", "prompt_token_count"):
        val = usage.get(key)
        if val is not None:
            try:
                budget.last_prompt_tokens = int(val)
            except (TypeError, ValueError):
                pass
            break
    for key in ("completion_tokens", "output_tokens"):
        val = usage.get(key)
        if val is not None:
            try:
                budget.last_completion_tokens = int(val)
            except (TypeError, ValueError):
                pass
            break


def _apply_decision(
    observation: str, decision: HookDecision | None
) -> tuple[str, str | None]:
    """Return (observation, abort_reason_or_None)."""
    if decision is None:
        return observation, None
    if decision.append_observation:
        extra = decision.append_observation
        observation = f"{observation}\n{extra}" if observation else extra
    if decision.abort:
        return observation, decision.abort_reason or "aborted by hook"
    return observation, None


def _call_hook(
    hooks: LoopHooks,
    method: str,
    *args: Any,
) -> HookDecision | None:
    fn = getattr(hooks, method, None)
    if fn is None or not callable(fn):
        return None
    result = fn(*args)
    if result is None:
        return None
    if isinstance(result, HookDecision):
        return result
    return None
