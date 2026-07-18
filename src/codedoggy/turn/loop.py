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
from codedoggy.turn.executor import execute_tool_call, parse_tool_arguments
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
        ``on_turn_end`` (flush deferred non-P0 audit notes).
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

    from codedoggy.context.live_history import copy_message, seed_messages

    messages = seed_messages(
        system_prompt=system_prompt,
        user_text=user_text,
        prior_messages=prior_messages,
    )
    # Archive only *this prompt's* user line (priors already archived earlier).
    if on_archive_message is not None:
        try:
            on_archive_message(copy_message(messages[-1]))
        except Exception:  # noqa: BLE001
            logger.exception("on_archive_message failed for user prompt")

    tools_called: list[str] = []
    rounds = 0
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
    ) -> LoopResult:
        meta = dict(metadata or {})
        if compact_meta.get("compactions"):
            meta["context_compactions"] = compact_meta["compactions"]
            meta["context_last"] = compact_meta.get("last")
        body = text if text is not None else final_text
        deferred = _call_on_turn_end(hook_impl, _hctx(rounds))
        if deferred:
            # Product name: shadow (影子). Keep audit_deferred key for compat.
            meta["shadow_deferred"] = deferred
            meta["audit_deferred"] = deferred
            body = f"{body}\n\n{deferred}" if body else deferred
            # Close the non-P0 loop: land in live transcript so SessionStore
            # FTS and any consumer of loop.messages can recall the notes.
            note_msg = Message(
                role=Role.USER,
                content=(
                    "[shadow — end-of-turn notes]\n"
                    + deferred.lstrip()
                ),
            )
            messages.append(note_msg)
            _archive(on_archive_message, note_msg)
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
            metadata=meta,
        )

    while True:
        if is_cancelled is not None and is_cancelled():
            return _finish(completed=False, cancelled=True)

        if max_turns is not None and rounds >= max_turns:
            partial = final_text or _last_assistant_text(messages)
            return _finish(
                completed=False,
                max_turns_reached=True,
                text=partial,
                metadata={"hint": f"stopped after {max_turns} sampling round(s)"},
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

        tool_specs = tools.tool_definitions()
        try:
            sample = sampler.sample(messages, tool_specs)
        except Exception as e:
            logger.exception("sampler failed")
            return _finish(completed=False, error=f"sampler error: {e}")

        sample = _normalize_sample(sample)
        rounds += 1
        _note_sample_usage(context_compactor, sample)
        # Hermes ContextEngine.update_from_response
        if context_compactor is not None:
            upd = getattr(context_compactor, "update_from_response", None)
            if callable(upd) and isinstance(sample.raw, dict):
                usage = sample.raw.get("usage")
                if isinstance(usage, dict):
                    upd(usage)

        messages.append(
            Message(
                role=Role.ASSISTANT,
                content=sample.content,
                tool_calls=list(sample.tool_calls) if sample.tool_calls else None,
            )
        )

        hctx = _hctx(rounds)
        decision = _call_hook(hook_impl, "after_sample", sample, hctx)
        if decision and decision.append_observation and sample.content:
            messages[-1] = Message(
                role=Role.ASSISTANT,
                content=f"{sample.content}\n{decision.append_observation}",
                tool_calls=messages[-1].tool_calls,
            )
        elif decision and decision.append_observation and not sample.content:
            messages[-1] = Message(
                role=Role.ASSISTANT,
                content=decision.append_observation,
                tool_calls=messages[-1].tool_calls,
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

        for call in sample.tool_calls:
            if is_cancelled is not None and is_cancelled():
                return _finish(completed=False, cancelled=True, text=sample.content)

            record = execute_tool_call(
                tools,
                call,
                cwd=cwd_path,
                session_id=session_id,
                extra=extra,
            )
            tools_called.append(record.call.name)
            observation = record.content

            decision = _call_hook(hook_impl, "after_tool", record, hctx)
            observation, abort = _apply_decision(observation, decision)
            if abort is not None:
                _append_tool_message(messages, record, observation, on_archive_message)
                return _finish(
                    completed=False,
                    aborted=True,
                    text=sample.content,
                    error=abort,
                )

            if record.ok and record.mutation is not None:
                mut_decision = _call_hook(hook_impl, "after_mutation", record, hctx)
                observation, abort = _apply_decision(observation, mut_decision)
                if abort is not None:
                    _append_tool_message(messages, record, observation, on_archive_message)
                    return _finish(
                        completed=False,
                        aborted=True,
                        text=sample.content,
                        error=abort,
                        metadata={"mutation_path": record.mutation.path},
                    )

            _append_tool_message(messages, record, observation, on_archive_message)

        # After tools: window prune/fold. Prefire was started pre-tools and may
        # already be done (try_join); only block-join under hard pressure.
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
    return SampleResult(content=sample.content, tool_calls=calls, raw=sample.raw)


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


def _note_sample_usage(compactor: ContextCompactor | None, sample: SampleResult) -> None:
    """Feed model-reported prompt_tokens into the budget when present."""
    if compactor is None:
        return
    budget = getattr(compactor, "budget", None)
    if budget is None:
        return
    raw = sample.raw or {}
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if not isinstance(usage, dict):
        return
    for key in ("prompt_tokens", "input_tokens", "prompt_token_count"):
        val = usage.get(key)
        if val is not None:
            try:
                budget.last_prompt_tokens = int(val)
            except (TypeError, ValueError):
                pass
            return


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
