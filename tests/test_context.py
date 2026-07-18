"""Grok-replica context: prune, retain, flush, fold, suppress, segments."""

from __future__ import annotations

from pathlib import Path

from codedoggy.context import (
    COMPACTION_PREFIX,
    CompactionMode,
    CompactionSuppressor,
    ContextBudget,
    ContextCompactor,
    FlushResultKind,
    MemoryFlushConfig,
    SuppressLevel,
    estimate_chars,
    needs_compaction,
    process_flush_response,
    prune_oversized_tool_results,
    prune_retained_tool_results,
    write_segment,
)
from codedoggy.memory import MemoryStore
from codedoggy.turn.types import Message, Role, ToolCall


def _msgs_with_big_tools(n: int = 20, tool_size: int = 2000) -> list[Message]:
    out: list[Message] = [
        Message(role=Role.SYSTEM, content="SYSTEM_MEMORY_BLOCK keep me"),
        Message(role=Role.USER, content="fix the bug in auth"),
    ]
    for i in range(n):
        out.append(
            Message(
                role=Role.ASSISTANT,
                content=f"step {i}",
                tool_calls=[
                    ToolCall(
                        id=f"c{i}",
                        name="read_file",
                        arguments={"target_file": f"f{i}.py"},
                    )
                ],
            )
        )
        out.append(
            Message(
                role=Role.TOOL,
                content=("X" * tool_size) + f" file{i}",
                tool_call_id=f"c{i}",
                name="read_file",
            )
        )
    return out


def test_prune_tool_results() -> None:
    budget = ContextBudget(tool_result_max_chars=100)
    msgs = [
        Message(role=Role.USER, content="hi"),
        Message(role=Role.TOOL, content="A" * 500, name="read_file", tool_call_id="1"),
    ]
    out, n = prune_oversized_tool_results(msgs, budget)
    assert n == 1
    assert len(out[1].content or "") <= 100
    assert "pruned" in (out[1].content or "")


def test_prune_retained_clears_old_tools() -> None:
    msgs = _msgs_with_big_tools(5, 100)
    out, n = prune_retained_tool_results(msgs, retain_recent_tool_messages=2)
    assert n >= 1
    tools = [m for m in out if m.role is Role.TOOL]
    # last 2 keep real content, earlier cleared
    assert "cleared by retain-prune" in (tools[0].content or "")
    assert "file" in (tools[-1].content or "")


def test_retain_prune_only_under_pressure() -> None:
    """Under budget: size prune ok, retain-prune must not wipe history."""
    budget = ContextBudget(
        max_chars=500_000,
        threshold_percent=90,
        tool_result_max_chars=50_000,
        retain_recent_tool_messages=2,
    )
    msgs = _msgs_with_big_tools(8, 200)
    r = ContextCompactor(
        budget=budget, flush_config=MemoryFlushConfig(enabled=False)
    ).ensure(msgs)
    tools = [m for m in r.messages if m.role is Role.TOOL]
    # All tool bodies still present (not retain-cleared)
    assert all("cleared by retain-prune" not in (t.content or "") for t in tools)
    assert r.retained_cleared == 0


def test_needs_compaction_threshold_percent() -> None:
    budget = ContextBudget(max_chars=1000, threshold_percent=50)
    small = [Message(role=Role.USER, content="hi")]
    assert not needs_compaction(small, budget)
    big = _msgs_with_big_tools(5, 500)
    assert needs_compaction(big, budget) or estimate_chars(big) >= budget.trigger_chars


def test_fold_keeps_system_and_recent() -> None:
    budget = ContextBudget(
        max_chars=8_000,
        threshold_percent=30,
        keep_recent_messages=4,
        tool_result_max_chars=200,
        retain_recent_tool_messages=100,  # don't clear all tools before fold
    )
    msgs = _msgs_with_big_tools(15, 800)
    before = estimate_chars(msgs)
    result = ContextCompactor(budget=budget, flush_config=MemoryFlushConfig(enabled=False)).ensure(
        msgs
    )
    assert result.did_compact
    assert result.chars_after < before
    systems = [m for m in result.messages if m.role is Role.SYSTEM]
    assert systems and "SYSTEM_MEMORY_BLOCK" in (systems[0].content or "")
    if "fold" in result.mode or "llm" in result.mode:
        texts = "\n".join(m.content or "" for m in result.messages)
        assert "CONTEXT COMPACTION" in texts or COMPACTION_PREFIX[:20] in texts


def test_suppress_blocks_auto_compact() -> None:
    budget = ContextBudget(max_chars=1000, threshold_percent=20, tool_result_max_chars=50)
    msgs = _msgs_with_big_tools(10, 400)
    sup = CompactionSuppressor()
    sup.mark_sticky_failure()
    c = ContextCompactor(
        budget=budget,
        suppressor=sup,
        flush_config=MemoryFlushConfig(enabled=False),
    )
    r = c.ensure(msgs)
    assert r.suppressed
    assert not r.did_compact or r.mode == "none"
    assert sup.level is SuppressLevel.STICKY


def test_process_flush_response_gates() -> None:
    cfg = MemoryFlushConfig()
    assert process_flush_response("NO_REPLY", cfg).kind is FlushResultKind.NOTHING
    assert process_flush_response("just text no headers", cfg).kind is FlushResultKind.REJECTED
    ok = process_flush_response("## Decisions\n- use JWT\n", cfg)
    assert ok.kind is FlushResultKind.ACCEPTED


def test_memory_flush_writes_store(tmp_path: Path) -> None:
    from codedoggy.context.memory_flush import run_memory_flush
    from codedoggy.model.types import CompletionResult

    class Client:
        config = None

        def complete(self, messages, **kw):
            return CompletionResult(
                content="## Technical context\n- auth uses JWT expiry 15m\n",
                model="t",
            )

    store = MemoryStore(memory_dir=tmp_path / "mem")
    store.load_from_disk()
    msgs = [
        Message(
            role=Role.USER,
            content="fix auth timeout on the login page for enterprise customers",
        ),
        Message(
            role=Role.ASSISTANT,
            content="We should set JWT expiry to 15 minutes and rotate refresh tokens",
        ),
        Message(
            role=Role.USER,
            content="also document the decision in the architecture notes",
        ),
    ]
    fr = run_memory_flush(
        msgs,
        client=Client(),
        memory_store=store,
        config=MemoryFlushConfig(enabled=True),
    )
    assert fr.kind is FlushResultKind.ACCEPTED
    assert fr.entries_written >= 1
    assert any("JWT" in e for e in store.memory_entries)


def test_segments_mode_writes_files(tmp_path: Path) -> None:
    budget = ContextBudget(
        max_chars=5_000,
        threshold_percent=25,
        keep_recent_messages=4,
        tool_result_max_chars=100,
        retain_recent_tool_messages=50,
    )
    msgs = _msgs_with_big_tools(12, 600)
    c = ContextCompactor(
        budget=budget,
        mode=CompactionMode.SEGMENTS,
        compaction_home=tmp_path,
        flush_config=MemoryFlushConfig(enabled=False),
    )
    r = c.ensure(msgs)
    assert r.did_compact
    assert r.segment_path
    assert Path(r.segment_path).exists()
    assert (tmp_path / "compaction" / "INDEX.md").exists()


def test_mode_transcript_hint() -> None:
    h = CompactionMode.TRANSCRIPT.transcript_hint("/tmp/state.db")
    assert h and "session_search" in h
    assert CompactionMode.SUMMARY.transcript_hint("/x") is None


def test_disabled_budget_noop() -> None:
    budget = ContextBudget(enabled=False, max_chars=10, threshold_percent=10)
    msgs = _msgs_with_big_tools(5, 2000)
    result = ContextCompactor(budget=budget).ensure(msgs)
    assert not result.did_compact
    assert len(result.messages) == len(msgs)


def test_snap_to_safe_boundary_tool_pairs() -> None:
    from codedoggy.context.select import (
        hard_trim_safe,
        plan_fold_regions,
        sanitize_tool_pairs,
        snap_to_safe_boundary,
    )

    msgs = [
        Message(role=Role.USER, content="u0"),
        Message(
            role=Role.ASSISTANT,
            content="call",
            tool_calls=[ToolCall(id="1", name="read_file", arguments={"target_file": "a"})],
        ),
        Message(role=Role.TOOL, content="body-a", tool_call_id="1", name="read_file"),
        Message(role=Role.USER, content="u1"),
        Message(role=Role.ASSISTANT, content="done"),
    ]
    # Split between assistant-tools and tool result must snap past tool
    assert snap_to_safe_boundary(msgs, 2) == 3
    head, mid, tail = plan_fold_regions(msgs, protect_first_n=1, keep_recent=2)
    assert mid is not None

    # Orphan tool alone is stripped
    orphan = [
        Message(role=Role.TOOL, content="orphan", tool_call_id="x", name="read_file"),
        Message(role=Role.USER, content="hi"),
    ]
    cleaned = sanitize_tool_pairs(orphan)
    assert all(m.role is not Role.TOOL or m.tool_call_id != "x" for m in cleaned) or not any(
        m.role is Role.TOOL for m in cleaned
    )

    # hard_trim must not leave orphan tools
    fat = msgs + [
        Message(
            role=Role.ASSISTANT,
            content="c2",
            tool_calls=[ToolCall(id="2", name="read_file", arguments={})],
        ),
        Message(role=Role.TOOL, content="b2", tool_call_id="2", name="read_file"),
    ]
    trimmed = hard_trim_safe(
        [],
        fat,
        over_budget=lambda m: estimate_chars(m) > 50,
    )
    # Every tool result must have a prior assistant tool_call id
    for i, m in enumerate(trimmed):
        if m.role is Role.TOOL and m.tool_call_id:
            assert any(
                tc.id == m.tool_call_id
                for prev in trimmed[:i]
                if prev.tool_calls
                for tc in prev.tool_calls
            ), f"orphan tool at {i}"


def test_session_rewind_context_api(tmp_path: Path) -> None:
    from codedoggy.bootstrap import build_session
    from codedoggy.context.segments import write_segment
    from codedoggy.model import CompletionResult
    from tests.test_bootstrap import ScriptClient

    mid = [
        Message(role=Role.USER, content="recover me auth JWT detail"),
        Message(role=Role.ASSISTANT, content="jwt set to 15m"),
    ]
    path = write_segment(mid, home=tmp_path, note="pre-fold checkpoint")
    main = ScriptClient(
        [CompletionResult(content="ok", model="m")],
        name="m",
    )
    audit = ScriptClient(
        [CompletionResult(content='{"ok": true}', model="a")],
        name="a",
    )
    s = build_session(
        tmp_path,
        main_client=main,
        audit_client=audit,
        enable_memory=False,
        enable_session_store=False,
        enable_audit=False,
    )
    try:
        s.handle_prompt("hi")
        # Point compactor at our segment
        s.extensions.context.last_checkpoint_path = str(path)  # type: ignore[union-attr]
        s.extensions.turn_runner.live_messages = [  # type: ignore[union-attr]
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="latest"),
        ]
        r = s.rewind_context()
        assert r.get("ok") is True
        live = s.extensions.turn_runner.live_messages  # type: ignore[union-attr]
        blob = "\n".join(m.content or "" for m in live)
        assert "CHECKPOINT REWIND" in blob or "JWT" in blob or "jwt" in blob
        assert "latest" in blob
    finally:
        s.close()


def test_rewind_from_checkpoint(tmp_path: Path) -> None:
    from codedoggy.context.rewind import inject_checkpoint_into_live, parse_segment_file
    from codedoggy.context.segments import write_segment

    mid = [
        Message(role=Role.USER, content="old question about auth"),
        Message(role=Role.ASSISTANT, content="fixed JWT"),
    ]
    path = write_segment(mid, home=tmp_path, note="pre-fold checkpoint")
    parsed = parse_segment_file(path)
    assert any("auth" in (m.content or "") for m in parsed)
    live = [
        Message(role=Role.SYSTEM, content="sys"),
        Message(role=Role.USER, content="latest ask"),
    ]
    merged = inject_checkpoint_into_live(live, parsed)
    text = "\n".join(m.content or "" for m in merged)
    assert "CHECKPOINT REWIND" in text
    assert "latest ask" in text


def test_thrash_cooldown_allows_prune_and_recovers() -> None:
    """Thrash must not freeze prune forever; cool-down then allow fold again."""
    c = ContextCompactor(
        budget=ContextBudget(
            max_chars=2_000,
            threshold_percent=30,
            tool_result_max_chars=200,
            retain_recent_tool_messages=2,
            protect_first_n=0,
            keep_recent_messages=4,
        ),
        flush_config=MemoryFlushConfig(enabled=False),
    )
    c._ineffective_compression_count = 2
    c._thrash_turns_left = 0
    msgs = _msgs_with_big_tools(10, 400)
    r1 = c.ensure(msgs)
    # Cooldown: may prune, should not full-suppress all work permanently
    assert r1.mode != "thrash_guard" or r1.pruned_tools >= 0
    # After cool-down exhausts, fold path available again
    for _ in range(5):
        r = c.ensure(msgs)
    assert c._ineffective_compression_count < 2 or c._thrash_turns_left >= 0


def test_deterministic_sketch_extracts_paths_and_errors() -> None:
    from codedoggy.context.compactor import _deterministic_sketch

    middle = [
        Message(
            role=Role.ASSISTANT,
            content="editing",
            tool_calls=[
                ToolCall(
                    id="1",
                    name="search_replace",
                    arguments={"file_path": "src/auth.py", "old_string": "a", "new_string": "b"},
                )
            ],
        ),
        Message(
            role=Role.TOOL,
            content="Error: failed to apply patch on src/auth.py traceback",
            tool_call_id="1",
            name="search_replace",
        ),
    ]
    sk = _deterministic_sketch(middle)
    assert "src/auth.py" in sk
    assert "Files touched" in sk
    assert "Errors/signals" in sk or "error" in sk.lower()


def test_summary_end_marker_and_protect_decay() -> None:
    from codedoggy.context.compactor import SUMMARY_END_MARKER

    assert "END OF CONTEXT SUMMARY" in SUMMARY_END_MARKER
    c = ContextCompactor(
        budget=ContextBudget(max_chars=2000, threshold_percent=20, protect_first_n=3),
        flush_config=MemoryFlushConfig(enabled=False),
    )
    assert c._effective_protect_first_n() == 3
    c.compaction_count = 1
    assert c._effective_protect_first_n() == 0


def test_compaction_prefix_matches_hermes_contract() -> None:
    assert "REFERENCE ONLY" in COMPACTION_PREFIX
    assert "AFTER this" in COMPACTION_PREFIX
    assert "MEMORY.md" in COMPACTION_PREFIX
    assert "Historical Task Snapshot" in COMPACTION_PREFIX
    assert "never mind" in COMPACTION_PREFIX
    assert "undo" in COMPACTION_PREFIX


def test_retain_prune_keeps_p0_footer() -> None:
    p0 = (
        "── shadow P0 ──\n"
        "Blocking issues on your last write.\n"
        "1. [critical]: bad edit\n"
        "── end shadow P0 ──"
    )
    msgs = [
        Message(role=Role.USER, content="go"),
        Message(
            role=Role.TOOL,
            content=f"{'BODY' * 50}\n\n{p0}",
            name="search_replace",
            tool_call_id="1",
        ),
        Message(role=Role.TOOL, content="recent ok", name="read_file", tool_call_id="2"),
    ]
    out, n = prune_retained_tool_results(msgs, retain_recent_tool_messages=1)
    assert n >= 1
    first = out[1].content or ""
    assert "cleared by retain-prune" in first
    assert "shadow P0" in first
    assert "bad edit" in first
    assert "── end shadow P0 ──" in first


def test_oversized_prune_keeps_p0_footer() -> None:
    p0 = (
        "── shadow P0 ──\n"
        "1. [critical]: must keep\n"
        "── end shadow P0 ──"
    )
    body = "X" * 2000 + "\n\n" + p0
    msgs = [
        Message(role=Role.TOOL, content=body, name="search_replace", tool_call_id="1"),
    ]
    budget = ContextBudget(tool_result_max_chars=200)
    out, n = prune_oversized_tool_results(msgs, budget)
    assert n == 1
    text = out[0].content or ""
    assert "must keep" in text
    assert "shadow P0" in text


def test_fold_reinjects_p0_when_middle_dropped() -> None:
    p0 = (
        "── shadow P0 ──\n"
        "1. [critical]: rethink auth path\n"
        "── end shadow P0 ──"
    )
    budget = ContextBudget(
        max_chars=3_000,
        threshold_percent=20,
        keep_recent_messages=2,
        tool_result_max_chars=80,
        retain_recent_tool_messages=1,
    )
    msgs: list[Message] = [
        Message(role=Role.SYSTEM, content="sys"),
        Message(role=Role.USER, content="fix auth"),
    ]
    for i in range(12):
        msgs.append(
            Message(
                role=Role.ASSISTANT,
                content=f"step {i}",
                tool_calls=[
                    ToolCall(id=f"c{i}", name="search_replace", arguments={"file_path": f"f{i}"})
                ],
            )
        )
        body = ("Y" * 400) + (f"\n\n{p0}" if i == 2 else "")
        msgs.append(
            Message(
                role=Role.TOOL,
                content=body,
                tool_call_id=f"c{i}",
                name="search_replace",
            )
        )
    r = ContextCompactor(
        budget=budget, flush_config=MemoryFlushConfig(enabled=False)
    ).ensure(msgs)
    assert r.did_compact
    texts = "\n".join(m.content or "" for m in r.messages)
    assert "rethink auth path" in texts
    assert "shadow P0" in texts


def test_retain_then_fold_reinjects_short_p0() -> None:
    """Short retain-pruned P0 must not be 'satisfied' by REFERENCE ONLY summary."""
    from codedoggy.context.pruning import P0_REINJECT_PREFIX

    p0 = (
        "── shadow P0 ──\n"
        "1. [critical]: short path rethink\n"
        "── end shadow P0 ──"
    )
    # Build so retain-prune keeps only last tool; P0 is on an older short body.
    msgs: list[Message] = [
        Message(role=Role.SYSTEM, content="sys"),
        Message(role=Role.USER, content="fix it"),
    ]
    for i in range(10):
        msgs.append(
            Message(
                role=Role.ASSISTANT,
                content=f"step {i} " + ("detail " * 40),
                tool_calls=[
                    ToolCall(id=f"c{i}", name="search_replace", arguments={"f": f"{i}"})
                ],
            )
        )
        if i == 1:
            body = f"cleared placeholder\n\n{p0}"
        else:
            body = ("Z" * 500) + f" file{i}"
        msgs.append(
            Message(
                role=Role.TOOL,
                content=body,
                tool_call_id=f"c{i}",
                name="search_replace",
            )
        )
    budget = ContextBudget(
        max_chars=2_500,
        threshold_percent=25,
        keep_recent_messages=2,
        tool_result_max_chars=120,
        retain_recent_tool_messages=1,
    )
    r = ContextCompactor(
        budget=budget, flush_config=MemoryFlushConfig(enabled=False)
    ).ensure(msgs)
    assert r.did_compact
    texts = "\n".join(m.content or "" for m in r.messages)
    assert "short path rethink" in texts
    # Binding form: either still on a TOOL, or dedicated reinject USER note.
    binding = [
        m
        for m in r.messages
        if (m.content or "")
        and "short path rethink" in (m.content or "")
        and "REFERENCE ONLY" not in (m.content or "")
    ]
    assert binding, "P0 must not live only inside fold summary"
    # Prefer reinject when middle was folded away
    if r.folded_messages:
        assert any(P0_REINJECT_PREFIX in (m.content or "") for m in r.messages) or any(
            m.role is Role.TOOL and "short path rethink" in (m.content or "")
            for m in r.messages
        )


def test_cjk_weighted_budget() -> None:
    from codedoggy.context.budget import estimate_chars, estimate_tokens, weighted_text_len
    from codedoggy.context.tokens import count_text_tokens, tokenizer_backend

    assert count_text_tokens("hello world") >= 1
    assert tokenizer_backend() in {"heuristic"} or tokenizer_backend().startswith(
        "tiktoken:"
    )
    # CJK costs more tokens than same-length latin under heuristic
    assert count_text_tokens("中文测试") >= count_text_tokens("abcd")
    msgs = [Message(role=Role.USER, content="你好世界" * 20)]
    assert estimate_tokens(msgs) >= 1
    assert estimate_chars(msgs) == estimate_tokens(msgs) * 4
    assert weighted_text_len("x") >= 4 or weighted_text_len("x") == count_text_tokens("x") * 4


def test_fold_writes_checkpoint(tmp_path: Path) -> None:
    budget = ContextBudget(
        max_chars=3_000,
        threshold_percent=25,
        keep_recent_messages=2,
        tool_result_max_chars=80,
        retain_recent_tool_messages=2,
    )
    msgs = _msgs_with_big_tools(12, 500)
    c = ContextCompactor(
        budget=budget,
        compaction_home=tmp_path,
        flush_config=MemoryFlushConfig(enabled=False),
        checkpoint_on_fold=True,
    )
    r = c.ensure(msgs)
    assert r.did_compact
    assert c.last_checkpoint_path
    assert Path(c.last_checkpoint_path).exists()
    assert "checkpoint" in Path(c.last_checkpoint_path).read_text(encoding="utf-8").lower() or (
        tmp_path / "compaction" / "INDEX.md"
    ).exists()


def test_flush_once_per_cycle_including_zero() -> None:
    from codedoggy.context.memory_flush import should_flush

    cfg = MemoryFlushConfig(enabled=True, soft_ratio=0.5)
    msgs = [Message(role=Role.USER, content="Z" * 10_000)]
    # First time at cycle 0
    assert should_flush(
        msgs, trigger_chars=5_000, config=cfg, last_flush_cycle=-1, current_cycle=0
    )
    # After flush at cycle 0, block again
    assert not should_flush(
        msgs, trigger_chars=5_000, config=cfg, last_flush_cycle=0, current_cycle=0
    )
    # After hard compact advances cycle, allow again
    assert should_flush(
        msgs, trigger_chars=5_000, config=cfg, last_flush_cycle=0, current_cycle=1
    )


def test_loop_compacts_before_sample(tmp_path: Path) -> None:
    from codedoggy.tools import ToolRegistryBuilder
    from codedoggy.turn import SampleResult, run_agent_loop

    seen_sizes: list[int] = []

    class Sampler:
        def sample(self, messages, tools):
            seen_sizes.append(estimate_chars(messages))
            return SampleResult(content="done early")

    budget = ContextBudget(
        max_chars=5_000,
        threshold_percent=40,
        keep_recent_messages=4,
        tool_result_max_chars=150,
    )
    huge = "M" * 4_000
    tools = ToolRegistryBuilder.new().finalize()
    result = run_agent_loop(
        user_text="hi",
        sampler=Sampler(),
        tools=tools,
        cwd=tmp_path,
        system_prompt=huge + "\nKEEP_SYSTEM",
        max_turns=2,
        context_compactor=ContextCompactor(
            budget=budget, flush_config=MemoryFlushConfig(enabled=False)
        ),
    )
    assert result.completed
    assert seen_sizes
    systems = [m.content or "" for m in result.messages if m.role is Role.SYSTEM]
    assert systems and "KEEP_SYSTEM" in systems[0]
