"""Focused tests for Grok update_goal pure logic + tool wire.

Mirrors crates/codegen/xai-grok-tools/src/implementations/grok_build/update_goal/mod.rs
tests (build_summary_*) and render_ack / local complete-blocked semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.orchestration.session_mode import SessionModeState
from codedoggy.tools.builtins.update_goal import UpdateGoalTool
from codedoggy.tools.grok_build.update_goal_logic import (
    BLOCKED_REASON_PARAM_DESC,
    COMPLETED_PARAM_DESC,
    DESCRIPTION_TEMPLATE,
    MESSAGE_PARAM_DESC,
    Accepted,
    ClassifierAchieved,
    ClassifierBlocked,
    ClassifierCapReached,
    ClassifierConcurrentInFlight,
    ClassifierFailOpenAchieved,
    ClassifierNotAchieved,
    ClassifierStalled,
    CompletedWithoutClassifier,
    DeferredToTurnEnd,
    RejectReason,
    Rejected,
    UpdateGoalInput,
    blocked_success_summary,
    blocked_streak_summary,
    build_summary,
    local_ack_for_input,
    message_only_summary,
    parse_input,
    render_ack_into_output,
)
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import ToolCallContext, ToolError


def _empty() -> UpdateGoalInput:
    return UpdateGoalInput()


# ── build_summary (mod.rs tests) ──────────────────────────────────────


def test_build_summary_empty_input() -> None:
    assert build_summary(_empty()) == "Goal updated."


def test_build_summary_completed() -> None:
    inp = UpdateGoalInput(completed=True)
    assert build_summary(inp) == "Goal marked complete."


def test_build_summary_message_only() -> None:
    inp = UpdateGoalInput(message="Working on it")
    assert build_summary(inp) == "Working on it."


def test_build_summary_blocked_reason_only() -> None:
    inp = UpdateGoalInput(blocked_reason="no windows sdk")
    assert build_summary(inp) == "Goal blocked: no windows sdk."


def test_build_summary_blocked_reason_with_message() -> None:
    inp = UpdateGoalInput(blocked_reason="X", message="longer body")
    summary = build_summary(inp)
    assert "Goal blocked: X" in summary
    assert "longer body" in summary


def test_build_summary_completed_with_message() -> None:
    inp = UpdateGoalInput(completed=True, message="All done")
    summary = build_summary(inp)
    assert "Goal marked complete" in summary
    assert "All done" in summary


def test_build_summary_completed_false_treated_as_noop() -> None:
    inp = UpdateGoalInput(completed=False)
    assert build_summary(inp) == "Goal updated."


# ── render_ack_into_output ────────────────────────────────────────────


def test_render_accepted() -> None:
    out = render_ack_into_output(Accepted(summary="Working on it."))
    assert out.success is True
    assert out.summary == "Working on it."


def test_render_completed_without_classifier() -> None:
    out = render_ack_into_output(CompletedWithoutClassifier())
    assert out.success is True
    assert out.summary == "Goal marked complete."


def test_render_classifier_achieved() -> None:
    out = render_ack_into_output(ClassifierAchieved(details_path="/tmp/v.md"))
    assert out.success is True
    assert out.summary == (
        "Goal classifier verdict: Achieved. Goal complete. See /tmp/v.md"
    )


def test_render_classifier_fail_open() -> None:
    out = render_ack_into_output(ClassifierFailOpenAchieved(reason="timeout"))
    assert "fail-open" in out.summary
    assert "timeout" in out.summary


def test_render_classifier_not_achieved() -> None:
    with pytest.raises(ToolError) as ei:
        render_ack_into_output(
            ClassifierNotAchieved(details_path="d.md", attempt=1, max_runs=3)
        )
    assert ei.value.code == "goal_classifier_not_achieved"
    assert "1/3" in ei.value.message
    assert "d.md" in ei.value.message


def test_render_classifier_cap_reached_with_path() -> None:
    with pytest.raises(ToolError) as ei:
        render_ack_into_output(ClassifierCapReached(details_path="cap.md", attempt=3))
    assert ei.value.code == "goal_classifier_cap_reached"
    assert "3 times" in ei.value.message
    assert "See cap.md" in ei.value.message


def test_render_classifier_cap_reached_empty_path_omits_pointer() -> None:
    with pytest.raises(ToolError) as ei:
        render_ack_into_output(ClassifierCapReached(details_path="  ", attempt=3))
    assert "See " not in ei.value.message
    assert "auto-paused." in ei.value.message


def test_render_classifier_stalled() -> None:
    with pytest.raises(ToolError) as ei:
        render_ack_into_output(ClassifierStalled(details_path="s.md", attempt=2))
    assert ei.value.code == "goal_classifier_stalled"
    assert "s.md" in ei.value.message


def test_render_classifier_blocked() -> None:
    with pytest.raises(ToolError) as ei:
        render_ack_into_output(ClassifierBlocked(details_path="b.md"))
    assert ei.value.code == "goal_classifier_blocked"
    assert "no model-fixable path" in ei.value.message
    assert "b.md" in ei.value.message


def test_render_concurrent_in_flight_empty_path() -> None:
    with pytest.raises(ToolError) as ei:
        render_ack_into_output(
            ClassifierConcurrentInFlight(details_path="", attempt=2, max_runs=5)
        )
    assert ei.value.code == "goal_classifier_in_flight"
    assert "do NOT call" in ei.value.message
    assert "; see " not in ei.value.message


def test_render_deferred_to_turn_end() -> None:
    out = render_ack_into_output(DeferredToTurnEnd(pending_depth=1))
    assert out.success is True
    assert "pending_depth=1" in out.summary
    assert "do NOT call update_goal" in out.summary


def test_render_rejected_uses_error_code() -> None:
    with pytest.raises(ToolError) as ei:
        render_ack_into_output(
            Rejected(
                reason=RejectReason.HarnessDisabled,
                detail="Goal mode is not active for this session (no /goal run in "
                "progress); update_goal has no effect.",
            )
        )
    assert ei.value.code == "goal_update_harness_disabled"
    assert "no /goal run" in ei.value.message


def test_reject_reason_error_codes() -> None:
    assert RejectReason.BlockSeenInDrain.error_code() == "goal_update_block_seen"
    assert (
        RejectReason.BlockedAgainstNonActive.error_code()
        == "goal_update_blocked_against_non_active"
    )
    assert RejectReason.PostCap.error_code() == "goal_update_post_cap"
    assert RejectReason.NonActive.error_code() == "goal_update_non_active"
    assert RejectReason.HarnessDisabled.error_code() == "goal_update_harness_disabled"
    assert RejectReason.PendingQueueEvicted.error_code() == "goal_update_evicted"
    assert (
        RejectReason.DroppedAfterPauseInDrain.error_code()
        == "goal_update_dropped_after_pause"
    )
    assert (
        RejectReason.OrchestrationVanished.error_code()
        == "goal_update_no_orchestration"
    )
    assert (
        RejectReason.StatusChangedDuringClassifier.error_code()
        == "goal_update_status_changed"
    )
    assert (
        RejectReason.InFlightOrchestrationVanished.error_code()
        == "goal_update_in_flight_orchestration_vanished"
    )


# ── local_ack complete / blocked semantics ────────────────────────────


def test_local_ack_message_only() -> None:
    ack = local_ack_for_input(UpdateGoalInput(message="halfway"))
    assert isinstance(ack, Accepted)
    assert ack.summary == "halfway."


def test_local_ack_empty_message() -> None:
    ack = local_ack_for_input(_empty())
    assert isinstance(ack, Accepted)
    assert ack.summary == "Goal updated."


def test_local_ack_completed() -> None:
    ack = local_ack_for_input(UpdateGoalInput(completed=True, message="done"))
    assert isinstance(ack, CompletedWithoutClassifier)
    out = render_ack_into_output(ack)
    assert out.summary == "Goal marked complete."


def test_local_ack_blocked_streak_1_and_2() -> None:
    a1 = local_ack_for_input(
        UpdateGoalInput(blocked_reason="sdk"),
        blocked_streak_before=0,
    )
    assert isinstance(a1, Accepted)
    assert a1.summary == blocked_streak_summary(1)
    assert "1/3" in a1.summary

    a2 = local_ack_for_input(
        UpdateGoalInput(blocked_reason="sdk"),
        blocked_streak_before=1,
    )
    assert isinstance(a2, Accepted)
    assert "2/3" in a2.summary


def test_local_ack_blocked_streak_3_pauses() -> None:
    ack = local_ack_for_input(
        UpdateGoalInput(blocked_reason="no windows sdk"),
        blocked_streak_before=2,
        goal_active=True,
    )
    assert isinstance(ack, Accepted)
    assert ack.summary == blocked_success_summary("no windows sdk")
    assert ack.summary == "Goal blocked: no windows sdk."


def test_local_ack_blocked_against_non_active() -> None:
    ack = local_ack_for_input(
        UpdateGoalInput(blocked_reason="X"),
        blocked_streak_before=2,
        goal_active=False,
    )
    assert isinstance(ack, Rejected)
    assert ack.reason is RejectReason.BlockedAgainstNonActive


def test_message_only_summary_helpers() -> None:
    assert message_only_summary(None) == "Goal updated."
    assert message_only_summary("hi") == "hi."


# ── schema / tool metadata ────────────────────────────────────────────


def test_tool_metadata_exact_grok_strings() -> None:
    tool = UpdateGoalTool()
    assert tool.id() == "update_goal"
    assert tool.kind() is ToolKind.GoalUpdate
    assert tool.tool_namespace() is ToolNamespace.Doggy
    assert tool.description(None).description == DESCRIPTION_TEMPLATE
    schema = tool.parameters_schema()
    props = schema["properties"]
    assert props["completed"]["description"] == COMPLETED_PARAM_DESC
    assert props["message"]["description"] == MESSAGE_PARAM_DESC
    assert props["blocked_reason"]["description"] == BLOCKED_REASON_PARAM_DESC
    assert schema["required"] == []


def test_parse_input_lenient_completed() -> None:
    assert parse_input({"completed": "true"}).completed is True
    assert parse_input({"completed": 0}).completed is False
    with pytest.raises(ToolError):
        parse_input({"completed": "maybe"})


# ── tool wire ─────────────────────────────────────────────────────────


def test_tool_message_progress(tmp_path: Path) -> None:
    tool = UpdateGoalTool()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    out = tool.run(ctx, {"message": "halfway"})
    assert out == "halfway."


def test_tool_completed_exits_goal_mode(tmp_path: Path) -> None:
    state = SessionModeState()
    state.enter_goal()
    tool = UpdateGoalTool()
    ctx = ToolCallContext(cwd=tmp_path, extra={"session_mode_state": state})
    out = tool.run(ctx, {"completed": True, "message": "done"})
    assert out == "Goal marked complete."
    assert not state.is_goal()


def test_tool_blocked_streak_then_pause(tmp_path: Path) -> None:
    state = SessionModeState()
    state.enter_goal()
    tool = UpdateGoalTool()
    bag: dict = {"session_mode_state": state, "goal_blocked_streak": 0}
    ctx = ToolCallContext(cwd=tmp_path, extra=bag)

    out1 = tool.run(ctx, {"blocked_reason": "missing sdk"})
    assert "1/3" in out1
    assert bag["goal_blocked_streak"] == 1

    out2 = tool.run(ctx, {"blocked_reason": "missing sdk"})
    assert "2/3" in out2
    assert bag["goal_blocked_streak"] == 2

    out3 = tool.run(ctx, {"blocked_reason": "missing sdk"})
    assert out3 == "Goal blocked: missing sdk."
    assert bag["goal_blocked_streak"] == 3
    assert state.metadata.get("goal_blocked") is True


def test_tool_host_ack_fn(tmp_path: Path) -> None:
    tool = UpdateGoalTool()

    def ack_fn(inp: UpdateGoalInput):
        return DeferredToTurnEnd(pending_depth=2)

    ctx = ToolCallContext(cwd=tmp_path, extra={"goal_ack_fn": ack_fn})
    out = tool.run(ctx, {"completed": True})
    assert "pending_depth=2" in out


def test_tool_host_ack_rejected(tmp_path: Path) -> None:
    tool = UpdateGoalTool()

    def ack_fn(inp: UpdateGoalInput):
        return Rejected(
            reason=RejectReason.HarnessDisabled,
            detail="Goal mode is not active for this session (no /goal run in "
            "progress); update_goal has no effect.",
        )

    ctx = ToolCallContext(cwd=tmp_path, extra={"goal_ack_fn": ack_fn})
    with pytest.raises(ToolError) as ei:
        tool.run(ctx, {"message": "x"})
    assert ei.value.code == "goal_update_harness_disabled"
