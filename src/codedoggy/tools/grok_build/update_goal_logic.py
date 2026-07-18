"""update_goal pure logic — source port from Grok.

Ported from:
  crates/codegen/xai-grok-tools/src/implementations/grok_build/update_goal/mod.rs

Maps 1:1 where practical:
  UpdateGoalInput (schema descriptions)
  UpdateGoalAck / RejectReason / RejectReason::error_code
  UpdateGoalOutput
  build_summary
  render_ack_into_output

SessionActor drain (classifier, 3× blocked streak, harness channel) lives in
xai-grok-shell and is **not** ported here — host may inject an ack via the
tool shell. Without a harness, the tool shell uses Accepted /
CompletedWithoutClassifier paths with these same model-facing strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from codedoggy.tools.runtime import ToolError

# ── description_template (ToolMetadata) ───────────────────────────────

DESCRIPTION_TEMPLATE = (
    "Report progress on the active goal. Use the parameters to log a status "
    "message, mark the goal completed, or flag that you're blocked."
)

# Schemars descriptions from UpdateGoalInput
COMPLETED_PARAM_DESC = (
    "Set to true ONLY when the goal is fully achieved. This ends goal mode. "
    "Use together with `message` to include a completion summary."
)
MESSAGE_PARAM_DESC = (
    "Optional short message logged as progress (visible in tool response, "
    "not surfaced to the pager dashboard). Use with `completed: true` for a "
    "completion summary."
)
BLOCKED_REASON_PARAM_DESC = (
    "Set only when truly stuck after 3+ consecutive failed attempts at the "
    "same problem. If set, the goal is paused as blocked. This is a FAILURE "
    "signal — never put success text here. For success, use "
    "`completed: true` with `message`."
)

# Harness-disabled reject detail (goal.rs drain when !goal_harness_enabled)
HARNESS_DISABLED_DETAIL = (
    "Goal mode is not active for this session (no /goal run in "
    "progress); update_goal has no effect."
)

# Blocked streak message (goal.rs: streak < 3 before actual pause)
# {streak} is 1-based count after increment.
BLOCKED_STREAK_SUMMARY = (
    "Blocked attempt {streak}/3 recorded. The harness needs "
    "3 consecutive blocked attempts before pausing the goal — "
    "continue retrying or refining the approach."
)


# ── Input ────────────────────────────────────────────────────────────


@dataclass
class UpdateGoalInput:
    """Grok UpdateGoalInput."""

    completed: Optional[bool] = None
    message: Optional[str] = None
    blocked_reason: Optional[str] = None


# ── RejectReason + error_code ─────────────────────────────────────────


class RejectReason(str, Enum):
    """Structured cause for UpdateGoalAck::Rejected."""

    BlockSeenInDrain = "BlockSeenInDrain"
    BlockedAgainstNonActive = "BlockedAgainstNonActive"
    PostCap = "PostCap"
    NonActive = "NonActive"
    HarnessDisabled = "HarnessDisabled"
    PendingQueueEvicted = "PendingQueueEvicted"
    DroppedAfterPauseInDrain = "DroppedAfterPauseInDrain"
    OrchestrationVanished = "OrchestrationVanished"
    StatusChangedDuringClassifier = "StatusChangedDuringClassifier"
    InFlightOrchestrationVanished = "InFlightOrchestrationVanished"

    def error_code(self) -> str:
        """Stable error code surfaced as the tool's ToolError.kind."""
        return {
            RejectReason.BlockSeenInDrain: "goal_update_block_seen",
            RejectReason.BlockedAgainstNonActive: "goal_update_blocked_against_non_active",
            RejectReason.PostCap: "goal_update_post_cap",
            RejectReason.NonActive: "goal_update_non_active",
            RejectReason.HarnessDisabled: "goal_update_harness_disabled",
            RejectReason.PendingQueueEvicted: "goal_update_evicted",
            RejectReason.DroppedAfterPauseInDrain: "goal_update_dropped_after_pause",
            RejectReason.OrchestrationVanished: "goal_update_no_orchestration",
            RejectReason.StatusChangedDuringClassifier: "goal_update_status_changed",
            RejectReason.InFlightOrchestrationVanished: (
                "goal_update_in_flight_orchestration_vanished"
            ),
        }[self]


# ── UpdateGoalAck variants ────────────────────────────────────────────


@dataclass(frozen=True)
class Accepted:
    """message-only or blocked_reason update accepted."""

    summary: str


@dataclass(frozen=True)
class ClassifierAchieved:
    details_path: str


@dataclass(frozen=True)
class ClassifierFailOpenAchieved:
    reason: str


@dataclass(frozen=True)
class ClassifierNotAchieved:
    details_path: str
    attempt: int
    max_runs: int


@dataclass(frozen=True)
class ClassifierCapReached:
    details_path: str
    attempt: int


@dataclass(frozen=True)
class ClassifierStalled:
    details_path: str
    attempt: int


@dataclass(frozen=True)
class ClassifierBlocked:
    details_path: str


@dataclass(frozen=True)
class CompletedWithoutClassifier:
    """Classifier disabled by policy; goal marked complete directly."""


@dataclass(frozen=True)
class ClassifierConcurrentInFlight:
    details_path: str
    attempt: int
    max_runs: int


@dataclass(frozen=True)
class DeferredToTurnEnd:
    pending_depth: int


@dataclass(frozen=True)
class Rejected:
    reason: RejectReason
    detail: str


UpdateGoalAck = Union[
    Accepted,
    ClassifierAchieved,
    ClassifierFailOpenAchieved,
    ClassifierNotAchieved,
    ClassifierCapReached,
    ClassifierStalled,
    ClassifierBlocked,
    CompletedWithoutClassifier,
    ClassifierConcurrentInFlight,
    DeferredToTurnEnd,
    Rejected,
]


@dataclass
class UpdateGoalOutput:
    """Grok UpdateGoalOutput."""

    success: bool
    summary: str


# ── build_summary (mod.rs) ────────────────────────────────────────────


def build_summary(inp: UpdateGoalInput) -> str:
    """Fallback / local summary from input fields (Grok build_summary)."""
    parts: list[str] = []
    if inp.completed is True:
        parts.append("Goal marked complete")
    if inp.blocked_reason is not None:
        parts.append(f"Goal blocked: {inp.blocked_reason}")
    if inp.message is not None:
        parts.append(inp.message)
    if not parts:
        return "Goal updated."
    return ". ".join(parts) + "."


# ── render_ack_into_output (mod.rs) ───────────────────────────────────


def render_ack_into_output(ack: UpdateGoalAck) -> UpdateGoalOutput:
    """Map UpdateGoalAck to model-facing tool result.

    On reject / classifier-not-achieved paths raises ToolError with Grok
    error codes and exact detail strings.
    """
    if isinstance(ack, Accepted):
        return UpdateGoalOutput(success=True, summary=ack.summary)

    if isinstance(ack, CompletedWithoutClassifier):
        return UpdateGoalOutput(success=True, summary="Goal marked complete.")

    if isinstance(ack, ClassifierAchieved):
        return UpdateGoalOutput(
            success=True,
            summary=(
                f"Goal classifier verdict: Achieved. Goal complete. "
                f"See {ack.details_path}"
            ),
        )

    if isinstance(ack, ClassifierFailOpenAchieved):
        return UpdateGoalOutput(
            success=True,
            summary=(
                f"Goal marked complete via fail-open (reason: {ack.reason}). "
                f"No classifier verdict was produced."
            ),
        )

    if isinstance(ack, ClassifierNotAchieved):
        raise ToolError(
            (
                f"Goal classifier rejected this completion attempt "
                f"({ack.attempt}/{ack.max_runs}). "
                f"Review {ack.details_path} and continue working; another "
                f"attempt is available."
            ),
            code="goal_classifier_not_achieved",
        )

    if isinstance(ack, ClassifierCapReached):
        # Empty path ⇒ omit the "See …" pointer entirely.
        pointer = (
            ""
            if not ack.details_path.strip()
            else f" See {ack.details_path}"
        )
        raise ToolError(
            (
                f"Goal classifier rejected completion {ack.attempt} times — "
                f"goal auto-paused.{pointer}"
            ),
            code="goal_classifier_cap_reached",
        )

    if isinstance(ack, ClassifierStalled):
        raise ToolError(
            (
                f"Goal verification saw no change in the flagged gaps across "
                f"{ack.attempt} attempts — goal auto-paused. Review "
                f"{ack.details_path}; the user must resume."
            ),
            code="goal_classifier_stalled",
        )

    if isinstance(ack, ClassifierBlocked):
        raise ToolError(
            (
                "Goal verification found no model-fixable path "
                "(objective/plan contradiction or "
                "evidence that cannot be captured here) — goal paused for "
                "your decision. "
                f"See {ack.details_path}"
            ),
            code="goal_classifier_blocked",
        )

    if isinstance(ack, ClassifierConcurrentInFlight):
        pointer = (
            ""
            if not ack.details_path.strip()
            else f"; see {ack.details_path}"
        )
        raise ToolError(
            (
                "Goal classifier is still verifying a previous completion — "
                "do NOT call update_goal(completed: true) again until you "
                "receive a verdict reminder. "
                f"This attempt was recorded as Not Achieved "
                f"({ack.attempt}/{ack.max_runs}){pointer}"
            ),
            code="goal_classifier_in_flight",
        )

    if isinstance(ack, DeferredToTurnEnd):
        return UpdateGoalOutput(
            success=True,
            summary=(
                "Goal completion queued for classifier verification at end of "
                f"turn (pending_depth={ack.pending_depth}). The verdict will "
                "be delivered as a system reminder before your next reply; "
                "do NOT call update_goal again until you see it."
            ),
        )

    if isinstance(ack, Rejected):
        raise ToolError(ack.detail, code=ack.reason.error_code())

    raise TypeError(f"unknown UpdateGoalAck: {type(ack)!r}")


# ── Local (no-classifier) ack selection ───────────────────────────────
# Mirrors xai-grok-shell goal.rs drain branches when classifier is off:
#   blocked_reason → Accepted("Goal blocked: {reason}.") after streak>=3
#                    Accepted(BLOCKED_STREAK_SUMMARY) for streak 1..2
#   completed true → CompletedWithoutClassifier
#   else           → Accepted("{message}." | "Goal updated.")


def blocked_streak_summary(streak: int) -> str:
    """Model-facing summary for blocked attempts 1 and 2 (goal.rs)."""
    return BLOCKED_STREAK_SUMMARY.format(streak=streak)


def message_only_summary(message: Optional[str]) -> str:
    """Accepted summary for non-completed, non-blocked updates (goal.rs)."""
    if message is not None:
        return f"{message}."
    return "Goal updated."


def blocked_success_summary(reason: str) -> str:
    """Accepted summary after a successful block pause (goal.rs)."""
    return f"Goal blocked: {reason}."


def local_ack_for_input(
    inp: UpdateGoalInput,
    *,
    blocked_streak_before: int = 0,
    goal_active: bool = True,
    pause_on_block: bool = True,
) -> UpdateGoalAck:
    """Select an UpdateGoalAck for CodeDoggy's no-classifier host path.

    Semantics aligned with ``drain_goal_updates`` when classifier policy is
    disabled (and harness is enabled):

    - ``blocked_reason``: streak 1–2 → Accepted(streak text); streak ≥ 3
      and active → Accepted("Goal blocked: …"); non-active → Rejected
    - ``completed is True``: CompletedWithoutClassifier (streak reset is
      caller's job)
    - otherwise: Accepted(message_only_summary)

    When ``goal_active`` is False and input is not a no-op harness case,
    callers that want Grok harness-disabled rejection should pass that
    separately; this helper assumes harness is on when goal_active is True.
    """
    if inp.blocked_reason is not None:
        streak = blocked_streak_before + 1
        if streak < 3:
            return Accepted(summary=blocked_streak_summary(streak))
        if not goal_active or not pause_on_block:
            return Rejected(
                reason=RejectReason.BlockedAgainstNonActive,
                detail="Goal is not Active; blocked_reason ignored",
            )
        return Accepted(summary=blocked_success_summary(inp.blocked_reason))

    if inp.completed is True:
        return CompletedWithoutClassifier()

    return Accepted(summary=message_only_summary(inp.message))


def parse_input(args: dict) -> UpdateGoalInput:
    """Parse tool args into UpdateGoalInput (lenient completed bool)."""
    completed_raw = args.get("completed")
    completed: Optional[bool]
    if completed_raw is None:
        completed = None
    elif isinstance(completed_raw, bool):
        completed = completed_raw
    elif isinstance(completed_raw, (int, float)) and completed_raw in (0, 1):
        completed = bool(completed_raw)
    elif isinstance(completed_raw, str):
        s = completed_raw.strip().lower()
        if s in {"true", "1", "yes"}:
            completed = True
        elif s in {"false", "0", "no"}:
            completed = False
        else:
            raise ToolError.invalid_arguments("completed must be a boolean")
    else:
        raise ToolError.invalid_arguments("completed must be a boolean")

    message = args.get("message")
    if message is not None and not isinstance(message, str):
        raise ToolError.invalid_arguments("message must be a string")

    blocked = args.get("blocked_reason")
    if blocked is not None and not isinstance(blocked, str):
        raise ToolError.invalid_arguments("blocked_reason must be a string")

    return UpdateGoalInput(
        completed=completed,
        message=message,
        blocked_reason=blocked,
    )
