"""update_goal — model-driven goal progress reporting.

Ported from:
  crates/codegen/xai-grok-tools/src/implementations/grok_build/update_goal/mod.rs

Pure logic: ``codedoggy.tools.grok_build.update_goal_logic``
  UpdateGoalInput / UpdateGoalAck / RejectReason
  build_summary, render_ack_into_output, local_ack_for_input

Runtime divergence (documented):
  Grok posts ``UpdateGoalEnvelope`` to ``GoalUpdateHandle`` and blocks on
  ``UpdateGoalAck`` from SessionActor (classifier, 3× blocked streak, drain).
  CodeDoggy applies the no-classifier local ack path (same model-facing
  strings) and session/kernel glue on ``ctx.extra``. Host may inject
  ``extra["goal_ack_fn"](input) -> UpdateGoalAck`` to supply harness acks.
"""

from __future__ import annotations

from typing import Any, Callable

from codedoggy.tools.grok_build.update_goal_logic import (
    BLOCKED_REASON_PARAM_DESC,
    COMPLETED_PARAM_DESC,
    DESCRIPTION_TEMPLATE,
    MESSAGE_PARAM_DESC,
    Accepted,
    ClassifierAchieved,
    ClassifierFailOpenAchieved,
    CompletedWithoutClassifier,
    UpdateGoalAck,
    UpdateGoalInput,
    local_ack_for_input,
    parse_input,
    render_ack_into_output,
)
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolId,
)


class UpdateGoalTool(Tool):
    def id(self) -> ToolId:
        return ToolId("update_goal")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.GoalUpdate

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(
            name="update_goal",
            description=DESCRIPTION_TEMPLATE,
        )

    def parameters_schema(self) -> dict[str, Any]:
        # Schemars descriptions from UpdateGoalInput (mod.rs)
        return {
            "type": "object",
            "properties": {
                "completed": {
                    "type": "boolean",
                    "description": COMPLETED_PARAM_DESC,
                },
                "message": {
                    "type": "string",
                    "description": MESSAGE_PARAM_DESC,
                },
                "blocked_reason": {
                    "type": "string",
                    "description": BLOCKED_REASON_PARAM_DESC,
                },
            },
            "required": [],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        inp = parse_input(args if isinstance(args, dict) else {})
        bag = ctx.extra if ctx.extra is not None else {}

        kernel = bag.get("kernel")
        mode_state = bag.get("session_mode_state")
        if mode_state is None and kernel is not None:
            mode_state = getattr(kernel, "session_mode_state", None)

        # completed=true must consult the same incomplete-work gate as the loop.
        if inp.completed is True:
            from codedoggy.orchestration.incomplete_work import (
                format_incomplete_work_nudge,
                incomplete_work_reasons,
            )

            open_reasons = incomplete_work_reasons(bag)
            if open_reasons:
                summary = format_incomplete_work_nudge(open_reasons)
                refuse = UpdateGoalInput(
                    completed=False,
                    message=summary,
                    blocked_reason=inp.blocked_reason,
                )
                ack = Accepted(summary=summary)
                self._apply_side_effects(
                    refuse, ack, bag=bag, kernel=kernel, mode_state=mode_state
                )
                return summary

        # Optional host harness: (UpdateGoalInput) -> UpdateGoalAck
        host_ack: Callable[[UpdateGoalInput], UpdateGoalAck] | None = bag.get(
            "goal_ack_fn"
        )
        if callable(host_ack):
            ack = host_ack(inp)
        else:
            ack = self._local_ack(inp, bag=bag, kernel=kernel, mode_state=mode_state)

        self._apply_side_effects(inp, ack, bag=bag, kernel=kernel, mode_state=mode_state)
        out = render_ack_into_output(ack)
        return out.summary

    def _local_ack(
        self,
        inp: UpdateGoalInput,
        *,
        bag: dict[str, Any],
        kernel: Any,
        mode_state: Any,
    ) -> UpdateGoalAck:
        """No-classifier host path (Grok drain with classifier disabled)."""
        goal_active = False
        if mode_state is not None:
            is_goal = getattr(mode_state, "is_goal", None)
            if callable(is_goal):
                goal_active = bool(is_goal())
            else:
                goal_active = bool(getattr(mode_state, "mode", None) == "goal")
        if kernel is not None and not goal_active:
            # Kernel flag used by session enter_goal_mode
            if getattr(kernel, "goal_mode", False) or getattr(
                kernel, "goal_active", False
            ):
                goal_active = True

        # Auto-enter goal when host already tracks goal_log / kernel mode
        # (CodeDoggy bootstrap may enter goal without SessionModeState).
        # Grok rejects with HarnessDisabled when harness is off; we only
        # auto-enter when a kernel or mode_state bag is present so bare
        # unit calls still work as progress logging.
        if not goal_active and (kernel is not None or mode_state is not None):
            if mode_state is not None:
                enter = getattr(mode_state, "enter_goal", None)
                if callable(enter):
                    enter()
                    goal_active = True
            if kernel is not None:
                enter_k = getattr(kernel, "enter_goal_mode", None)
                if callable(enter_k):
                    try:
                        enter_k()
                        goal_active = True
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    kernel.goal_active = True
                    goal_active = True
                except Exception:  # noqa: BLE001
                    pass

        streak = 0
        if kernel is not None:
            raw = getattr(kernel, "goal_blocked_streak", 0)
            if isinstance(raw, int):
                streak = raw
        if isinstance(bag.get("goal_blocked_streak"), int):
            streak = int(bag["goal_blocked_streak"])

        # Bare unit calls (no kernel/mode) still accept progress/block/complete
        # with Grok model-facing strings; full HarnessDisabled only when a
        # host injects goal_ack_fn.
        return local_ack_for_input(
            inp,
            blocked_streak_before=streak,
            goal_active=True if (kernel is None and mode_state is None) else goal_active,
            pause_on_block=True,
        )

    def _apply_side_effects(
        self,
        inp: UpdateGoalInput,
        ack: UpdateGoalAck,
        *,
        bag: dict[str, Any],
        kernel: Any,
        mode_state: Any,
    ) -> None:
        """Session/kernel bookkeeping (CodeDoggy storage, not Grok Resources)."""
        log: list[dict[str, Any]] = []
        if kernel is not None:
            existing = getattr(kernel, "goal_log", None)
            if isinstance(existing, list):
                log = existing
            else:
                try:
                    kernel.goal_log = log
                except Exception:  # noqa: BLE001
                    pass
        elif "goal_log" in bag and isinstance(bag["goal_log"], list):
            log = bag["goal_log"]
        else:
            bag["goal_log"] = log

        entry: dict[str, Any] = {}
        if inp.message is not None:
            entry["message"] = inp.message
        if inp.completed is True:
            entry["completed"] = True
        if inp.blocked_reason is not None:
            entry["blocked_reason"] = inp.blocked_reason
        if entry:
            log.append(entry)

        # Blocked streak bookkeeping (shell goal_blocked_streak spirit)
        if inp.blocked_reason is not None and isinstance(ack, Accepted):
            prev = 0
            if kernel is not None:
                raw = getattr(kernel, "goal_blocked_streak", 0)
                if isinstance(raw, int):
                    prev = raw
            if isinstance(bag.get("goal_blocked_streak"), int):
                prev = int(bag["goal_blocked_streak"])
            new_streak = prev + 1
            bag["goal_blocked_streak"] = new_streak
            if kernel is not None:
                try:
                    kernel.goal_blocked_streak = new_streak
                except Exception:  # noqa: BLE001
                    pass

            if new_streak >= 3 and "Goal blocked:" in ack.summary:
                if kernel is not None:
                    try:
                        kernel.goal_blocked = True
                        kernel.goal_blocked_reason = inp.blocked_reason
                    except Exception:  # noqa: BLE001
                        pass
                if mode_state is not None:
                    meta = getattr(mode_state, "metadata", None)
                    if isinstance(meta, dict):
                        meta["goal_blocked"] = True
                        meta["goal_blocked_reason"] = inp.blocked_reason
            return

        if isinstance(
            ack,
            (CompletedWithoutClassifier, ClassifierAchieved, ClassifierFailOpenAchieved),
        ):
            # Reset streak on completed (goal.rs)
            bag["goal_blocked_streak"] = 0
            if kernel is not None:
                try:
                    kernel.goal_blocked_streak = 0
                    kernel.goal_blocked = False
                    kernel.goal_completed = True
                    if inp.message:
                        kernel.goal_completion_message = inp.message
                    exit_km = getattr(kernel, "exit_goal_mode", None)
                    if callable(exit_km):
                        exit_km()
                except Exception:  # noqa: BLE001
                    pass
            if mode_state is not None:
                exit_g = getattr(mode_state, "exit_goal", None)
                if callable(exit_g):
                    try:
                        exit_g(reason="completed")
                    except TypeError:
                        exit_g()
            return

        # Message-only: clear blocked flags (progress continues)
        if kernel is not None:
            try:
                kernel.goal_blocked = False
                kernel.goal_blocked_reason = None
            except Exception:  # noqa: BLE001
                pass
        if mode_state is not None:
            meta = getattr(mode_state, "metadata", None)
            if isinstance(meta, dict):
                meta.pop("goal_blocked", None)
                meta.pop("goal_blocked_reason", None)
