"""Session mode state + plan-mode edit gate.

Source (do not invent tool gates):
  - Plan gate: Grok PlanModeTracker spirit / plan_mode_edit_gate
  - Goal mode: session flag + update_goal tool surface (state only)
  - Deleted: invented ``goal_mode_tool_gate`` allowlists (not in Grok source)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codedoggy.orchestration.types import SessionMode
from codedoggy.tools.kinds import ToolKind


@dataclass
class SessionModeState:
    """Mutable mode for one session (Grok plan / goal session flags)."""

    mode: SessionMode = SessionMode.NORMAL
    # Relative to cwd unless absolute.
    plan_file: str = "plan.md"
    awaiting_plan_approval: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_plan(self) -> bool:
        return self.mode is SessionMode.PLAN

    def enter_plan(self, plan_file: str | None = None) -> None:
        self.mode = SessionMode.PLAN
        if plan_file:
            self.plan_file = plan_file
        self.awaiting_plan_approval = False

    def exit_plan(self, *, approved: bool = True) -> None:
        self.mode = SessionMode.NORMAL
        self.awaiting_plan_approval = False
        self.metadata["last_plan_exit"] = "approved" if approved else "cancelled"

    def enter_goal(self) -> None:
        """Session goal-mode flag (state only — no invented tool allowlist)."""
        self.mode = SessionMode.GOAL
        self.awaiting_plan_approval = False
        self.metadata["goal_active"] = True

    def exit_goal(self, *, reason: str = "exit") -> None:
        self.mode = SessionMode.NORMAL
        self.metadata["last_goal_exit"] = reason
        self.metadata["goal_active"] = False

    def is_goal(self) -> bool:
        return self.mode is SessionMode.GOAL

    def plan_path(self, cwd: Path) -> Path:
        p = Path(self.plan_file)
        if p.is_absolute():
            return p
        return (cwd / p).resolve()


class PlanEditGate:
    """Verdict for an edit under plan mode (Grok plan_mode_edit_gate)."""

    ALLOW = "allow"
    REJECT_NON_PLAN_FILE = "reject_non_plan_file"


def plan_mode_edit_gate(
    state: SessionModeState,
    *,
    cwd: Path,
    kind: ToolKind | None,
    tool_name: str,
    args: dict[str, Any],
) -> str:
    """Return PlanEditGate.ALLOW or REJECT_NON_PLAN_FILE.

    Rules (Grok):
    - Inactive plan mode → allow
    - Non-edit tools → allow
    - Edit/Write only allowed for the plan file itself
    - apply_patch rejected in plan mode (conservative)
    """
    if not state.is_plan():
        return PlanEditGate.ALLOW

    write_kinds = {ToolKind.Edit, ToolKind.Write, ToolKind.Delete, ToolKind.Move}
    write_names = {
        "search_replace",
        "write",
        "write_file",
        "delete_file",
        "apply_patch",
    }
    is_write = (kind in write_kinds) or (tool_name in write_names)
    if not is_write:
        return PlanEditGate.ALLOW

    if tool_name == "apply_patch":
        return PlanEditGate.REJECT_NON_PLAN_FILE

    path = _extract_path(args)
    if path is None:
        return PlanEditGate.REJECT_NON_PLAN_FILE

    plan = state.plan_path(cwd)
    try:
        target = Path(path)
        if not target.is_absolute():
            target = (cwd / target).resolve()
        else:
            target = target.resolve()
    except OSError:
        return PlanEditGate.REJECT_NON_PLAN_FILE

    if target == plan:
        return PlanEditGate.ALLOW
    return PlanEditGate.REJECT_NON_PLAN_FILE if target != plan else PlanEditGate.ALLOW


def _extract_path(args: dict[str, Any]) -> str | None:
    for key in ("file_path", "target_file", "path", "destination"):
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


PLAN_REJECT_MESSAGE = (
    "Plan mode is active: only the plan file may be edited "
    "({plan_file}). Other workspace edits are blocked by the plan gate "
    "(not optional — independent of yolo/auto-approve)."
)
