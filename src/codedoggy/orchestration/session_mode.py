"""Session mode state + plan-mode edit gate.

Source (do not invent tool gates):
  - Plan gate: Grok PlanModeTracker / plan_mode_edit_gate
  - Lifecycle: Inactive → Pending → Active → ExitPending (plan_mode.rs)
  - Goal mode: session flag + update_goal tool surface (state only)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codedoggy.orchestration.types import SessionMode
from codedoggy.tools.kinds import ToolKind

logger = logging.getLogger(__name__)

# Grok plan_mode.rs reminder templates (static English, no MiniJinja).
PLAN_REMINDER_FULL = (
    "Plan mode is active. Do not make any edits or writes to the system.\n\n"
    "You should build your plan by writing to or editing the plan file. "
    "Note that this is the only file you are allowed to edit.\n\n"
    "Your turn should only end with either ask_user_question to clarify "
    "requirements or exit_plan_mode to present your plan to the user."
)
PLAN_REMINDER_SPARSE = (
    "Plan mode is still active. Do not make any edits or writes to the system "
    "except for the plan file."
)
PLAN_REMINDER_REENTRY = (
    "## Returning to Plan Mode\n\n"
    "You are entering plan mode again after having previously exited it. "
    "A plan file may already exist from your previous planning session.\n\n"
    "Your turn should only end with either ask_user_question to clarify "
    "requirements or exit_plan_mode to present your plan to the user."
)
PLAN_REMINDER_EXIT = (
    "You have exited plan mode. You can now make edits, run tools, and take actions."
)


@dataclass
class SessionModeState:
    """Mutable mode for one session (Grok plan / goal + PlanModeTracker subset)."""

    mode: SessionMode = SessionMode.NORMAL
    # Absolute or session-relative plan path. Product default is
    # ``{cwd}/.grok/sessions/<id>/plan.md`` (Grok PlanModeTracker session_dir).
    # Tool-only fallback without a session is cwd/.grok/plan.md (Grok resources).
    plan_file: str = ""
    awaiting_plan_approval: bool = False
    # inactive | pending | active | exit_pending
    plan_phase: str = "inactive"
    was_previously_active: bool = False
    reminder_count: int = 0
    pending_exit_reminder: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_plan(self) -> bool:
        """Edit gate / write constraints: Active only (Grok ``is_active()``).

        ExitPending does **not** gate edits — Grok turns the plan edit gate
        off when the user has toggled out and the in-flight turn is finishing.
        """
        return self.plan_phase == "active"

    def is_plan_pending(self) -> bool:
        return self.plan_phase == "pending"

    def is_plan_ui(self) -> bool:
        """Status chrome: any non-inactive plan lifecycle (incl. ExitPending)."""
        return self.plan_phase in {"pending", "active", "exit_pending"}

    def enter_plan(self, plan_file: str | None = None) -> None:
        """Agent tool / hard activate → Active (Grok activate_from_tool)."""
        self.mode = SessionMode.PLAN
        self.plan_phase = "active"
        if plan_file:
            self.plan_file = plan_file
        self.awaiting_plan_approval = False
        self.was_previously_active = True
        self.reminder_count = 0
        self.pending_exit_reminder = False

    def enter_plan_pending(self, plan_file: str | None = None) -> bool:
        """User toggled plan ON while idle (Grok enter_pending).

        Returns True if state changed.
        """
        if plan_file:
            self.plan_file = plan_file
        if self.plan_phase == "exit_pending":
            # Cancel deferred exit — model already has plan context.
            self.plan_phase = "active"
            self.mode = SessionMode.PLAN
            self.pending_exit_reminder = False
            return True
        if self.plan_phase in {"pending", "active"}:
            return False
        self.plan_phase = "pending"
        self.mode = SessionMode.PLAN
        self.pending_exit_reminder = False
        self.awaiting_plan_approval = False
        return True

    def exit_plan(self, *, approved: bool = True, reason: str | None = None) -> None:
        """Leave plan immediately (tool approve/abandon or idle user toggle)."""
        self.mode = SessionMode.NORMAL
        self.plan_phase = "inactive"
        self.awaiting_plan_approval = False
        self.pending_exit_reminder = False
        if reason:
            self.metadata["last_plan_exit"] = reason
        else:
            self.metadata["last_plan_exit"] = "approved" if approved else "cancelled"

    def user_exit(self, *, turn_in_flight: bool) -> None:
        """Client toggled plan OFF (Grok user_exit)."""
        self.awaiting_plan_approval = False
        if self.plan_phase == "pending":
            self.plan_phase = "inactive"
            self.mode = SessionMode.NORMAL
            return
        if self.plan_phase == "active":
            if turn_in_flight:
                self.plan_phase = "exit_pending"
                # Keep mode=PLAN for session chrome; edit gate is off (is_plan=False).
            else:
                self.plan_phase = "inactive"
                self.mode = SessionMode.NORMAL
                self.pending_exit_reminder = True
            return
        if self.plan_phase == "exit_pending":
            return

    def begin_turn(self) -> str | None:
        """Turn start: Pending→Active, drain exit reminder, alternate plan reminders.

        Returns optional ``<system-reminder>`` body (without tags).
        """
        parts: list[str] = []

        if self.pending_exit_reminder:
            parts.append(PLAN_REMINDER_EXIT)
            self.pending_exit_reminder = False

        if self.plan_phase == "pending":
            reentry = self.was_previously_active
            self.plan_phase = "active"
            self.mode = SessionMode.PLAN
            self.was_previously_active = True
            self.reminder_count = 0
            parts.append(PLAN_REMINDER_REENTRY if reentry else PLAN_REMINDER_FULL)
            self.reminder_count = 1
            return "\n\n".join(parts) if parts else None

        if self.plan_phase == "active":
            if self.reminder_count % 2 == 0:
                parts.append(PLAN_REMINDER_FULL)
            else:
                parts.append(PLAN_REMINDER_SPARSE)
            self.reminder_count += 1
            return "\n\n".join(parts) if parts else None

        # exit_pending / inactive: only exit reminder (if any)
        return "\n\n".join(parts) if parts else None

    def end_turn(self) -> None:
        """Turn finished — complete ExitPending (Grok complete_deferred_exit)."""
        if self.plan_phase == "exit_pending":
            self.plan_phase = "inactive"
            self.mode = SessionMode.NORMAL
            self.pending_exit_reminder = True
            self.awaiting_plan_approval = False

    def reset_after_compaction(self) -> None:
        """Grok PlanModeTracker::reset_after_compaction — next reminder full."""
        if self.plan_phase == "active":
            self.reminder_count = 0

    def enter_goal(self) -> None:
        """Session goal-mode flag (state only — no invented tool allowlist)."""
        self.mode = SessionMode.GOAL
        # Goal supersedes plan UI; clear plan lifecycle.
        self.plan_phase = "inactive"
        self.awaiting_plan_approval = False
        self.metadata["goal_active"] = True

    def exit_goal(self, *, reason: str = "exit") -> None:
        self.mode = SessionMode.NORMAL
        self.metadata["last_goal_exit"] = reason
        self.metadata["goal_active"] = False

    def is_goal(self) -> bool:
        return self.mode is SessionMode.GOAL

    def plan_path(self, cwd: Path) -> Path:
        if not (self.plan_file or "").strip():
            # Tool-only fallback when session path not yet bound (Grok resources).
            from codedoggy.tools.grok_build.plan_mode import PLAN_FILE_RELATIVE_PATH

            return (cwd / PLAN_FILE_RELATIVE_PATH).resolve()
        p = Path(self.plan_file)
        if p.is_absolute():
            return p
        return (cwd / p).resolve()

    def snapshot(self) -> dict[str, Any]:
        """Grok PlanModeSnapshot fields (plan_file path not included)."""
        return {
            "state": self.plan_phase,
            "was_previously_active": bool(self.was_previously_active),
            "reminder_count": int(self.reminder_count),
            "pending_exit_reminder": bool(self.pending_exit_reminder),
            "awaiting_plan_approval": bool(self.awaiting_plan_approval),
            "mode": self.mode.value if hasattr(self.mode, "value") else str(self.mode),
            # Display path only; plan body always re-read from disk on re-park.
            "plan_file": self.plan_file,
        }

    @classmethod
    def from_snapshot(
        cls,
        data: dict[str, Any] | None,
        *,
        plan_file: str | None = None,
    ) -> SessionModeState:
        """Restore from plan_mode.json; collapse transient phases (Grok).

        Pending → Inactive (toggle never activated).
        ExitPending → Inactive + pending_exit_reminder (turn gone).
        Active / awaiting_plan_approval preserved.
        """
        state = cls()
        if plan_file:
            state.plan_file = plan_file
        elif data and data.get("plan_file"):
            state.plan_file = str(data["plan_file"])
        if not data:
            return state

        phase = str(data.get("state") or data.get("plan_phase") or "inactive").lower()
        # Accept Grok enum casing
        phase_map = {
            "inactive": "inactive",
            "pending": "pending",
            "active": "active",
            "exitpending": "exit_pending",
            "exit_pending": "exit_pending",
        }
        phase = phase_map.get(phase.replace("-", "_"), "inactive")

        state.was_previously_active = bool(data.get("was_previously_active", False))
        state.reminder_count = int(data.get("reminder_count") or 0)
        state.pending_exit_reminder = bool(data.get("pending_exit_reminder", False))
        state.awaiting_plan_approval = bool(data.get("awaiting_plan_approval", False))

        if phase == "pending":
            phase = "inactive"
            state.awaiting_plan_approval = False
        elif phase == "exit_pending":
            phase = "inactive"
            state.pending_exit_reminder = True
            state.awaiting_plan_approval = False

        state.plan_phase = phase
        if phase == "active":
            state.mode = SessionMode.PLAN
        else:
            # Goal mode is not persisted here; resume as normal unless active plan.
            mode_raw = str(data.get("mode") or "normal").lower()
            if mode_raw == "goal" and phase == "inactive":
                state.mode = SessionMode.GOAL
            else:
                state.mode = SessionMode.NORMAL
        return state


def plan_mode_session_dir(cwd: Path | str, session_id: str) -> Path:
    """``{cwd}/.grok/sessions/<session_id>/`` (Grok-style session dir)."""
    # Sanitize for Windows (``parent:child`` session ids).
    s = str(session_id or "").strip() or "default"
    for ch in (":", "/", "\\", "<", ">", "|", "*", "?", '"'):
        s = s.replace(ch, "__")
    return Path(cwd).resolve() / ".grok" / "sessions" / s


def plan_file_for_session(cwd: Path | str, session_id: str) -> Path:
    """Grok ``PlanModeTracker`` plan body: ``session_dir/plan.md``.

    Source: xai-grok-shell ``plan_mode.rs`` — ``session_dir.join("plan.md")``.
    Lifecycle JSON lives beside it as ``plan_mode.json``.
    """
    return plan_mode_session_dir(cwd, session_id) / "plan.md"


def plan_mode_json_path(cwd: Path | str, session_id: str) -> Path:
    return plan_mode_session_dir(cwd, session_id) / "plan_mode.json"


def save_plan_mode_state(
    state: SessionModeState,
    *,
    cwd: Path | str,
    session_id: str,
) -> Path | None:
    """Write plan_mode.json; best-effort, never raises to callers."""
    try:
        path = plan_mode_json_path(cwd, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = state.snapshot()
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path
    except OSError:
        logger.debug("save_plan_mode_state failed", exc_info=True)
        return None


def load_plan_mode_state(
    *,
    cwd: Path | str,
    session_id: str,
    plan_file: str | None = None,
) -> SessionModeState | None:
    """Load plan_mode.json if present; None if missing/unreadable."""
    path = plan_mode_json_path(cwd, session_id)
    try:
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        return SessionModeState.from_snapshot(raw, plan_file=plan_file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        logger.debug("load_plan_mode_state failed path=%s", path, exc_info=True)
        return None


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

    Rules (Grok plan_mode_edit_gate / tool_calls.rs):
    - Inactive / Pending / ExitPending → allow (gate only when Active)
    - Non-edit tools (bash, read, MCP, spawn, web…) → allow
    - Edit/Write only allowed for the plan file itself
    - apply_patch rejected in plan mode (conservative; no per-file parse)
    """
    # Grok: tracker.is_active() only — ExitPending does not gate.
    if not state.is_plan():
        return PlanEditGate.ALLOW

    from codedoggy.tools.kinds import FILE_MUTATING_KINDS

    write_kinds = set(FILE_MUTATING_KINDS)
    write_names = {
        "search_replace",
        "write",
        "write_file",
        "delete_file",
        "apply_patch",
    }
    wire_name = tool_name or ""
    if ":" in wire_name:
        wire_name = wire_name.split(":", 1)[-1]
    try:
        from codedoggy.tools.grok_surface import CLIENT_ALIASES

        wire_name = CLIENT_ALIASES.get(wire_name, wire_name)
    except Exception:  # noqa: BLE001
        pass
    is_write = (kind in write_kinds) or (wire_name in write_names)
    if not is_write:
        return PlanEditGate.ALLOW

    if wire_name == "apply_patch" or (
        isinstance(tool_name, str) and tool_name.endswith("apply_patch")
    ):
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
    return PlanEditGate.REJECT_NON_PLAN_FILE


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
