"""Plan-mode pure logic (Grok EnterPlanMode / ExitPlanMode).

Ported from grok-build:
  crates/codegen/xai-grok-tools/src/implementations/grok_build/enter_plan_mode/mod.rs
  crates/codegen/xai-grok-tools/src/implementations/grok_build/exit_plan_mode/mod.rs
  crates/codegen/xai-grok-tools/src/types/output.rs
    EnterPlanModeOutput / ExitPlanModeOutput / PlanFileSeed* / to_prompt_format
  crates/codegen/xai-grok-tools/src/types/resources.rs
    PLAN_FILE_RELATIVE_PATH / resolve_plan_file_path / require_plan_file_path

Storage / Resources divergence (honest X/C):
  Grok: PlanFilePath, Cwd, FileSystem, NotificationHandle, TemplateRenderer
        via shared Resources; mode flip is orchestration on PlanModeEntered/Exited.
  CodeDoggy: path from ctx.extra["plan_file_path"] else ctx.cwd / .grok/plan.md;
        FS is pathlib; session mode via host-injected kernel / session_mode_state
        (no invented plan kernel). Tool-name hints from extra["plan_tool_hints"]
        or EnterPlanModeToolHints defaults (no TemplateRenderer).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# resources.rs
PLAN_FILE_RELATIVE_PATH = ".grok/plan.md"

# EnterPlanModeOutput::Entered.message (exact)
ENTERED_MESSAGE = (
    "You have entered plan mode. You should now focus on exploring the codebase "
    "and creating an implementation plan."
)

# ExitPlanModeOutput messages (exact)
PLAN_READY_MESSAGE = "Your plan has been approved. You can now start coding."
EMPTY_PLAN_MESSAGE = (
    "Plan mode exit approved. No plan content was found — you can proceed."
)

# User consent decline (enter_plan_mode mod.rs docs / UI contract)
USER_DECLINED_ENTER = "User declined to enter plan mode."

# require_plan_file_path error
MISSING_PLAN_PATH_RESOURCE = (
    "missing required resource: PlanFilePath or an absolute Cwd"
)


class PlanFileSeedFailure(str, Enum):
    """Why the session plan file is not a ready file (output.rs)."""

    NOT_CREATED = "not_created"
    NOT_A_FILE = "not_a_file"
    INACCESSIBLE = "inaccessible"
    UNAVAILABLE = "unavailable"


class PlanFileSeedStatusKind(str, Enum):
    """Probe / seed outcome discriminator."""

    MISSING = "missing"
    EMPTY = "empty"
    NON_EMPTY = "non_empty"


@dataclass(frozen=True, slots=True)
class PlanFileSeedStatus:
    kind: PlanFileSeedStatusKind
    failure: PlanFileSeedFailure | None = None

    @classmethod
    def empty(cls) -> PlanFileSeedStatus:
        return cls(PlanFileSeedStatusKind.EMPTY)

    @classmethod
    def non_empty(cls) -> PlanFileSeedStatus:
        return cls(PlanFileSeedStatusKind.NON_EMPTY)

    @classmethod
    def missing(cls, reason: PlanFileSeedFailure) -> PlanFileSeedStatus:
        return cls(PlanFileSeedStatusKind.MISSING, reason)


@dataclass(frozen=True, slots=True)
class EnterPlanModeToolHints:
    """Pre-resolved tool name hints (output.rs EnterPlanModeToolHints)."""

    ask_user: str = "ask_user_question"
    exit_plan: str = "exit_plan_mode"
    task: str = ""


def resolve_plan_file_path(
    *,
    cwd: Path | None,
    plan_file_path: str | Path | None = None,
) -> tuple[Path | None, str]:
    """Grok resolve_plan_file_path.

    Returns (absolute_target_or_None, display). absolute_target is Some only when
    the resolved path is absolute so seed/write never hits a relative process-CWD path.
    """
    if plan_file_path is not None:
        path = Path(plan_file_path)
    elif cwd is not None:
        path = Path(cwd) / PLAN_FILE_RELATIVE_PATH
    else:
        path = Path(PLAN_FILE_RELATIVE_PATH)
    display = str(path)
    absolute_target = path if path.is_absolute() else None
    return absolute_target, display


def require_plan_file_path(
    *,
    cwd: Path | None,
    plan_file_path: str | Path | None = None,
) -> tuple[Path, str]:
    """Grok require_plan_file_path — error when no absolute target resolves."""
    target, display = resolve_plan_file_path(cwd=cwd, plan_file_path=plan_file_path)
    if target is None:
        raise ValueError(MISSING_PLAN_PATH_RESOURCE)
    return target, display


def probe_or_create_empty_plan_file(path: Path) -> PlanFileSeedStatus:
    """Probe plan file; create empty only on not-found. Never truncate.

    Port of enter_plan_mode::probe_or_create_empty_plan_file.
    """
    try:
        if path.is_dir():
            return PlanFileSeedStatus.missing(PlanFileSeedFailure.NOT_A_FILE)
        if path.exists():
            try:
                data = path.read_bytes()
            except OSError:
                return PlanFileSeedStatus.missing(PlanFileSeedFailure.INACCESSIBLE)
            if len(data) == 0:
                return PlanFileSeedStatus.empty()
            return PlanFileSeedStatus.non_empty()
        # Not found → create empty (parents as needed; Grok LocalFs creates parents)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
            return PlanFileSeedStatus.empty()
        except OSError:
            return PlanFileSeedStatus.missing(PlanFileSeedFailure.NOT_CREATED)
    except OSError:
        return PlanFileSeedStatus.missing(PlanFileSeedFailure.INACCESSIBLE)


def _plan_status_line(plan_file_path: str, seed: PlanFileSeedStatus) -> str:
    """plan_status branch of EnterPlanMode to_prompt_format."""
    if seed.kind is PlanFileSeedStatusKind.EMPTY:
        return f"Write your plan to {plan_file_path}. The file exists and is empty."
    if seed.kind is PlanFileSeedStatusKind.NON_EMPTY:
        return (
            f"Write your plan to {plan_file_path}. The file exists but is not empty."
        )
    reason = seed.failure or PlanFileSeedFailure.NOT_CREATED
    if reason is PlanFileSeedFailure.NOT_CREATED:
        detail = "The file has not yet been created."
    elif reason is PlanFileSeedFailure.NOT_A_FILE:
        detail = "A directory already exists at that path."
    elif reason is PlanFileSeedFailure.INACCESSIBLE:
        detail = "The file could not be accessed."
    else:  # UNAVAILABLE
        detail = "The plan file location is unavailable."
    return f"Write your plan to {plan_file_path}. {detail}"


def format_enter_plan_prompt(
    *,
    message: str = ENTERED_MESSAGE,
    plan_file_path: str,
    tool_hints: EnterPlanModeToolHints | None = None,
    plan_file_seed: PlanFileSeedStatus,
) -> str:
    """Grok ToolOutput::EnterPlanMode → to_prompt_format."""
    hints = tool_hints or EnterPlanModeToolHints()
    ask = hints.ask_user
    exit_name = hints.exit_plan
    if hints.task:
        task_hint = (
            f'\n     You can use the {hints.task} tool with subagent_type="explore" to '
            "parallelize codebase exploration without filling your context window."
        )
    else:
        task_hint = ""
    plan_status = _plan_status_line(plan_file_path, plan_file_seed)
    return (
        f"{message}\n\n"
        f"{plan_status}\n\n"
        "In plan mode, you should:\n"
        f"1. Thoroughly explore the codebase to understand existing patterns{task_hint}\n"
        "2. Identify similar features, codebase architecture, and understand trade-offs\n"
        f"3. Use {ask} if you need to clarify the approach\n"
        "4. Design a concrete implementation strategy\n"
        "5. Write your plan to the plan file above\n"
        f"6. When ready, use {exit_name} to present your plan to the user."
    )


def read_plan_content(path: Path) -> str | None:
    """Read plan file; None if missing, unreadable, or whitespace-only."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.strip():
        return None
    return text


def format_exit_plan_ready(
    *,
    message: str = PLAN_READY_MESSAGE,
    plan_content: str,
    plan_file_path: str,
) -> str:
    """ExitPlanModeOutput::PlanReady → to_prompt_format."""
    return (
        f"{message}\n\nYour plan has been saved at: {plan_file_path}\n\n"
        f"## Plan:\n{plan_content}"
    )


def format_exit_plan_empty(*, message: str = EMPTY_PLAN_MESSAGE) -> str:
    """ExitPlanModeOutput::EmptyPlan → to_prompt_format (message only)."""
    return message


def tool_hints_from_extra(extra: dict[str, Any] | None) -> EnterPlanModeToolHints:
    """Resolve hints from host bag (TemplateRenderer stand-in)."""
    bag = extra or {}
    raw = bag.get("plan_tool_hints")
    if isinstance(raw, EnterPlanModeToolHints):
        return raw
    if isinstance(raw, dict):
        return EnterPlanModeToolHints(
            ask_user=str(raw.get("ask_user") or "ask_user_question"),
            exit_plan=str(raw.get("exit_plan") or "exit_plan_mode"),
            task=str(raw.get("task") or ""),
        )
    return EnterPlanModeToolHints()


def resolve_configured_plan_path(extra: dict[str, Any] | None) -> str | Path | None:
    """PlanFilePath resource stand-in: extra['plan_file_path']."""
    bag = extra or {}
    configured = bag.get("plan_file_path")
    if configured is not None and str(configured).strip():
        return str(configured)
    return None
