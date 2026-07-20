"""Capability-mode → tool-kind filtering (Grok SubagentCapabilityMode)."""

from __future__ import annotations

from codedoggy.orchestration.types import CapabilityMode
from codedoggy.tools.kinds import (
    FILE_MUTATING_KINDS,
    HARD_MUTATING_TOOL_NAMES,
    ToolKind,
)

# Kinds allowed per mode (inclusive ladder), matching Grok tool-class constraints.
# ToolKind.Other is not on any restricted ladder (fail closed).
_READ_KINDS = frozenset(
    {
        ToolKind.Read,
        ToolKind.ListDir,
        ToolKind.Search,
        ToolKind.Lsp,
        ToolKind.WebSearch,
        ToolKind.WebFetch,
        ToolKind.Todo,
        ToolKind.BackgroundTaskAction,
        ToolKind.WaitTasksAction,
        ToolKind.AskUser,
        ToolKind.EnterPlan,
        ToolKind.ExitPlan,
        ToolKind.MemorySearch,
        ToolKind.MemoryGet,
        ToolKind.SearchTool,
        ToolKind.Skill,
    }
)
_WRITE_KINDS = frozenset(
    {
        ToolKind.Edit,
        ToolKind.Write,
        ToolKind.Delete,
        ToolKind.Move,
        ToolKind.GoalUpdate,
    }
)
_EXECUTE_KINDS = frozenset(
    {
        ToolKind.Execute,
        ToolKind.KillTaskAction,
        ToolKind.Monitor,
    }
)
_TASK_KINDS = frozenset({ToolKind.Task, ToolKind.Plan})


def kinds_for_capability(mode: CapabilityMode) -> frozenset[ToolKind] | None:
    """Return allowed kinds, or ``None`` meaning *all* kinds (mode ALL)."""
    if mode is CapabilityMode.ALL:
        return None
    if mode is CapabilityMode.READ_ONLY:
        return _READ_KINDS
    if mode is CapabilityMode.READ_WRITE:
        return _READ_KINDS | _WRITE_KINDS
    if mode is CapabilityMode.EXECUTE:
        return _READ_KINDS | _WRITE_KINDS | _EXECUTE_KINDS
    return None


def kind_allowed(mode: CapabilityMode, kind: ToolKind | None) -> bool:
    allowed = kinds_for_capability(mode)
    if allowed is None:
        return True
    if kind is None:
        return False
    return kind in allowed


def is_mutating_kind(kind: ToolKind | None) -> bool:
    return kind is not None and kind in _WRITE_KINDS


def is_read_only_kind(kind: ToolKind | None) -> bool:
    if kind is None:
        return False
    return kind in _READ_KINDS


def _wire_tool_name(tool_name: str | None) -> str:
    name = (tool_name or "").strip()
    if ":" in name:
        name = name.split(":", 1)[-1]
    if not name:
        return ""
    try:
        from codedoggy.tools.grok_surface import CLIENT_ALIASES

        return CLIENT_ALIASES.get(name, name)
    except Exception:  # noqa: BLE001
        return name


# Kinds that always mutate / side-effect outside pure research.
_MUTATING_KINDS = frozenset(
    FILE_MUTATING_KINDS
    | {
        ToolKind.Execute,
        ToolKind.Task,
        ToolKind.UseTool,
        ToolKind.GoalUpdate,
        ToolKind.Monitor,
        ToolKind.KillTaskAction,
        ToolKind.ImageGen,
        ToolKind.ImageEdit,
        ToolKind.VideoGen,
    }
)


def is_mutating_action(
    kind: ToolKind | None,
    tool_name: str | None = None,
) -> bool:
    """Single truth for writes_paused / plan-mode / pause gates.

    Kind ladder + HARD_MUTATING_TOOL_NAMES (via CLIENT_ALIASES → wire id).
    """
    if kind in _MUTATING_KINDS:
        return True
    wire = _wire_tool_name(tool_name)
    if wire and wire in HARD_MUTATING_TOOL_NAMES:
        return True
    # scheduler_* are ToolKind.Other but listed in HARD_WRITE_TOOL_NAMES
    if wire in {"scheduler_create", "scheduler_delete", "parallel_tasks"}:
        return True
    return False
