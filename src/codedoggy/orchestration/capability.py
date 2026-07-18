"""Capability-mode → tool-kind filtering (Grok SubagentCapabilityMode)."""

from __future__ import annotations

from codedoggy.orchestration.types import CapabilityMode
from codedoggy.tools.kinds import ToolKind

# Kinds allowed per mode (inclusive ladder), matching Grok tool-class constraints.
_READ_KINDS = frozenset(
    {
        ToolKind.Read,
        ToolKind.ListDir,
        ToolKind.Search,
        ToolKind.Lsp,
        ToolKind.WebSearch,
        ToolKind.WebFetch,
        ToolKind.Other,  # memory / session_search treated as non-mutating
        ToolKind.Todo,
        ToolKind.BackgroundTaskAction,
        ToolKind.WaitTasksAction,
        ToolKind.AskUser,
        ToolKind.EnterPlan,
        ToolKind.ExitPlan,
        ToolKind.MemorySearch,
        ToolKind.MemoryGet,
        ToolKind.SearchTool,
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
