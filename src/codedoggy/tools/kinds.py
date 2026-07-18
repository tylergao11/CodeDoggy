"""Tool namespace and capability kind enums (Grok ToolKind subset + Doggy)."""

from __future__ import annotations

from enum import Enum


class ToolNamespace(str, Enum):
    """Prefix used in qualified tool ids (`Doggy:read_file`)."""

    Doggy = "Doggy"
    MCP = "MCP"

    def __str__(self) -> str:
        return self.value


class ToolKind(str, Enum):
    """High-level capability class for filtering and config.

    Aligns with Grok `ToolKind` for the GrokBuild default surface.
    Unknown wire values should be treated as Other by consumers.
    """

    Read = "read"
    Edit = "edit"
    Delete = "delete"
    ListDir = "list_dir"
    Write = "write"
    Move = "move"
    Search = "search"
    Lsp = "lsp"
    Execute = "execute"
    Plan = "plan"
    WebSearch = "web_search"
    WebFetch = "web_fetch"
    Todo = "todo"
    Task = "task"
    BackgroundTaskAction = "background_task_action"
    WaitTasksAction = "wait_tasks_action"
    KillTaskAction = "kill_task_action"
    EnterPlan = "enter_plan"
    ExitPlan = "exit_plan"
    AskUser = "ask_user"
    ImageGen = "image_gen"
    ImageEdit = "image_edit"
    VideoGen = "video_gen"
    Monitor = "monitor"
    GoalUpdate = "goal_update"
    MemorySearch = "memory_search"
    MemoryGet = "memory_get"
    SearchTool = "search_tool"
    UseTool = "use_tool"
    Skill = "skill"
    Other = "other"

    def is_read_only(self) -> bool:
        """Kind-level default for read-only (tools may override)."""
        return self in {
            ToolKind.Read,
            ToolKind.Search,
            ToolKind.Lsp,
            ToolKind.ListDir,
            ToolKind.MemorySearch,
            ToolKind.MemoryGet,
            ToolKind.WebSearch,
            ToolKind.WebFetch,
            ToolKind.EnterPlan,
            ToolKind.ExitPlan,
            ToolKind.AskUser,
            ToolKind.BackgroundTaskAction,
            ToolKind.WaitTasksAction,
            ToolKind.Todo,
        }


# File-mutation kinds — config cannot downgrade these to Search/Other/etc.
FILE_MUTATING_KINDS = frozenset(
    {
        ToolKind.Edit,
        ToolKind.Write,
        ToolKind.Delete,
        ToolKind.Move,
    }
)

# Registration kinds that always win over config (includes shell Execute).
REGISTRATION_AUTHORITATIVE_KINDS = FILE_MUTATING_KINDS | frozenset({ToolKind.Execute})

# Allowlist: wire/client short-ids that must never be masked as read-only.
HARD_WRITE_TOOL_NAMES = frozenset(
    {
        "search_replace",
        "write",
        "write_file",
        "delete_file",
        "apply_patch",
        "memory",  # curated MEMORY/USER mutations
        "scheduler_create",
        "scheduler_delete",
    }
)

HARD_EXECUTE_TOOL_NAMES = frozenset(
    {
        "run_terminal_cmd",
        "run_terminal_command",  # product client name
        "bash",
        "shell",
    }
)

HARD_MUTATING_TOOL_NAMES = HARD_WRITE_TOOL_NAMES | HARD_EXECUTE_TOOL_NAMES


def is_registration_authoritative_kind(kind: ToolKind | None) -> bool:
    """True when registration kind must win over config (no downgrade)."""
    return kind is not None and kind in REGISTRATION_AUTHORITATIVE_KINDS


def resolve_authoritative_kind(
    *,
    short_id: str,
    registered_kind: ToolKind,
    config_kind: ToolKind | None,
) -> ToolKind:
    """Finalize-time kind: registration wins for mutating tools / hard names.

    Config may set kind for non-mutating tools, but cannot downgrade Write/Edit/
    Delete/Move/Execute (or hard-named write/execute tools) to Search/Other.
    """
    if is_registration_authoritative_kind(registered_kind):
        return registered_kind
    if short_id in HARD_MUTATING_TOOL_NAMES:
        return registered_kind
    return config_kind if config_kind is not None else registered_kind
