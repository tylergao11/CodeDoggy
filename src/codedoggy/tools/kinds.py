"""Tool namespace and capability kind enums."""

from __future__ import annotations

from enum import Enum


class ToolNamespace(str, Enum):
    """Prefix used in qualified tool ids (`Doggy:read_file`)."""

    Doggy = "Doggy"
    MCP = "MCP"

    def __str__(self) -> str:
        return self.value


class ToolKind(str, Enum):
    """High-level capability class for filtering and config."""

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
    Other = "other"
