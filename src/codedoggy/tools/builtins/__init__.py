"""Default builtin tools registered by ToolRegistryBuilder.new()."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codedoggy.tools.registry import ToolRegistryBuilder


def register_builtins(builder: ToolRegistryBuilder) -> None:
    from codedoggy.tools.builtins.code_nav import CodeNavTool
    from codedoggy.tools.builtins.grep import GrepTool
    from codedoggy.tools.builtins.list_dir import ListDirTool
    from codedoggy.tools.builtins.memory import MemoryTool
    from codedoggy.tools.builtins.read_file import ReadFileTool
    from codedoggy.tools.builtins.run_terminal_cmd import RunTerminalCmdTool
    from codedoggy.tools.builtins.search_replace import SearchReplaceTool
    from codedoggy.tools.builtins.session_search import SessionSearchTool

    builder.register(ReadFileTool())
    builder.register(SearchReplaceTool())
    builder.register(ListDirTool())
    builder.register(GrepTool())
    builder.register(RunTerminalCmdTool())
    builder.register(MemoryTool())
    builder.register(SessionSearchTool())
    builder.register(CodeNavTool())
