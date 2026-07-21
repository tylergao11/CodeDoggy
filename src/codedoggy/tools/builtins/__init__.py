"""Default builtin tools registered by ToolRegistryBuilder.new()."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codedoggy.tools.registry import ToolRegistryBuilder


def register_optional_grok_memory_tools(builder: "ToolRegistryBuilder") -> None:
    """Opt-in Grok memory_search / memory_get (NOT Hermes product memory).

    Product fusion: Hermes owns memory. These tools are Grok wire surfaces
    only — register explicitly in tests or experimental hosts; never default.
    """
    from codedoggy.tools.builtins.memory_get import MemoryGetTool
    from codedoggy.tools.builtins.memory_search import MemorySearchTool

    builder.register(MemorySearchTool())
    builder.register(MemoryGetTool())


def register_builtins(builder: ToolRegistryBuilder) -> None:
    from codedoggy.tools.builtins.apply_patch import ApplyPatchTool
    from codedoggy.tools.builtins.ask_user_question import AskUserQuestionTool
    from codedoggy.tools.builtins.code_nav import CodeNavTool
    from codedoggy.tools.builtins.enter_plan_mode import EnterPlanModeTool
    from codedoggy.tools.builtins.exit_plan_mode import ExitPlanModeTool
    from codedoggy.tools.builtins.get_task_output import GetTaskOutputTool
    from codedoggy.tools.builtins.grep import GrepTool
    from codedoggy.tools.builtins.image_gen import ImageEditTool, ImageGenTool
    from codedoggy.tools.builtins.video_gen import ImageToVideoTool, ReferenceToVideoTool
    from codedoggy.tools.builtins.kill_task import KillTaskTool
    from codedoggy.tools.builtins.list_dir import ListDirTool
    from codedoggy.tools.builtins.lsp import LspTool
    from codedoggy.tools.builtins.memory import MemoryTool
    from codedoggy.tools.builtins.monitor import MonitorTool
    from codedoggy.tools.builtins.read_file import ReadFileTool
    from codedoggy.tools.builtins.run_terminal_cmd import RunTerminalCmdTool
    from codedoggy.tools.builtins.scheduler_tools import (
        SchedulerCreateTool,
        SchedulerDeleteTool,
        SchedulerListTool,
    )
    from codedoggy.tools.builtins.search_replace import SearchReplaceTool
    from codedoggy.tools.builtins.search_tool import SearchToolTool
    from codedoggy.tools.builtins.session_search import SessionSearchTool
    from codedoggy.tools.builtins.skill import SkillTool
    from codedoggy.tools.builtins.merge_subagent_worktree import (
        MergeSubagentWorktreeTool,
    )
    from codedoggy.tools.builtins.parallel_tasks import ParallelTasksTool
    from codedoggy.tools.builtins.spawn_subagent import (
        GetSubagentOutputTool,
        TaskTool,
    )
    from codedoggy.tools.builtins.todo_write import TodoWriteTool
    from codedoggy.tools.builtins.update_goal import UpdateGoalTool
    from codedoggy.tools.builtins.use_tool import UseToolTool
    from codedoggy.tools.builtins.wait_tasks import WaitTasksTool
    from codedoggy.tools.builtins.web_fetch import WebFetchTool
    from codedoggy.tools.builtins.web_search import WebSearchTool
    from codedoggy.tools.builtins.write import WriteTool

    # Core coding surface (GrokBuild wire ids)
    builder.register(ReadFileTool())
    builder.register(SearchReplaceTool())
    builder.register(WriteTool())
    builder.register(ApplyPatchTool())
    builder.register(ListDirTool())
    builder.register(GrepTool())
    builder.register(RunTerminalCmdTool())
    builder.register(LspTool())
    builder.register(ImageGenTool())
    builder.register(ImageEditTool())
    builder.register(ImageToVideoTool())
    builder.register(ReferenceToVideoTool())
    # Background task subsystem
    builder.register(GetTaskOutputTool())
    builder.register(WaitTasksTool())
    builder.register(KillTaskTool())
    builder.register(MonitorTool())
    # Orchestration
    builder.register(TodoWriteTool())
    builder.register(UpdateGoalTool())
    builder.register(EnterPlanModeTool())
    builder.register(ExitPlanModeTool())
    builder.register(AskUserQuestionTool())
    builder.register(TaskTool())  # wire id `task` → product spawn_subagent
    builder.register(ParallelTasksTool())  # MAIN-opt-in multi-spawn (not auto)
    builder.register(MergeSubagentWorktreeTool())  # explicit worktree land
    builder.register(GetSubagentOutputTool())  # legacy; not in product list
    # Web
    builder.register(WebFetchTool())
    builder.register(WebSearchTool())
    # MCP discovery / dispatch (host injects mcp_tools + mcp_dispatch)
    builder.register(SearchToolTool())
    builder.register(UseToolTool())
    # Skills (SKILL.md discovery + substitution)
    builder.register(SkillTool())
    # Scheduler
    builder.register(SchedulerCreateTool())
    builder.register(SchedulerDeleteTool())
    builder.register(SchedulerListTool())
    # Memory (Hermes only on product surface): write `memory` + `session_search`.
    # Grok memory_search / memory_get are NOT registered by default — see
    # register_optional_grok_memory_tools() for wire-fidelity tests only.
    builder.register(MemoryTool())
    builder.register(SessionSearchTool())
    # CodeDoggy enhancements
    builder.register(CodeNavTool())
