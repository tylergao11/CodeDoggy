"""Grok-aligned orchestration layer for CodeDoggy.

Faithful port of Grok Build orchestration concepts:

* **SessionActor spine** → ``RuntimeKernel`` + turn loop (host)
* **Agent** = config package (prompt + tools + capability), not the loop
* **Turn** = multi-step sample → tools → sample with first-class exits
* **Tool pipeline** = two-phase precheck then execute (path locks)
* **Subagent** = child session + summary fold-back
* **Session mode** = Plan hard gate independent of yolo (Grok enter/exit plan mode)
* **Interjection** = mid-turn followup drain

Source map (grok-build):
  shell/session/acp_session_impl/{turn,tool_calls,tool_dispatch,run_loop}.rs
  agent/mvp_agent/subagent_coordinator.rs
  xai-grok-agent AgentBuilder / AgentDefinition
  xai-tool-types SubagentCapabilityMode
"""

from codedoggy.orchestration.agent_def import (
    BUILTIN_AGENTS,
    Agent,
    AgentDefinition,
    build_agent,
    builtin_explore,
    builtin_plan,
    filter_toolset,
    load_agent_definition_file,
    resolve_agent_definition,
)
from codedoggy.orchestration.capability import kind_allowed, kinds_for_capability
from codedoggy.orchestration.interjection import (
    LARGE_PROMPT_THRESHOLD,
    format_interjection,
    user_query,
)
from codedoggy.orchestration.prompt_queue import InterjectionBuffer, PromptQueue, PromptQueueItem
from codedoggy.orchestration.session_mode import (
    SessionModeState,
    load_plan_mode_state,
    plan_mode_edit_gate,
    plan_mode_json_path,
    save_plan_mode_state,
)
from codedoggy.orchestration.subagent import (
    SubagentCoordinator,
    SubagentRequest,
    SubagentSnapshot,
    format_parallel_aggregate,
    format_parallel_dispatched,
    make_child_runner,
)
from codedoggy.orchestration.worktree import (
    MergeResult,
    WorktreeError,
    WorktreeHandle,
    branch_for_subagent,
    commit_worktree_changes,
    create_worktree,
    find_git_root,
    merge_worktree_into_parent,
    reattach_worktree,
    remove_worktree,
)
from codedoggy.orchestration.tool_pipeline import (
    execute_approved_batch,
    execute_prepared,
    execute_tool_calls_two_phase,
    prepare_tool_batch,
    prepare_tool_call,
)
from codedoggy.orchestration.types import (
    CapabilityMode,
    IsolationMode,
    SessionMode,
    ToolBatchResult,
    ToolLoopOutcome,
    TurnExit,
)

__all__ = [
    "BUILTIN_AGENTS",
    "Agent",
    "AgentDefinition",
    "CapabilityMode",
    "InterjectionBuffer",
    "IsolationMode",
    "LARGE_PROMPT_THRESHOLD",
    "PromptQueue",
    "PromptQueueItem",
    "format_interjection",
    "user_query",
    "SessionMode",
    "SessionModeState",
    "SubagentCoordinator",
    "SubagentRequest",
    "SubagentSnapshot",
    "format_parallel_aggregate",
    "format_parallel_dispatched",
    "ToolBatchResult",
    "ToolLoopOutcome",
    "TurnExit",
    "MergeResult",
    "WorktreeError",
    "WorktreeHandle",
    "branch_for_subagent",
    "build_agent",
    "builtin_explore",
    "builtin_plan",
    "commit_worktree_changes",
    "create_worktree",
    "execute_approved_batch",
    "execute_prepared",
    "execute_tool_calls_two_phase",
    "filter_toolset",
    "find_git_root",
    "kind_allowed",
    "kinds_for_capability",
    "load_agent_definition_file",
    "make_child_runner",
    "merge_worktree_into_parent",
    "plan_mode_edit_gate",
    "plan_mode_json_path",
    "load_plan_mode_state",
    "save_plan_mode_state",
    "prepare_tool_batch",
    "prepare_tool_call",
    "reattach_worktree",
    "remove_worktree",
    "resolve_agent_definition",
]
