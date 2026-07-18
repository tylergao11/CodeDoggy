"""Agentic turn loop: sample → tools → writeback."""

from codedoggy.turn.executor import (
    execute_tool_batch,
    execute_tool_call,
    is_mutating_kind,
)
from codedoggy.turn.hooks import HookContext, LoopHooks, NoopHooks
from codedoggy.turn.loop import run_agent_loop
from codedoggy.turn.runner import AgentTurnRunner
from codedoggy.turn.sampler import Sampler
from codedoggy.turn.types import (
    FileMutation,
    HookDecision,
    LoopResult,
    Message,
    Role,
    SampleResult,
    ToolCall,
    ToolResultRecord,
)

__all__ = [
    "AgentTurnRunner",
    "FileMutation",
    "HookContext",
    "HookDecision",
    "LoopHooks",
    "LoopResult",
    "Message",
    "NoopHooks",
    "Role",
    "SampleResult",
    "Sampler",
    "ToolCall",
    "ToolResultRecord",
    "execute_tool_batch",
    "execute_tool_call",
    "is_mutating_kind",
    "run_agent_loop",
]
