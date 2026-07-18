"""CodeDoggy — coding agent harness."""

from codedoggy.session import Session, SessionConfig, SessionId
from codedoggy.session.types import SessionPhase, TurnRequest, TurnResult, TurnStatus
from codedoggy.tools import (
    FinalizedToolset,
    ToolCallContext,
    ToolRegistryBuilder,
    ToolServerConfig,
    register_tool_pack,
)
from codedoggy.memory import HermesMemorySelector, MemoryStore, SessionStore
from codedoggy.bootstrap import build_session
from codedoggy.context import CompactionMode, ContextBudget, ContextCompactor
from codedoggy.model import (
    ChatSampler,
    ModelConfig,
    ModelProfiles,
    create_client,
    list_providers,
    model_config_from_env,
    model_profiles_from_env,
    register_provider,
)
from codedoggy.turn import AgentTurnRunner, run_agent_loop

__all__ = [
    "AgentTurnRunner",
    "ChatSampler",
    "CompactionMode",
    "ContextBudget",
    "ContextCompactor",
    "FinalizedToolset",
    "HermesMemorySelector",
    "MemoryStore",
    "SessionStore",
    "ModelConfig",
    "ModelProfiles",
    "Session",
    "SessionConfig",
    "SessionId",
    "SessionPhase",
    "ToolCallContext",
    "ToolRegistryBuilder",
    "ToolServerConfig",
    "TurnRequest",
    "TurnResult",
    "TurnStatus",
    "build_session",
    "create_client",
    "list_providers",
    "model_config_from_env",
    "model_profiles_from_env",
    "register_provider",
    "register_tool_pack",
    "run_agent_loop",
]

__version__ = "0.1.0"
