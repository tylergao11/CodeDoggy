"""Tool registration and dispatch."""

from codedoggy.tools.config import ToolConfig, ToolServerConfig
from codedoggy.tools.grok_surface import (
    codedoggy_product_config,
    grok_build_product_config,
)
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.registry import (
    FinalizedToolset,
    ToolRegistryBuilder,
    register_tool_pack,
)
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolDispatch,
    ToolError,
    ToolId,
    ToolSpec,
)

__all__ = [
    "FinalizedToolset",
    "ListToolsContext",
    "Tool",
    "ToolCallContext",
    "ToolConfig",
    "ToolDescription",
    "ToolDispatch",
    "ToolError",
    "ToolId",
    "ToolKind",
    "ToolNamespace",
    "ToolRegistryBuilder",
    "ToolServerConfig",
    "ToolSpec",
    "codedoggy_product_config",
    "grok_build_product_config",
    "register_tool_pack",
]
