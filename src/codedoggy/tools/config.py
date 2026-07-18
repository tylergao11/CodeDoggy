"""Per-tool and toolset enable configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from codedoggy.tools.kinds import ToolKind
from codedoggy.tools.runtime import Tool


@dataclass
class ToolConfig:
    """One enabled tool and optional client-facing overrides."""

    id: str
    """Fully-qualified id, e.g. ``Doggy:read_file``."""

    params: dict[str, Any] | None = None
    name_override: str | None = None
    params_name_overrides: dict[str, str] | None = None
    description_override: str | None = None
    behavior_version: str | None = None
    kind: ToolKind | None = None

    @classmethod
    def for_tool(cls, tool: Tool) -> ToolConfig:
        return cls(id=tool.qualified_id(), kind=tool.kind())

    @classmethod
    def from_id(cls, tool_id: str) -> ToolConfig:
        return cls(id=tool_id, kind=None)

    def with_name(self, name: str) -> ToolConfig:
        self.name_override = name
        return self

    def with_description(self, desc: str) -> ToolConfig:
        self.description_override = desc
        return self

    def resolve_client_name(self, default_id: str) -> str:
        return self.name_override if self.name_override is not None else default_id


@dataclass
class ToolServerConfig:
    """Which tools to enable for a session."""

    tools: list[ToolConfig] = field(default_factory=list)
    behavior_preset: str | None = None

    @classmethod
    def enable_all_from(cls, tools: list[Tool]) -> ToolServerConfig:
        return cls(tools=[ToolConfig.for_tool(t) for t in tools])
