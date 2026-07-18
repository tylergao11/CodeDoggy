"""Tool registry builder, packs, and finalized dispatch table."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from codedoggy.tools.config import ToolConfig, ToolServerConfig
from codedoggy.tools.kinds import ToolKind
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDispatch,
    ToolError,
    ToolSpec,
)

ToolPack = Callable[["ToolRegistryBuilder"], None]

_pack_lock = threading.Lock()
_TOOL_PACKS: list[ToolPack] = []
_builder_constructed = False


def register_tool_pack(pack: ToolPack) -> None:
    """Register a pack applied on every subsequent `ToolRegistryBuilder.new()`.

    Prefer calling this at process startup, before the first builder is created.
    """
    with _pack_lock:
        _TOOL_PACKS.append(pack)


def _reset_packs_for_tests() -> None:
    global _builder_constructed
    with _pack_lock:
        _TOOL_PACKS.clear()
        _builder_constructed = False


@dataclass
class _ToolEntry:
    tool: Tool
    namespace: str
    short_id: str
    kind: ToolKind
    input_schema: dict[str, Any]


class ToolRegistryBuilder:
    """Collect tool implementations, then finalize into a dispatch table."""

    def __init__(self, *, empty: bool = False) -> None:
        global _builder_constructed
        self._tools: dict[str, _ToolEntry] = {}
        with _pack_lock:
            _builder_constructed = True
            packs = list(_TOOL_PACKS)
        if not empty:
            from codedoggy.tools.builtins import register_builtins

            register_builtins(self)
            for pack in packs:
                pack(self)

    @classmethod
    def new(cls) -> ToolRegistryBuilder:
        """Builder with builtins and process packs."""
        return cls(empty=False)

    @classmethod
    def empty(cls) -> ToolRegistryBuilder:
        """Builder with no builtins or packs."""
        return cls(empty=True)

    def register(self, tool: Tool) -> None:
        qid = tool.qualified_id()
        self._tools[qid] = _ToolEntry(
            tool=tool,
            namespace=str(tool.tool_namespace()),
            short_id=str(tool.id()),
            kind=tool.kind(),
            input_schema=tool.parameters_schema(),
        )

    def has_tool_id(self, qualified_id: str) -> bool:
        return qualified_id in self._tools

    def known_tool_ids(self) -> set[str]:
        return set(self._tools.keys())

    def known_tool_kinds(self) -> dict[str, ToolKind]:
        return {qid: e.kind for qid, e in self._tools.items()}

    def finalize(
        self,
        config: ToolServerConfig | None = None,
        *,
        list_ctx: ListToolsContext | None = None,
    ) -> FinalizedToolset:
        """Build an immutable toolset.

        If ``config.tools`` is empty, every registered tool is enabled.
        Rejects unknown ids and duplicate client-facing names.
        """
        config = config or ToolServerConfig()
        list_ctx = list_ctx or ListToolsContext()

        if config.tools:
            selected: list[tuple[ToolConfig, _ToolEntry]] = []
            for tc in config.tools:
                entry = self._tools.get(tc.id)
                if entry is None:
                    raise FinalizeError(f"unknown tool id in config: {tc.id}")
                selected.append((tc, entry))
        else:
            selected = [
                (ToolConfig(id=qid, kind=e.kind), e) for qid, e in self._tools.items()
            ]

        by_client: dict[str, _FinalizedTool] = {}
        for tc, entry in selected:
            client_name = tc.resolve_client_name(entry.short_id)
            if client_name in by_client:
                raise FinalizeError(f"duplicate client-facing tool name: {client_name}")
            desc = entry.tool.description(list_ctx)
            description = (
                tc.description_override
                if tc.description_override is not None
                else desc.description
            )
            if not entry.tool.should_list(list_ctx):
                continue
            by_client[client_name] = _FinalizedTool(
                qualified_id=tc.id,
                short_id=entry.short_id,
                client_name=client_name,
                kind=tc.kind or entry.kind,
                description=description,
                parameters=entry.input_schema,
                tool=entry.tool,
            )

        return FinalizedToolset(by_client_name=by_client)


class FinalizeError(Exception):
    """Invalid tool configuration at finalize time."""


@dataclass
class _FinalizedTool:
    qualified_id: str
    short_id: str
    client_name: str
    kind: ToolKind
    description: str
    parameters: dict[str, Any]
    tool: Tool


@dataclass
class FinalizedToolset(ToolDispatch):
    """Immutable tool table ready for listing and dispatch."""

    by_client_name: dict[str, _FinalizedTool] = field(default_factory=dict)

    def tool_definitions(self) -> list[ToolSpec]:
        specs = [
            ToolSpec(
                name=ft.client_name,
                description=ft.description,
                parameters=ft.parameters,
            )
            for ft in self.by_client_name.values()
        ]
        specs.sort(key=lambda s: s.name)
        return specs

    def client_names(self) -> list[str]:
        return sorted(self.by_client_name.keys())

    def kind_of(self, tool_name: str) -> ToolKind | None:
        """Capability kind for a client-facing name, or None if unknown."""
        ft = self.by_client_name.get(tool_name)
        return ft.kind if ft is not None else None

    def call(
        self,
        tool_name: str,
        args: dict[str, Any],
        ctx: ToolCallContext,
    ) -> str:
        ft = self.by_client_name.get(tool_name)
        if ft is None:
            raise ToolError.not_found(tool_name)
        if not isinstance(args, dict):
            raise ToolError.invalid_arguments("args must be a JSON object")
        return ft.tool.run(ctx, args)
