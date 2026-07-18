"""Tool registry builder, packs, and finalized dispatch table."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from codedoggy.tools.config import ToolConfig, ToolServerConfig
from codedoggy.tools.kinds import ToolKind, resolve_authoritative_kind
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
    """Register a pack applied on every subsequent `ToolRegistryBuilder.new()`."""
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
        return cls(empty=False)

    @classmethod
    def empty(cls) -> ToolRegistryBuilder:
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
        product_surface: bool = True,
    ) -> FinalizedToolset:
        """Build an immutable toolset.

        Default: Grok product surface (client renames + product tool list).
        ``product_surface=False`` with empty config enables all wire ids.
        """
        from codedoggy.tools.grok_surface import (
            apply_product_rename,
            codedoggy_product_config,
            remap_schema_properties,
        )

        list_ctx = list_ctx or ListToolsContext()
        using_product = False
        if config is None and product_surface:
            # Default finalize = CodeDoggy pack (Grok + Doggy extras).
            # Pure Grok: finalize(grok_build_product_config()).
            config = codedoggy_product_config()
            using_product = True
        elif config is None:
            config = ToolServerConfig()
        elif config.behavior_preset == "grok-build":
            using_product = True

        if config.tools:
            selected: list[tuple[ToolConfig, _ToolEntry]] = []
            for tc in config.tools:
                entry = self._tools.get(tc.id)
                if entry is None:
                    if using_product:
                        continue
                    raise FinalizeError(f"unknown tool id in config: {tc.id}")
                selected.append((apply_product_rename(tc), entry))
        else:
            selected = [
                (apply_product_rename(ToolConfig(id=qid, kind=e.kind)), e)
                for qid, e in self._tools.items()
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
            overrides = dict(tc.params_name_overrides or {})
            client_schema = remap_schema_properties(entry.input_schema, overrides or None)
            if overrides:
                for internal, client in overrides.items():
                    description = description.replace(internal, client)
            # Registration kind wins for mutating tools; config cannot downgrade.
            final_kind = resolve_authoritative_kind(
                short_id=entry.short_id,
                registered_kind=entry.kind,
                config_kind=tc.kind,
            )
            ft = _FinalizedTool(
                qualified_id=tc.id,
                short_id=entry.short_id,
                client_name=client_name,
                kind=final_kind,
                description=description,
                parameters=client_schema,
                internal_parameters=entry.input_schema,
                tool=entry.tool,
                params_name_overrides=overrides,
            )
            by_client[client_name] = ft
            # Wire-id alias for callers/tests still using short ids
            if entry.short_id != client_name:
                by_client.setdefault(entry.short_id, ft)

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
    internal_parameters: dict[str, Any] = field(default_factory=dict)
    params_name_overrides: dict[str, str] = field(default_factory=dict)


@dataclass
class FinalizedToolset(ToolDispatch):
    """Immutable tool table ready for listing and dispatch."""

    by_client_name: dict[str, _FinalizedTool] = field(default_factory=dict)

    def tool_definitions(self) -> list[ToolSpec]:
        by_qid: dict[str, ToolSpec] = {}
        for name, ft in self.by_client_name.items():
            if name != ft.client_name:
                continue
            by_qid[ft.qualified_id] = ToolSpec(
                name=ft.client_name,
                description=ft.description,
                parameters=ft.parameters,
            )
        specs = list(by_qid.values())
        specs.sort(key=lambda s: s.name)
        return specs

    def client_names(self) -> list[str]:
        return sorted(
            {
                ft.client_name
                for name, ft in self.by_client_name.items()
                if name == ft.client_name
            }
        )

    def kind_of(self, tool_name: str) -> ToolKind | None:
        ft = self._resolve(tool_name)
        return ft.kind if ft is not None else None

    def _resolve(self, tool_name: str) -> _FinalizedTool | None:
        ft = self.by_client_name.get(tool_name)
        if ft is not None:
            return ft
        from codedoggy.tools.grok_surface import CLIENT_ALIASES

        short = CLIENT_ALIASES.get(tool_name)
        if not short:
            return None
        for cand in self.by_client_name.values():
            if cand.short_id == short:
                return cand
        return None

    def inject_client_tool(
        self,
        *,
        client_name: str,
        tool: Tool,
        kind: ToolKind | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> bool:
        """Post-finalize glue registration (memory provider tools).

        CodeDoggy glue: product finalize is a fixed allow-list; provider tools
        arrive after MemoryManager load. Returns True if newly added.
        Does not overwrite an existing primary client_name entry.
        """
        if not client_name or not isinstance(client_name, str):
            return False
        existing = self.by_client_name.get(client_name)
        if existing is not None and existing.client_name == client_name:
            return False
        short_id = str(tool.id())
        params = parameters if parameters is not None else tool.parameters_schema()
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        desc = description
        if desc is None:
            try:
                desc = tool.description(ListToolsContext()).description
            except Exception:  # noqa: BLE001
                desc = client_name
        ft = _FinalizedTool(
            qualified_id=tool.qualified_id(),
            short_id=short_id,
            client_name=client_name,
            kind=kind if kind is not None else tool.kind(),
            description=desc or client_name,
            parameters=params,
            internal_parameters=params,
            tool=tool,
            params_name_overrides={},
        )
        self.by_client_name[client_name] = ft
        if short_id != client_name:
            self.by_client_name.setdefault(short_id, ft)
        return True

    def call(
        self,
        tool_name: str,
        args: dict[str, Any],
        ctx: ToolCallContext,
    ) -> str:
        """Central gate: schema + policy, then tool.run."""
        from codedoggy.tools.gate import enforce_policy, validate_args_against_schema
        from codedoggy.tools.grok_surface import remap_args_client_to_internal

        ft = self._resolve(tool_name)
        if ft is None:
            raise ToolError.not_found(tool_name)
        if not isinstance(args, dict):
            raise ToolError.invalid_arguments("args must be a JSON object")

        # Normalize: accept both product and wire param names
        internal_args = remap_args_client_to_internal(
            dict(args),
            ft.params_name_overrides or None,
            short_id=ft.short_id,
        )
        # Validate against *internal* schema (implementation contract)
        schema = ft.internal_parameters or ft.parameters or {}
        validate_args_against_schema(internal_args, schema)

        # Prefer wire short_id for hard-name checks; tool.kind() is registration truth.
        enforce_policy(
            tool_name=ft.short_id,
            kind=ft.kind,
            args=internal_args,
            ctx=ctx,
            registered_kind=ft.tool.kind(),
        )
        if (ctx.extra or {}).get("writes_paused"):
            if ft.kind in {
                ToolKind.Edit,
                ToolKind.Write,
                ToolKind.Delete,
                ToolKind.Move,
            } or ft.short_id in {"run_terminal_cmd", "search_replace"}:
                raise ToolError(
                    "writes paused — fix the blocking issue before more writes",
                    code="writes_paused",
                )
        ctx.extra.pop("mutations", None)
        ctx.extra.pop("mutation", None)
        return ft.tool.run(ctx, internal_args)
