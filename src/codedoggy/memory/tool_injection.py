"""CodeDoggy glue: wire MemoryManager provider tools into FinalizedToolset.

Hermes source spirit: agent/memory_manager.inject_memory_provider_tools —
append provider tool schemas so the model can call them; dispatch routes
through MemoryManager.handle_tool_call (no second tool protocol).
"""

from __future__ import annotations

import logging
from typing import Any

from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolId,
)

logger = logging.getLogger(__name__)


class MemoryProviderDispatchTool(Tool):
    """Dispatch wrapper: model tool name → MemoryManager.handle_tool_call."""

    def __init__(
        self,
        tool_name: str,
        *,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        memory_manager: Any = None,
    ) -> None:
        self._tool_name = tool_name
        self._description = description or tool_name
        self._parameters = parameters if isinstance(parameters, dict) else {
            "type": "object",
            "properties": {},
        }
        self._memory_manager = memory_manager

    def id(self) -> ToolId:
        return ToolId(self._tool_name)

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        # Provider tools typically mutate external durable state
        return ToolKind.Edit

    def description(self, ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name=self._tool_name, description=self._description)

    def parameters_schema(self) -> dict[str, Any]:
        return self._parameters

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        mm = self._memory_manager
        live = (ctx.extra or {}).get("memory_manager")
        if live is not None:
            has = getattr(live, "has_tool", None)
            if callable(has) and has(self._tool_name):
                mm = live
            elif mm is None:
                mm = live
        if mm is None:
            import json

            return json.dumps(
                {
                    "success": False,
                    "error": f"No memory manager for tool {self._tool_name!r}",
                }
            )
        handle = getattr(mm, "handle_tool_call", None)
        if not callable(handle):
            import json

            return json.dumps(
                {
                    "success": False,
                    "error": f"Memory manager cannot handle {self._tool_name!r}",
                }
            )
        return handle(self._tool_name, args or {})


def inject_memory_provider_tools(toolset: Any, memory_manager: Any) -> int:
    """Hermes inject_memory_provider_tools — append provider schemas to toolset.

    ``toolset`` is a FinalizedToolset (CodeDoggy dispatch table). Returns count
    of newly injected client-facing tools.
    """
    if toolset is None or memory_manager is None:
        return 0
    get_schemas = getattr(memory_manager, "get_all_tool_schemas", None)
    if not callable(get_schemas):
        return 0
    inject = getattr(toolset, "inject_client_tool", None)
    if not callable(inject):
        logger.warning(
            "toolset has no inject_client_tool; cannot register memory provider tools"
        )
        return 0

    existing: set[str] = set()
    client_names = getattr(toolset, "client_names", None)
    if callable(client_names):
        existing.update(client_names())
    by = getattr(toolset, "by_client_name", None)
    if isinstance(by, dict):
        existing.update(by.keys())

    added = 0
    try:
        schemas = get_schemas() or []
    except Exception as e:  # noqa: BLE001
        logger.warning("get_all_tool_schemas failed: %s", e)
        return 0

    for raw in schemas:
        if not isinstance(raw, dict):
            continue
        schema = raw
        if raw.get("type") == "function" and isinstance(raw.get("function"), dict):
            schema = raw["function"]
        tool_name = schema.get("name")
        if not tool_name or not isinstance(tool_name, str):
            logger.warning(
                "Memory provider tool schema with no resolvable name; skipping (%r)",
                raw,
            )
            continue
        if tool_name in existing:
            continue
        desc = str(schema.get("description") or tool_name)
        params = schema.get("parameters")
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        dispatch = MemoryProviderDispatchTool(
            tool_name,
            description=desc,
            parameters=params,
            memory_manager=memory_manager,
        )
        try:
            ok = inject(
                client_name=tool_name,
                tool=dispatch,
                kind=ToolKind.Edit,
                description=desc,
                parameters=params,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to inject memory tool %r: %s", tool_name, e)
            continue
        if ok:
            existing.add(tool_name)
            added += 1
    if added:
        logger.info("Injected %d memory provider tool(s) into toolset", added)
    return added
