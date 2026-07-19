"""memory — curated persistent MEMORY.md / USER.md."""

from __future__ import annotations

import json
import logging
from typing import Any

from codedoggy.memory.store import MemoryStore
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolError,
    ToolId,
)

logger = logging.getLogger(__name__)

_DESCRIPTION = """\
Manage curated persistent memory that survives across sessions.

Two targets:
  - memory: your personal notes (environment, project conventions, lessons learned)
  - user: profile about the user (preferences, communication style, habits)

Actions:
  - add: append a new entry (content required)
  - replace: replace one entry matched by unique old_text substring
  - remove: delete one entry matched by unique old_text substring
  - batch: apply a list of {action, content?, old_text?} ops atomically

There is no separate read action — frozen memory is injected into the system
prompt at session start. Tool responses confirm writes and show live usage.
Character budgets are finite; when full, consolidate (replace/remove) then retry.
Entries are separated by § on disk; keep each entry short and durable.
"""


class MemoryTool(Tool):
    """Model-facing memory mutations against a shared MemoryStore."""

    def __init__(self, store: MemoryStore | None = None) -> None:
        self._store = store

    def bind_store(self, store: MemoryStore) -> None:
        self._store = store

    def id(self) -> ToolId:
        return ToolId("memory")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        # Mutates curated MEMORY/USER — not read-only (capability filter)
        return ToolKind.Edit

    def description(self, ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="memory", description=_DESCRIPTION.strip())

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "replace", "remove", "batch"],
                    "description": "Mutation to perform.",
                },
                "target": {
                    "type": "string",
                    "enum": ["memory", "user"],
                    "description": "Which store: memory (agent notes) or user (profile).",
                },
                "content": {
                    "type": "string",
                    "description": "New entry text for add/replace.",
                },
                "old_text": {
                    "type": "string",
                    "description": (
                        "Unique substring identifying the entry for replace/remove."
                    ),
                },
                "operations": {
                    "type": "array",
                    "description": (
                        "For action=batch: list of {action, content?, old_text?} objects."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["add", "replace", "remove"],
                            },
                            "content": {"type": "string"},
                            "old_text": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["action", "target"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        store = self._resolve_store(ctx)
        action = args.get("action")
        target = args.get("target")
        if action not in {"add", "replace", "remove", "batch"}:
            raise ToolError.invalid_arguments(
                "action must be add, replace, remove, or batch"
            )
        if target not in {"memory", "user"}:
            raise ToolError.invalid_arguments("target must be 'memory' or 'user'")

        if action == "add":
            content = args.get("content")
            if not isinstance(content, str):
                raise ToolError.invalid_arguments("content is required for add")
            result = store.add(target, content)
        elif action == "replace":
            old = args.get("old_text")
            content = args.get("content")
            if not isinstance(old, str):
                raise ToolError.invalid_arguments("old_text is required for replace")
            if not isinstance(content, str):
                raise ToolError.invalid_arguments("content is required for replace")
            result = store.replace(target, old, content)
        elif action == "remove":
            old = args.get("old_text")
            if not isinstance(old, str):
                raise ToolError.invalid_arguments("old_text is required for remove")
            result = store.remove(target, old)
        else:
            ops = args.get("operations")
            if not isinstance(ops, list):
                raise ToolError.invalid_arguments("operations list is required for batch")
            result = store.apply_batch(target, ops)

        # Hermes bridge contract: the built-in tool hands the raw result and
        # args to MemoryManager. The manager owns success/staged gating, batch
        # expansion, frozen-snapshot refresh, and external-provider mirroring.
        manager = (ctx.extra or {}).get("memory_manager")
        notify = getattr(manager, "notify_memory_tool_write", None)
        if callable(notify):
            try:
                notify(
                    result,
                    args,
                    build_metadata=lambda: self._build_write_metadata(ctx),
                )
            except Exception:  # noqa: BLE001
                # The curated write already committed. A mirror failure must
                # not turn that durable success into a tool-level failure.
                logger.debug("memory write notification failed", exc_info=True)

        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _build_write_metadata(ctx: ToolCallContext) -> dict[str, Any]:
        extra = ctx.extra or {}
        metadata: dict[str, Any] = {
            "write_origin": "assistant_tool",
            "execution_context": "foreground",
            "session_id": ctx.session_id or "",
            "platform": str(extra.get("platform") or "codedoggy"),
            "tool_name": "memory",
            "cwd": str(ctx.cwd),
        }
        for key in (
            "parent_session_id",
            "prompt_id",
            "task_id",
            "tool_call_id",
        ):
            value = extra.get(key)
            if value not in {None, ""}:
                metadata[key] = value
        return {key: value for key, value in metadata.items() if value not in {None, ""}}

    def _resolve_store(self, ctx: ToolCallContext) -> MemoryStore:
        if self._store is not None:
            return self._store
        extra = ctx.extra or {}
        store = extra.get("memory_store")
        if isinstance(store, MemoryStore):
            return store
        raise ToolError(
            "No MemoryStore bound. Create MemoryStore, load_from_disk(), "
            "and pass it to MemoryTool or ToolCallContext.extra['memory_store'].",
            code="memory_not_configured",
        )
