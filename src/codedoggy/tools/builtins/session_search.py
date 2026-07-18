"""session_search — Hermes-style discovery / scroll / browse over SessionStore."""

from __future__ import annotations

import json
from typing import Any

from codedoggy.memory.session_store import SessionStore
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolError,
    ToolId,
)

_DESCRIPTION = """\
Search past CodeDoggy sessions in the local SQLite store (FTS5 when available).
No LLM calls — returns real stored messages.

Shapes:
  1) discovery: session_search(query="auth refactor", limit=5)
  2) scroll:    session_search(session_id="...", around_message_id=12, window=8)
  3) read:      session_search(session_id="...")
  4) browse:    session_search()  — recent sessions

Use for "what did we do about X" / past decisions. Prefer live workspace tools
for current file state. Curated MEMORY.md is separate and always in the prompt.
"""


class SessionSearchTool(Tool):
    def __init__(self, store: SessionStore | None = None) -> None:
        self._store = store

    def bind_store(self, store: SessionStore | None) -> None:
        self._store = store

    def id(self) -> ToolId:
        return ToolId("session_search")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Search

    def description(self, ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="session_search", description=_DESCRIPTION.strip())

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Discovery query (keywords / phrase). Omit to browse.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Target session for scroll/read shapes.",
                },
                "around_message_id": {
                    "type": "integer",
                    "description": "Anchor message id for scroll window.",
                },
                "window": {
                    "type": "integer",
                    "description": "±window messages around anchor (default 5).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max discovery hits (default 5, max 30).",
                },
            },
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        store = self._resolve(ctx)
        query = args.get("query")
        session_id = args.get("session_id")
        around = args.get("around_message_id")
        window = int(args.get("window") or 5)
        limit = int(args.get("limit") or 5)
        limit = max(1, min(limit, 30))

        # scroll
        if isinstance(session_id, str) and session_id.strip() and around is not None:
            data = store.get_messages_around(
                session_id.strip(), int(around), window=window
            )
            return json.dumps({"shape": "scroll", **data}, ensure_ascii=False, default=str)

        # read whole session
        if isinstance(session_id, str) and session_id.strip() and around is None and not query:
            msgs = store.get_messages(session_id.strip())
            if len(msgs) > 30:
                head, tail = msgs[:20], msgs[-10:]
                payload = {
                    "shape": "read",
                    "session_id": session_id.strip(),
                    "total": len(msgs),
                    "messages_head": head,
                    "messages_tail": tail,
                    "note": "truncated large session to first 20 + last 10",
                }
            else:
                payload = {
                    "shape": "read",
                    "session_id": session_id.strip(),
                    "total": len(msgs),
                    "messages": msgs,
                }
            return json.dumps(payload, ensure_ascii=False, default=str)

        # discovery
        if isinstance(query, str) and query.strip():
            exclude = None
            if ctx.session_id:
                exclude = ctx.session_id
            hits = store.search(query.strip(), limit=limit, exclude_session_id=exclude)
            results = []
            for h in hits:
                around = store.get_messages_around(
                    h.session_id, h.message_id, window=5
                )
                results.append(
                    {
                        "session_id": h.session_id,
                        "title": h.title,
                        "goal": h.goal,
                        "snippet": h.snippet,
                        "match_message_id": h.message_id,
                        "role": h.role,
                        "messages_before": around["messages_before"],
                        "messages_after": around["messages_after"],
                        "messages": around["window"],
                    }
                )
            return json.dumps(
                {"shape": "discovery", "query": query.strip(), "results": results},
                ensure_ascii=False,
                default=str,
            )

        # browse
        recent = store.list_recent_sessions(limit=limit)
        return json.dumps(
            {"shape": "browse", "sessions": recent},
            ensure_ascii=False,
            default=str,
        )

    def _resolve(self, ctx: ToolCallContext) -> SessionStore:
        if self._store is not None:
            return self._store
        store = (ctx.extra or {}).get("session_store")
        if isinstance(store, SessionStore):
            return store
        raise ToolError(
            "No SessionStore bound. Wire via build_session or ToolCallContext.extra['session_store'].",
            code="session_store_not_configured",
        )
