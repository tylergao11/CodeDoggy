"""session_search — Hermes-style discovery / scroll / browse over SessionStore."""

from __future__ import annotations

import json
from pathlib import Path
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
        cwd_scope = str(Path(ctx.cwd).resolve()) if ctx.cwd is not None else None

        if isinstance(session_id, str) and session_id.strip():
            self._require_workspace_session(
                store,
                session_id.strip(),
                cwd_scope=cwd_scope,
            )

        # scroll
        if isinstance(session_id, str) and session_id.strip() and around is not None:
            data = store.get_messages_around(
                session_id.strip(), int(around), window=window
            )
            data = _public_transcript_payload(data)
            return json.dumps({"shape": "scroll", **data}, ensure_ascii=False, default=str)

        # read whole session
        if isinstance(session_id, str) and session_id.strip() and around is None and not query:
            msgs = store.get_messages(session_id.strip())
            msgs = [_public_message(m) for m in msgs]
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
            hits = store.search(
                query.strip(),
                limit=limit,
                exclude_session_id=exclude,
                cwd=cwd_scope,
            )
            results = []
            for h in hits:
                around = store.get_messages_around(
                    h.session_id, h.message_id, window=5
                )
                around = _public_transcript_payload(around)
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
                {
                    "shape": "discovery",
                    "query": query.strip(),
                    "cwd": cwd_scope,
                    "results": results,
                },
                ensure_ascii=False,
                default=str,
            )
        # browse — same cwd scope when store supports it
        try:
            recent = store.list_recent_sessions(limit=limit, cwd=cwd_scope)
        except TypeError:
            recent = store.list_recent_sessions(limit=limit)
            if cwd_scope and isinstance(recent, list):
                recent = [
                    s
                    for s in recent
                    if str(s.get("cwd") or "") == cwd_scope
                    or not s.get("cwd")
                ]
        return json.dumps(
            {"shape": "browse", "cwd": cwd_scope, "sessions": recent},
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

    @staticmethod
    def _require_workspace_session(
        store: SessionStore,
        session_id: str,
        *,
        cwd_scope: str | None,
    ) -> None:
        """Fail closed for explicit read/scroll across workspace boundaries."""
        if not cwd_scope:
            raise ToolError(
                "Explicit session reads require a workspace cwd.",
                code="session_scope_required",
            )
        check = store.validate_session_cwd(
            session_id,
            cwd_scope,
            allow_missing=False,
            allow_unbound=False,
        )
        if not check.allowed:
            raise ToolError(
                f"Session {session_id!r} is outside the current workspace "
                f"({check.reason}).",
                code="session_cwd_mismatch",
            )


def _public_message(message: dict[str, Any]) -> dict[str, Any]:
    """Hide model-internal reasoning/signatures from the searchable tool view."""
    return {
        key: value
        for key, value in message.items()
        if key not in {"reasoning_content", "provider_data"}
    }


def _public_transcript_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    for key in ("window", "messages"):
        value = out.get(key)
        if isinstance(value, list):
            out[key] = [
                _public_message(item) if isinstance(item, dict) else item
                for item in value
            ]
    return out
