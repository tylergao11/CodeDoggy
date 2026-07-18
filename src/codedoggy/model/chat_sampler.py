"""Turn-loop Sampler backed by a ChatClient."""

from __future__ import annotations

import json
from typing import Any

from codedoggy.model.provider import ChatClient
from codedoggy.model.types import ChatMessage
from codedoggy.tools.runtime import ToolSpec
from codedoggy.turn.types import Message, Role, SampleResult, ToolCall


class ChatSampler:
    """Adapter: ``turn.Sampler`` ← ``ChatClient`` (tools optional)."""

    def __init__(self, client: ChatClient) -> None:
        self.client = client

    def sample(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> SampleResult:
        chat_msgs = [_to_chat(m) for m in messages]
        tool_schemas = [_tool_schema(t) for t in tools] if tools else None
        result = self.client.complete(chat_msgs, tools=tool_schemas or None)
        calls: list[ToolCall] = []
        for i, tc in enumerate(result.tool_calls):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name") or ""
            args_raw = fn.get("arguments") or tc.get("arguments") or {}
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw) if args_raw.strip() else {}
                except json.JSONDecodeError:
                    args = {"_raw": args_raw}
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}
            calls.append(
                ToolCall(
                    id=str(tc.get("id") or f"call_{i}"),
                    name=str(name),
                    arguments=args,
                )
            )
        raw = dict(result.raw or {})
        if result.usage:
            raw.setdefault("usage", result.usage)
        return SampleResult(
            content=result.content,
            tool_calls=calls,
            raw=raw,
        )


def _to_chat(m: Message) -> ChatMessage:
    tool_calls = None
    if m.tool_calls:
        tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments)
                    if isinstance(tc.arguments, dict)
                    else str(tc.arguments),
                },
            }
            for tc in m.tool_calls
        ]
    role = m.role.value if isinstance(m.role, Role) else str(m.role)
    return ChatMessage(
        role=role,
        content=m.content,
        name=m.name,
        tool_call_id=m.tool_call_id,
        tool_calls=tool_calls,
    )


def _tool_schema(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description or "",
            "parameters": spec.parameters or {"type": "object", "properties": {}},
        },
    }
