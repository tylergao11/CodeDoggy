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

    def __init__(
        self,
        client: ChatClient,
        *,
        stream: bool = False,
        on_delta: Any | None = None,
    ) -> None:
        self.client = client
        # Host opt-in: stream deltas when client supports complete_stream
        self.stream = stream
        self.on_delta = on_delta
        # P0: monotonic fallback tool-call ids across samples (never reuse call_0)
        self._call_seq = 0

    def sample(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> SampleResult:
        chat_msgs = [_to_chat(m) for m in messages]
        tool_schemas = [_tool_schema(t) for t in tools] if tools else None
        result = self._complete(chat_msgs, tool_schemas)
        return self._sample_from_completion(result)

    def _complete(self, chat_msgs: list[ChatMessage], tool_schemas: list | None):
        if self.stream:
            stream_fn = getattr(self.client, "complete_stream", None)
            if callable(stream_fn):
                return stream_fn(
                    chat_msgs,
                    tools=tool_schemas or None,
                    on_delta=self.on_delta,
                )
        return self.client.complete(chat_msgs, tools=tool_schemas or None)

    def _next_call_id(self) -> str:
        self._call_seq += 1
        return f"call_{self._call_seq}"

    def _sample_from_completion(self, result: Any) -> SampleResult:
        calls: list[ToolCall] = []
        for i, tc in enumerate(result.tool_calls or []):
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
            raw_id = tc.get("id")
            # Empty / missing id → unique sequential (never per-sample call_0 reuse)
            if raw_id is None or str(raw_id).strip() == "":
                tid = self._next_call_id()
            else:
                tid = str(raw_id)
            calls.append(
                ToolCall(
                    id=tid,
                    name=str(name),
                    arguments=args,
                )
            )
        raw = dict(result.raw or {})
        if result.usage:
            raw.setdefault("usage", result.usage)
        reasoning = getattr(result, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning.strip():
            raw.setdefault("reasoning_content", reasoning)
        pdata = getattr(result, "provider_data", None)
        if isinstance(pdata, dict) and pdata:
            raw.setdefault("provider_data", pdata)
        return SampleResult(
            content=result.content,
            tool_calls=calls,
            raw=raw,
            reasoning_content=reasoning if isinstance(reasoning, str) else None,
            provider_data=dict(pdata) if isinstance(pdata, dict) else None,
        )


def _sample_from_completion(result: Any) -> SampleResult:
    """Module-level helper for tests; uses a throwaway sequence (not session-stable)."""
    sampler = ChatSampler.__new__(ChatSampler)
    sampler._call_seq = 0
    return ChatSampler._sample_from_completion(sampler, result)


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
    reasoning = getattr(m, "reasoning_content", None)
    pdata = getattr(m, "provider_data", None)
    return ChatMessage(
        role=role,
        content=m.content,
        name=m.name,
        tool_call_id=m.tool_call_id,
        tool_calls=tool_calls,
        reasoning_content=reasoning if isinstance(reasoning, str) else None,
        provider_data=dict(pdata) if isinstance(pdata, dict) else None,
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
