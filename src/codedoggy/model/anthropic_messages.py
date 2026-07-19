"""Anthropic Messages API transport (stdlib HTTP).

Converts OpenAI-shaped messages/tools → Anthropic wire format and back.
Auth headers come from the auth layer (x-api-key vs Bearer).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from codedoggy.model.errors import ModelStreamCancelled
from codedoggy.model.openai_compat import ModelError, scrub_model_content
from codedoggy.model.profile import ProviderProfile
from codedoggy.model.profile_registry import get_profile
from codedoggy.model.protocol_context import assistant_blocks_for_anthropic
from codedoggy.model.stream_cancel import (
    HTTPErrorSnapshot,
    cancellable_read,
    cancellable_readline,
    run_cancellable_request,
    snapshot_http_error,
)
from codedoggy.model.types import ChatMessage, CompletionResult, ModelConfig

logger = logging.getLogger(__name__)

_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"

_STOP_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


class AnthropicMessagesClient:
    """POST ``{base_url}/v1/messages`` (or ``{base}/messages``)."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        profile: ProviderProfile | None = None,
    ) -> None:
        self._config = config
        self._profile = profile or get_profile(config.provider)

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def profile(self) -> ProviderProfile | None:
        return self._profile

    def complete(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        cancel_event: Any | None = None,
    ) -> CompletionResult:
        cfg = self._config
        url = _messages_url(cfg.normalized_base_url())
        body = self._build_body(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        headers = self._headers()
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        def _request() -> bytes:
            try:
                with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                    return cancellable_read(resp, cancel_event)
            except urllib.error.HTTPError as exc:
                raise snapshot_http_error(exc, cancel_event, max_body_bytes=500) from None

        try:
            raw_bytes = run_cancellable_request(_request, cancel_event)
        except HTTPErrorSnapshot as e:
            raise ModelError(
                f"HTTP {e.status} from {url}: {e.body[:500]}",
                status=e.status,
            ) from e
        except urllib.error.URLError as e:
            raise ModelError(f"Failed to reach {url}: {e.reason}") from e

        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise ModelError(f"Invalid JSON from Anthropic: {e}") from e

        return normalize_anthropic_response(payload, model=cfg.model)

    def complete_stream(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_delta: Any | None = None,
        cancel_event: Any | None = None,
    ) -> CompletionResult:
        """Anthropic SSE stream; falls back to complete() if rejected."""
        cfg = self._config
        url = _messages_url(cfg.normalized_base_url())
        body = self._build_body(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            stream=True,
        )
        headers = self._headers(accept="text/event-stream")
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        def _request() -> CompletionResult:
            try:
                with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                    return _consume_anthropic_sse(
                        resp,
                        model=cfg.model,
                        on_delta=on_delta,
                        cancel_event=cancel_event,
                    )
            except urllib.error.HTTPError as exc:
                raise snapshot_http_error(
                    exc,
                    cancel_event,
                    read_body=exc.code not in {400, 404, 422},
                    max_body_bytes=500,
                ) from None

        try:
            return run_cancellable_request(_request, cancel_event)
        except HTTPErrorSnapshot as e:
            if e.status in {400, 404, 422}:
                logger.debug("anthropic stream rejected (%s); complete()", e.status)
                return self.complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    cancel_event=cancel_event,
                )
            raise ModelError(
                f"HTTP {e.status} from {url}: {e.body[:500]}",
                status=e.status,
            ) from e
        except urllib.error.URLError as e:
            raise ModelError(f"Failed to reach {url}: {e.reason}") from e

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        cfg = self._config
        headers = {
            "Content-Type": "application/json",
            "Accept": accept,
            "anthropic-version": _DEFAULT_ANTHROPIC_VERSION,
            **(self._profile.default_headers if self._profile else {}),
            **cfg.extra_headers,
        }
        token = (cfg.api_key or "").strip()
        if token:
            # Console keys: x-api-key. OAuth / setup tokens: Bearer.
            if token.startswith("sk-ant-api") or (
                token.startswith("sk-ant-") and not token.startswith("sk-ant-oat")
            ):
                # sk-ant-api* → x-api-key; other sk-ant- (oat) → Bearer
                if token.startswith("sk-ant-api"):
                    headers["x-api-key"] = token
                else:
                    headers["Authorization"] = f"Bearer {token}"
            elif token.startswith(("sk-ant-oat", "eyJ", "cc-")):
                headers["Authorization"] = f"Bearer {token}"
            else:
                # Default API key style
                headers["x-api-key"] = token
        return headers

    def _build_body(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        cfg = self._config
        wire = _as_dicts(messages)
        # Reasoning strip + Anthropic cache_control (Hermes prepare_wire_messages)
        if self._profile is not None:
            wire = self._profile.prepare_messages(wire, model=cfg.model)
        system, anth_msgs = convert_messages_to_anthropic(
            wire,
            base_url=cfg.base_url,
        )
        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": anth_msgs,
            "max_tokens": int(max_tokens if max_tokens is not None else (cfg.max_tokens or 8192)),
            "stream": stream,
        }
        if system:
            body["system"] = system
        if isinstance(cfg.extra, dict) and cfg.extra.get("auth_kind") == "oauth":
            identity = {
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI for Claude.",
            }
            current = body.get("system")
            if isinstance(current, list):
                body["system"] = [identity, *current]
            elif isinstance(current, str) and current:
                body["system"] = [identity, {"type": "text", "text": current}]
            else:
                body["system"] = [identity]
        temp = temperature if temperature is not None else cfg.temperature
        if temp is not None:
            body["temperature"] = temp
        if tools:
            anth_tools = convert_tools_to_anthropic(tools)
            # Anthropic accepts at most four cache breakpoints.  Preserve the
            # Hermes last-tool marker when capacity exists, but never create
            # the invalid fifth marker after system/message policy used all
            # four slots.
            cache_points = _count_cache_controls(body.get("system")) + _count_cache_controls(
                body.get("messages")
            )
            if (
                self._profile is not None
                and self._profile.prompt_cache
                and anth_tools
                and cache_points < 4
            ):
                anth_tools[-1] = dict(anth_tools[-1])
                anth_tools[-1]["cache_control"] = {
                    "type": "ephemeral",
                    **(
                        {"ttl": self._profile.prompt_cache_ttl}
                        if self._profile.prompt_cache_ttl == "1h"
                        else {}
                    ),
                }
            body["tools"] = anth_tools
        return body


def _count_cache_controls(value: Any) -> int:
    if isinstance(value, dict):
        return int("cache_control" in value) + sum(
            _count_cache_controls(item)
            for key, item in value.items()
            if key != "cache_control"
        )
    if isinstance(value, list):
        return sum(_count_cache_controls(item) for item in value)
    return 0


def _messages_url(base: str) -> str:
    b = base.rstrip("/")
    if b.endswith("/v1"):
        return f"{b}/messages"
    if b.endswith("/messages"):
        return b
    return f"{b}/v1/messages"


def _as_dicts(
    messages: list[ChatMessage] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, ChatMessage):
            d: dict[str, Any] = {"role": m.role}
            if m.content is not None:
                d["content"] = m.content
            if m.name:
                d["name"] = m.name
            if m.tool_call_id:
                d["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                d["tool_calls"] = m.tool_calls
            if m.reasoning_content is not None:
                d["reasoning_content"] = m.reasoning_content
            if m.provider_data:
                d["provider_data"] = dict(m.provider_data)
            out.append(d)
        else:
            out.append(dict(m))
    return out


def convert_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI tools[] → Anthropic tools[] with input_schema."""
    out: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" or "function" in t:
            fn = t.get("function") or {}
            name = fn.get("name") or t.get("name") or ""
            desc = fn.get("description") or t.get("description") or ""
            params = fn.get("parameters") or t.get("input_schema") or {
                "type": "object",
                "properties": {},
            }
            out.append(
                {
                    "name": str(name),
                    "description": str(desc),
                    "input_schema": params,
                }
            )
        elif t.get("name"):
            out.append(
                {
                    "name": str(t["name"]),
                    "description": str(t.get("description") or ""),
                    "input_schema": t.get("input_schema")
                    or t.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
    return out


def convert_messages_to_anthropic(
    messages: list[dict[str, Any]],
    *,
    base_url: str | None = None,
    preserve_unsigned_thinking: bool = False,
) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Split system; convert tool/assistant turns; run Hermes hygiene pipeline."""
    from codedoggy.model.anthropic_hygiene import finalize_anthropic_messages

    system_parts: list[str] = []
    system_blocks: list[dict[str, Any]] | None = None
    out: list[dict[str, Any]] = []

    for msg in messages:
        role = str(msg.get("role") or "")
        if role == "system":
            c = msg.get("content")
            # Preserve cache_control content-block form (Hermes)
            if isinstance(c, list) and any(
                isinstance(p, dict) and p.get("cache_control") for p in c
            ):
                system_blocks = [p for p in c if isinstance(p, dict)]
            elif isinstance(c, str) and c.strip():
                system_parts.append(c)
            continue

        if role == "tool":
            # Anthropic: user message with tool_result blocks
            tool_result = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id") or msg.get("id") or "",
                "content": msg.get("content") if msg.get("content") is not None else "",
            }
            # Forward message-level cache_control onto last tool_result when present
            if isinstance(msg.get("cache_control"), dict):
                tool_result["cache_control"] = dict(msg["cache_control"])
            if out and out[-1].get("role") == "user" and isinstance(out[-1].get("content"), list):
                content_list = out[-1]["content"]
                if all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content_list):
                    content_list.append(tool_result)
                    continue
            out.append({"role": "user", "content": [tool_result]})
            continue

        if role == "assistant":
            # Prefer stored ordered blocks (signed thinking + tool_use).
            # Hermes: wrong order → HTTP 400 thinking signature invalid.
            ordered = assistant_blocks_for_anthropic(msg)
            if ordered:
                # Move envelope cache_control onto last block (native layout)
                cc = msg.get("cache_control")
                if isinstance(cc, dict) and ordered:
                    ordered[-1] = dict(ordered[-1])
                    ordered[-1]["cache_control"] = dict(cc)
                out.append({"role": "assistant", "content": ordered})
            else:
                out.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})
            continue

        if role == "user":
            c = msg.get("content")
            if isinstance(c, list):
                out.append({"role": "user", "content": c})
            else:
                block: dict[str, Any] = {
                    "role": "user",
                    "content": c if c is not None else "",
                }
                # content str + cache_control → promote to list for native
                if isinstance(c, str) and isinstance(msg.get("cache_control"), dict):
                    block["content"] = [
                        {
                            "type": "text",
                            "text": c,
                            "cache_control": dict(msg["cache_control"]),
                        }
                    ]
                out.append(block)
            continue

        # developer / other → system-ish
        c = msg.get("content")
        if isinstance(c, str) and c.strip():
            system_parts.append(c)

    if system_blocks is not None:
        system: str | list[dict[str, Any]] | None = system_blocks
    else:
        system = "\n\n".join(system_parts) if system_parts else None
    # Anthropic requires messages non-empty and alternating starting with user
    if not out:
        out = [{"role": "user", "content": ""}]

    out = finalize_anthropic_messages(
        out,
        base_url=base_url,
        preserve_unsigned_thinking=preserve_unsigned_thinking,
    )
    return system, out


def _consume_anthropic_sse(
    resp: Any,
    *,
    model: str,
    on_delta: Any | None,
    cancel_event: Any | None = None,
) -> CompletionResult:
    """Parse Anthropic Messages SSE into CompletionResult."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    blocks: dict[int, dict[str, Any]] = {}
    tool_json: dict[int, str] = {}
    stop_reason: str | None = None
    usage: dict[str, Any] = {}
    model_name = model
    event_type = ""
    saw_message_stop = False

    while True:
        line = cancellable_readline(resp, cancel_event)
        if not line:
            break
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if not line:
            continue
        if line.startswith("event:"):
            event_type = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        et = data.get("type") or event_type
        if et == "error":
            error = data.get("error") or data
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise ModelError(f"Anthropic stream error: {message or error!r}"[:400])
        if et == "message_start":
            msg = data.get("message") or {}
            if msg.get("model"):
                model_name = str(msg["model"])
            if isinstance(msg.get("usage"), dict):
                usage.update(msg["usage"])
        elif et == "content_block_start":
            idx = int(data.get("index") or 0)
            block = data.get("content_block") or {}
            if isinstance(block, dict):
                blocks[idx] = dict(block)
                if block.get("type") == "tool_use":
                    tool_json[idx] = ""
        elif et == "content_block_delta":
            idx = int(data.get("index") or 0)
            delta = data.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                piece = delta.get("text") or ""
                if piece:
                    text_parts.append(piece)
                    if callable(on_delta):
                        try:
                            if on_delta(piece) is False:
                                raise ModelStreamCancelled()
                        except ModelStreamCancelled:
                            raise
                        except Exception:  # noqa: BLE001
                            logger.debug("on_delta failed", exc_info=True)
                    b = blocks.setdefault(idx, {"type": "text", "text": ""})
                    b["text"] = str(b.get("text") or "") + piece
            elif dtype == "thinking_delta":
                th = delta.get("thinking") or ""
                if th:
                    thinking_parts.append(th)
                    b = blocks.setdefault(idx, {"type": "thinking", "thinking": ""})
                    b["thinking"] = str(b.get("thinking") or "") + th
            elif dtype == "signature_delta":
                sig = delta.get("signature") or ""
                b = blocks.setdefault(idx, {"type": "thinking", "thinking": ""})
                b["signature"] = str(b.get("signature") or "") + sig
            elif dtype == "input_json_delta":
                pj = delta.get("partial_json") or ""
                tool_json[idx] = tool_json.get(idx, "") + pj
        elif et == "content_block_stop":
            idx = int(data.get("index") or 0)
            b = blocks.get(idx)
            if b and b.get("type") == "tool_use":
                raw_json = tool_json.get(idx, "") or "{}"
                try:
                    b["input"] = json.loads(raw_json) if raw_json.strip() else {}
                except json.JSONDecodeError:
                    b["input"] = {"_raw": raw_json}
        elif et == "message_delta":
            delta = data.get("delta") or {}
            if delta.get("stop_reason"):
                stop_reason = str(delta["stop_reason"])
            if isinstance(data.get("usage"), dict):
                usage.update(data["usage"])
        elif et == "message_stop":
            saw_message_stop = True
            break

    if not saw_message_stop:
        raise ModelError("Anthropic stream ended before message_stop")
    ordered = [blocks[i] for i in sorted(blocks)]
    # synthesize payload for normalize path
    payload = {
        "model": model_name,
        "stop_reason": stop_reason or "end_turn",
        "content": ordered,
        "usage": usage,
    }
    return normalize_anthropic_response(payload, model=model_name)


def normalize_anthropic_response(
    payload: dict[str, Any],
    *,
    model: str,
) -> CompletionResult:
    content_blocks = payload.get("content") or []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    ordered_blocks: list[dict[str, Any]] = []

    if isinstance(content_blocks, list):
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            # Keep a clean copy for exact replay (signatures + order)
            ordered_blocks.append(dict(block))
            btype = block.get("type")
            if btype == "text":
                t = block.get("text")
                if isinstance(t, str):
                    text_parts.append(t)
            elif btype in {"thinking", "redacted_thinking"}:
                th = block.get("thinking") or block.get("data")
                if isinstance(th, str) and th.strip():
                    reasoning_parts.append(th)
            elif btype == "tool_use":
                inp = block.get("input") if isinstance(block.get("input"), dict) else {}
                tool_calls.append(
                    {
                        "id": str(block.get("id") or ""),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name") or ""),
                            "arguments": json.dumps(inp),
                        },
                    }
                )

    text = scrub_model_content("\n".join(text_parts) if text_parts else None)
    stop = payload.get("stop_reason")
    finish = _STOP_MAP.get(str(stop), str(stop) if stop else None)
    usage_raw = payload.get("usage") or {}
    usage: dict[str, Any] = {}
    if isinstance(usage_raw, dict):
        if "input_tokens" in usage_raw:
            usage["prompt_tokens"] = usage_raw.get("input_tokens")
        if "output_tokens" in usage_raw:
            usage["completion_tokens"] = usage_raw.get("output_tokens")
        # cache metrics when present
        if "cache_read_input_tokens" in usage_raw:
            usage["cache_read_input_tokens"] = usage_raw.get("cache_read_input_tokens")
        if "cache_creation_input_tokens" in usage_raw:
            usage["cache_creation_input_tokens"] = usage_raw.get(
                "cache_creation_input_tokens"
            )
        usage.update({k: v for k, v in usage_raw.items()})

    provider_data: dict[str, Any] | None = None
    if ordered_blocks:
        provider_data = {"anthropic_content_blocks": ordered_blocks}

    return CompletionResult(
        content=text,
        model=str(payload.get("model") or model),
        finish_reason=finish,
        tool_calls=tool_calls,
        raw=payload,
        usage=usage,
        reasoning_content="\n\n".join(reasoning_parts) if reasoning_parts else None,
        provider_data=provider_data,
    )
