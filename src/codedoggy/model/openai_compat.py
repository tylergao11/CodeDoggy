"""OpenAI-compatible chat/completions transport (stdlib HTTP, no SDK).

Reads a :class:`ProviderProfile` for message prep and request kwargs quirks
(Hermes transport + profile split, slimmed for CodeDoggy).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

from codedoggy.model.profile import ProviderProfile
from codedoggy.model.profile_registry import get_profile
from codedoggy.model.reasoning import extract_reasoning_from_message
from codedoggy.model.types import ChatMessage, CompletionResult, ModelConfig

logger = logging.getLogger(__name__)

# Local models (qwen3, etc.) often wrap reasoning; strip before use.
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_THINK_UNCLOSED_RE = re.compile(r"<think>[\s\S]*$", re.IGNORECASE)


class ModelError(Exception):
    """Transport or API failure talking to a model endpoint."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _as_dict_messages(
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


class OpenAICompatClient:
    """POST ``{base_url}/chat/completions`` with optional provider profile."""

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
    ) -> CompletionResult:
        cfg = self._config
        url = f"{cfg.normalized_base_url()}/chat/completions"
        body = self._build_body(
            messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        headers = self._headers(accept="application/json")
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                raw_bytes = resp.read()
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise ModelError(
                f"HTTP {e.code} from {url}: {err_body[:500]}",
                status=e.code,
            ) from e
        except urllib.error.URLError as e:
            raise ModelError(f"Failed to reach {url}: {e.reason}") from e

        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise ModelError(f"Invalid JSON from model endpoint: {e}") from e

        return self._completion_from_payload(payload)

    def complete_stream(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_delta: Any | None = None,
    ) -> CompletionResult:
        """SSE streaming completion (host progressive content).

        Assembles a final ``CompletionResult``. ``on_delta(text_chunk)`` is
        invoked for content deltas when provided.
        Falls back to non-stream ``complete`` if the endpoint rejects stream.
        """
        cfg = self._config
        url = f"{cfg.normalized_base_url()}/chat/completions"
        body = self._build_body(
            messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        body["stream_options"] = {"include_usage": True}

        headers = self._headers(accept="text/event-stream")
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                return _consume_sse_completion(
                    resp,
                    model=cfg.model,
                    on_delta=on_delta,
                )
        except urllib.error.HTTPError as e:
            if e.code in {400, 404, 422}:
                logger.debug("stream rejected (%s); falling back to complete", e.code)
                return self.complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                )
            err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise ModelError(
                f"HTTP {e.code} from {url}: {err_body[:500]}",
                status=e.code,
            ) from e
        except urllib.error.URLError as e:
            raise ModelError(f"Failed to reach {url}: {e.reason}") from e

    def _headers(self, *, accept: str) -> dict[str, str]:
        cfg = self._config
        headers = {
            "Content-Type": "application/json",
            "Accept": accept,
            **(self._profile.default_headers if self._profile else {}),
            **cfg.extra_headers,
        }
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        return headers

    def _build_body(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        cfg = self._config
        wire_msgs = _as_dict_messages(messages)
        if self._profile is not None:
            wire_msgs = self._profile.prepare_messages(wire_msgs, model=cfg.model)

        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": wire_msgs,
            "stream": stream,
        }
        temp = temperature if temperature is not None else cfg.temperature
        if temp is not None:
            body["temperature"] = temp
        mt = max_tokens if max_tokens is not None else cfg.max_tokens
        if mt is not None:
            body["max_tokens"] = mt
        if tools:
            body["tools"] = tools

        # Profile + config.extra reasoning knobs
        reasoning_config = None
        if isinstance(cfg.extra, dict):
            rc = cfg.extra.get("reasoning")
            if isinstance(rc, dict):
                reasoning_config = rc

        if self._profile is not None:
            extra_body, top_level = self._profile.build_api_kwargs_extras(
                model=cfg.model,
                reasoning_config=reasoning_config,
            )
            if extra_body:
                # OpenAI-compat: vendor-specific keys usually sit top-level or
                # under a nested bag; DeepSeek expects ``thinking`` top-level
                # when using raw HTTP (same as extra_body merge in SDK).
                body.update(extra_body)
            if top_level:
                body.update(top_level)

        # Opaque extra passthrough (num_ctx, etc.) — never override core keys
        if isinstance(cfg.extra, dict):
            for k, v in cfg.extra.items():
                if k in {"reasoning", "messages", "model", "stream", "tools"}:
                    continue
                if k not in body:
                    body[k] = v

        return body

    def _completion_from_payload(self, payload: dict[str, Any]) -> CompletionResult:
        cfg = self._config
        choices = payload.get("choices") or []
        if not choices:
            raise ModelError(f"No choices in model response: {payload!r}"[:400])
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        tool_calls = msg.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            tool_calls = []
        text = content if isinstance(content, str) else (str(content) if content else None)
        text = scrub_model_content(text)
        reasoning = extract_reasoning_from_message(msg)
        if not text:
            for key in ("refusal",):
                alt = msg.get(key)
                if isinstance(alt, str) and alt.strip():
                    text = alt.strip()
                    break
        usage = normalize_openai_usage(payload.get("usage") or {})
        return CompletionResult(
            content=text,
            model=str(payload.get("model") or cfg.model),
            finish_reason=choices[0].get("finish_reason"),
            tool_calls=tool_calls,
            raw=payload,
            usage=usage,
            reasoning_content=reasoning,
            provider_data=None,
        )


def _consume_sse_completion(
    resp: Any,
    *,
    model: str,
    on_delta: Any | None,
) -> CompletionResult:
    """Parse OpenAI-compatible SSE chat.completion.chunk stream."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_acc: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    model_name = model
    raw_chunks: list[dict[str, Any]] = []

    while True:
        line = resp.readline()
        if not line:
            break
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        raw_chunks.append(chunk)
        if chunk.get("model"):
            model_name = str(chunk["model"])
        if isinstance(chunk.get("usage"), dict):
            usage = dict(chunk["usage"])
        choices = chunk.get("choices") or []
        if not choices:
            continue
        ch0 = choices[0] or {}
        if ch0.get("finish_reason"):
            finish_reason = ch0.get("finish_reason")
        delta = ch0.get("delta") or {}
        piece = delta.get("content")
        if isinstance(piece, str) and piece:
            content_parts.append(piece)
            if callable(on_delta):
                try:
                    on_delta(piece)
                except Exception:  # noqa: BLE001
                    logger.debug("on_delta failed", exc_info=True)
        for rkey in ("reasoning_content", "reasoning"):
            rp = delta.get(rkey)
            if isinstance(rp, str) and rp:
                reasoning_parts.append(rp)
                break
        for tc in delta.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            idx = int(tc.get("index") or 0)
            slot = tool_acc.setdefault(
                idx,
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if tc.get("id"):
                slot["id"] = str(tc["id"])
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["function"]["name"] = str(fn["name"])
            if fn.get("arguments"):
                slot["function"]["arguments"] = (
                    str(slot["function"].get("arguments") or "") + str(fn["arguments"])
                )

    text = scrub_model_content("".join(content_parts) or None)
    reasoning = "".join(reasoning_parts).strip() or None
    tool_calls = [tool_acc[i] for i in sorted(tool_acc)]
    return CompletionResult(
        content=text,
        model=model_name,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
        raw={"stream": True, "chunks": len(raw_chunks), "usage": usage},
        usage=usage,
        reasoning_content=reasoning,
    )


def normalize_openai_usage(usage: Any) -> dict[str, Any]:
    """Normalize OpenAI / DeepSeek usage including cache hit fields.

    DeepSeek reports ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``.
    OpenAI may report ``prompt_tokens_details.cached_tokens``.
    """
    if not isinstance(usage, dict):
        return {}
    out = dict(usage)
    # DeepSeek disk cache
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if hit is not None:
        out["cache_read_input_tokens"] = hit
        out["cached_tokens"] = hit
    if miss is not None:
        out["cache_miss_input_tokens"] = miss
    # OpenAI nested details
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        out.setdefault("cached_tokens", details.get("cached_tokens"))
        out.setdefault("cache_read_input_tokens", details.get("cached_tokens"))
    return out


def scrub_model_content(text: str | None) -> str | None:
    """Strip think-blocks; return None when nothing usable remains."""
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    cleaned = _THINK_RE.sub("", text)
    cleaned = _THINK_UNCLOSED_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    return cleaned if cleaned else None
