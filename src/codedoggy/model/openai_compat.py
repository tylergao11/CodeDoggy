"""OpenAI-compatible chat/completions client (stdlib HTTP, no SDK).

Hermes talks to Ollama/local endpoints the same way: ``base_url`` + optional
key on the OpenAI wire format. Grok's sampler is richer (streaming, auth
schemes); we keep the same *config* axes without the full stack.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

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
            out.append(d)
        else:
            out.append(dict(m))
    return out


class OpenAICompatClient:
    """POST ``{base_url}/chat/completions``."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    @property
    def config(self) -> ModelConfig:
        return self._config

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
        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": _as_dict_messages(messages),
            "stream": False,
        }
        temp = temperature if temperature is not None else cfg.temperature
        if temp is not None:
            body["temperature"] = temp
        mt = max_tokens if max_tokens is not None else cfg.max_tokens
        if mt is not None:
            body["max_tokens"] = mt
        if tools:
            body["tools"] = tools

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **cfg.extra_headers,
        }
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

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

        choices = payload.get("choices") or []
        if not choices:
            raise ModelError(f"No choices in model response: {payload!r}"[:400])
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            # Some multimodal endpoints return content parts.
            content = "".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        tool_calls = msg.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            tool_calls = []
        text = content if isinstance(content, str) else (str(content) if content else None)
        text = scrub_model_content(text)
        # Some endpoints put visible answer in alternate fields after thinking.
        if not text:
            for key in ("reasoning_content", "reasoning"):
                alt = msg.get(key)
                if isinstance(alt, str) and alt.strip() and not tool_calls:
                    # Prefer not to surface pure chain-of-thought as final answer.
                    break
            for key in ("refusal",):
                alt = msg.get(key)
                if isinstance(alt, str) and alt.strip():
                    text = alt.strip()
                    break
        return CompletionResult(
            content=text,
            model=str(payload.get("model") or cfg.model),
            finish_reason=choices[0].get("finish_reason"),
            tool_calls=tool_calls,
            raw=payload,
            usage=dict(payload.get("usage") or {}),
        )


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
