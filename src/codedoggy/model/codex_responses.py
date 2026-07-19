"""OpenAI Responses API transport (Codex / ChatGPT / compatible).

Ported from Hermes ``codex_responses_adapter`` + ``transports/codex``:
chat messages → Responses ``input`` items, tools → function tools,
POST ``/v1/responses`` (store=false), normalize ``output`` → CompletionResult.

Also stores ``codex_reasoning_items`` / ``codex_message_items`` on
``provider_data`` so multi-turn encrypted reasoning can be replayed
(Hermes cross-turn coherence; wrong-issuer blobs are filtered).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from codedoggy.model.openai_compat import ModelError, normalize_openai_usage, scrub_model_content
from codedoggy.model.profile import ProviderProfile
from codedoggy.model.profile_registry import get_profile
from codedoggy.model.types import ChatMessage, CompletionResult, ModelConfig

logger = logging.getLogger(__name__)

_MAX_ITEM_ID_LEN = 64
_TOOL_CALL_LEAK = re.compile(
    r"(?:^|[\s>|])to=functions\.[A-Za-z_][\w.]*",
    re.IGNORECASE,
)


class CodexResponsesClient:
    """POST ``{base_url}/responses`` (OpenAI Responses API)."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        profile: ProviderProfile | None = None,
        issuer_kind: str | None = None,
    ) -> None:
        self._config = config
        self._profile = profile or get_profile(config.provider)
        self._issuer_kind = issuer_kind or classify_responses_issuer(
            base_url=config.base_url,
            provider=config.provider,
        )

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
        url = responses_url(cfg.normalized_base_url())
        body = self._build_body(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        headers = self._headers()
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                raw_bytes = resp.read()
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise ModelError(
                f"HTTP {e.code} from {url}: {err_body[:600]}",
                status=e.code,
            ) from e
        except urllib.error.URLError as e:
            raise ModelError(f"Failed to reach {url}: {e.reason}") from e

        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise ModelError(f"Invalid JSON from Responses API: {e}") from e

        return normalize_responses_payload(
            payload,
            model=cfg.model,
            issuer_kind=self._issuer_kind,
        )

    def complete_stream(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_delta: Any | None = None,
    ) -> CompletionResult:
        """Responses SSE stream; falls back to non-stream complete()."""
        cfg = self._config
        url = responses_url(cfg.normalized_base_url())
        body = self._build_body(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        body["stream"] = True
        headers = self._headers(accept="text/event-stream")
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                return _consume_responses_sse(
                    resp,
                    model=cfg.model,
                    issuer_kind=self._issuer_kind,
                    on_delta=on_delta,
                )
        except urllib.error.HTTPError as e:
            if e.code in {400, 404, 422}:
                logger.debug("responses stream rejected (%s); complete()", e.code)
                body.pop("stream", None)
                return self.complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                )
            err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise ModelError(
                f"HTTP {e.code} from {url}: {err_body[:600]}",
                status=e.code,
            ) from e
        except urllib.error.URLError as e:
            raise ModelError(f"Failed to reach {url}: {e.reason}") from e

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        import os

        cfg = self._config
        headers = {
            "Content-Type": "application/json",
            "Accept": accept,
            **(self._profile.default_headers if self._profile else {}),
            **cfg.extra_headers,
        }
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        acct = (
            os.environ.get("CHATGPT_ACCOUNT_ID")
            or os.environ.get("OPENAI_ACCOUNT_ID")
            or ""
        ).strip()
        if acct:
            headers.setdefault("ChatGPT-Account-Id", acct)
        # Hermes xAI: x-grok-conv-id from session id when present
        if self._issuer_kind == "xai_responses":
            sid = ""
            if isinstance(cfg.extra, dict):
                sid = str(cfg.extra.get("session_id") or cfg.extra.get("conversation_id") or "")
            if sid:
                headers.setdefault("x-grok-conv-id", sid)
        return headers

    def _build_body(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        cfg = self._config
        wire = _as_dicts(messages)
        if self._profile is not None:
            # Responses still needs strip/require reasoning for any OpenAI-shell
            # fields before conversion (strip is default for codex).
            wire = self._profile.prepare_messages(wire, model=cfg.model)

        instructions, chat_msgs = split_system_instructions(wire)
        input_items = chat_messages_to_responses_input(
            chat_msgs,
            current_issuer_kind=self._issuer_kind,
        )
        body: dict[str, Any] = {
            "model": cfg.model,
            "instructions": instructions or "You are a helpful coding assistant.",
            "input": input_items,
            "store": False,
        }
        resp_tools = responses_tools(tools)
        if resp_tools:
            body["tools"] = resp_tools
            body["parallel_tool_calls"] = True

        temp = temperature if temperature is not None else cfg.temperature
        if temp is not None:
            body["temperature"] = temp
        mt = max_tokens if max_tokens is not None else cfg.max_tokens
        if mt is not None:
            body["max_output_tokens"] = int(mt)

        # Reasoning effort when configured
        if isinstance(cfg.extra, dict):
            rc = cfg.extra.get("reasoning")
            if isinstance(rc, dict) and rc.get("enabled") is not False:
                effort = (rc.get("effort") or "medium").strip().lower()
                if effort in {"low", "medium", "high", "xhigh", "minimal"}:
                    body["reasoning"] = {"effort": effort if effort != "minimal" else "low"}
                    body["include"] = ["reasoning.encrypted_content"]

        # Stable prompt_cache_key from instructions + tools (Hermes)
        pck = content_cache_key(body["instructions"], body.get("tools"))
        if pck:
            body["prompt_cache_key"] = pck

        return body


# ── URL / issuer ─────────────────────────────────────────────────────


def responses_url(base: str) -> str:
    b = base.rstrip("/")
    if b.endswith("/responses"):
        return b
    if b.endswith("/v1"):
        return f"{b}/responses"
    return f"{b}/v1/responses"


def classify_responses_issuer(
    *,
    base_url: str | None = None,
    provider: str | None = None,
) -> str:
    host = (urlparse(base_url or "").hostname or "").lower()
    prov = (provider or "").lower()
    if "x.ai" in host or prov in {"xai", "grok"}:
        return "xai_responses"
    if "github" in host or "copilot" in host:
        return "github_responses"
    if "openai.com" in host or "chatgpt.com" in host or prov in {"codex", "openai-codex"}:
        return "codex_backend"
    if base_url:
        return f"other:{base_url}"
    return "other"


def content_cache_key(instructions: str, tools: list[dict[str, Any]] | None) -> str | None:
    if not instructions and not tools:
        return None
    tools_part = ""
    if tools:
        sorted_tools = sorted(
            (t for t in tools if isinstance(t, dict)),
            key=lambda t: str(t.get("name") or t.get("type") or ""),
        )
        tools_part = json.dumps(
            sorted_tools, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
    content = f"{instructions or ''}\x00{tools_part}"
    digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"pck_{digest}"


# ── conversion ───────────────────────────────────────────────────────


def split_system_instructions(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    rest: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                parts.append(c)
            elif isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text":
                        t = p.get("text")
                        if isinstance(t, str) and t.strip():
                            parts.append(t)
        else:
            rest.append(m)
    return "\n\n".join(parts), rest


def responses_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in tools:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if item.get("type") == "function" or "function" in item else item
        if not isinstance(fn, dict):
            continue
        name = fn.get("name") or item.get("name")
        if not isinstance(name, str) or not name.strip() or name in seen:
            continue
        seen.add(name)
        converted.append(
            {
                "type": "function",
                "name": name.strip(),
                "description": str(fn.get("description") or item.get("description") or ""),
                "strict": False,
                "parameters": fn.get("parameters")
                or item.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return converted or None


def chat_messages_to_responses_input(
    messages: list[dict[str, Any]],
    *,
    current_issuer_kind: str | None = None,
    replay_encrypted_reasoning: bool = True,
) -> list[dict[str, Any]]:
    """OpenAI chat messages → Responses ``input`` items (Hermes core)."""
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            continue

        if role in {"user", "assistant"}:
            content = msg.get("content", "")
            content_text = content if isinstance(content, str) else (
                "" if content is None else str(content)
            )
            if isinstance(content, list):
                parts = _content_to_parts(content, role=str(role))
                text_type = "output_text" if role == "assistant" else "input_text"
                content_text = "".join(
                    p.get("text", "") for p in parts if p.get("type") == text_type
                )
            else:
                parts = []

            if role == "assistant":
                # Replay encrypted reasoning (issuer-filtered)
                if replay_encrypted_reasoning:
                    pdata = msg.get("provider_data") if isinstance(msg.get("provider_data"), dict) else {}
                    reasoning_items = (
                        msg.get("codex_reasoning_items")
                        or pdata.get("codex_reasoning_items")
                    )
                    has_reasoning = False
                    if isinstance(reasoning_items, list):
                        for ri in reasoning_items:
                            if not isinstance(ri, dict) or not ri.get("encrypted_content"):
                                continue
                            item_id = ri.get("id")
                            if item_id and item_id in seen_ids:
                                continue
                            item_issuer = ri.get("_issuer_kind")
                            if (
                                current_issuer_kind is not None
                                and item_issuer is not None
                                and item_issuer != current_issuer_kind
                            ):
                                logger.debug(
                                    "drop cross-issuer reasoning %s vs %s",
                                    item_issuer,
                                    current_issuer_kind,
                                )
                                continue
                            replay = {
                                k: v
                                for k, v in ri.items()
                                if k not in {"id", "_issuer_kind"}
                            }
                            items.append(replay)
                            if item_id:
                                seen_ids.add(str(item_id))
                            has_reasoning = True

                    msg_items = (
                        msg.get("codex_message_items")
                        or pdata.get("codex_message_items")
                    )
                    replayed = 0
                    if isinstance(msg_items, list):
                        for raw in msg_items:
                            if not isinstance(raw, dict):
                                continue
                            if raw.get("type") != "message" or raw.get("role") != "assistant":
                                continue
                            raw_parts = raw.get("content")
                            if not isinstance(raw_parts, list):
                                continue
                            norm_parts = []
                            for part in raw_parts:
                                if not isinstance(part, dict):
                                    continue
                                if str(part.get("type") or "") not in {"output_text", "text"}:
                                    continue
                                text = part.get("text", "")
                                if not isinstance(text, str):
                                    text = str(text) if text is not None else ""
                                norm_parts.append({"type": "output_text", "text": text})
                            if not norm_parts:
                                continue
                            ri = {
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": norm_parts,
                            }
                            iid = raw.get("id")
                            if isinstance(iid, str) and iid.strip() and len(iid) <= _MAX_ITEM_ID_LEN:
                                ri["id"] = iid.strip()
                            phase = raw.get("phase")
                            if isinstance(phase, str) and phase.strip():
                                ri["phase"] = phase.strip()
                            items.append(ri)
                            replayed += 1

                    if replayed == 0:
                        if parts:
                            items.append({"role": "assistant", "content": parts})
                        elif content_text.strip():
                            items.append({"role": "assistant", "content": content_text})
                        elif has_reasoning:
                            items.append({"role": "assistant", "content": ""})
                else:
                    if parts:
                        items.append({"role": "assistant", "content": parts})
                    elif content_text.strip():
                        items.append({"role": "assistant", "content": content_text})

                # function_call items
                for tc in msg.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    if not isinstance(name, str) or not name.strip():
                        continue
                    call_id = tc.get("call_id") or tc.get("id") or f"call_{name}"
                    if not isinstance(call_id, str) or not call_id.strip():
                        call_id = f"call_{name}"
                    args = fn.get("arguments", "{}")
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    elif not isinstance(args, str):
                        args = str(args)
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": str(call_id).strip(),
                            "name": name.strip(),
                            "arguments": (args or "{}").strip() or "{}",
                        }
                    )
                continue

            # user
            if parts:
                items.append({"role": "user", "content": parts})
            else:
                items.append({"role": "user", "content": content_text})
            continue

        if role == "tool":
            call_id = msg.get("tool_call_id") or msg.get("call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                continue
            tool_content = msg.get("content")
            if isinstance(tool_content, list):
                converted = _content_to_parts(tool_content, role="user")
                output_value: Any = converted if converted else ""
            else:
                output_value = str(tool_content or "")
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id.strip(),
                    "output": output_value,
                }
            )

    return items


def _content_to_parts(content: list[Any], *, role: str) -> list[dict[str, Any]]:
    text_type = "output_text" if role == "assistant" else "input_text"
    out: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str) and part:
            out.append({"type": text_type, "text": part})
            continue
        if not isinstance(part, dict):
            continue
        ptype = str(part.get("type") or "").lower()
        if ptype in {"text", "input_text", "output_text"}:
            t = part.get("text")
            if isinstance(t, str) and t:
                out.append({"type": text_type, "text": t})
        elif ptype in {"image_url", "input_image"}:
            ref = part.get("image_url")
            url = ref.get("url") if isinstance(ref, dict) else ref
            if isinstance(url, str) and url:
                out.append({"type": "input_image", "image_url": url})
    return out


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
                # hoist for converter convenience
                for k in ("codex_reasoning_items", "codex_message_items"):
                    if k in m.provider_data:
                        d[k] = m.provider_data[k]
            out.append(d)
        else:
            out.append(dict(m))
    return out


# ── normalize response ───────────────────────────────────────────────


def _consume_responses_sse(
    resp: Any,
    *,
    model: str,
    issuer_kind: str | None,
    on_delta: Any | None,
) -> CompletionResult:
    """Parse OpenAI Responses SSE; assemble a final payload for normalize."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    reasoning_items: list[dict[str, Any]] = []
    message_items: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    model_name = model
    status = "completed"
    # Accumulate function call args by item_id
    fc_buf: dict[str, dict[str, Any]] = {}

    while True:
        line = resp.readline()
        if not line:
            break
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            continue
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            break
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        et = str(data.get("type") or "")
        if et == "response.created" or et == "response.in_progress":
            resp_obj = data.get("response") or {}
            if resp_obj.get("model"):
                model_name = str(resp_obj["model"])
        elif et in {
            "response.output_text.delta",
            "response.output_text.delta.delta",  # defensive
        }:
            piece = data.get("delta") or data.get("text") or ""
            if isinstance(piece, str) and piece:
                text_parts.append(piece)
                if callable(on_delta):
                    try:
                        on_delta(piece)
                    except Exception:  # noqa: BLE001
                        logger.debug("on_delta failed", exc_info=True)
        elif et == "response.function_call_arguments.delta":
            item_id = str(data.get("item_id") or data.get("output_index") or "0")
            slot = fc_buf.setdefault(
                item_id,
                {"name": data.get("name") or "", "arguments": "", "call_id": data.get("call_id") or item_id},
            )
            if data.get("name"):
                slot["name"] = data["name"]
            if data.get("call_id"):
                slot["call_id"] = data["call_id"]
            piece = data.get("delta") or ""
            if isinstance(piece, str):
                slot["arguments"] += piece
        elif et == "response.function_call_arguments.done":
            item_id = str(data.get("item_id") or data.get("output_index") or "0")
            slot = fc_buf.setdefault(
                item_id,
                {
                    "name": data.get("name") or "",
                    "arguments": data.get("arguments") or "",
                    "call_id": data.get("call_id") or item_id,
                },
            )
            if data.get("arguments"):
                slot["arguments"] = data["arguments"]
            if data.get("name"):
                slot["name"] = data["name"]
        elif et == "response.output_item.done":
            item = data.get("item") or {}
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "function_call":
                tool_calls.append(
                    {
                        "id": str(item.get("call_id") or item.get("id") or ""),
                        "call_id": str(item.get("call_id") or item.get("id") or ""),
                        "type": "function",
                        "function": {
                            "name": str(item.get("name") or ""),
                            "arguments": str(item.get("arguments") or "{}"),
                        },
                    }
                )
            elif itype == "reasoning":
                ri = dict(item)
                if issuer_kind:
                    ri["_issuer_kind"] = issuer_kind
                reasoning_items.append(ri)
            elif itype == "message":
                message_items.append(dict(item))
                # also pull final text if no deltas
                for p in item.get("content") or []:
                    if isinstance(p, dict) and p.get("type") in {"output_text", "text"}:
                        t = p.get("text")
                        if isinstance(t, str) and t and t not in "".join(text_parts):
                            # only append if not already streamed
                            pass
        elif et == "response.completed":
            resp_obj = data.get("response") or {}
            status = str(resp_obj.get("status") or "completed")
            if isinstance(resp_obj.get("usage"), dict):
                usage = dict(resp_obj["usage"])
            # prefer full response if present
            if isinstance(resp_obj.get("output"), list) and resp_obj["output"]:
                return normalize_responses_payload(
                    resp_obj,
                    model=model_name,
                    issuer_kind=issuer_kind,
                )
        elif et == "error":
            raise ModelError(f"Responses stream error: {data!r}"[:400])

    # Assemble from deltas if no full response.completed payload
    if not tool_calls and fc_buf:
        for slot in fc_buf.values():
            tool_calls.append(
                {
                    "id": str(slot.get("call_id") or ""),
                    "call_id": str(slot.get("call_id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(slot.get("name") or ""),
                        "arguments": str(slot.get("arguments") or "{}"),
                    },
                }
            )

    text = scrub_model_content("".join(text_parts) or None)
    if not text and not tool_calls:
        # empty stream — still return structure
        pass
    finish = "tool_calls" if tool_calls else "stop"
    if status == "incomplete":
        finish = "length"
    pdata: dict[str, Any] = {}
    if reasoning_items:
        pdata["codex_reasoning_items"] = reasoning_items
    if message_items:
        pdata["codex_message_items"] = message_items
    if issuer_kind:
        pdata["responses_issuer_kind"] = issuer_kind
    return CompletionResult(
        content=text,
        model=model_name,
        finish_reason=finish,
        tool_calls=tool_calls,
        raw={"stream": True, "status": status},
        usage=normalize_openai_usage(usage),
        reasoning_content=None,
        provider_data=pdata or None,
    )


def normalize_responses_payload(
    payload: dict[str, Any],
    *,
    model: str,
    issuer_kind: str | None = None,
) -> CompletionResult:
    status = str(payload.get("status") or "").strip().lower()
    if status in {"failed", "cancelled"}:
        err = payload.get("error") or {}
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise ModelError(f"Responses API {status}: {msg or payload!r}"[:400])

    output = payload.get("output")
    if not isinstance(output, list) or not output:
        out_text = payload.get("output_text")
        if isinstance(out_text, str) and out_text.strip():
            output = [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": out_text.strip()}],
                }
            ]
        else:
            raise ModelError("Responses API returned no output items")

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_items: list[dict[str, Any]] = []
    message_items: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []

    for item in output:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            phase = str(item.get("phase") or "").lower()
            text = _extract_message_text(item)
            if text:
                if phase in {"commentary", "analysis"}:
                    reasoning_parts.append(text)
                else:
                    content_parts.append(text)
                mi: dict[str, Any] = {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text}],
                }
                if isinstance(item.get("id"), str) and item["id"]:
                    mi["id"] = item["id"]
                if phase:
                    mi["phase"] = phase
                message_items.append(mi)
        elif itype == "reasoning":
            rtext = _extract_reasoning_text(item)
            if rtext:
                reasoning_parts.append(rtext)
            # keep encrypted blob for replay
            enc = item.get("encrypted_content")
            if enc or rtext:
                ri = {k: v for k, v in item.items() if k != "_issuer_kind"}
                if issuer_kind:
                    ri["_issuer_kind"] = issuer_kind
                reasoning_items.append(ri)
        elif itype == "function_call":
            name = item.get("name") or ""
            call_id = item.get("call_id") or item.get("id") or f"call_{name}"
            args = item.get("arguments") or "{}"
            if isinstance(args, dict):
                args = json.dumps(args)
            tool_calls.append(
                {
                    "id": str(call_id),
                    "call_id": str(call_id),
                    "type": "function",
                    "function": {
                        "name": str(name),
                        "arguments": str(args),
                    },
                }
            )

    text = scrub_model_content("\n".join(content_parts) if content_parts else None)
    # scrub leaked tool markup
    if text and _TOOL_CALL_LEAK.search(text):
        text = scrub_model_content(_TOOL_CALL_LEAK.sub("", text))

    finish = "tool_calls" if tool_calls else "stop"
    if status == "incomplete":
        finish = "length"

    usage = normalize_openai_usage(payload.get("usage") or {})
    # Responses usage shape
    u = payload.get("usage")
    if isinstance(u, dict):
        if "input_tokens" in u:
            usage.setdefault("prompt_tokens", u.get("input_tokens"))
        if "output_tokens" in u:
            usage.setdefault("completion_tokens", u.get("output_tokens"))
        details = u.get("input_tokens_details")
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            usage.setdefault("cached_tokens", details.get("cached_tokens"))
            usage.setdefault("cache_read_input_tokens", details.get("cached_tokens"))

    provider_data: dict[str, Any] = {}
    if reasoning_items:
        provider_data["codex_reasoning_items"] = reasoning_items
    if message_items:
        provider_data["codex_message_items"] = message_items
    if issuer_kind:
        provider_data["responses_issuer_kind"] = issuer_kind

    return CompletionResult(
        content=text,
        model=str(payload.get("model") or model),
        finish_reason=finish,
        tool_calls=tool_calls,
        raw=payload,
        usage=usage,
        reasoning_content="\n\n".join(reasoning_parts) if reasoning_parts else None,
        provider_data=provider_data or None,
    )


def _extract_message_text(item: dict[str, Any]) -> str:
    parts = item.get("content")
    if not isinstance(parts, list):
        return ""
    texts: list[str] = []
    for p in parts:
        if isinstance(p, dict) and p.get("type") in {"output_text", "text"}:
            t = p.get("text")
            if isinstance(t, str) and t:
                texts.append(t)
    return "".join(texts)


def _extract_reasoning_text(item: dict[str, Any]) -> str:
    summary = item.get("summary")
    if isinstance(summary, list):
        texts = []
        for p in summary:
            if isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str) and t:
                    texts.append(t)
        return "\n".join(texts)
    if isinstance(summary, str):
        return summary
    return ""
