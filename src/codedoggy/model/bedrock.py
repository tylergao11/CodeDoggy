"""AWS Bedrock Converse API client (Hermes bedrock_adapter).

Optional dependency: ``boto3>=1.34.59``. Conversion helpers work without boto3;
``complete()`` / discovery raise ImportError with install hint when missing.

Features ported from Hermes:
  - OpenAI ↔ Converse message/tool conversion with strict role alternation
  - Image data-URL decoding (bytes, not double-encoded base64)
  - Cached boto3 clients + stale-connection eviction/retry
  - Stream access-denied fallback to non-streaming converse
  - Guardrails, model tool denylist, region resolution
  - Foundation model / inference profile discovery
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any, Callable

from codedoggy.model.openai_compat import ModelError
from codedoggy.model.profile import ProviderProfile
from codedoggy.model.profile_registry import get_profile
from codedoggy.model.types import ChatMessage, CompletionResult, ModelConfig

logger = logging.getLogger(__name__)

_MIN_BOTO3_VERSION = (1, 34, 59)
_bedrock_runtime_client_cache: dict[str, Any] = {}
_bedrock_control_client_cache: dict[str, Any] = {}
_discovery_cache: dict[str, Any] = {}
_DISCOVERY_CACHE_TTL_SECONDS = 3600

_NON_TOOL_CALLING_PATTERNS = (
    "deepseek.r1",
    "deepseek-r1",
    "stability.",
    "cohere.embed",
    "amazon.titan-embed",
    "titan-embed",
)

_STALE_LIB_MODULE_PREFIXES = ("urllib3.", "botocore.", "boto3.")

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
}


# ---------------------------------------------------------------------------
# boto3 client cache
# ---------------------------------------------------------------------------


def _require_boto3() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for Bedrock. Install: pip install 'boto3>=1.34.59'"
        ) from exc
    try:
        version = tuple(int(x) for x in boto3.__version__.split(".")[:3])
    except (AttributeError, ValueError):
        return boto3
    if version < _MIN_BOTO3_VERSION:
        raise RuntimeError(
            f"boto3 {boto3.__version__} does not support converse_stream "
            f"(minimum 1.34.59 required). Upgrade with: pip install --upgrade boto3"
        )
    return boto3


def _get_bedrock_runtime_client(region: str) -> Any:
    if region not in _bedrock_runtime_client_cache:
        boto3 = _require_boto3()
        _bedrock_runtime_client_cache[region] = boto3.client(
            "bedrock-runtime", region_name=region
        )
    return _bedrock_runtime_client_cache[region]


def _get_bedrock_control_client(region: str) -> Any:
    if region not in _bedrock_control_client_cache:
        boto3 = _require_boto3()
        _bedrock_control_client_cache[region] = boto3.client(
            "bedrock", region_name=region
        )
    return _bedrock_control_client_cache[region]


def reset_client_cache() -> None:
    _bedrock_runtime_client_cache.clear()
    _bedrock_control_client_cache.clear()


def invalidate_runtime_client(region: str) -> bool:
    existed = region in _bedrock_runtime_client_cache
    _bedrock_runtime_client_cache.pop(region, None)
    return existed


def reset_discovery_cache() -> None:
    _discovery_cache.clear()


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def _traceback_frames_modules(exc: BaseException):
    tb = getattr(exc, "__traceback__", None)
    while tb is not None:
        frame = tb.tb_frame
        module = frame.f_globals.get("__name__", "")
        yield module or ""
        tb = tb.tb_next


def is_stale_connection_error(exc: BaseException) -> bool:
    try:
        from botocore.exceptions import (
            ConnectionError as BotoConnectionError,
            HTTPClientError,
        )

        botocore_errors: tuple = (BotoConnectionError, HTTPClientError)
    except ImportError:
        botocore_errors = ()
    if botocore_errors and isinstance(exc, botocore_errors):
        return True
    try:
        from urllib3.exceptions import (
            ConnectionError as Urllib3ConnectionError,
            NewConnectionError,
            ProtocolError,
        )

        urllib3_errors = (ProtocolError, NewConnectionError, Urllib3ConnectionError)
    except ImportError:
        urllib3_errors = ()
    if urllib3_errors and isinstance(exc, urllib3_errors):
        return True
    if isinstance(exc, AssertionError):
        for module in _traceback_frames_modules(exc):
            if any(module.startswith(p) for p in _STALE_LIB_MODULE_PREFIXES):
                return True
    return False


def is_streaming_access_denied_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "invokemodelwithresponsestream" not in msg:
        return False
    try:
        from botocore.exceptions import ClientError
    except ImportError:
        ClientError = None  # type: ignore[assignment,misc]
    if ClientError is not None and isinstance(exc, ClientError):
        code = (getattr(exc, "response", None) or {}).get("Error", {}).get("Code", "")
        return code in ("AccessDeniedException", "UnauthorizedException")
    return "not authorized" in msg or "accessdenied" in msg


# ---------------------------------------------------------------------------
# Credentials / region
# ---------------------------------------------------------------------------


def resolve_aws_auth_env_var(env: dict[str, str] | None = None) -> str | None:
    env_map = env if env is not None else os.environ
    if env_map.get("AWS_BEARER_TOKEN_BEDROCK", "").strip():
        return "AWS_BEARER_TOKEN_BEDROCK"
    if (
        env_map.get("AWS_ACCESS_KEY_ID", "").strip()
        and env_map.get("AWS_SECRET_ACCESS_KEY", "").strip()
    ):
        return "AWS_ACCESS_KEY_ID"
    if env_map.get("AWS_PROFILE", "").strip():
        return "AWS_PROFILE"
    if env_map.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "").strip():
        return "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"
    if env_map.get("AWS_WEB_IDENTITY_TOKEN_FILE", "").strip():
        return "AWS_WEB_IDENTITY_TOKEN_FILE"
    try:
        import botocore.session

        creds = botocore.session.get_session().get_credentials()
        if creds is not None:
            frozen = creds.get_frozen_credentials()
            if frozen and frozen.access_key:
                return "iam-role"
    except Exception:
        pass
    return None


def has_aws_credentials(env: dict[str, str] | None = None) -> bool:
    return resolve_aws_auth_env_var(env) is not None


def resolve_bedrock_region(env: dict[str, str] | None = None) -> str:
    env_map = env if env is not None else os.environ
    explicit = (
        env_map.get("AWS_REGION", "").strip()
        or env_map.get("AWS_DEFAULT_REGION", "").strip()
    )
    if explicit:
        return explicit
    try:
        import botocore.session

        region = botocore.session.get_session().get_config_variable("region")
        if region:
            return region
    except Exception:
        pass
    return "us-east-1"


def model_supports_tools(model_id: str) -> bool:
    m = model_id.lower()
    return not any(pat in m for pat in _NON_TOOL_CALLING_PATTERNS)


def is_anthropic_bedrock_model(model_id: str) -> bool:
    model_lower = model_id.lower()
    for prefix in ("us.", "global.", "eu.", "ap.", "jp."):
        if model_lower.startswith(prefix):
            model_lower = model_lower[len(prefix) :]
            break
    return model_lower.startswith("anthropic.claude")


# ---------------------------------------------------------------------------
# Conversion: tools / messages
# ---------------------------------------------------------------------------


def convert_tools_to_converse(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tools:
        return []
    result = []
    for t in tools:
        fn = t.get("function") or t
        if not isinstance(fn, dict):
            continue
        result.append(
            {
                "toolSpec": {
                    "name": fn.get("name") or t.get("name") or "",
                    "description": fn.get("description") or "",
                    "inputSchema": {
                        "json": fn.get("parameters")
                        or {"type": "object", "properties": {}}
                    },
                }
            }
        )
    return result


def _convert_content_to_converse(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return [{"text": " "}]
    if isinstance(content, str):
        return [{"text": content}] if content.strip() else [{"text": " "}]
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                blocks.append({"text": part if part.strip() else " "})
                continue
            if not isinstance(part, dict):
                continue
            part_type = part.get("type", "")
            if part_type == "text":
                text = part.get("text", "")
                blocks.append({"text": text if text else " "})
            elif part_type == "image_url":
                image_url = part.get("image_url", {})
                url = (
                    image_url.get("url", "")
                    if isinstance(image_url, dict)
                    else ""
                )
                if url.startswith("data:"):
                    header, _, data = url.partition(",")
                    media_type = "image/jpeg"
                    if header.startswith("data:"):
                        mime_part = header[5:].split(";")[0]
                        if mime_part:
                            media_type = mime_part
                    try:
                        raw_bytes = base64.b64decode(data)
                    except Exception:
                        raw_bytes = data.encode("utf-8")
                    blocks.append(
                        {
                            "image": {
                                "format": (
                                    media_type.split("/")[-1]
                                    if "/" in media_type
                                    else "jpeg"
                                ),
                                "source": {"bytes": raw_bytes},
                            }
                        }
                    )
                else:
                    blocks.append({"text": f"[Image: {url}]"})
            elif "text" in part:
                blocks.append({"text": str(part.get("text") or " ")})
        return blocks if blocks else [{"text": " "}]
    return [{"text": str(content)}]


def convert_messages_to_converse(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """OpenAI messages → (system blocks | None, converse messages).

    Enforces strict user/assistant alternation by merging consecutive same-role
    messages; ensures first/last message is user (Converse requirement).
    """
    system_blocks: list[dict[str, Any]] = []
    converse_msgs: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "system":
            if isinstance(content, str) and content.strip():
                system_blocks.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_blocks.append({"text": part.get("text", "")})
                    elif isinstance(part, str):
                        system_blocks.append({"text": part})
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            if isinstance(content, str):
                result_content = content
            else:
                result_content = json.dumps(content) if content is not None else " "
            tool_result_block = {
                "toolResult": {
                    "toolUseId": str(tool_call_id or ""),
                    "content": [{"text": result_content if result_content else " "}],
                }
            }
            if converse_msgs and converse_msgs[-1]["role"] == "user":
                converse_msgs[-1]["content"].append(tool_result_block)
            else:
                converse_msgs.append(
                    {"role": "user", "content": [tool_result_block]}
                )
            continue

        if role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            if isinstance(content, str) and content.strip():
                content_blocks.append({"text": content})
            elif isinstance(content, list):
                content_blocks.extend(_convert_content_to_converse(content))

            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                args_str = fn.get("arguments", "{}")
                try:
                    args_dict = (
                        json.loads(args_str)
                        if isinstance(args_str, str)
                        else args_str
                    )
                except (json.JSONDecodeError, TypeError):
                    args_dict = {"_raw": args_str} if args_str else {}
                if not isinstance(args_dict, dict):
                    args_dict = {"value": args_dict}
                content_blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": str(tc.get("id") or ""),
                            "name": str(fn.get("name") or ""),
                            "input": args_dict,
                        }
                    }
                )

            if not content_blocks:
                content_blocks = [{"text": " "}]

            if converse_msgs and converse_msgs[-1]["role"] == "assistant":
                converse_msgs[-1]["content"].extend(content_blocks)
            else:
                converse_msgs.append(
                    {"role": "assistant", "content": content_blocks}
                )
            continue

        if role == "user":
            content_blocks = _convert_content_to_converse(content)
            if converse_msgs and converse_msgs[-1]["role"] == "user":
                converse_msgs[-1]["content"].extend(content_blocks)
            else:
                converse_msgs.append({"role": "user", "content": content_blocks})
            continue

    if converse_msgs and converse_msgs[0]["role"] != "user":
        converse_msgs.insert(0, {"role": "user", "content": [{"text": " "}]})
    if converse_msgs and converse_msgs[-1]["role"] != "user":
        converse_msgs.append({"role": "user", "content": [{"text": " "}]})

    return (system_blocks if system_blocks else None), converse_msgs


def converse_stop_reason_to_openai(stop_reason: str) -> str:
    return _STOP_REASON_MAP.get(stop_reason, "stop")


def normalize_converse_response(
    response: dict[str, Any],
    *,
    model: str = "",
) -> CompletionResult:
    output = response.get("output") or {}
    message = output.get("message") or {}
    content = message.get("content") or []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            text_parts.append(str(block["text"]))
        elif "reasoningContent" in block:
            reasoning = block["reasoningContent"]
            if isinstance(reasoning, dict):
                thinking_text = reasoning.get("text", "")
                if thinking_text:
                    reasoning_parts.append(str(thinking_text))
        tu = block.get("toolUse")
        if isinstance(tu, dict):
            tool_calls.append(
                {
                    "id": str(tu.get("toolUseId") or ""),
                    "type": "function",
                    "function": {
                        "name": str(tu.get("name") or ""),
                        "arguments": json.dumps(tu.get("input") or {}),
                    },
                }
            )
    stop = response.get("stopReason") or "end_turn"
    finish = converse_stop_reason_to_openai(str(stop))
    if tool_calls and finish == "stop":
        finish = "tool_calls"
    usage_raw = response.get("usage") or {}
    usage = {
        "prompt_tokens": usage_raw.get("inputTokens"),
        "completion_tokens": usage_raw.get("outputTokens"),
        "total_tokens": usage_raw.get("totalTokens")
        or (
            (usage_raw.get("inputTokens") or 0)
            + (usage_raw.get("outputTokens") or 0)
        ),
    }
    return CompletionResult(
        content="\n".join(text_parts) if text_parts else None,
        model=model,
        finish_reason=finish,
        tool_calls=tool_calls,
        raw=response if isinstance(response, dict) else {"raw": str(response)},
        usage={k: v for k, v in usage.items() if v is not None},
        reasoning_content="\n\n".join(reasoning_parts) if reasoning_parts else None,
    )


def stream_converse_to_result(
    event_stream: dict[str, Any] | Any,
    *,
    model: str = "",
    on_text_delta: Callable[[str], None] | None = None,
    on_tool_start: Callable[[str], None] | None = None,
    on_reasoning_delta: Callable[[str], None] | None = None,
    on_interrupt_check: Callable[[], bool] | None = None,
) -> CompletionResult:
    """Consume Bedrock ConverseStream events into CompletionResult."""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    current_tool: dict[str, Any] | None = None
    current_text_buffer: list[str] = []
    has_tool_use = False
    stop_reason = "end_turn"
    usage_data: dict[str, int] = {}

    stream = (
        event_stream.get("stream", [])
        if isinstance(event_stream, dict)
        else getattr(event_stream, "get", lambda k, d=None: d)("stream", [])
    )
    if not stream and hasattr(event_stream, "__iter__") and not isinstance(
        event_stream, dict
    ):
        stream = event_stream

    for event in stream:
        if on_interrupt_check and on_interrupt_check():
            break
        if not isinstance(event, dict):
            continue

        if "contentBlockStart" in event:
            start = event["contentBlockStart"].get("start", {})
            if "toolUse" in start:
                has_tool_use = True
                if current_text_buffer:
                    text_parts.append("".join(current_text_buffer))
                    current_text_buffer = []
                current_tool = {
                    "toolUseId": start["toolUse"].get("toolUseId", ""),
                    "name": start["toolUse"].get("name", ""),
                    "input_json": "",
                }
                if on_tool_start:
                    on_tool_start(current_tool["name"])

        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                text = delta["text"]
                current_text_buffer.append(text)
                if on_text_delta and not has_tool_use:
                    on_text_delta(text)
            elif "toolUse" in delta and current_tool is not None:
                current_tool["input_json"] += delta["toolUse"].get("input", "")
            elif "reasoningContent" in delta:
                reasoning = delta["reasoningContent"]
                if isinstance(reasoning, dict):
                    thinking_text = reasoning.get("text", "")
                    if thinking_text:
                        reasoning_parts.append(str(thinking_text))
                        if on_reasoning_delta:
                            on_reasoning_delta(thinking_text)

        elif "contentBlockStop" in event:
            if current_tool is not None:
                try:
                    input_dict = (
                        json.loads(current_tool["input_json"])
                        if current_tool["input_json"]
                        else {}
                    )
                except (json.JSONDecodeError, TypeError):
                    input_dict = {}
                tool_calls.append(
                    {
                        "id": str(current_tool["toolUseId"]),
                        "type": "function",
                        "function": {
                            "name": str(current_tool["name"]),
                            "arguments": json.dumps(input_dict),
                        },
                    }
                )
                current_tool = None
            elif current_text_buffer:
                text_parts.append("".join(current_text_buffer))
                current_text_buffer = []

        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason", "end_turn")

        elif "metadata" in event:
            meta_usage = event["metadata"].get("usage", {})
            usage_data = {
                "inputTokens": meta_usage.get("inputTokens", 0),
                "outputTokens": meta_usage.get("outputTokens", 0),
            }

    if current_text_buffer:
        text_parts.append("".join(current_text_buffer))

    finish = converse_stop_reason_to_openai(str(stop_reason))
    if tool_calls and finish == "stop":
        finish = "tool_calls"
    usage = {
        "prompt_tokens": usage_data.get("inputTokens"),
        "completion_tokens": usage_data.get("outputTokens"),
        "total_tokens": (usage_data.get("inputTokens") or 0)
        + (usage_data.get("outputTokens") or 0),
    }
    return CompletionResult(
        content="\n".join(text_parts) if text_parts else None,
        model=model,
        finish_reason=finish,
        tool_calls=tool_calls,
        raw={"streamed": True},
        usage={k: v for k, v in usage.items() if v is not None},
        reasoning_content="\n\n".join(reasoning_parts) if reasoning_parts else None,
    )


def build_converse_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 4096,
    temperature: float | None = None,
    top_p: float | None = None,
    stop_sequences: list[str] | None = None,
    guardrail_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    system_prompt, converse_messages = convert_messages_to_converse(messages)
    kwargs: dict[str, Any] = {
        "modelId": model,
        "messages": converse_messages,
        "inferenceConfig": {"maxTokens": int(max_tokens)},
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if temperature is not None:
        kwargs["inferenceConfig"]["temperature"] = float(temperature)
    if top_p is not None:
        kwargs["inferenceConfig"]["topP"] = float(top_p)
    if stop_sequences:
        kwargs["inferenceConfig"]["stopSequences"] = stop_sequences
    if tools:
        converse_tools = convert_tools_to_converse(tools)
        if converse_tools:
            if model_supports_tools(model):
                kwargs["toolConfig"] = {"tools": converse_tools}
            else:
                logger.warning(
                    "Model %s does not support tool calling — tools stripped.",
                    model,
                )
    if guardrail_config:
        kwargs["guardrailConfig"] = guardrail_config
    return kwargs


def discover_bedrock_models(
    region: str,
    provider_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Discover foundation models + inference profiles (1h cache per region)."""
    cache_key = f"{region}:{','.join(sorted(provider_filter or []))}"
    cached = _discovery_cache.get(cache_key)
    if cached and (time.time() - cached["timestamp"]) < _DISCOVERY_CACHE_TTL_SECONDS:
        return list(cached["models"])

    try:
        client = _get_bedrock_control_client(region)
    except Exception as exc:
        logger.warning("Bedrock discovery client failed: %s", exc)
        return []

    models: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    filter_set = {f.lower() for f in (provider_filter or [])}

    try:
        response = client.list_foundation_models()
        for summary in response.get("modelSummaries", []):
            model_id = (summary.get("modelId") or "").strip()
            if not model_id or model_id in seen_ids:
                continue
            if filter_set:
                provider_name = (summary.get("providerName") or "").lower()
                model_prefix = (
                    model_id.split(".")[0].lower() if "." in model_id else ""
                )
                if provider_name not in filter_set and model_prefix not in filter_set:
                    continue
            lifecycle = summary.get("modelLifecycle") or {}
            if str(lifecycle.get("status", "")).upper() not in {"", "ACTIVE"}:
                continue
            output_mods = summary.get("outputModalities") or []
            if output_mods and "TEXT" not in output_mods:
                continue
            models.append(
                {
                    "id": model_id,
                    "name": (summary.get("modelName") or model_id).strip(),
                    "provider": (summary.get("providerName") or "").strip(),
                    "input_modalities": summary.get("inputModalities") or [],
                    "output_modalities": output_mods,
                    "streaming": bool(
                        summary.get("responseStreamingSupported", False)
                    ),
                }
            )
            seen_ids.add(model_id)
    except Exception as exc:
        logger.warning("list_foundation_models failed: %s", exc)

    # Inference profiles (cross-region)
    try:
        if hasattr(client, "list_inference_profiles"):
            resp = client.list_inference_profiles()
            for profile in resp.get("inferenceProfileSummaries") or []:
                pid = (
                    profile.get("inferenceProfileId")
                    or profile.get("inferenceProfileArn")
                    or ""
                ).strip()
                if not pid or pid in seen_ids:
                    continue
                models.append(
                    {
                        "id": pid,
                        "name": (
                            profile.get("inferenceProfileName") or pid
                        ).strip(),
                        "provider": "inference-profile",
                        "input_modalities": [],
                        "output_modalities": ["TEXT"],
                        "streaming": True,
                    }
                )
                seen_ids.add(pid)
    except Exception as exc:
        logger.debug("list_inference_profiles skipped: %s", exc)

    _discovery_cache[cache_key] = {"timestamp": time.time(), "models": models}
    return list(models)


def bedrock_model_ids_or_none(region: str | None = None) -> list[str] | None:
    try:
        discovered = discover_bedrock_models(region or resolve_bedrock_region())
        if discovered:
            return [m["id"] for m in discovered]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _as_dicts(
    messages: list[ChatMessage] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, ChatMessage):
            d: dict[str, Any] = {"role": m.role}
            if m.content is not None:
                d["content"] = m.content
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


class BedrockConverseClient:
    """Bedrock Converse — OpenAI-shaped messages in, CompletionResult out."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        profile: ProviderProfile | None = None,
        region: str | None = None,
    ) -> None:
        self._config = config
        self._profile = profile or get_profile(config.provider)
        self._region = (
            region
            or (config.extra.get("region") if isinstance(config.extra, dict) else None)
            or resolve_bedrock_region()
        )

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def profile(self) -> ProviderProfile | None:
        return self._profile

    @property
    def region(self) -> str:
        return self._region

    def complete(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResult:
        wire = _as_dicts(messages)
        if self._profile is not None:
            wire = self._profile.prepare_messages(wire, model=self._config.model)
        mt = max_tokens if max_tokens is not None else self._config.max_tokens
        temp = temperature if temperature is not None else self._config.temperature
        guardrail = None
        if isinstance(self._config.extra, dict):
            guardrail = self._config.extra.get("guardrail_config")
        kwargs = build_converse_kwargs(
            model=self._config.model,
            messages=wire,
            tools=tools,
            max_tokens=int(mt or 4096),
            temperature=temp,
            guardrail_config=guardrail,
        )
        return self._call_converse(kwargs)

    def complete_stream(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> CompletionResult:
        wire = _as_dicts(messages)
        if self._profile is not None:
            wire = self._profile.prepare_messages(wire, model=self._config.model)
        mt = max_tokens if max_tokens is not None else self._config.max_tokens
        temp = temperature if temperature is not None else self._config.temperature
        guardrail = None
        if isinstance(self._config.extra, dict):
            guardrail = self._config.extra.get("guardrail_config")
        kwargs = build_converse_kwargs(
            model=self._config.model,
            messages=wire,
            tools=tools,
            max_tokens=int(mt or 4096),
            temperature=temp,
            guardrail_config=guardrail,
        )
        return self._call_converse_stream(kwargs, on_text_delta=on_delta)

    def _call_converse(self, kwargs: dict[str, Any]) -> CompletionResult:
        client = _get_bedrock_runtime_client(self._region)
        try:
            response = client.converse(**kwargs)
        except Exception as exc:
            if is_stale_connection_error(exc):
                logger.warning(
                    "bedrock: stale connection on converse(region=%s): %s — retrying",
                    self._region,
                    type(exc).__name__,
                )
                invalidate_runtime_client(self._region)
                client = _get_bedrock_runtime_client(self._region)
                try:
                    response = client.converse(**kwargs)
                except Exception as exc2:
                    raise ModelError(f"Bedrock converse failed: {exc2}") from exc2
            else:
                raise ModelError(f"Bedrock converse failed: {exc}") from exc
        return normalize_converse_response(response, model=self._config.model)

    def _call_converse_stream(
        self,
        kwargs: dict[str, Any],
        *,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> CompletionResult:
        client = _get_bedrock_runtime_client(self._region)
        try:
            response = client.converse_stream(**kwargs)
        except Exception as exc:
            if is_streaming_access_denied_error(exc):
                logger.info(
                    "bedrock: converse_stream IAM denied — falling back to converse()"
                )
                return self._call_converse(kwargs)
            if is_stale_connection_error(exc):
                invalidate_runtime_client(self._region)
                client = _get_bedrock_runtime_client(self._region)
                try:
                    response = client.converse_stream(**kwargs)
                except Exception as exc2:
                    if is_streaming_access_denied_error(exc2):
                        return self._call_converse(kwargs)
                    raise ModelError(
                        f"Bedrock converse_stream failed: {exc2}"
                    ) from exc2
            else:
                raise ModelError(f"Bedrock converse_stream failed: {exc}") from exc
        return stream_converse_to_result(
            response,
            model=self._config.model,
            on_text_delta=on_text_delta,
        )


# Back-compat private name used by older code
def _bedrock_runtime(region: str) -> Any:
    return _get_bedrock_runtime_client(region)


def _model_supports_tools(model_id: str) -> bool:
    return model_supports_tools(model_id)
