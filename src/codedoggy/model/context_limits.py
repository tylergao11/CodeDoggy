"""Resolve context window for the *active* model — not a blind 32k default.

Order:
  1. Explicit value (env / ModelConfig already set by caller)
  2. Known model-id table (flagships + common local tags)
  3. Provider-level default
  4. Live probe (Ollama ``/api/show`` when provider is ollama)
  5. Last-resort DEFAULT_CONTEXT_WINDOW

Connecting a model must never leave the budget on a random fallback when we
already know the provider + model id.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

# urlparse used by ollama root + local-host detection

logger = logging.getLogger(__name__)

# Only when provider+model are unknown. Not a product claim about Qwen.
DEFAULT_CONTEXT_WINDOW = 32_768

# Prefix / exact model id → context tokens (best-known public limits).
# Longer / more specific keys should be checked first via sorted prefix match.
_MODEL_CONTEXT: dict[str, int] = {
    # xAI Grok
    "grok-4.5": 256_000,
    "grok-4": 256_000,
    "grok-4-0709": 256_000,
    "grok-3": 131_072,
    "grok-3-mini": 131_072,
    "grok-2": 131_072,
    "grok-2-vision": 32_768,
    "grok-imagine": 32_768,
    # Anthropic Claude
    "claude-opus-4": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4": 200_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-5": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    # OpenAI / Codex family
    "gpt-5.6": 256_000,
    "gpt-5.5": 256_000,
    "gpt-5.4": 256_000,
    "gpt-5.1": 256_000,
    "gpt-5": 256_000,
    "gpt-4.1": 1_047_576,
    "gpt-4o": 128_000,
    "o4-mini": 200_000,
    "o3": 200_000,
    "o1": 200_000,
    # DeepSeek
    "deepseek-reasoner": 128_000,
    "deepseek-chat": 128_000,
    "deepseek-r1": 128_000,
    # Qwen (local tags often advertise 32k–128k; prefer larger common default)
    "qwen3": 131_072,
    "qwen2.5": 131_072,
    "qwen2.5-coder": 131_072,
    "qwen2": 131_072,
    # Llama / Mistral common Ollama tags
    "llama3.3": 128_000,
    "llama3.2": 128_000,
    "llama3.1": 128_000,
    "llama3": 8_192,
    "mistral": 32_768,
    "mixtral": 32_768,
    # Gemini
    "gemini-2.5": 1_048_576,
    "gemini-2.0": 1_048_576,
    "gemini-1.5": 1_048_576,
}

# When model id is unknown, use a sane provider-wide floor (not 32k for cloud).
_PROVIDER_CONTEXT: dict[str, int] = {
    "grok": 256_000,
    "xai": 256_000,
    "claude": 200_000,
    "anthropic": 200_000,
    "codex": 256_000,
    "openai": 128_000,
    "openai_compat": 128_000,
    "openai-codex": 256_000,
    "deepseek": 128_000,
    "ollama": 131_072,  # modern local defaults; probe overrides when possible
    "custom": 128_000,
    "bedrock": 200_000,
    "vertex": 1_048_576,
    "codex_app_server": 256_000,
}


def resolve_context_window(
    provider: str | None,
    model: str | None,
    *,
    explicit: int | None = None,
    base_url: str | None = None,
    probe: bool = True,
) -> int:
    """Return tokens of context for this connection.

    ``explicit`` is env/caller override (wins when > 0).
    """
    if explicit is not None and int(explicit) > 0:
        return max(1024, int(explicit))

    env = _env_window()
    if env is not None:
        return env

    prov = (provider or "").strip().lower()
    mid = (model or "").strip()

    # Local Ollama: live /api/show beats any static tag table.
    if probe and prov == "ollama":
        probed = probe_ollama_context(base_url or "http://127.0.0.1:11434", mid)
        if probed is not None:
            return probed

    known = lookup_model_context(mid)
    if known is not None:
        return known

    # OpenAI-compat custom base that is actually Ollama
    if probe and base_url and _looks_like_ollama(base_url):
        probed = probe_ollama_context(base_url, mid)
        if probed is not None:
            return probed

    if prov in {"x-ai", "x.ai"}:
        prov = "grok"
    if prov in _PROVIDER_CONTEXT:
        return _PROVIDER_CONTEXT[prov]

    return DEFAULT_CONTEXT_WINDOW


def ensure_model_context_window(config: Any) -> Any:
    """Rewrite ``config.context_window`` from provider+model (unless env forces).

    Never treat an already-stale 32k on the config as authoritative — that was
    the original bug (fallback left on the object after connect).
    """
    from codedoggy.model.types import ModelConfig

    if not isinstance(config, ModelConfig):
        return config

    env = _env_window()
    if env is not None:
        if config.context_window == env:
            return config
        return ModelConfig(
            provider=config.provider,
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
            context_window=env,
            extra_headers=dict(config.extra_headers),
            extra=dict(config.extra),
        )

    resolved = resolve_context_window(
        config.provider,
        config.model,
        base_url=config.base_url,
        probe=True,
    )
    if config.context_window == resolved:
        return config
    return ModelConfig(
        provider=config.provider,
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
        context_window=resolved,
        extra_headers=dict(config.extra_headers),
        extra=dict(config.extra),
    )


def _looks_like_ollama(base_url: str) -> bool:
    raw = (base_url or "").strip()
    if ":11434" in raw:
        return True
    try:
        host = (urlparse(raw if "://" in raw else f"http://{raw}").hostname or "").lower()
    except Exception:  # noqa: BLE001
        return False
    return host in {"127.0.0.1", "localhost", "::1"} or host.endswith(".local")


def lookup_model_context(model: str | None) -> int | None:
    """Match model id / tag against known limits (case-insensitive prefix)."""
    mid = (model or "").strip().lower()
    if not mid:
        return None
    # Strip common size tags for matching: "qwen3:8b" → try full then "qwen3"
    candidates = [mid]
    if ":" in mid:
        candidates.append(mid.split(":", 1)[0])
    if "/" in mid:
        candidates.append(mid.rsplit("/", 1)[-1])
        candidates.append(mid.split(":", 1)[0].rsplit("/", 1)[-1])

    # Exact first
    for c in candidates:
        if c in _MODEL_CONTEXT:
            return _MODEL_CONTEXT[c]

    # Longest prefix key wins
    best_key = ""
    best_val: int | None = None
    for key, val in _MODEL_CONTEXT.items():
        k = key.lower()
        for c in candidates:
            if c.startswith(k) and len(k) > len(best_key):
                best_key = k
                best_val = val
            # also key as prefix of candidate with separators
            if c.startswith(k + "-") or c.startswith(k + ":"):
                if len(k) > len(best_key):
                    best_key = k
                    best_val = val
    return best_val


def probe_ollama_context(base_url: str | None, model: str | None) -> int | None:
    """Best-effort Ollama ``/api/show`` → context_length / num_ctx."""
    name = (model or "").strip()
    if not name:
        return None
    root = _ollama_root(base_url)
    if not root:
        return None
    url = f"{root}/api/show"
    payload = json.dumps({"name": name}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError) as e:
        logger.debug("ollama context probe failed for %s: %s", name, e)
        return None
    except Exception:  # noqa: BLE001
        logger.debug("ollama context probe failed", exc_info=True)
        return None

    return _extract_ollama_context(body)


def _extract_ollama_context(body: dict[str, Any]) -> int | None:
    # model_info: {"qwen3.context_length": 40960, ...}
    info = body.get("model_info") or body.get("modelinfo") or {}
    if isinstance(info, dict):
        for k, v in info.items():
            key = str(k).lower()
            if key.endswith("context_length") or key.endswith(".context_length"):
                n = _as_positive_int(v)
                if n:
                    return n
        for k, v in info.items():
            key = str(k).lower()
            if "context" in key and "length" in key:
                n = _as_positive_int(v)
                if n:
                    return n

    # parameters string often includes "num_ctx                        8192"
    params = body.get("parameters")
    if isinstance(params, str):
        m = re.search(r"num_ctx\s+(\d+)", params, flags=re.I)
        if m:
            n = _as_positive_int(m.group(1))
            if n:
                return n
    if isinstance(params, dict):
        n = _as_positive_int(params.get("num_ctx") or params.get("context_length"))
        if n:
            return n

    details = body.get("details")
    if isinstance(details, dict):
        n = _as_positive_int(details.get("context_length") or details.get("num_ctx"))
        if n:
            return n
    return None


def _ollama_root(base_url: str | None) -> str:
    raw = (base_url or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").strip()
    if not raw:
        return "http://127.0.0.1:11434"
    # Accept openai-compat style .../v1
    if raw.rstrip("/").endswith("/v1"):
        raw = raw.rstrip("/")[:-3]
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    if not parsed.scheme or not parsed.netloc:
        return "http://127.0.0.1:11434"
    return f"{parsed.scheme}://{parsed.netloc}"


def _env_window() -> int | None:
    for key in ("CODEDOGGY_CONTEXT_WINDOW", "CODEDOGGY_CONTEXT_MAX_TOKENS"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        n = _as_positive_int(raw)
        if n:
            return max(1024, n)
    return None


def _as_positive_int(value: Any) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None
