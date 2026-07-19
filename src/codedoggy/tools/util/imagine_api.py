"""Image generation HTTP client (tool-layer).

Ported from:
  crates/codegen/xai-grok-tools/src/implementations/grok_build/image_gen/mod.rs
  crates/codegen/xai-grok-tools/src/implementations/grok_build/image_edit/mod.rs

Wire:
  POST {base_url}/images/generations
  POST {base_url}/images/edits

Auth product rule
-----------------
Config follows the session **ActiveConnection** (Ctrl+L / login wizard pick).
Tools never steal another provider's env key or OAuth session.

Resolution order for ``ImagineConfig.resolve(extra)``:
  1. ``extra['imagine_config']`` (test/host override)
  2. ``extra['connection']`` / ``kernel.connection`` → ``from_connection``
  3. ``from_env`` (CLI / unit tests without a session)

Payload family (endpoint-derived via ``image_api_family``):
  - xai     → Grok Imagine fields (aspect_ratio, resolution)
  - openai  → OpenAI Images fields (size)
  - unsupported → disabled with a clear reason (no silent Grok fallback)
"""

from __future__ import annotations

import base64
import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from codedoggy.tools.util.active_auth import (
    ImageApiFamily,
    connection_fields,
    connection_from_extra,
    image_api_family,
    is_xai_endpoint,
    resolve_base_for_provider,
    resolve_provider_token,
    unsupported_image_reason,
)

DEFAULT_XAI_BASE = "https://api.x.ai/v1"
DEFAULT_XAI_IMAGE_MODEL = "grok-imagine-image-quality"
DEFAULT_OPENAI_COMPAT_IMAGE_MODEL = "gpt-image-1"
# Back-compat names used by tests / older imports
DEFAULT_BASE = DEFAULT_XAI_BASE
DEFAULT_MODEL = DEFAULT_XAI_IMAGE_MODEL
DEFAULT_TIMEOUT_S = 300.0

# OpenAI Images size map from aspect_ratio hints.
_ASPECT_TO_OPENAI_SIZE: dict[str, str] = {
    "auto": "1024x1024",
    "1:1": "1024x1024",
    "16:9": "1792x1024",
    "9:16": "1024x1792",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
    "4:3": "1536x1024",
    "3:4": "1024x1536",
}


class ImagineNotSupported(Exception):
    """API missing, misconfigured, or endpoint does not support image gen."""

    def __init__(self, message: str, *, code: str = "not_supported") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ImagineError(Exception):
    def __init__(self, message: str, *, code: str = "image_gen_error") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ImagineConfig:
    enabled: bool
    base_url: str
    api_key: str | None
    model: str
    timeout_s: float = DEFAULT_TIMEOUT_S
    reason_disabled: str = ""
    provider: str = ""
    family: ImageApiFamily = "unsupported"

    @classmethod
    def resolve(cls, extra: dict[str, Any] | None = None) -> ImagineConfig:
        """Prefer session connection; fall back to env for headless/tests."""
        bag = extra or {}
        override = bag.get("imagine_config")
        if isinstance(override, ImagineConfig):
            return override
        conn = connection_from_extra(bag)
        if conn is not None:
            return cls.from_connection(conn)
        return cls.from_env()

    @classmethod
    def from_connection(cls, connection: Any) -> ImagineConfig:
        """Bind image API to the user's selected login/API connection only."""
        if _flag_off("CODEDOGGY_IMAGINE_ENABLED"):
            return cls._disabled("CODEDOGGY_IMAGINE_ENABLED is off")

        provider, conn_base, _chat_model = connection_fields(connection)
        if not provider:
            return cls._disabled(
                "Image generation has no active connection. "
                "Log in via Ctrl+L and select a provider first."
            )

        base = resolve_base_for_provider(provider, conn_base)
        # Explicit media base only — never XAI_BASE_URL (chat profile env).
        media_base = (os.environ.get("CODEDOGGY_IMAGINE_BASE_URL") or "").strip().rstrip("/")
        if media_base:
            base = media_base

        if not base:
            return cls(
                enabled=False,
                base_url="",
                api_key=None,
                model="",
                timeout_s=_timeout_from_env(),
                provider=provider,
                family="unsupported",
                reason_disabled=(
                    f"Image generation follows connection ({provider}) but has no "
                    f"base_url. Set the provider endpoint or CODEDOGGY_IMAGINE_BASE_URL."
                ),
            )

        family = image_api_family(base, provider=provider)
        model = _image_model_for_family(family, base)
        if family == "unsupported":
            return cls(
                enabled=False,
                base_url=base,
                api_key=None,
                model=model,
                timeout_s=_timeout_from_env(),
                provider=provider,
                family=family,
                reason_disabled=unsupported_image_reason(provider, base),
            )

        key, key_source = resolve_provider_token(provider)
        if not key:
            return cls(
                enabled=False,
                base_url=base,
                api_key=None,
                model=model,
                timeout_s=_timeout_from_env(),
                provider=provider,
                family=family,
                reason_disabled=(
                    f"Image generation follows your active connection ({provider}), "
                    f"but that login has no usable credential. Re-auth via Ctrl+L "
                    f"or set the API key for {provider}."
                ),
            )

        return cls(
            enabled=True,
            base_url=base,
            api_key=key,
            model=model,
            timeout_s=_timeout_from_env(),
            provider=provider,
            family=family,
            reason_disabled=f"auth={key_source} provider={provider} family={family}",
        )

    @classmethod
    def from_env(cls) -> ImagineConfig:
        """Headless / test path when no session connection is injected.

        Credential order (never steals another provider's key when provider is set):
          1. CODEDOGGY_IMAGINE_API_KEY (dedicated media override)
          2. resolve_credential(CODEDOGGY_PROVIDER) when provider is set
        """
        if _flag_off("CODEDOGGY_IMAGINE_ENABLED"):
            return cls._disabled("CODEDOGGY_IMAGINE_ENABLED is off")

        provider = (
            os.environ.get("CODEDOGGY_PROVIDER")
            or os.environ.get("CODEDOGGY_MODEL_PROVIDER")
            or ""
        ).strip().lower()

        key = (os.environ.get("CODEDOGGY_IMAGINE_API_KEY") or "").strip() or None
        key_source = "env:CODEDOGGY_IMAGINE_API_KEY" if key else ""
        if not key and provider:
            key, key_source = resolve_provider_token(provider)

        base = (os.environ.get("CODEDOGGY_IMAGINE_BASE_URL") or "").strip().rstrip("/")
        if not base and provider:
            base = resolve_base_for_provider(provider, "")
        # Headless tests often set only the dedicated media key + optional base.
        if not base and key and not provider:
            base = DEFAULT_XAI_BASE

        family: ImageApiFamily = (
            image_api_family(base, provider=provider) if base else "unsupported"
        )
        # Dedicated IMAGINE key + base always treated as intentional media call.
        if key and base and key_source.startswith("env:CODEDOGGY_IMAGINE"):
            if family == "unsupported":
                family = "xai" if is_xai_endpoint(base) else "openai"

        model = _image_model_for_family(family, base)
        timeout = _timeout_from_env()

        if not key:
            return cls(
                enabled=False,
                base_url=base or "",
                api_key=None,
                model=model,
                timeout_s=timeout,
                provider=provider,
                family=family,
                reason_disabled=(
                    "Image generation is not supported: no API key or login for the "
                    "active provider. Log in via Ctrl+L (same connection as chat) or "
                    "set CODEDOGGY_IMAGINE_API_KEY, or set CODEDOGGY_PROVIDER and log in."
                ),
            )
        if not base:
            return cls(
                enabled=False,
                base_url="",
                api_key=None,
                model=model,
                timeout_s=timeout,
                provider=provider,
                family="unsupported",
                reason_disabled=(
                    "Image generation has a credential but no base_url. "
                    "Set CODEDOGGY_IMAGINE_BASE_URL or log in so the connection "
                    "supplies the provider endpoint."
                ),
            )
        if family == "unsupported":
            return cls(
                enabled=False,
                base_url=base,
                api_key=None,
                model=model,
                timeout_s=timeout,
                provider=provider,
                family=family,
                reason_disabled=unsupported_image_reason(provider, base),
            )
        return cls(
            enabled=True,
            base_url=base,
            api_key=key,
            model=model,
            timeout_s=timeout,
            provider=provider,
            family=family,
            reason_disabled=f"auth={key_source}" if key_source else "",
        )

    @classmethod
    def _disabled(cls, reason: str) -> ImagineConfig:
        return cls(
            enabled=False,
            base_url="",
            api_key=None,
            model=DEFAULT_XAI_IMAGE_MODEL,
            reason_disabled=reason,
            family="unsupported",
        )


def _flag_off(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() in {"0", "false", "off", "no"}


def _timeout_from_env() -> float:
    timeout = DEFAULT_TIMEOUT_S
    raw_t = os.environ.get("CODEDOGGY_IMAGINE_TIMEOUT_S", "").strip()
    if raw_t:
        try:
            timeout = float(raw_t)
        except ValueError:
            pass
    return timeout


def _image_model_for_family(family: ImageApiFamily, base_url: str) -> str:
    explicit = (os.environ.get("CODEDOGGY_IMAGINE_MODEL") or "").strip()
    if explicit:
        return explicit
    if family == "xai" or is_xai_endpoint(base_url):
        return DEFAULT_XAI_IMAGE_MODEL
    if family == "openai":
        return DEFAULT_OPENAI_COMPAT_IMAGE_MODEL
    return DEFAULT_XAI_IMAGE_MODEL


def _openai_size(aspect_ratio: str) -> str:
    key = (aspect_ratio or "auto").strip()
    return _ASPECT_TO_OPENAI_SIZE.get(key, "1024x1024")


def generate_image(
    prompt: str,
    *,
    aspect_ratio: str = "auto",
    config: ImagineConfig | None = None,
) -> bytes:
    """Call POST /images/generations; return raw image bytes."""
    cfg = config or ImagineConfig.from_env()
    if not cfg.enabled:
        raise ImagineNotSupported(
            cfg.reason_disabled or "Image generation is not supported on this API."
        )
    url = f"{cfg.base_url.rstrip('/')}/images/generations"
    family = cfg.family if cfg.family != "unsupported" else image_api_family(
        cfg.base_url, provider=cfg.provider
    )
    if family == "xai":
        payload: dict[str, Any] = {
            "model": cfg.model,
            "prompt": prompt,
            "n": 1,
            "aspect_ratio": aspect_ratio or "auto",
            "resolution": "1k",
            "response_format": "b64_json",
        }
    else:
        # OpenAI Images / OpenAI-compat proxies
        payload = {
            "model": cfg.model,
            "prompt": prompt,
            "n": 1,
            "size": _openai_size(aspect_ratio or "auto"),
            "response_format": "b64_json",
        }

    body = _post_json(url, payload, api_key=cfg.api_key or "", timeout_s=cfg.timeout_s)
    return _extract_b64_image(body, empty_msg="Image generation returned no image data.")


def edit_image(
    prompt: str,
    image_data_urls: list[str],
    *,
    aspect_ratio: str = "auto",
    config: ImagineConfig | None = None,
) -> bytes:
    """Call POST /images/edits; return raw image bytes."""
    cfg = config or ImagineConfig.from_env()
    if not cfg.enabled:
        raise ImagineNotSupported(
            cfg.reason_disabled or "Image edit is not supported on this API."
        )
    if not image_data_urls:
        raise ImagineError(
            "image_edit requires at least one reference image. "
            "Use image_gen for text-only generation.",
            code="invalid_arguments",
        )

    url = f"{cfg.base_url.rstrip('/')}/images/edits"
    family = cfg.family if cfg.family != "unsupported" else image_api_family(
        cfg.base_url, provider=cfg.provider
    )

    if family == "xai":
        payload: dict[str, Any] = {
            "model": cfg.model or DEFAULT_XAI_IMAGE_MODEL,
            "prompt": prompt,
            "n": 1,
            "resolution": "1k",
            "response_format": "b64_json",
        }
        imgs = [{"url": u} for u in image_data_urls]
        if len(imgs) == 1:
            payload["image"] = imgs[0]
        else:
            payload["images"] = imgs
            payload["aspect_ratio"] = aspect_ratio or "auto"
    else:
        # OpenAI edits: single image as data URL field; multi-image is xAI-only shape.
        payload = {
            "model": cfg.model or DEFAULT_OPENAI_COMPAT_IMAGE_MODEL,
            "prompt": prompt,
            "n": 1,
            "size": _openai_size(aspect_ratio or "auto"),
            "response_format": "b64_json",
            "image": image_data_urls[0],
        }

    body = _post_json(url, payload, api_key=cfg.api_key or "", timeout_s=cfg.timeout_s)
    return _extract_b64_image(body, empty_msg="Image edit returned no image data.")


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout_s: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "CodeDoggy-imagine/0.1",
            "Accept": "application/json",
        },
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:  # noqa: BLE001
            pass
        if e.code in {404, 405, 501}:
            raise ImagineNotSupported(
                f"Image generation is not supported by this API endpoint "
                f"({url}, HTTP {e.code}). {err_body}".strip(),
                code="not_supported",
            ) from e
        if e.code in {401, 403}:
            raise ImagineError(
                f"Image API auth failed (HTTP {e.code}). "
                f"Check the credential for the active connection. {err_body}",
                code="auth_failed",
            ) from e
        raise ImagineError(
            f"Image generation failed with HTTP {e.code}: {err_body or e.reason}",
            code="http_failure",
        ) from e
    except urllib.error.URLError as e:
        raise ImagineError(
            f"Image generation API request failed: {e.reason}",
            code="network_error",
        ) from e
    except TimeoutError as e:
        raise ImagineError("Image API request timed out", code="timeout") from e

    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        preview = raw[:500].decode("utf-8", errors="replace")
        raise ImagineError(
            f"Failed to parse image generation response: {e} — body preview: {preview}",
            code="bad_response",
        ) from e


def _extract_b64_image(body: dict[str, Any], *, empty_msg: str) -> bytes:
    """Grok/OpenAI ImageGenResponse: data[0].b64_json (+ lenient variants)."""
    data = body.get("data")
    b64 = ""
    if isinstance(data, list) and data:
        item = data[0] if isinstance(data[0], dict) else {}
        b64 = (
            item.get("b64_json")
            or item.get("b64")
            or item.get("image")
            or ""
        )
        if not b64 and isinstance(item.get("url"), str) and item["url"].startswith("data:"):
            part = item["url"].split(",", 1)
            if len(part) == 2:
                b64 = part[1]
    if not b64:
        b64 = body.get("b64_json") or body.get("image") or ""
    if not b64:
        raise ImagineError(empty_msg, code="empty_image")
    try:
        return base64.b64decode(b64)
    except Exception as e:  # noqa: BLE001
        raise ImagineError(f"Failed to decode base64 image data: {e}", code="bad_b64") from e
