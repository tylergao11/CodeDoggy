"""xAI Imagine API client (tool-layer).

Ported from:
  crates/codegen/xai-grok-tools/src/implementations/grok_build/image_gen/mod.rs
  crates/codegen/xai-grok-tools/src/implementations/grok_build/image_edit/mod.rs

Grok image_gen / image_edit use the real xAI HTTP API:
  POST {base_url}/images/generations
  POST {base_url}/images/edits

Default base: https://api.x.ai/v1
Default model: grok-imagine-image-quality

Config (env, first wins for key):
  CODEDOGGY_IMAGINE_API_KEY | XAI_API_KEY | CODEDOGGY_API_KEY | OPENAI_API_KEY
  CODEDOGGY_IMAGINE_BASE_URL | XAI_BASE_URL  (default https://api.x.ai/v1)
  CODEDOGGY_IMAGINE_MODEL     (default grok-imagine-image-quality)
  CODEDOGGY_IMAGINE_ENABLED   (0/false to force disable)
  CODEDOGGY_IMAGINE_TIMEOUT_S (default 300)
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

# Grok: XAI_IMAGINE_MODEL
DEFAULT_BASE = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-imagine-image-quality"
# Grok: IMAGE_GEN_TIMEOUT_SECS = 300
DEFAULT_TIMEOUT_S = 300.0


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

    @classmethod
    def from_env(cls) -> ImagineConfig:
        flag = os.environ.get("CODEDOGGY_IMAGINE_ENABLED", "1").strip().lower()
        if flag in {"0", "false", "off", "no"}:
            return cls(
                enabled=False,
                base_url=DEFAULT_BASE,
                api_key=None,
                model=DEFAULT_MODEL,
                reason_disabled="CODEDOGGY_IMAGINE_ENABLED is off",
            )
        key = (
            os.environ.get("CODEDOGGY_IMAGINE_API_KEY")
            or os.environ.get("XAI_API_KEY")
            or os.environ.get("CODEDOGGY_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip() or None
        base = (
            os.environ.get("CODEDOGGY_IMAGINE_BASE_URL")
            or os.environ.get("XAI_BASE_URL")
            or DEFAULT_BASE
        ).strip().rstrip("/")
        model = (
            os.environ.get("CODEDOGGY_IMAGINE_MODEL") or DEFAULT_MODEL
        ).strip() or DEFAULT_MODEL
        timeout = DEFAULT_TIMEOUT_S
        raw_t = os.environ.get("CODEDOGGY_IMAGINE_TIMEOUT_S", "").strip()
        if raw_t:
            try:
                timeout = float(raw_t)
            except ValueError:
                pass
        if not key:
            return cls(
                enabled=False,
                base_url=base,
                api_key=None,
                model=model,
                timeout_s=timeout,
                reason_disabled=(
                    "Image generation is not supported: no API key configured. "
                    "Set CODEDOGGY_IMAGINE_API_KEY (or XAI_API_KEY / CODEDOGGY_API_KEY / "
                    "OPENAI_API_KEY) and optionally CODEDOGGY_IMAGINE_BASE_URL "
                    f"(default {DEFAULT_BASE})."
                ),
            )
        return cls(
            enabled=True,
            base_url=base,
            api_key=key,
            model=model,
            timeout_s=timeout,
        )


def generate_image(
    prompt: str,
    *,
    aspect_ratio: str = "auto",
    config: ImagineConfig | None = None,
) -> bytes:
    """Call POST /images/generations; return raw image bytes.

    Grok payload (ImageGenClient::generate):
      model, prompt, n=1, aspect_ratio, resolution="1k", response_format="b64_json"
    """
    cfg = config or ImagineConfig.from_env()
    if not cfg.enabled:
        raise ImagineNotSupported(
            cfg.reason_disabled or "Image generation is not supported on this API."
        )
    url = f"{cfg.base_url.rstrip('/')}/images/generations"
    # Match Grok exactly — always send aspect_ratio (default "auto").
    payload: dict[str, Any] = {
        "model": cfg.model,
        "prompt": prompt,
        "n": 1,
        "aspect_ratio": aspect_ratio or "auto",
        "resolution": "1k",
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
    """Call POST /images/edits; return raw image bytes.

    Grok payload (image_edit::run):
      model, prompt, n=1, resolution="1k", response_format="b64_json"
      single ref  → "image":  {"url": data_url}
      multi refs  → "images": [{"url": ...}, ...] + aspect_ratio
    """
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
    # Grok image_edit hard-codes XAI_IMAGINE_MODEL (quality); CodeDoggy uses
    # the same default, overridable via CODEDOGGY_IMAGINE_MODEL.
    payload: dict[str, Any] = {
        "model": cfg.model or DEFAULT_MODEL,
        "prompt": prompt,
        "n": 1,
        "resolution": "1k",
        "response_format": "b64_json",
    }

    imgs = [{"url": u} for u in image_data_urls]
    if len(imgs) == 1:
        # Single-image edits: API auto-detects aspect ratio; do not send field.
        payload["image"] = imgs[0]
    else:
        payload["images"] = imgs
        payload["aspect_ratio"] = aspect_ratio or "auto"

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
        # Assignment: not_supported on 404/501 (also 405 Method Not Allowed).
        if e.code in {404, 405, 501}:
            raise ImagineNotSupported(
                f"Image generation is not supported by this API endpoint "
                f"({url}, HTTP {e.code}). {err_body}".strip(),
                code="not_supported",
            ) from e
        if e.code in {401, 403}:
            raise ImagineError(
                f"Image API auth failed (HTTP {e.code}). Check API key. {err_body}",
                code="auth_failed",
            ) from e
        # Grok: code "http_failure" with status for other non-success.
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
    """Grok ImageGenResponse: data[0].b64_json (+ a few lenient variants)."""
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
