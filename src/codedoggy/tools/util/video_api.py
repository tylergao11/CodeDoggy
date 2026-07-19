"""xAI Video Generation API client (tool-layer).

Ported from:
  grok-build/crates/codegen/xai-grok-tools/src/implementations/grok_build/video_gen/mod.rs
    VideoGenClient::generate_with_images, download_video
    constants, GenerateVideoPayload, VideoGenStartResponse, VideoGenPollResponse
    validate_imagine_duration, validate_one_of

Function map:
  generate_video          ↔ VideoGenClient::generate_with_images (+ download)
  validate_imagine_duration ↔ validate_imagine_duration
  validate_one_of         ↔ validate_one_of
  VideoConfig.from_env    ↔ VideoGenConfig (env-driven; no host injection)

Gaps (honest X — not ported):
  - ZDR S3 presign / output.upload_url path (ZdrVideoOutputS3Config)
  - tier_restricted SuperGrok upsell short-circuit
  - SharedApiKeyProvider / 401 attribution callbacks
  - SessionFileWriter injection (tools write files themselves)
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from codedoggy.tools.util.active_auth import (
    unsupported_video_reason,
    video_api_family,
)
from codedoggy.tools.util.imagine_api import ImagineConfig, ImagineError, ImagineNotSupported

# --- constants from video_gen/mod.rs ---
XAI_VIDEO_BASE_MODEL = "grok-imagine-video"
XAI_VIDEO_QUALITY_MODEL = "grok-imagine-video-1.5-preview"
VIDEO_START_TIMEOUT_SECS = 60
VIDEO_GEN_TIMEOUT_SECS = 300
VIDEO_POLL_INTERVAL_SECS = 5
VIDEO_POLL_REQUEST_TIMEOUT_SECS = 30
VIDEO_DOWNLOAD_TIMEOUT_SECS = 120
DEFAULT_RESOLUTION = "480p"
DEFAULT_IMAGINE_VIDEO_DURATION_SECS = 6
MAX_R2V_REFERENCE_IMAGES = 7
VALID_IMAGINE_VIDEO_ASPECT_RATIOS = ("1:1", "16:9", "9:16", "3:2", "2:3")
VALID_VIDEO_RESOLUTIONS = ("480p", "720p")
IMAGINE_VIDEO_DURATIONS_SECS = (6, 10)

# Back-compat aliases used by older call sites / docs
DEFAULT_VIDEO_MODEL = XAI_VIDEO_BASE_MODEL
DEFAULT_QUALITY_MODEL = XAI_VIDEO_QUALITY_MODEL


@dataclass
class VideoConfig:
    enabled: bool
    base_url: str
    api_key: str | None
    model: str
    reason_disabled: str = ""
    provider: str = ""

    @classmethod
    def resolve(cls, extra: dict[str, Any] | None = None) -> VideoConfig:
        """Follow ActiveConnection via ImagineConfig (same login as chat/images)."""
        bag = extra or {}
        override = bag.get("video_config")
        if isinstance(override, VideoConfig):
            return override
        if _video_flag_off():
            return cls(
                enabled=False,
                base_url="",
                api_key=None,
                model=XAI_VIDEO_BASE_MODEL,
                reason_disabled="CODEDOGGY_VIDEO_ENABLED is off",
            )
        return cls._from_imagine(ImagineConfig.resolve(bag))

    @classmethod
    def from_connection(cls, connection: Any) -> VideoConfig:
        if _video_flag_off():
            return cls(
                enabled=False,
                base_url="",
                api_key=None,
                model=XAI_VIDEO_BASE_MODEL,
                reason_disabled="CODEDOGGY_VIDEO_ENABLED is off",
            )
        return cls._from_imagine(ImagineConfig.from_connection(connection))

    @classmethod
    def from_env(cls) -> VideoConfig:
        if _video_flag_off():
            return cls(
                enabled=False,
                base_url="",
                api_key=None,
                model=XAI_VIDEO_BASE_MODEL,
                reason_disabled="CODEDOGGY_VIDEO_ENABLED is off",
            )
        return cls._from_imagine(ImagineConfig.from_env())

    @classmethod
    def _from_imagine(cls, img: ImagineConfig) -> VideoConfig:
        if not img.enabled or not img.api_key:
            return cls(
                enabled=False,
                base_url=img.base_url or "",
                api_key=None,
                model=XAI_VIDEO_BASE_MODEL,
                provider=img.provider,
                reason_disabled=(
                    img.reason_disabled
                    or "Video generation is not supported: no credential on the "
                    "active connection. Log in via Ctrl+L (same as chat) or set "
                    "CODEDOGGY_IMAGINE_API_KEY."
                ),
            )
        # Only xAI /videos/* is implemented. Follow connection credentials but
        # refuse early on other endpoints (never steal a Grok session).
        if video_api_family(img.base_url) != "xai":
            return cls(
                enabled=False,
                base_url=img.base_url or "",
                api_key=None,
                model=XAI_VIDEO_BASE_MODEL,
                provider=img.provider,
                reason_disabled=unsupported_video_reason(img.provider, img.base_url),
            )
        model = (
            os.environ.get("CODEDOGGY_VIDEO_MODEL") or XAI_VIDEO_BASE_MODEL
        ).strip() or XAI_VIDEO_BASE_MODEL
        return cls(
            enabled=True,
            base_url=img.base_url,
            api_key=img.api_key,
            model=model,
            provider=img.provider,
        )


def _video_flag_off() -> bool:
    return os.environ.get("CODEDOGGY_VIDEO_ENABLED", "1").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }


def validate_one_of(field: str, value: str, allowed: tuple[str, ...] | list[str]) -> None:
    """Grok ``validate_one_of`` — exact error string."""
    if value in allowed:
        return
    raise ImagineError(
        f"`{field}` must be one of: {', '.join(allowed)}. Got {value}.",
        code="invalid_arguments",
    )


def validate_imagine_duration(duration: int | None) -> None:
    """Grok ``validate_imagine_duration`` — exact error string."""
    if duration is None:
        return
    if duration not in IMAGINE_VIDEO_DURATIONS_SECS:
        raise ImagineError(
            f"`duration` must be either 6 or 10 seconds. Got {duration}.",
            code="invalid_arguments",
        )


def generate_video(
    *,
    prompt: str = "",
    image_url: str | None = None,
    reference_urls: list[str] | None = None,
    duration: int | None = None,
    aspect_ratio: str | None = None,
    resolution: str = DEFAULT_RESOLUTION,
    model: str | None = None,
    config: VideoConfig | None = None,
    # legacy kwargs accepted by older video_gen.py
    image_data_url: str | None = None,
    reference_data_urls: list[str] | None = None,
) -> bytes:
    """Start video job, poll until done, download mp4 bytes.

    Mirrors ``VideoGenClient::generate_with_images`` (non-ZDR path only).
    """
    cfg = config or VideoConfig.from_env()
    if not cfg.enabled:
        raise ImagineNotSupported(
            cfg.reason_disabled or "Video generation is not supported on this API."
        )

    if image_url is None and image_data_url is not None:
        image_url = image_data_url
    if reference_urls is None and reference_data_urls is not None:
        reference_urls = reference_data_urls

    validate_imagine_duration(duration)
    validate_one_of("resolution_name", resolution, list(VALID_VIDEO_RESOLUTIONS))
    if aspect_ratio is not None:
        validate_one_of("aspect_ratio", aspect_ratio, list(VALID_IMAGINE_VIDEO_ASPECT_RATIOS))

    # Tools always pass an explicit duration (default 6). On the wire, None is
    # omitted so the server default applies (Grok payload skip_serializing_if).
    wire_model = model or cfg.model
    payload: dict[str, Any] = {
        "model": wire_model,
        "prompt": prompt if prompt is not None else "",
        "resolution": resolution,
    }
    if duration is not None:
        payload["duration"] = duration
    if image_url:
        payload["image"] = {"url": image_url}
    if reference_urls:
        payload["reference_images"] = [{"url": u} for u in reference_urls]
    if aspect_ratio:
        payload["aspect_ratio"] = aspect_ratio
    # ZDR ``output.upload_url`` intentionally not ported (X).

    base = cfg.base_url.rstrip("/")
    start_url = f"{base}/videos/generations"
    start_body = _http_json(
        "POST",
        start_url,
        payload=payload,
        api_key=cfg.api_key or "",
        timeout_s=float(VIDEO_START_TIMEOUT_SECS),
        phase="start",
    )
    request_id = str(start_body.get("request_id") or "").strip()
    if not request_id:
        raise ImagineError(
            "No request_id received from the video generation API.",
            code="invalid_arguments",
        )

    poll_url = f"{base}/videos/{request_id}"
    deadline = time.monotonic() + float(VIDEO_GEN_TIMEOUT_SECS)
    video_url: str | None = None

    while True:
        time.sleep(float(VIDEO_POLL_INTERVAL_SECS))
        if time.monotonic() >= deadline:
            raise ImagineError(
                f"Video generation did not complete within {VIDEO_GEN_TIMEOUT_SECS}s "
                f"(request_id={request_id})",
                code="invalid_arguments",
            )

        poll, poll_http_status, poll_raw = _http_json_ex(
            "GET",
            poll_url,
            payload=None,
            api_key=cfg.api_key or "",
            timeout_s=float(VIDEO_POLL_REQUEST_TIMEOUT_SECS),
            phase="poll",
        )
        # Grok: non-success && != 202 → error
        if poll_http_status is not None and poll_http_status not in {200, 202}:
            truncated = (poll_raw or "")[:200]
            raise ImagineError(
                f"Video poll failed with HTTP {poll_http_status}: {truncated}",
                code="http_failure",
            )

        status = str(poll.get("status") or "")
        if status == "done":
            video = poll.get("video") or {}
            if isinstance(video, dict):
                video_url = video.get("url") or ""
            else:
                video_url = ""
            if not video_url:
                raise ImagineError(
                    "Video generation completed but no download URL was returned.",
                    code="invalid_arguments",
                )
            break
        if status == "failed":
            preview = (poll_raw or json.dumps(poll))[:300]
            raise ImagineError(
                f"Video generation failed on the server "
                f"(request_id={request_id}): {preview}",
                code="invalid_arguments",
            )
        if status == "expired":
            raise ImagineError(
                f"Video generation request expired (request_id={request_id}).",
                code="invalid_arguments",
            )
        # any other status → still in progress (Grok debug path)

    return _download_video(video_url)


def _http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None,
    api_key: str,
    timeout_s: float,
    phase: str = "start",
) -> dict[str, Any]:
    body, _status, _raw = _http_json_ex(
        method,
        url,
        payload=payload,
        api_key=api_key,
        timeout_s=timeout_s,
        phase=phase,
    )
    return body


def _http_json_ex(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None,
    api_key: str,
    timeout_s: float,
    phase: str = "start",
) -> tuple[dict[str, Any], int | None, str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
            raw_bytes = resp.read()
            status_code = getattr(resp, "status", None) or resp.getcode()
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        truncated = err_body[:200]
        if e.code in {401, 403}:
            raise ImagineError(
                f"Video generation failed with HTTP {e.code}: {truncated}",
                code="http_failure",
            ) from e
        if e.code in {404, 405, 501}:
            raise ImagineNotSupported(
                f"Video generation is not supported by this API endpoint "
                f"({url}, HTTP {e.code}). {truncated}".strip(),
                code="not_supported",
            ) from e
        # Poll accepts 202 via success path; other codes:
        label = "Video generation" if phase == "start" else "Video poll"
        raise ImagineError(
            f"{label} failed with HTTP {e.code}: {truncated}",
            code="http_failure",
        ) from e
    except urllib.error.URLError as e:
        label = (
            "Video generation API request failed"
            if phase == "start"
            else "Video poll request failed"
        )
        raise ImagineError(f"{label}: {e.reason}", code="invalid_arguments") from e
    except TimeoutError as e:
        label = (
            "Video generation API request failed"
            if phase == "start"
            else "Video poll request failed"
        )
        raise ImagineError(f"{label}: timed out", code="invalid_arguments") from e

    raw_text = raw_bytes.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        preview = raw_text[:500]
        if phase == "start":
            raise ImagineError(
                f"Failed to parse video generation start response: {e} — "
                f"body preview: {preview}",
                code="invalid_arguments",
            ) from e
        raise ImagineError(
            f"Failed to parse video poll response: {e} — body preview: {preview}",
            code="invalid_arguments",
        ) from e
    if not isinstance(parsed, dict):
        raise ImagineError(
            "Failed to parse video generation start response: expected object",
            code="invalid_arguments",
        )
    return parsed, int(status_code) if status_code is not None else 200, raw_text


def _download_video(url: str) -> bytes:
    """Grok ``download_video``: no auth headers (presigned / CDN URL)."""
    headers = {"User-Agent": "CodeDoggy-video/0.1"}
    req = urllib.request.Request(url, method="GET", headers=headers)
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(
            req, timeout=float(VIDEO_DOWNLOAD_TIMEOUT_SECS), context=ctx
        ) as resp:
            if hasattr(resp, "status") and resp.status and not (200 <= int(resp.status) < 300):
                raise ImagineError(
                    f"Video download failed (HTTP {resp.status})",
                    code="http_failure",
                )
            return resp.read(200 * 1024 * 1024)
    except ImagineError:
        raise
    except urllib.error.HTTPError as e:
        raise ImagineError(
            f"Video download failed (HTTP {e.code})",
            code="http_failure",
        ) from e
    except Exception as e:  # noqa: BLE001
        raise ImagineError(f"Failed to download video: {e}", code="invalid_arguments") from e
