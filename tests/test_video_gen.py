"""Source-level tests for image_to_video / reference_to_video + video_api.

Mirrors Grok video_gen/mod.rs unit tests (validation + payload) and exercises
the generations + poll HTTP flow with mocked urllib.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from codedoggy.tools.builtins.video_gen import (
    ImageToVideoTool,
    ReferenceToVideoTool,
    resolve_image_reference,
)
from codedoggy.tools.registry import ToolRegistryBuilder
from codedoggy.tools.runtime import ToolCallContext, ToolError
from codedoggy.tools.util import video_api
from codedoggy.tools.util.imagine_api import ImagineError, ImagineNotSupported
from codedoggy.tools.util.video_api import (
    DEFAULT_IMAGINE_VIDEO_DURATION_SECS,
    DEFAULT_RESOLUTION,
    IMAGINE_VIDEO_DURATIONS_SECS,
    VALID_IMAGINE_VIDEO_ASPECT_RATIOS,
    VALID_VIDEO_RESOLUTIONS,
    VIDEO_GEN_TIMEOUT_SECS,
    VIDEO_POLL_INTERVAL_SECS,
    XAI_VIDEO_BASE_MODEL,
    XAI_VIDEO_QUALITY_MODEL,
    VideoConfig,
    generate_video,
    validate_imagine_duration,
    validate_one_of,
)

# Minimal valid PNG (1x1) for local path resolution.
_MIN_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
    b"\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_image_to_video_name_and_description() -> None:
    tool = ImageToVideoTool()
    assert str(tool.id()) == "image_to_video"
    desc = tool.description().description
    assert "single source image" in desc
    assert "image_to_video" in desc


def test_reference_to_video_name_and_description() -> None:
    tool = ReferenceToVideoTool()
    assert str(tool.id()) == "reference_to_video"
    desc = tool.description().description
    assert "multiple reference images" in desc
    assert "reference_to_video" in desc


def test_imagine_duration_validation_allows_only_toolbox_values() -> None:
    validate_imagine_duration(None)
    validate_imagine_duration(6)
    validate_imagine_duration(10)
    with pytest.raises(ImagineError, match="either 6 or 10"):
        validate_imagine_duration(8)


def test_validate_one_of_resolution_and_aspect() -> None:
    validate_one_of("resolution_name", "480p", list(VALID_VIDEO_RESOLUTIONS))
    with pytest.raises(ImagineError, match="resolution_name"):
        validate_one_of("resolution_name", "1080p", list(VALID_VIDEO_RESOLUTIONS))
    with pytest.raises(ImagineError, match="aspect_ratio"):
        validate_one_of("aspect_ratio", "4:3", list(VALID_IMAGINE_VIDEO_ASPECT_RATIOS))


def test_image_to_video_rejects_bad_duration(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    (tmp_path / "a.png").write_bytes(_MIN_PNG)
    with pytest.raises(ToolError, match="either 6 or 10"):
        tools.call(
            "image_to_video",
            {"image": "a.png", "duration": 8},
            ctx,
        )


def test_image_to_video_rejects_bad_resolution(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    (tmp_path / "a.png").write_bytes(_MIN_PNG)
    with pytest.raises(ToolError, match="resolution_name"):
        tools.call(
            "image_to_video",
            {"image": "a.png", "resolution_name": "1080p"},
            ctx,
        )


def test_reference_to_video_rejects_bad_aspect_ratio(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    with pytest.raises(ToolError, match="aspect_ratio"):
        tools.call(
            "reference_to_video",
            {
                "prompt": "blend",
                "images": ["a.png", "b.png"],
                "aspect_ratio": "4:3",
            },
            ctx,
        )


def test_reference_to_video_rejects_too_few_images(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    with pytest.raises(ToolError, match="at least two"):
        tools.call(
            "reference_to_video",
            {"prompt": "blend", "images": ["only.png"], "aspect_ratio": "16:9"},
            ctx,
        )


def test_reference_to_video_rejects_empty_prompt(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    with pytest.raises(ToolError, match="prompt"):
        tools.call(
            "reference_to_video",
            {"prompt": "  ", "images": ["a.png", "b.png"], "aspect_ratio": "16:9"},
            ctx,
        )


def test_resolve_https_passthrough() -> None:
    url = "https://cdn.example.com/photo.jpg"
    assert resolve_image_reference(url) == url


def test_resolve_data_url_requires_base64() -> None:
    ok = "data:image/png;base64,abc"
    assert resolve_image_reference(ok) == ok
    with pytest.raises(ToolError, match="base64"):
        resolve_image_reference("data:image/png,notbase64")
    with pytest.raises(ToolError, match="malformed"):
        resolve_image_reference("data:image/png;base64")


def test_resolve_local_png(tmp_path: Path) -> None:
    p = tmp_path / "x.png"
    p.write_bytes(_MIN_PNG)
    out = resolve_image_reference(str(p))
    assert out.startswith("data:image/png;base64,")


def test_resolve_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.png"
    p.write_bytes(b"")
    with pytest.raises(ToolError, match="contained no data"):
        resolve_image_reference(str(p))


def test_video_config_missing_key_not_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "CODEDOGGY_IMAGINE_API_KEY",
        "XAI_API_KEY",
        "CODEDOGGY_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CODEDOGGY_VIDEO_ENABLED", "1")
    cfg = VideoConfig.from_env()
    assert not cfg.enabled
    with pytest.raises(ImagineNotSupported):
        generate_video(prompt="x", image_url="data:image/png;base64,aa", config=cfg)


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status
        self._fp = io.BytesIO(body)

    def read(self, n: int = -1) -> bytes:
        return self._fp.read(n)

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_generate_video_start_poll_download_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST generations → poll done → download bytes (Grok non-ZDR path)."""
    calls: list[tuple[str, str]] = []
    payloads: list[dict[str, Any] | None] = []

    def fake_urlopen(req: Any, timeout: float = 0, context: Any = None) -> _FakeResponse:
        method = getattr(req, "get_method", lambda: req.method)()
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        calls.append((method, url))
        body = req.data
        if body:
            payloads.append(json.loads(body.decode("utf-8")))
        else:
            payloads.append(None)

        if url.endswith("/videos/generations") and method == "POST":
            return _FakeResponse(json.dumps({"request_id": "req-abc"}).encode())
        if "/videos/req-abc" in url and method == "GET":
            return _FakeResponse(
                json.dumps(
                    {
                        "status": "done",
                        "video": {"url": "https://cdn.example.com/out.mp4"},
                    }
                ).encode()
            )
        if "cdn.example.com/out.mp4" in url:
            return _FakeResponse(b"FAKE_MP4_BYTES")
        raise AssertionError(f"unexpected request {method} {url}")

    monkeypatch.setattr(video_api.urllib.request, "urlopen", fake_urlopen)
    # skip real sleep in poll loop
    monkeypatch.setattr(video_api.time, "sleep", lambda _s: None)

    cfg = VideoConfig(
        enabled=True,
        base_url="https://api.x.ai/v1",
        api_key="sk-test",
        model=XAI_VIDEO_BASE_MODEL,
    )
    raw = generate_video(
        prompt="animate",
        image_url="data:image/png;base64,aa",
        duration=DEFAULT_IMAGINE_VIDEO_DURATION_SECS,
        resolution=DEFAULT_RESOLUTION,
        model=XAI_VIDEO_QUALITY_MODEL,
        config=cfg,
    )
    assert raw == b"FAKE_MP4_BYTES"
    assert any(u.endswith("/videos/generations") for _, u in calls)
    assert any("req-abc" in u for _, u in calls)
    start_payload = next(p for p in payloads if p and "model" in p)
    assert start_payload["model"] == XAI_VIDEO_QUALITY_MODEL
    assert start_payload["image"]["url"].startswith("data:image/png")
    assert start_payload["duration"] == 6
    assert start_payload["resolution"] == "480p"
    assert "aspect_ratio" not in start_payload
    assert "output" not in start_payload  # ZDR not ported


def test_generate_video_reference_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: Any, timeout: float = 0, context: Any = None) -> _FakeResponse:
        method = getattr(req, "get_method", lambda: req.method)()
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        body = req.data
        if url.endswith("/videos/generations") and method == "POST":
            payload = json.loads(body.decode("utf-8"))
            assert payload["model"] == XAI_VIDEO_BASE_MODEL
            assert len(payload["reference_images"]) == 2
            assert payload["aspect_ratio"] == "16:9"
            assert "image" not in payload
            return _FakeResponse(json.dumps({"request_id": "r2"}).encode())
        if "/videos/r2" in url:
            return _FakeResponse(
                json.dumps(
                    {"status": "done", "video": {"url": "https://cdn.example.com/v.mp4"}}
                ).encode()
            )
        if "cdn.example.com" in url:
            return _FakeResponse(b"mp4")
        raise AssertionError(url)

    monkeypatch.setattr(video_api.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(video_api.time, "sleep", lambda _s: None)

    cfg = VideoConfig(
        enabled=True,
        base_url="https://api.x.ai/v1",
        api_key="sk-test",
        model=XAI_VIDEO_BASE_MODEL,
    )
    out = generate_video(
        prompt="blend",
        reference_urls=["data:image/png;base64,a", "data:image/png;base64,b"],
        duration=6,
        aspect_ratio="16:9",
        model=XAI_VIDEO_BASE_MODEL,
        config=cfg,
    )
    assert out == b"mp4"


def test_generate_video_server_failed_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: Any, timeout: float = 0, context: Any = None) -> _FakeResponse:
        method = getattr(req, "get_method", lambda: req.method)()
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        if url.endswith("/videos/generations"):
            return _FakeResponse(json.dumps({"request_id": "fail1"}).encode())
        if "/videos/fail1" in url:
            return _FakeResponse(
                json.dumps({"status": "failed", "error": "boom"}).encode()
            )
        raise AssertionError(url)

    monkeypatch.setattr(video_api.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(video_api.time, "sleep", lambda _s: None)

    cfg = VideoConfig(
        enabled=True,
        base_url="https://api.x.ai/v1",
        api_key="sk-test",
        model=XAI_VIDEO_BASE_MODEL,
    )
    with pytest.raises(ImagineError, match="failed on the server"):
        generate_video(
            prompt="x",
            image_url="data:image/png;base64,aa",
            duration=6,
            config=cfg,
        )


def test_generate_video_missing_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: Any, timeout: float = 0, context: Any = None) -> _FakeResponse:
        return _FakeResponse(json.dumps({}).encode())

    monkeypatch.setattr(video_api.urllib.request, "urlopen", fake_urlopen)
    cfg = VideoConfig(
        enabled=True,
        base_url="https://api.x.ai/v1",
        api_key="sk-test",
        model=XAI_VIDEO_BASE_MODEL,
    )
    with pytest.raises(ImagineError, match="No request_id"):
        generate_video(prompt="x", image_url="data:image/png;base64,aa", duration=6, config=cfg)


def test_image_to_video_tool_end_to_end_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "src.png").write_bytes(_MIN_PNG)

    def fake_urlopen(req: Any, timeout: float = 0, context: Any = None) -> _FakeResponse:
        method = getattr(req, "get_method", lambda: req.method)()
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        if url.endswith("/videos/generations") and method == "POST":
            payload = json.loads(req.data.decode("utf-8"))
            assert payload["model"] == XAI_VIDEO_QUALITY_MODEL
            return _FakeResponse(json.dumps({"request_id": "t1"}).encode())
        if "/videos/t1" in url:
            return _FakeResponse(
                json.dumps(
                    {"status": "done", "video": {"url": "https://cdn.example.com/o.mp4"}}
                ).encode()
            )
        if "cdn.example.com" in url:
            return _FakeResponse(b"VIDEO_BYTES")
        raise AssertionError(url)

    monkeypatch.setattr(video_api.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(video_api.time, "sleep", lambda _s: None)
    monkeypatch.setenv("CODEDOGGY_IMAGINE_API_KEY", "sk-test")
    monkeypatch.setenv("CODEDOGGY_VIDEO_ENABLED", "1")

    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    out = tools.call(
        "image_to_video",
        {"image": "src.png", "prompt": "zoom in", "duration": 6},
        ctx,
    )
    assert "videos/" in out
    saved = tmp_path / "videos" / "1.mp4"
    assert saved.read_bytes() == b"VIDEO_BYTES"


def test_constants_match_grok() -> None:
    assert XAI_VIDEO_BASE_MODEL == "grok-imagine-video"
    assert XAI_VIDEO_QUALITY_MODEL == "grok-imagine-video-1.5-preview"
    assert DEFAULT_RESOLUTION == "480p"
    assert DEFAULT_IMAGINE_VIDEO_DURATION_SECS == 6
    assert IMAGINE_VIDEO_DURATIONS_SECS == (6, 10)
    assert VALID_VIDEO_RESOLUTIONS == ("480p", "720p")
    assert VIDEO_GEN_TIMEOUT_SECS == 300
    assert VIDEO_POLL_INTERVAL_SECS == 5
