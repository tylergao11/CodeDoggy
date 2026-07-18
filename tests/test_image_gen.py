"""Focused tests for image_gen / image_edit source port (mock HTTP)."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.builtins.image_gen import (
    compress_reference,
    parse_attachment_token,
    resolve_attachment_reference,
    resolve_to_data_url,
)
from codedoggy.tools.runtime import ToolCallContext, ToolError
from codedoggy.tools.util.imagine_api import (
    DEFAULT_BASE,
    DEFAULT_MODEL,
    ImagineConfig,
    ImagineError,
    ImagineNotSupported,
    edit_image,
    generate_image,
)


def _tiny_jpeg() -> bytes:
    try:
        from PIL import Image
    except ImportError:
        # Minimal JPEG SOI + EOI markers (not a full image; for passthrough only)
        return b"\xff\xd8\xff\xd9"
    img = Image.new("RGB", (2, 2), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _tiny_png() -> bytes:
    try:
        from PIL import Image
    except ImportError:
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    img = Image.new("RGBA", (2, 2), color=(10, 20, 30, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _b64_response(jpeg: bytes | None = None) -> bytes:
    raw = jpeg if jpeg is not None else _tiny_jpeg()
    payload = {
        "data": [{"b64_json": base64.b64encode(raw).decode("ascii")}],
    }
    return json.dumps(payload).encode("utf-8")


class _FakeHTTPResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


# ── config ──────────────────────────────────────────────────────────────


def test_config_missing_key_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "CODEDOGGY_IMAGINE_API_KEY",
        "XAI_API_KEY",
        "CODEDOGGY_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    cfg = ImagineConfig.from_env()
    assert not cfg.enabled
    assert "API key" in cfg.reason_disabled
    assert cfg.base_url == DEFAULT_BASE or cfg.base_url.endswith("/v1")


def test_config_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEDOGGY_IMAGINE_API_KEY", "sk-x")
    monkeypatch.delenv("CODEDOGGY_IMAGINE_MODEL", raising=False)
    cfg = ImagineConfig.from_env()
    assert cfg.model == DEFAULT_MODEL
    assert DEFAULT_MODEL == "grok-imagine-image-quality"


# ── generate payload (Grok ImageGenClient::generate) ────────────────────


def test_generate_payload_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEDOGGY_IMAGINE_API_KEY", "sk-test")
    monkeypatch.setenv("CODEDOGGY_IMAGINE_BASE_URL", "https://api.x.ai/v1")
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0, context: Any = None) -> _FakeHTTPResponse:
        captured["url"] = req.full_url
        captured["headers"] = {k: v for k, v in req.header_items()}
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(_b64_response())

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        raw = generate_image("a cat", aspect_ratio="16:9")

    assert captured["url"] == "https://api.x.ai/v1/images/generations"
    body = captured["body"]
    assert body["model"] == DEFAULT_MODEL
    assert body["prompt"] == "a cat"
    assert body["n"] == 1
    assert body["aspect_ratio"] == "16:9"
    assert body["resolution"] == "1k"
    assert body["response_format"] == "b64_json"
    assert "Authorization" in str(captured["headers"]) or any(
        "authorization" in k.lower() for k in captured["headers"]
    )
    assert raw[:3] == b"\xff\xd8\xff" or len(raw) > 0


def test_generate_404_is_not_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    monkeypatch.setenv("CODEDOGGY_IMAGINE_API_KEY", "sk-test")

    def boom(*a: Any, **k: Any) -> Any:
        raise urllib.error.HTTPError(
            "https://api.x.ai/v1/images/generations",
            404,
            "Not Found",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"error":"nope"}'),
        )

    with patch("urllib.request.urlopen", side_effect=boom):
        with pytest.raises(ImagineNotSupported) as ei:
            generate_image("x")
    assert ei.value.code == "not_supported"


def test_generate_500_is_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    monkeypatch.setenv("CODEDOGGY_IMAGINE_API_KEY", "sk-test")

    def boom(*a: Any, **k: Any) -> Any:
        raise urllib.error.HTTPError(
            "https://api.x.ai/v1/images/generations",
            500,
            "err",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b"boom"),
        )

    with patch("urllib.request.urlopen", side_effect=boom):
        with pytest.raises(ImagineError) as ei:
            generate_image("x")
    assert ei.value.code == "http_failure"


# ── edit payload (Grok image_edit) ──────────────────────────────────────


def test_edit_single_image_object_no_aspect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEDOGGY_IMAGINE_API_KEY", "sk-test")
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0, context: Any = None) -> _FakeHTTPResponse:
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(_b64_response())

    data_url = "data:image/jpeg;base64," + base64.b64encode(_tiny_jpeg()).decode("ascii")
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        edit_image("anime style", [data_url], aspect_ratio="1:1")

    assert captured["url"] == "https://api.x.ai/v1/images/edits"
    body = captured["body"]
    assert body["model"] == DEFAULT_MODEL
    assert body["prompt"] == "anime style"
    assert body["n"] == 1
    assert body["resolution"] == "1k"
    assert body["response_format"] == "b64_json"
    assert "image" in body
    assert body["image"] == {"url": data_url}
    assert "images" not in body
    # Grok: single-image → aspect_ratio omitted
    assert "aspect_ratio" not in body


def test_edit_multi_images_array_with_aspect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEDOGGY_IMAGINE_API_KEY", "sk-test")
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0, context: Any = None) -> _FakeHTTPResponse:
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(_b64_response())

    urls = [
        "data:image/jpeg;base64," + base64.b64encode(_tiny_jpeg()).decode("ascii"),
        "data:image/png;base64," + base64.b64encode(_tiny_png()).decode("ascii"),
    ]
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        edit_image("blend", urls, aspect_ratio="16:9")

    body = captured["body"]
    assert "images" in body
    assert body["images"] == [{"url": urls[0]}, {"url": urls[1]}]
    assert "image" not in body
    assert body["aspect_ratio"] == "16:9"


# ── tool wire ───────────────────────────────────────────────────────────


def test_image_gen_description_grok_wording() -> None:
    tools = ToolRegistryBuilder.new().finalize()
    defs = {d.name: d for d in tools.tool_definitions()}
    assert "images/1.jpg" in (defs["image_gen"].description or "")
    assert "Imagine" in (defs["image_gen"].description or "")
    assert "images/1.jpg" in (defs["image_edit"].description or "")
    assert "xAI Imagine" in (defs["image_edit"].description or "")


def test_image_edit_empty_array(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    with pytest.raises(ToolError) as ei:
        tools.call("image_edit", {"prompt": "x", "image": []}, ctx)
    assert ei.value.code == "invalid_arguments"
    assert "at least one reference image" in ei.value.message


def test_image_edit_saves_jpg_with_grok_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEDOGGY_IMAGINE_API_KEY", "sk-test")
    jpeg = _tiny_jpeg()

    def fake_urlopen(req: Any, timeout: float = 0, context: Any = None) -> _FakeHTTPResponse:
        return _FakeHTTPResponse(_b64_response(jpeg))

    src = tmp_path / "ref.jpg"
    src.write_bytes(jpeg)

    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        out = tools.call(
            "image_edit",
            {"prompt": "make blue", "image": [str(src)]},
            ctx,
        )
    data = json.loads(out)
    assert data["filename"] == "1.jpg"
    assert data["session_folder"] == "images"
    assert "Image edited and saved" in data["message"]
    assert "Do not read or re-display" in data["message"]
    assert (tmp_path / "images" / "1.jpg").is_file()


def test_image_gen_tool_not_supported_without_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for k in (
        "CODEDOGGY_IMAGINE_API_KEY",
        "XAI_API_KEY",
        "CODEDOGGY_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    tools = ToolRegistryBuilder.new().finalize()
    with pytest.raises(ToolError) as ei:
        tools.call("image_gen", {"prompt": "cube"}, ToolCallContext(cwd=tmp_path))
    assert ei.value.code == "not_supported"


# ── attachment tokens / refs ────────────────────────────────────────────


def test_parse_attachment_token_forms() -> None:
    assert parse_attachment_token("[Image #1]") == 1
    assert parse_attachment_token("Image #2") == 2
    assert parse_attachment_token("image #3") == 3
    assert parse_attachment_token("#6") == 6
    assert parse_attachment_token("/Users/me/photo.jpg") is None
    assert parse_attachment_token("data:image/png;base64,AAAA") is None
    assert parse_attachment_token("[Image #0]") is None


def test_resolve_attachment_maps_token() -> None:
    attached = {1: "/tmp/a.png", 3: "/tmp/c.png"}
    assert resolve_attachment_reference("[Image #1]", attached) == "/tmp/a.png"
    assert resolve_attachment_reference("[Image #3]", attached) == "/tmp/c.png"
    with pytest.raises(ToolError, match="does not match"):
        resolve_attachment_reference("[Image #2]", attached)
    with pytest.raises(ToolError, match="re-attach"):
        resolve_attachment_reference("[Image #1]", None)


def test_resolve_file_uri_and_data_url(tmp_path: Path) -> None:
    jpeg = _tiny_jpeg()
    path = tmp_path / "t.jpg"
    path.write_bytes(jpeg)
    url = resolve_to_data_url(f"file://{path}", cwd=tmp_path)
    assert url.startswith("data:image/jpeg;base64,")

    b64 = base64.b64encode(jpeg).decode("ascii")
    url2 = resolve_to_data_url(f"data:image/jpeg;base64,{b64}", cwd=tmp_path)
    assert url2.startswith("data:image/jpeg;base64,")


def test_compress_small_jpeg_passthrough() -> None:
    jpeg = _tiny_jpeg()
    if len(jpeg) > 400 * 1024:
        pytest.skip("fixture too large")
    out, mime = compress_reference(jpeg)
    # Only assert passthrough when under limit and jpeg sniff works
    if jpeg[:3] == b"\xff\xd8\xff" and len(jpeg) <= 400 * 1024:
        assert out == jpeg
        assert mime == "image/jpeg"


def test_image_edit_attachment_token_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEDOGGY_IMAGINE_API_KEY", "sk-test")
    jpeg = _tiny_jpeg()
    src = tmp_path / "att.jpg"
    src.write_bytes(jpeg)

    def fake_urlopen(req: Any, timeout: float = 0, context: Any = None) -> _FakeHTTPResponse:
        body = json.loads(req.data.decode("utf-8"))
        # single ref → image.url data URL
        assert "image" in body
        assert body["image"]["url"].startswith("data:image/")
        return _FakeHTTPResponse(_b64_response(jpeg))

    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={"attached_images": {1: str(src)}},
    )
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        out = tools.call(
            "image_edit",
            {"prompt": "style", "image": ["[Image #1]"]},
            ctx,
        )
    data = json.loads(out)
    assert data["filename"] == "1.jpg"
