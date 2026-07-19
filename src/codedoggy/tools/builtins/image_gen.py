"""image_gen / image_edit — Grok Imagine tools via real xAI HTTP API.

Ported from:
  crates/codegen/xai-grok-tools/src/implementations/grok_build/image_gen/mod.rs
  crates/codegen/xai-grok-tools/src/implementations/grok_build/image_edit/mod.rs

Uses:
  POST {base}/images/generations
  POST {base}/images/edits

Config follows the session ActiveConnection (same login as chat). See
``codedoggy.tools.util.imagine_api.ImagineConfig.resolve``.
Missing credential or 404/501 from the endpoint → ToolError code ``not_supported``.

Optional test override: extra['image_gen_client'] with generate/edit methods.
Optional attachment registry: extra['attached_images'] as {1: path, ...} or
list of (n, path) for ``[Image #N]`` tokens (Grok AttachedImages).
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolError,
    ToolId,
)
from codedoggy.tools.util.imagine_api import (
    ImagineConfig,
    ImagineError,
    ImagineNotSupported,
    edit_image,
    generate_image,
)

# Grok description_template for image_gen
_IMAGE_GEN_DESC = (
    "Generate a new image from a text description using Imagine; returns the saved "
    "image's absolute path. When telling the user where it was saved, refer to it by "
    "its short session-relative path (e.g. `images/1.jpg`) rather than the absolute "
    "path, so it renders as a clickable link that opens the image. To produce multiple "
    "images, emit multiple tool calls with distinct prompts."
)

# Grok description_template for image_edit
_IMAGE_EDIT_DESC = (
    'Edit or transform existing image(s) via the xAI Imagine API; use instead of '
    "image_gen for image-to-image work (preserve likeness, transfer style, remix). "
    "Returns the saved image's absolute path. When telling the user where it was "
    "saved, refer to it by its short session-relative path (e.g. `images/1.jpg`) "
    "rather than the absolute path, so it renders as a clickable link that opens the "
    "image. Each required `image` is one reference — a user-attachment token "
    '(e.g. "[Image #1]"), an absolute filesystem path, or a '
    "`data:image/...;base64,...` URL (see the `image` parameter for the resolution "
    "order and details)."
)

# Grok SessionFileWriter defaults
_DEFAULT_IMAGE_DIR = "images"
_DEFAULT_EXT = "jpg"

# Grok image_edit compression limits
_MAX_REF_RAW_BYTES = 400 * 1024
_MAX_REF_DIMENSION = 768
_MIN_REF_DIMENSION = 256
_REF_QUALITY_STEPS = (80, 65, 50, 35)
_MAX_REF_DECODE_PIXELS = 12_000_000


class ImageGenTool(Tool):
    def id(self) -> ToolId:
        return ToolId("image_gen")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.ImageGen

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="image_gen", description=_IMAGE_GEN_DESC)

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text description of the image to generate.",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": (
                        "Aspect ratio of the generated image, decide it based on the "
                        "user's request. Defaults to 'auto'. 1:1 for square (icons, "
                        "profiles), 16:9 for wide (landscapes, cinematic), 9:16 for "
                        "tall (phone wallpapers, stories), 3:2 for horizontal photos, "
                        "2:3 for vertical (portraits, posters)."
                    ),
                },
            },
            "required": ["prompt"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        prompt = args.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ToolError.invalid_arguments("prompt is required")
        aspect = str(args.get("aspect_ratio") or "auto")

        client = (ctx.extra or {}).get("image_gen_client")
        if client is not None and callable(getattr(client, "generate", None)):
            try:
                result = client.generate(prompt.strip(), aspect)
            except Exception as e:  # noqa: BLE001
                raise ToolError(f"image_gen failed: {e}", code="image_gen_error") from e
            return _save_and_report(ctx, result, action="Image generated")

        cfg = ImagineConfig.resolve(ctx.extra)
        try:
            raw = generate_image(prompt.strip(), aspect_ratio=aspect, config=cfg)
        except ImagineNotSupported as e:
            raise ToolError(e.message, code=e.code) from e
        except ImagineError as e:
            raise ToolError(e.message, code=e.code) from e
        return _save_and_report(ctx, raw, action="Image generated")


class ImageEditTool(Tool):
    def id(self) -> ToolId:
        return ToolId("image_edit")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.ImageEdit

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="image_edit", description=_IMAGE_EDIT_DESC)

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "A text description of the desired edit or transformation. "
                        "Describe what the output image should look like, referencing "
                        "the input image(s)."
                    ),
                },
                "image": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Reference image(s) to condition the edit on. Each is one "
                        "reference, in priority order: (1) a user attachment — its "
                        'placeholder token, e.g. "[Image #1]" (attachments have no '
                        "path you can see, so never invent one); (2) an absolute "
                        "filesystem path the user gave you; (3) a "
                        "`data:image/...;base64,...` URL."
                    ),
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": (
                        "The aspect ratio of the output image. For single-image edits "
                        "this is ignored — the output matches the input image's aspect "
                        "ratio. For multi-image edits, defaults to 'auto'. Supported "
                        "values: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 2:1, 1:2, "
                        "19.5:9, 9:19.5, 20:9, 9:20, auto."
                    ),
                },
            },
            "required": ["prompt", "image"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        prompt = args.get("prompt")
        images = args.get("image")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ToolError.invalid_arguments("prompt is required")
        if not isinstance(images, list) or not images:
            # Grok: empty array runtime check
            raise ToolError.invalid_arguments(
                "image_edit requires at least one reference image. "
                "Use image_gen for text-only generation."
            )

        client = (ctx.extra or {}).get("image_edit_client") or (ctx.extra or {}).get(
            "image_gen_client"
        )
        if client is not None and callable(getattr(client, "edit", None)):
            try:
                result = client.edit(
                    prompt.strip(), images, str(args.get("aspect_ratio") or "auto")
                )
            except Exception as e:  # noqa: BLE001
                raise ToolError(f"image_edit failed: {e}", code="image_gen_error") from e
            return _save_and_report(ctx, result, action="Image edited")

        aspect = str(args.get("aspect_ratio") or "auto")
        attached = (ctx.extra or {}).get("attached_images")
        try:
            data_urls = [
                resolve_to_data_url(
                    resolve_attachment_reference(str(ref), attached),
                    cwd=ctx.cwd,
                )
                for ref in images
            ]
        except ToolError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ToolError(
                f"failed to load reference image: {e}", code="invalid_arguments"
            ) from e

        cfg = ImagineConfig.resolve(ctx.extra)
        try:
            raw = edit_image(
                prompt.strip(), data_urls, aspect_ratio=aspect, config=cfg
            )
        except ImagineNotSupported as e:
            raise ToolError(e.message, code=e.code) from e
        except ImagineError as e:
            raise ToolError(e.message, code=e.code) from e
        return _save_and_report(ctx, raw, action="Image edited")


# ---------------------------------------------------------------------------
# Attachment tokens (Grok parse_attachment_token / resolve_attachment_reference)
# ---------------------------------------------------------------------------


def parse_attachment_token(value: str) -> int | None:
    """Parse ``[Image #1]`` / ``Image #1`` / ``#1`` → 1-based index, or None."""
    trimmed = value.strip()
    inner = trimmed[1:-1].strip() if trimmed.startswith("[") and trimmed.endswith("]") else trimmed
    rest = inner
    if len(inner) >= 5 and inner[:5].lower() == "image":
        rest = inner[5:].lstrip()
    if not rest.startswith("#"):
        return None
    digits = rest[1:].strip()
    try:
        n = int(digits)
    except ValueError:
        return None
    return n if n >= 1 else None


def resolve_attachment_reference(reference: str, attached: Any) -> str:
    """Map ``[Image #N]`` via attached registry; paths/data URLs pass through."""
    n = parse_attachment_token(reference)
    if n is None:
        return reference

    registry = _normalize_attached(attached)
    if not registry:
        raise ToolError.invalid_arguments(
            f"image reference {reference!r} matches no image attached to this message. "
            "If it was attached earlier in the conversation, ask the user to re-attach "
            "it here; otherwise pass an absolute filesystem path or a data: URL."
        )
    if n not in registry:
        available = ", ".join(f"[Image #{num}]" for num in sorted(registry))
        raise ToolError.invalid_arguments(
            f"image reference {reference!r} does not match any attached image. "
            f"Available: {available}."
        )
    return registry[n]


def _normalize_attached(attached: Any) -> dict[int, str]:
    if attached is None:
        return {}
    if isinstance(attached, dict):
        out: dict[int, str] = {}
        for k, v in attached.items():
            try:
                out[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(attached, (list, tuple)):
        out = {}
        for item in attached:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    out[int(item[0])] = str(item[1])
                except (TypeError, ValueError):
                    continue
        return out
    return {}


# ---------------------------------------------------------------------------
# Reference → compressed data URL (Grok resolve_to_data_url / compress_reference)
# ---------------------------------------------------------------------------


def resolve_to_data_url(value: str, *, cwd: Path) -> str:
    """Resolve path / data URL / file:// / HTTPS into a compressed data URL."""
    value = value.strip()
    # Grok: strip file:// prefix
    if value.startswith("file://"):
        value = value[len("file://") :]

    if value.startswith("data:image/"):
        comma = value.find(",")
        if comma < 0:
            raise ToolError.invalid_arguments("malformed data URL in image reference")
        if ";base64" not in value[:comma]:
            raise ToolError.invalid_arguments(
                "image references only support base64 data URLs"
            )
        try:
            raw_bytes = base64.b64decode(value[comma + 1 :])
        except Exception as e:  # noqa: BLE001
            raise ToolError.invalid_arguments(
                f"invalid base64 in image reference: {e}"
            ) from e
    elif value.startswith("http://") or value.startswith("https://"):
        # CodeDoggy extension (Grok wire schema lists path/data/token only)
        try:
            with urlopen(value, timeout=60) as resp:  # noqa: S310
                raw_bytes = resp.read(8 * 1024 * 1024)
        except Exception as e:  # noqa: BLE001
            raise ToolError.invalid_arguments(
                f"image reference not readable: {value} ({e})"
            ) from e
    else:
        p = Path(value)
        if not p.is_absolute():
            p = (Path(cwd) / p).resolve()
        try:
            raw_bytes = p.read_bytes()
        except OSError as e:
            raise ToolError.invalid_arguments(
                f"image reference not readable: {value} ({e})"
            ) from e

    if not raw_bytes:
        raise ToolError.invalid_arguments("image reference contained no data")

    compressed, mime = compress_reference(raw_bytes)
    b64 = base64.b64encode(compressed).decode("ascii")
    return f"data:{mime};base64,{b64}"


def compress_reference(raw_bytes: bytes) -> tuple[bytes, str]:
    """Compress a reference image to fit Imagine API limits (Grok limits).

    Small JPEG/PNG pass through. Other formats / oversized inputs are re-encoded
    with Pillow when available; otherwise oversized inputs are rejected.
    """
    kind = _sniff_mime(raw_bytes)
    if len(raw_bytes) <= _MAX_REF_RAW_BYTES and kind in {"image/jpeg", "image/png"}:
        return raw_bytes, kind

    try:
        from PIL import Image  # type: ignore[import-untyped]
    except ImportError as e:
        if len(raw_bytes) <= _MAX_REF_RAW_BYTES and kind:
            return raw_bytes, kind
        raise ToolError.invalid_arguments(
            "could not compress image reference small enough for Imagine API: "
            "Pillow is required for large or non-JPEG/PNG references"
        ) from e

    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
    except Exception as e:  # noqa: BLE001
        raise ToolError.invalid_arguments("failed to decode image reference") from e

    w, h = img.size
    if (w * h) > _MAX_REF_DECODE_PIXELS:
        raise ToolError.invalid_arguments(
            f"image reference is too large to process ({w}\u00d7{h} pixels)"
        )

    # Downscale long side to MAX_REF_DIMENSION, but not below MIN when possible.
    max_side = max(w, h)
    if max_side > _MAX_REF_DIMENSION:
        scale = _MAX_REF_DIMENSION / max_side
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        # Avoid shrinking below min side when the other side allows it
        if min(nw, nh) < _MIN_REF_DIMENSION and max(w, h) >= _MIN_REF_DIMENSION:
            # Prefer fitting max side; min may still be small for extreme ratios
            pass
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    last_err = "unknown"
    for quality in _REF_QUALITY_STEPS:
        buf = io.BytesIO()
        try:
            img.save(buf, format="JPEG", quality=quality, optimize=True)
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            continue
        data = buf.getvalue()
        if len(data) <= _MAX_REF_RAW_BYTES:
            return data, "image/jpeg"

    # Final PNG attempt (sometimes smaller for simple graphics)
    buf = io.BytesIO()
    try:
        img.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        if len(data) <= _MAX_REF_RAW_BYTES:
            return data, "image/png"
    except Exception as e:  # noqa: BLE001
        last_err = str(e)

    raise ToolError.invalid_arguments(
        f"could not compress image reference small enough for Imagine API: {last_err}"
    )


def _sniff_mime(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return None


# ---------------------------------------------------------------------------
# Save + Grok MediaGenOutput wording
# ---------------------------------------------------------------------------


def _save_and_report(ctx: ToolCallContext, result: Any, *, action: str) -> str:
    if isinstance(result, (str, Path)) and not isinstance(result, (bytes, bytearray)):
        path = Path(result)
        if path.is_file():
            return _media_gen_report(path, action=action)
        return str(result)
    if isinstance(result, (bytes, bytearray)):
        path = _write_image_bytes(ctx.cwd, bytes(result))
        return _media_gen_report(path, action=action)
    return str(result)


def _write_image_bytes(cwd: Path, data: bytes) -> Path:
    """Grok SessionFileWriter: ``<cwd>/images/<n>.jpg`` (always .jpg)."""
    out_dir = Path(cwd) / _DEFAULT_IMAGE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while (out_dir / f"{n}.{_DEFAULT_EXT}").exists():
        n += 1
    path = out_dir / f"{n}.{_DEFAULT_EXT}"
    path.write_bytes(data)
    return path.resolve()


def _media_gen_report(path: Path, *, action: str) -> str:
    """Mirror Grok MediaGenOutput.prompt_text JSON for the model."""
    abs_path = str(path.resolve())
    filename = path.name
    session_folder = path.parent.name
    message = (
        f"{action} and saved to {abs_path}. "
        "Do not read or re-display it, and do not describe how it appears to the user."
    )
    return json.dumps(
        {
            "path": abs_path,
            "filename": filename,
            "session_folder": session_folder,
            "message": message,
        },
        ensure_ascii=False,
    )
