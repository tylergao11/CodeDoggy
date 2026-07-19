"""image_to_video / reference_to_video — Grok video tools (xAI API).

Ported from:
  grok-build/crates/codegen/xai-grok-tools/src/implementations/grok_build/video_gen/mod.rs
    ImageToVideoTool, ReferenceToVideoTool
    resolve_image_reference, save_video_bytes / media_output_from_outcome
    constants (models, durations, aspect ratios, resolutions)

Function map:
  ImageToVideoTool.run      ↔ ImageToVideoTool::run
  ReferenceToVideoTool.run  ↔ ReferenceToVideoTool::run
  resolve_image_reference   ↔ resolve_image_reference
  _save_video               ↔ save_video_bytes (session videos/<n>.mp4)

Default API: POST {base}/videos/generations + poll GET {base}/videos/{id}.
Missing key → not_supported. ZDR/S3 upload path not ported (X).
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolError,
    ToolId,
)
from codedoggy.tools.util.imagine_api import ImagineError, ImagineNotSupported
from codedoggy.tools.util.video_api import (
    DEFAULT_IMAGINE_VIDEO_DURATION_SECS,
    DEFAULT_RESOLUTION,
    MAX_R2V_REFERENCE_IMAGES,
    VALID_IMAGINE_VIDEO_ASPECT_RATIOS,
    VALID_VIDEO_RESOLUTIONS,
    XAI_VIDEO_BASE_MODEL,
    XAI_VIDEO_QUALITY_MODEL,
    VideoConfig,
    generate_video,
    validate_imagine_duration,
    validate_one_of,
)

# Grok description_template strings (verbatim).
_I2V_DESC = (
    "Generate a video from a single source image; returns the saved video's absolute path. "
    "When telling the user where it was saved, refer to it by its short session-relative path "
    "(e.g. `videos/1.mp4`) rather than the absolute path, so it renders as a clickable link that "
    "opens the video. Provide `image` for the image to animate and optionally a `prompt` to guide "
    "the animation. Use this tool when the user provides an image and wants it animated, turned "
    "into a video, or used as the first frame. Example: image_to_video(image=\"/Users/me/photo.jpg\", "
    "prompt=\"gentle camera push-in with wind moving the hair\", duration=6, resolution_name=\"480p\")"
)

_R2V_DESC = (
    "Generate a video from multiple reference images guided by a text prompt; returns the saved "
    "video's absolute path. When telling the user where it was saved, refer to it by its short "
    "session-relative path (e.g. `videos/1.mp4`) rather than the absolute path, so it renders as a "
    "clickable link that opens the video. Provide `images` with 2 to 7 image references and a "
    "required `prompt` describing the desired video. Use this tool when the user wants a video "
    "using multiple images as style/content references. Example: reference_to_video("
    "prompt=\"blend these into a cinematic fashion shot with slow dolly movement\", "
    "images=[\"/Users/me/ref1.jpg\", \"/Users/me/ref2.jpg\"], aspect_ratio=\"16:9\", duration=6, "
    "resolution_name=\"480p\")"
)


class ImageToVideoTool(Tool):
    def id(self) -> ToolId:
        return ToolId("image_to_video")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.VideoGen

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="image_to_video", description=_I2V_DESC)

    def parameters_schema(self) -> dict[str, Any]:
        # schemars descriptions from ImageToVideoInput
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Optional prompt to guide the video generation model. If omitted, "
                        "a natural animation applies automatically."
                    ),
                },
                "image": {
                    "type": "string",
                    "description": (
                        "Source image to animate. Provide an absolute filesystem path, "
                        "HTTPS URL, or `data:image/...;base64,...` URL."
                    ),
                },
                "duration": {
                    "type": "integer",
                    "description": (
                        "Duration of the video generation, either 6 or 10 seconds. "
                        "Default to 6 unless the user requests longer."
                    ),
                },
                "resolution_name": {
                    "type": "string",
                    "description": (
                        "Resolution name of the video generation, only specify it when user "
                        "asks for a specific resolution, either 480p or 720p. Defaults to 480p "
                        "unless the user specifically requests for higher quality."
                    ),
                },
            },
            "required": ["image"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        image = args.get("image")
        if not isinstance(image, str) or not image.strip():
            raise ToolError.invalid_arguments("image is required")
        prompt = args.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise ToolError.invalid_arguments("prompt must be a string")

        duration = _parse_duration(args.get("duration"))
        resolution = str(args.get("resolution_name") or DEFAULT_RESOLUTION)

        try:
            validate_imagine_duration(duration)
            validate_one_of("resolution_name", resolution, list(VALID_VIDEO_RESOLUTIONS))
            image_url = resolve_image_reference(image, cwd=ctx.cwd)
            cfg = VideoConfig.resolve(ctx.extra)
            raw = generate_video(
                prompt=prompt if isinstance(prompt, str) else "",
                image_url=image_url,
                duration=duration if duration is not None else DEFAULT_IMAGINE_VIDEO_DURATION_SECS,
                resolution=resolution,
                model=XAI_VIDEO_QUALITY_MODEL,
                config=cfg,
            )
        except ImagineNotSupported as e:
            raise ToolError(e.message, code=e.code) from e
        except ImagineError as e:
            raise ToolError(e.message, code=e.code) from e
        except ToolError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"image_to_video failed: {e}", code="video_error") from e
        return _save_video(ctx, raw, label="image_to_video")


class ReferenceToVideoTool(Tool):
    def id(self) -> ToolId:
        return ToolId("reference_to_video")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.VideoGen

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="reference_to_video", description=_R2V_DESC)

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Prompt to guide the video generation model. Describe the desired video."
                    ),
                },
                "images": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Reference images. Provide 2 to 7 entries; the images are used as "
                        "style/content references for the generated video. Each entry may be an "
                        "absolute filesystem path, HTTPS URL, or `data:image/...;base64,...` URL."
                    ),
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": (
                        "Aspect ratio of the generated video, decide it based on the user's "
                        "request. 1:1 for square (icons, profiles), 16:9 for wide (landscapes, "
                        "cinematic), 9:16 for tall (phone wallpapers, stories), 3:2 for horizontal "
                        "photos, 2:3 for vertical (portraits, posters)."
                    ),
                },
                "duration": {
                    "type": "integer",
                    "description": (
                        "Duration of the video generation, either 6 or 10 seconds. Defaults to 6."
                    ),
                },
                "resolution_name": {
                    "type": "string",
                    "description": (
                        "Resolution name of the video generation, only specify it when user asks "
                        "for a specific resolution, either 480p or 720p. Defaults to 480p."
                    ),
                },
            },
            "required": ["prompt", "images", "aspect_ratio"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        prompt = args.get("prompt")
        images = args.get("images")
        aspect = args.get("aspect_ratio")

        if not isinstance(prompt, str) or not prompt.strip():
            raise ToolError.invalid_arguments("`prompt` must not be empty.")
        if not isinstance(images, list):
            raise ToolError.invalid_arguments("images must be an array")
        if len(images) < 2:
            raise ToolError.invalid_arguments(
                "`images` must contain at least two image references."
            )
        if len(images) > MAX_R2V_REFERENCE_IMAGES:
            raise ToolError.invalid_arguments(
                f"`images` must contain at most {MAX_R2V_REFERENCE_IMAGES} image references."
            )
        if not isinstance(aspect, str) or not aspect.strip():
            raise ToolError.invalid_arguments("aspect_ratio is required")

        duration = _parse_duration(args.get("duration"))
        resolution = str(args.get("resolution_name") or DEFAULT_RESOLUTION)

        try:
            validate_imagine_duration(duration)
            validate_one_of(
                "aspect_ratio", aspect.strip(), list(VALID_IMAGINE_VIDEO_ASPECT_RATIOS)
            )
            validate_one_of("resolution_name", resolution, list(VALID_VIDEO_RESOLUTIONS))
            refs = [resolve_image_reference(str(img), cwd=ctx.cwd) for img in images]
            cfg = VideoConfig.resolve(ctx.extra)
            raw = generate_video(
                prompt=prompt.strip(),
                reference_urls=refs,
                duration=duration if duration is not None else DEFAULT_IMAGINE_VIDEO_DURATION_SECS,
                aspect_ratio=aspect.strip(),
                resolution=resolution,
                model=XAI_VIDEO_BASE_MODEL,
                config=cfg,
            )
        except ImagineNotSupported as e:
            raise ToolError(e.message, code=e.code) from e
        except ImagineError as e:
            raise ToolError(e.message, code=e.code) from e
        except ToolError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"reference_to_video failed: {e}", code="video_error") from e
        return _save_video(ctx, raw, label="reference_to_video")


def _parse_duration(raw: Any) -> int | None:
    """Grok duration_from_json: accept int or numeric string; None if omitted."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        raise ToolError.invalid_arguments("duration must be 6 or 10")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw == int(raw):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw.strip())
        except ValueError as e:
            raise ToolError.invalid_arguments("duration must be 6 or 10") from e
    raise ToolError.invalid_arguments("duration must be 6 or 10")


def resolve_image_reference(value: str, *, cwd: Path | None = None) -> str:
    """Grok ``resolve_image_reference``.

    - ``data:image/...;base64,...`` → as-is
    - ``https://...`` → pass through (API fetches)
    - local path → read bytes, sniff mime, return data URL
    """
    value = value.strip()
    if not value:
        raise ToolError.invalid_arguments("image reference must not be empty")

    if value.startswith("data:image/"):
        comma = value.find(",")
        if comma < 0:
            raise ToolError.invalid_arguments("malformed data URL in image reference")
        if ";base64" not in value[:comma]:
            raise ToolError.invalid_arguments(
                "image references only support base64 data URLs"
            )
        return value

    if value.startswith("https://"):
        return value

    p = Path(value)
    if not p.is_absolute() and cwd is not None:
        p = (Path(cwd) / p).resolve()
    try:
        raw_bytes = p.read_bytes()
    except OSError as e:
        raise ToolError.invalid_arguments(
            f"image reference not readable: {value} ({e})"
        ) from e
    if not raw_bytes:
        raise ToolError.invalid_arguments("image reference contained no data")

    mime = _sniff_image_mime(raw_bytes)
    if mime is None:
        raise ToolError.invalid_arguments(
            "invalid image reference: unsupported or unrecognised image format"
        )
    b64 = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _sniff_image_mime(data: bytes) -> str | None:
    """Minimal magic-byte sniff (allow-list subset of Grok image_validate)."""
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 2 and data[:2] == b"BM":
        return "image/bmp"
    return None


def _save_video(ctx: ToolCallContext, data: bytes, *, label: str) -> str:
    """Write ``videos/<n>.mp4`` under cwd (session folder stand-in)."""
    out_dir = Path(ctx.cwd) / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while (out_dir / f"{n}.mp4").exists():
        n += 1
    path = out_dir / f"{n}.mp4"
    path.write_bytes(data)
    try:
        rel = path.resolve().relative_to(Path(ctx.cwd).resolve()).as_posix()
    except ValueError:
        rel = str(path)
    # Absolute path is what Grok MediaGenOutput exposes; also give short rel for UX.
    return f"{label} saved: {rel}\nfull_path: {path}"
