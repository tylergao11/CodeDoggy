"""Structured user attachments for model input.

The TUI owns how an attachment is displayed.  The turn layer owns the
model-facing image payload, so local paths never masquerade as vision input.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ARCHIVE_KEY = "_codedoggy_image_attachments"


class AttachmentError(ValueError):
    """A user attachment cannot be converted into model input."""


@dataclass(slots=True)
class ImageAttachment:
    """One local image, encoded lazily and reused across sampling rounds."""

    path: str
    media_type: str
    detail: str | None = None
    _data_url: str | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        detail: str | None = None,
    ) -> ImageAttachment:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise AttachmentError(f"image file does not exist: {resolved}")
        media_type, _ = mimetypes.guess_type(resolved.name)
        if not isinstance(media_type, str) or not media_type.startswith("image/"):
            raise AttachmentError(f"unsupported image type: {resolved.name}")
        return cls(
            path=str(resolved),
            media_type=media_type,
            detail=detail,
        )

    def as_content_part(self) -> dict[str, Any]:
        """Return an OpenAI-style image part understood by provider adapters."""
        if self._data_url is None:
            path = Path(self.path)
            if not path.is_file():
                raise AttachmentError(f"image file does not exist: {path}")
            try:
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            except OSError as exc:
                raise AttachmentError(f"cannot read image file: {path}") from exc
            self._data_url = f"data:{self.media_type};base64,{encoded}"

        image_url: dict[str, str] = {"url": self._data_url}
        if self.detail:
            image_url["detail"] = self.detail
        return {
            "type": "image_url",
            "image_url": image_url,
        }


def provider_data_with_attachments(
    provider_data: dict[str, Any] | None,
    attachments: tuple[ImageAttachment, ...],
) -> dict[str, Any] | None:
    """Embed lightweight attachment manifests in the existing archive envelope."""
    data = dict(provider_data or {})
    if attachments:
        data[_ARCHIVE_KEY] = [
            {
                "path": item.path,
                "media_type": item.media_type,
                **({"detail": item.detail} if item.detail else {}),
            }
            for item in attachments
        ]
    return data or None


def attachments_from_provider_data(
    provider_data: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, tuple[ImageAttachment, ...]]:
    """Restore attachments and remove their private archive envelope."""
    data = dict(provider_data or {})
    raw_items = data.pop(_ARCHIVE_KEY, None)
    attachments: list[ImageAttachment] = []
    if isinstance(raw_items, list):
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            path = raw.get("path")
            media_type = raw.get("media_type")
            if (
                not isinstance(path, str)
                or not Path(path).is_file()
                or not isinstance(media_type, str)
                or not media_type.startswith("image/")
            ):
                continue
            detail = raw.get("detail")
            attachments.append(
                ImageAttachment(
                    path=path,
                    media_type=media_type,
                    detail=detail if isinstance(detail, str) else None,
                )
            )
    return data or None, tuple(attachments)
