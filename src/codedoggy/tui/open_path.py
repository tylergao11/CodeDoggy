"""Open local files (images, etc.) with the OS default application.

Used by the TUI so image_gen/image_edit paths are clickable like Grok's
session-relative links.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff")

# Absolute Windows/Unix paths, or relative images/n.jpg
_PATH_RE = re.compile(
    r"(?P<path>"
    r"(?:[A-Za-z]:[\\/][^\s\"'<>|]+)"  # Windows abs
    r"|(?:/[^\s\"'<>|]+)"  # Unix abs
    r"|(?:images[/\\]\d+\.(?:jpe?g|png|webp|gif|bmp))"  # session relative
    r")",
    re.IGNORECASE,
)


def extract_image_paths(text: str | None) -> list[str]:
    """Pull image paths from tool JSON / prose (order-preserving, unique)."""
    if not text or not str(text).strip():
        return []
    raw = str(text)
    found: list[str] = []
    seen: set[str] = set()

    def add(p: str) -> None:
        key = p.replace("\\", "/").lower()
        if key in seen:
            return
        if not any(key.endswith(ext) for ext in _IMAGE_EXTS):
            return
        seen.add(key)
        found.append(p)

    # Prefer MediaGen JSON from image_gen
    stripped = raw.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                for key in ("path", "filename"):
                    val = data.get(key)
                    if isinstance(val, str) and val.strip():
                        add(val.strip())
                # relative folder + filename
                folder = data.get("session_folder")
                name = data.get("filename")
                if (
                    isinstance(folder, str)
                    and isinstance(name, str)
                    and folder.strip()
                    and name.strip()
                ):
                    add(f"{folder.strip()}/{name.strip()}")
        except json.JSONDecodeError:
            pass

    for match in _PATH_RE.finditer(raw):
        add(match.group("path").strip().rstrip(".,);]"))

    return found


def resolve_openable_path(
    path: str,
    *,
    cwd: str | Path | None = None,
) -> Path | None:
    """Resolve to an existing file, or None."""
    raw = (path or "").strip().strip("\"'")
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()
    if cwd is not None:
        joined = Path(cwd) / raw
        if joined.is_file():
            return joined.resolve()
        # images/1.jpg under cwd
        if not raw.startswith(("images/", "images\\")):
            alt = Path(cwd) / "images" / Path(raw).name
            if alt.is_file():
                return alt.resolve()
    return None


def open_local_path(
    path: str,
    *,
    cwd: str | Path | None = None,
) -> tuple[bool, str]:
    """Open with the OS default app. Returns (ok, message)."""
    resolved = resolve_openable_path(path, cwd=cwd)
    if resolved is None:
        return False, f"文件不存在: {path}"
    try:
        if sys.platform == "win32":
            os.startfile(str(resolved))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(resolved)])  # noqa: S603
        else:
            subprocess.Popen(["xdg-open", str(resolved)])  # noqa: S603
        return True, f"已打开 {resolved.name}"
    except OSError as exc:
        return False, f"无法打开: {exc}"


def paths_from_detail_record(record: Any) -> list[str]:
    """Collect image paths from a DetailRecord's blocks/title."""
    chunks: list[str] = []
    title = getattr(record, "title", None)
    if title:
        chunks.append(str(title))
    for block in getattr(record, "blocks", ()) or ():
        text = getattr(block, "text", None)
        if text:
            chunks.append(str(text))
    out: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        for p in extract_image_paths(chunk):
            key = p.replace("\\", "/").lower()
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out
