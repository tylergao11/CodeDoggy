"""Open local files (images, scripts, etc.) with the OS default application.

Used by the TUI so tool paths and pasted attachments are Ctrl+clickable.
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
_CODE_EXTS = (
    ".py",
    ".pyi",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".rs",
    ".go",
    ".java",
    ".kt",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".lua",
    ".sh",
    ".ps1",
    ".css",
    ".html",
    ".vue",
    ".svelte",
    ".sql",
    ".xml",
    ".ini",
    ".cfg",
    ".env",
)

VIEW_IMAGE_LABEL = "查看图片"
OPEN_FILE_LABEL = "打开文件"

_IMAGE_CHIP_RE = re.compile(
    rf"{re.escape(VIEW_IMAGE_LABEL)}\("
    r"(?P<path>\"[^\"]+\"|'[^']+'|[^)]+)"
    r"\)"
)

_TOOL_PATH_KEYS = (
    "path",
    "target_file",
    "file_path",
    "file",
    "filename",
)

# Absolute Windows/Unix paths, session-relative images, or attachment pastes.
_PATH_RE = re.compile(
    r"(?P<path>"
    r"(?:[A-Za-z]:[\\/][^\s\"'<>|]+)"  # Windows abs
    r"|(?:/[^\s\"'<>|]+)"  # Unix abs
    r"|(?:images[/\\]\d+\.(?:jpe?g|png|webp|gif|bmp))"  # session relative
    r"|(?:\.codedoggy[/\\]attachments[/\\][^\s\"'<>|]+)"  # pasted attachments
    r")",
    re.IGNORECASE,
)


def is_image_path(path: str | None) -> bool:
    if not path:
        return False
    key = str(path).replace("\\", "/").lower()
    return any(key.endswith(ext) for ext in _IMAGE_EXTS)


def is_openable_file_path(path: str | None) -> bool:
    """True for images, common source files, or known attachment pastes."""
    if not path:
        return False
    key = str(path).replace("\\", "/").lower().rstrip(".,);]")
    if any(key.endswith(ext) for ext in _IMAGE_EXTS + _CODE_EXTS):
        return True
    return "/.codedoggy/attachments/" in key or "\\.codedoggy\\attachments\\" in str(
        path
    ).lower()


def link_label_for_path(path: str) -> str:
    """Short filename (or image label) for TUI open links — no duplicated verbs.

    Callers compose a single verb once, e.g. ``打开 · {label}`` or
    ``查看图片 · {label}``. Do not embed "打开文件" here (it stacked with
    "点击打开" as "点击打开 打开文件 foo.py").
    """
    name = Path(str(path).strip().strip("\"'")).name
    if is_image_path(path):
        return name or VIEW_IMAGE_LABEL
    return name or "file"


def extract_image_paths(text: str | None) -> list[str]:
    """Pull image paths from tool JSON / prose (order-preserving, unique)."""
    return [p for p in extract_file_paths(text) if is_image_path(p)]


def extract_image_chip_paths(text: str | None) -> list[str]:
    """Return the canonical local paths represented by pasted image chips."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for match in _IMAGE_CHIP_RE.finditer(str(text)):
        path = match.group("path").strip().strip("\"'")
        key = path.replace("\\", "/").lower()
        if not path or key in seen:
            continue
        seen.add(key)
        found.append(path)
    return found


def strip_image_chips(text: str | None) -> str:
    """Remove display-only image chips before sending text to the model."""
    if not text:
        return ""
    return _IMAGE_CHIP_RE.sub("", str(text)).strip()


def extract_file_paths(text: str | None) -> list[str]:
    """Pull openable file paths from tool JSON / prose (order-preserving, unique)."""
    if not text or not str(text).strip():
        return []
    raw = str(text)
    found: list[str] = []
    seen: set[str] = set()

    def add(p: str) -> None:
        cleaned = p.strip().strip("\"'").rstrip(".,);]")
        if not cleaned or not is_openable_file_path(cleaned):
            return
        key = cleaned.replace("\\", "/").lower()
        if key in seen:
            return
        seen.add(key)
        found.append(cleaned)

    stripped = raw.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                for key in _TOOL_PATH_KEYS:
                    val = data.get(key)
                    if isinstance(val, str) and val.strip():
                        add(val.strip())
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
        add(match.group("path"))

    # ``path: foo.py`` / ``file_path = bar.ts`` lines from 调用参数 blocks
    for match in re.finditer(
        r"(?:path|target_file|file_path|file|filename)\s*[:=]\s*[\"']?([^\s\"']+)",
        raw,
        re.IGNORECASE,
    ):
        add(match.group(1))

    return found


def tool_paths_from_arguments(arguments: Any) -> tuple[str, ...]:
    """Full paths from a tool-call arguments mapping (Write / Read / edit…)."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return tuple(extract_file_paths(arguments))
    if not isinstance(arguments, dict):
        return ()
    found: list[str] = []
    seen: set[str] = set()
    for key in _TOOL_PATH_KEYS:
        val = arguments.get(key)
        if not isinstance(val, str) or not val.strip():
            continue
        cleaned = val.strip().strip("\"'")
        key_l = cleaned.replace("\\", "/").lower()
        if key_l in seen:
            continue
        # Accept any non-empty tool path (even without known extension).
        if not cleaned:
            continue
        seen.add(key_l)
        found.append(cleaned)
    return tuple(found)


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
        if not raw.startswith(("images/", "images\\")):
            alt = Path(cwd) / "images" / Path(raw).name
            if alt.is_file():
                return alt.resolve()
        # Bare basename often comes from collapsed tool headlines.
        name = Path(raw).name
        if name and name != raw:
            for root in (Path(cwd), Path(cwd) / ".codedoggy" / "attachments"):
                hit = root / name
                if hit.is_file():
                    return hit.resolve()
    return None


def open_local_path(
    path: str,
    *,
    cwd: str | Path | None = None,
) -> tuple[bool, str]:
    """Open with the OS default app without sharing the TUI terminal.

    GUI launchers may attach their stdout/stderr to the parent console even
    when invoked through ``os.startfile``.  In a full-screen prompt_toolkit
    application that writes straight through the alternate screen and corrupts
    the detail view.  Always launch through a detached, silenced helper.
    """
    resolved = resolve_openable_path(path, cwd=cwd)
    if resolved is None:
        return False, f"文件不存在: {path}"
    try:
        if sys.platform == "win32":
            system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
            helper = system_root / "System32" / "rundll32.exe"
            if not helper.is_file():
                raise OSError(f"系统打开器不存在: {helper}")
            creationflags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
            subprocess.Popen(  # noqa: S603
                [
                    str(helper),
                    "url.dll,FileProtocolHandler",
                    str(resolved),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                creationflags=creationflags,
            )
        elif sys.platform == "darwin":
            subprocess.Popen(  # noqa: S603
                ["open", str(resolved)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
        else:
            subprocess.Popen(  # noqa: S603
                ["xdg-open", str(resolved)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
        if is_image_path(str(resolved)):
            return True, f"已打开 {VIEW_IMAGE_LABEL} · {resolved.name}"
        return True, f"已打开 {resolved.name}"
    except OSError as exc:
        return False, f"无法打开: {exc}"


def paths_from_detail_record(record: Any) -> list[str]:
    """Collect openable paths from a DetailRecord (tool args + body text)."""
    out: list[str] = []
    seen: set[str] = set()

    def add_many(paths: list[str] | tuple[str, ...]) -> None:
        for p in paths:
            key = str(p).replace("\\", "/").lower()
            if key not in seen:
                seen.add(key)
                out.append(p)

    stored = getattr(record, "open_paths", None) or ()
    add_many(tuple(str(p) for p in stored if p))

    title = getattr(record, "title", None)
    if title:
        add_many(extract_file_paths(str(title)))
    for block in getattr(record, "blocks", ()) or ():
        text = getattr(block, "text", None)
        if text:
            add_many(extract_file_paths(str(text)))
    return out


def path_under_cursor(text: str, index: int) -> str | None:
    """Best-effort path token covering ``index`` in a prompt buffer."""
    if not text or index < 0 or index > len(text):
        return None
    for match in _IMAGE_CHIP_RE.finditer(text):
        if match.start() <= index <= match.end():
            return match.group("path").strip().strip("\"'")
    # Expand to whitespace-delimited token.
    left = index
    while left > 0 and not text[left - 1].isspace():
        left -= 1
    right = index
    while right < len(text) and not text[right].isspace():
        right += 1
    token = text[left:right].strip().strip("\"'")
    if not token:
        return None
    if is_openable_file_path(token) or "/" in token or "\\" in token or "." in token:
        return token
    return None
