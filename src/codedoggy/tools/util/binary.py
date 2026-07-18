"""Binary file detection for read tools."""

from __future__ import annotations

# PDF / pptx intentionally excluded — dedicated handlers elsewhere.
BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        "7z",
        "a",
        "avi",
        "avif",
        "bin",
        "bmp",
        "class",
        "dat",
        "dll",
        "doc",
        "docx",
        "dylib",
        "exe",
        "gif",
        "gz",
        "ico",
        "jar",
        "jpeg",
        "jpg",
        "lib",
        "mov",
        "mp3",
        "mp4",
        "o",
        "obj",
        "odp",
        "ods",
        "odt",
        "png",
        "ppt",
        "pyc",
        "pyd",
        "pyo",
        "qoi",
        "rar",
        "so",
        "tar",
        "tif",
        "tiff",
        "war",
        "wasm",
        "webp",
        "xls",
        "xlsx",
        "zip",
    }
)

_SAMPLE_SIZE = 8192
_NON_PRINTABLE_THRESHOLD = 0.3


def is_binary(extension: str, data: bytes) -> bool:
    """True if extension is known-binary or content looks non-text."""
    ext = extension.lower().lstrip(".")
    if ext in BINARY_EXTENSIONS:
        return True
    if not data:
        return False

    sample = data[: min(len(data), _SAMPLE_SIZE)]
    if 0x00 in sample:
        return True

    non_printable = sum(1 for b in sample if b < 9 or 14 <= b <= 31)
    return (non_printable / len(sample)) > _NON_PRINTABLE_THRESHOLD
