"""Rich file readers for read_file (PDF / PPTX / image) — tool-layer only.

Uses stdlib where possible. PDF prefers optional pypdf; otherwise a limited
stream extract with an honest capability note.

Ported from (subset):
  grok-build/.../implementations/read_file/{pdf,pptx,image,metadata}.rs
  grok-build/.../implementations/grok_build/read_file/mod.rs (handle_pptx)

Image path: metadata only here — Grok embeds compressed multimodal payloads
via host ImageContent. Fidelity: C for pdf/pptx text; X for image vision.
"""

from __future__ import annotations

import re
import struct
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

# Grok MAX_PDF_BYTES / MAX_PPTX_BYTES = 50 MiB
MAX_PDF_BYTES: int = 50 * 1024 * 1024
MAX_PPTX_BYTES: int = 50 * 1024 * 1024
MAX_IMAGE_BYTES: int = 20 * 1024 * 1024
# Grok PDF_AUTO_READ_THRESHOLD / PDF_MAX_PAGES_PER_READ
PDF_MAX_PAGES_DEFAULT: int = 10
PDF_MAX_PAGES_EXPLICIT: int = 20


def is_pdf(path: Path, data: bytes) -> bool:
    if path.suffix.lower() == ".pdf":
        return True
    return len(data) >= 5 and data[:5] == b"%PDF-"


def is_pptx(path: Path, data: bytes) -> bool:
    if path.suffix.lower() != ".pptx":
        return False
    return len(data) >= 2 and data[:2] == b"PK"


def is_image(path: Path, data: bytes) -> bool:
    ext = path.suffix.lower()
    if ext in {".png", ".gif", ".ico", ".jpeg", ".webp", ".jpg", ".bmp"}:
        return True
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return True
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return True
    return False


def read_pdf_text(
    data: bytes,
    pages: str | None = None,
    max_chars: int = 100_000,
) -> str:
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(
            f"PDF is {len(data) / 1_048_576:.1f} MB, exceeds the "
            f"{MAX_PDF_BYTES / 1_048_576:.0f} MB limit."
        )
    page_set = _parse_pages(pages)
    try:
        from pypdf import PdfReader  # type: ignore[import-untyped]
        import io

        reader = PdfReader(io.BytesIO(data))
        n = len(reader.pages)
        indices = _select_page_indices(n, page_set)
        parts: list[str] = []
        for i in indices:
            try:
                text = reader.pages[i].extract_text() or ""
            except Exception:
                text = ""
            parts.append(f"--- page {i + 1}/{n} ---\n{text.strip()}")
        body = "\n\n".join(parts)
        header = f"[PDF text extract via pypdf; {len(indices)} of {n} pages]\n\n"
        return _cap(header + body, max_chars)
    except ImportError:
        crude = _crude_pdf_strings(data)
        note = (
            "[PDF text extract (stdlib fallback — install pypdf for better quality); "
            "not a full layout engine]\n\n"
        )
        if page_set is not None:
            note += (
                "[pages= filter ignored without pypdf — showing whole-document "
                "crude extract]\n"
            )
        return _cap(note + crude, max_chars)


def read_pptx_text(data: bytes, max_chars: int = 100_000) -> str:
    if len(data) > MAX_PPTX_BYTES:
        raise ValueError(
            f"PPTX is {len(data) / 1_048_576:.1f} MB, exceeds the "
            f"{MAX_PPTX_BYTES / 1_048_576:.0f} MB limit."
        )
    import io

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ValueError(f"invalid PPTX (zip): {e}") from e

    names = sorted(
        n
        for n in zf.namelist()
        if n.startswith("ppt/slides/slide") and n.endswith(".xml")
    )
    # Numeric order (Grok: slide2 before slide10)
    def _slide_key(name: str) -> int:
        m = re.search(r"slide(\d+)", name)
        return int(m.group(1)) if m else 0

    names = sorted(names, key=_slide_key)
    parts: list[str] = []
    for name in names:
        xml = zf.read(name)
        texts = _xml_text_runs(xml)
        m = re.search(r"slide(\d+)", name)
        slide_no = m.group(1) if m else "?"
        body = "\n".join(t for t in texts if t.strip())
        parts.append(f"--- slide {slide_no} ---\n{body}")
    header = f"[PPTX text extract; {len(parts)} slide(s)]\n\n"
    return _cap(header + "\n\n".join(parts), max_chars)


def read_image_meta(path: Path, data: bytes) -> str:
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image is {len(data) / 1_048_576:.1f} MB, exceeds the "
            f"{MAX_IMAGE_BYTES / 1_048_576:.0f} MB limit."
        )
    w, h, kind = _image_size(data)
    size_kb = len(data) / 1024.0
    lines = [
        f"[image] path={path.name}",
        f"format: {kind or path.suffix.lstrip('.') or 'unknown'}",
        f"size_bytes: {len(data)} ({size_kb:.1f} KiB)",
    ]
    if w is not None and h is not None:
        lines.append(f"dimensions: {w}x{h}")
    lines.append(
        "Note: this tool returns image metadata only (no vision / no base64 payload). "
        "Use a multimodal host path if you need pixel content described."
    )
    return "\n".join(lines)


def _parse_pages(pages: str | None) -> set[int] | None:
    """Parse '1-5', '3', '10-' into 1-based page numbers set. None = default window."""
    if pages is None:
        return None
    s = str(pages).strip()
    if not s:
        return None
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a.strip()) if a.strip() else 1
            if b.strip():
                end = int(b.strip())
                out.update(range(start, end + 1))
            else:
                # open end: marker as negative of start for _select
                out.add(-start)
        else:
            out.add(int(part))
    return out


def _select_page_indices(n_pages: int, page_set: set[int] | None) -> list[int]:
    if n_pages <= 0:
        return []
    if page_set is None:
        return list(range(min(n_pages, PDF_MAX_PAGES_DEFAULT)))
    # open-ended markers stored as negative starts
    open_from = [abs(x) for x in page_set if x < 0]
    explicit = sorted(x for x in page_set if x > 0)
    indices: list[int] = []
    for p in explicit:
        if 1 <= p <= n_pages:
            indices.append(p - 1)
    for start in open_from:
        for p in range(start, n_pages + 1):
            idx = p - 1
            if idx not in indices:
                indices.append(idx)
    # de-dup preserve order
    seen: set[int] = set()
    ordered: list[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    if len(ordered) > PDF_MAX_PAGES_EXPLICIT:
        ordered = ordered[:PDF_MAX_PAGES_EXPLICIT]
    return ordered


def _crude_pdf_strings(data: bytes) -> str:
    try:
        text = data.decode("latin-1", errors="ignore")
    except Exception:
        return "(unable to decode PDF bytes)"
    chunks: list[str] = []
    for m in re.findall(r"\((?:\\.|[^\\)]){2,}\)", text)[:5000]:
        s = m[1:-1]
        s = (
            s.replace("\\n", "\n")
            .replace("\\r", "")
            .replace("\\t", "\t")
            .replace("\\(", "(")
            .replace("\\)", ")")
            .replace("\\\\", "\\")
        )
        s = re.sub(r"\\[0-7]{1,3}", "", s)
        s = s.strip()
        if s:
            chunks.append(s)
    if not chunks:
        return "(no extractable text streams found)"
    return "\n".join(chunks)


def _xml_text_runs(xml: bytes) -> list[str]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    texts: list[str] = []
    for el in root.iter():
        tag = el.tag
        if tag.endswith("}t") or tag == "t":
            if el.text:
                texts.append(el.text)
    return texts


def _image_size(data: bytes) -> tuple[int | None, int | None, str | None]:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h), "png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                break
            marker = data[i + 1]
            if marker in (192, 193, 194):  # SOF0/1/2
                h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                return int(w), int(h), "jpeg"
            if marker == 0xD9 or marker == 0xDA:
                break
            length = struct.unpack(">H", data[i + 2 : i + 4])[0]
            i += 2 + length
        return None, None, "jpeg"
    if len(data) >= 10 and data[:6] in (b"GIF87a", b"GIF89a"):
        w, h = struct.unpack("<HH", data[6:10])
        return int(w), int(h), "gif"
    return None, None, None


def _cap(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 40)] + "\n… (content truncated)"
