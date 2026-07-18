"""URL validation, content-type helpers, base64 strip, HTML→text for web_fetch.

Ported from:
  grok-build/.../implementations/grok_build/web_fetch/client.rs
    validate_url, upgrade_to_https, is_same_host
    is_html, is_pdf, is_image, is_video, is_binary_content_type
    validate_media_magic_bytes, media_extension
    strip_base64_data_uris
  (html_to_markdown is A-grade: no htmd; tag strip + structure approx)
"""

from __future__ import annotations

import html as html_lib
import re
from urllib.parse import urlparse, urlunparse

from codedoggy.tools.grok_build.web_fetch_config import MAX_URL_LENGTH
from codedoggy.tools.grok_build.web_fetch_error import WebFetchError

# ── URL validation (client.rs) ───────────────────────────────────────


def validate_url(raw: str) -> str:
    """Validate scheme, length, credentials, hostname labels. Returns normalized URL string."""
    if len(raw) > MAX_URL_LENGTH:
        raise WebFetchError.url_too_long(MAX_URL_LENGTH)

    # Absolute URL required (Rust url::Url::parse rejects relative / garbage).
    if "://" not in raw:
        raise WebFetchError.invalid_url("relative URL without a base")

    try:
        parsed = urlparse(raw)
    except Exception as e:  # noqa: BLE001
        raise WebFetchError.invalid_url(str(e)) from e

    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        if not scheme:
            raise WebFetchError.invalid_url("relative URL without a base")
        raise WebFetchError.unsupported_scheme(scheme)

    if not parsed.netloc and not parsed.hostname:
        raise WebFetchError.invalid_url("empty host")

    # Credentials
    if parsed.username or parsed.password:
        raise WebFetchError.credentials_in_url()

    host = parsed.hostname or ""
    # Grok: host.split('.').count() < 2  (IPv6 / localhost → single part → reject)
    if len(host.split(".")) < 2:
        raise WebFetchError.single_label_host(host)

    return raw


def upgrade_to_https(url: str) -> str:
    """Upgrade ``http://`` to ``https://`` (Grok ``upgrade_to_https``)."""
    parsed = urlparse(url)
    if parsed.scheme == "http":
        return urlunparse(parsed._replace(scheme="https"))
    return url


def is_same_host(a: str, b: str) -> bool:
    """Same-host redirect check with www. stripping."""

    def strip_www(h: str) -> str:
        return h[4:] if h.lower().startswith("www.") else h

    ha = urlparse(a).hostname or ""
    hb = urlparse(b).hostname or ""
    return strip_www(ha.lower()) == strip_www(hb.lower())


# ── Content type detection ───────────────────────────────────────────


def _mime_main(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def is_html(content_type: str) -> bool:
    return "text/html" in content_type or "application/xhtml" in content_type


def is_pdf(content_type: str) -> bool:
    return "application/pdf" in content_type


def is_image(content_type: str) -> bool:
    """Image types excluding SVG (XSS vector)."""
    mime = _mime_main(content_type)
    return mime.startswith("image/") and mime != "image/svg+xml"


def is_video(content_type: str) -> bool:
    return _mime_main(content_type).startswith("video/")


def is_binary_content_type(content_type: str) -> bool:
    """True for binary types that would garbage through lossy UTF-8."""
    mime = _mime_main(content_type)
    if mime.startswith("text/"):
        return False
    return mime not in {
        "application/json",
        "application/xml",
        "application/javascript",
        "application/ecmascript",
        "application/x-javascript",
        "application/xhtml+xml",
        "application/rss+xml",
        "application/atom+xml",
        "application/soap+xml",
        "application/xslt+xml",
        "application/mathml+xml",
        "application/svg+xml",
        "application/x-www-form-urlencoded",
        "application/graphql",
        "application/ld+json",
        "application/schema+json",
        "application/vnd.api+json",
        "application/x-yaml",
        "application/yaml",
        "application/toml",
    }


def validate_media_magic_bytes(content_type: str, body: bytes) -> bool:
    """Match magic bytes for claimed type; fail-open for unknown subtypes."""
    mime = _mime_main(content_type)
    if mime == "image/png":
        return body.startswith(b"\x89PNG")
    if mime == "image/jpeg":
        return body.startswith(b"\xff\xd8\xff")
    if mime == "image/gif":
        return body.startswith(b"GIF8")
    if mime == "image/webp":
        return len(body) >= 12 and body[:4] == b"RIFF" and body[8:12] == b"WEBP"
    if mime == "video/mp4":
        return len(body) >= 8 and body[4:8] == b"ftyp"
    if mime == "video/webm":
        return body.startswith(b"\x1a\x45\xdf\xa3")
    return True


def media_extension(content_type: str) -> str:
    mime = _mime_main(content_type)
    return {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/bmp": "bmp",
        "image/tiff": "tiff",
        "video/mp4": "mp4",
        "video/webm": "webm",
        "video/quicktime": "mov",
        "video/x-msvideo": "avi",
    }.get(mime, "bin")


# ── Base64 data URI stripping (client.rs strip_base64_data_uris) ─────

_MIN_BASE64_PAYLOAD = 4
_MAX_HEADER_LEN = 120


def strip_base64_data_uris(content: str) -> str:
    """Strip base64 data URIs; exact Grok manual scanner behavior."""
    if "data:" not in content:
        return content

    s = content
    result: list[str] = []
    last_end = 0
    search_from = 0

    while True:
        rel = s.find("data:", search_from)
        if rel < 0:
            break
        start = rel

        # "data:" must look like a URI scheme start, not mid-word.
        if start > 0 and s[start - 1].isalnum():
            search_from = start + 5
            continue

        comma_rel = s.find(",", start)
        if comma_rel < 0:
            search_from = start + 5
            continue
        comma = comma_rel
        header = s[start + 5 : comma]

        if len(header) > _MAX_HEADER_LEN or any(c.isspace() for c in header):
            search_from = start + 5
            continue

        parts = header.split(";")
        mime = parts[0] if parts[0] else "unknown"
        has_base64 = any(p.lower() == "base64" for p in parts[1:])
        if not has_base64:
            search_from = start + 5
            continue

        payload_start = comma + 1
        payload_len = 0
        for ch in s[payload_start:]:
            o = ord(ch)
            if (
                (65 <= o <= 90)
                or (97 <= o <= 122)
                or (48 <= o <= 57)
                or ch in "+/="
            ):
                payload_len += 1
            else:
                break

        if payload_len >= _MIN_BASE64_PAYLOAD:
            result.append(s[last_end:start])
            result.append(f"[base64 {mime} data removed]")
            last_end = payload_start + payload_len
            search_from = last_end
            continue

        search_from = start + 5

    if last_end == 0:
        return content
    result.append(s[last_end:])
    return "".join(result)


# ── HTML → markdown-ish (A-grade: no htmd) ───────────────────────────

_SKIP_BLOCK_RE = re.compile(
    r"<(script|style|noscript|svg|iframe|object|embed)\b[^>]*>[\s\S]*?</\1\s*>",
    re.I,
)
_BOILERPLATE_RE = re.compile(
    r"<(nav|header|footer)\b[^>]*>[\s\S]*?</\1\s*>",
    re.I,
)
_HEADING_RE = re.compile(r"<h([1-6])\b[^>]*>([\s\S]*?)</h\1\s*>", re.I)
_A_RE = re.compile(r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a\s*>', re.I)
_CODE_RE = re.compile(r"<pre\b[^>]*>\s*<code\b[^>]*>([\s\S]*?)</code\s*>\s*</pre\s*>", re.I)
_LI_RE = re.compile(r"<li\b[^>]*>([\s\S]*?)</li\s*>", re.I)
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_P_RE = re.compile(r"</p\s*>", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n{3,}")


def html_to_markdown(html_src: str) -> str:
    """Approximate HTML→markdown (Grok uses htmd; this is structure-preserving A)."""
    s = html_src
    s = _SKIP_BLOCK_RE.sub("", s)
    s = _BOILERPLATE_RE.sub("", s)

    def _heading(m: re.Match[str]) -> str:
        level = int(m.group(1))
        inner = _TAG_RE.sub("", m.group(2)).strip()
        return "\n" + ("#" * level) + " " + inner + "\n"

    s = _HEADING_RE.sub(_heading, s)

    def _link(m: re.Match[str]) -> str:
        href = m.group(1)
        text = _TAG_RE.sub("", m.group(2)).strip() or href
        return f"[{text}]({href})"

    s = _A_RE.sub(_link, s)

    def _code(m: re.Match[str]) -> str:
        body = html_lib.unescape(m.group(1))
        return f"\n```\n{body.strip()}\n```\n"

    s = _CODE_RE.sub(_code, s)
    s = _LI_RE.sub(lambda m: "\n- " + _TAG_RE.sub("", m.group(1)).strip(), s)
    s = _BR_RE.sub("\n", s)
    s = _P_RE.sub("\n\n", s)
    s = _TAG_RE.sub("\n", s)
    s = html_lib.unescape(s)
    s = _WS_RE.sub("\n\n", s)
    return s.strip()
