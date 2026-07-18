"""WebFetchClient — HTTP fetch with SSRF, redirects, cache, overflow.

Ported from:
  grok-build/.../implementations/grok_build/web_fetch/client.rs
    WebFetchClient::fetch, fetch_url, process_text_content
    media save helpers (simplified SessionFileWriter)
  types/output.rs WebFetchOutput::to_prompt_format strings
"""

from __future__ import annotations

import ssl
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

from codedoggy.tools.grok_build.web_fetch_cache import FetchCache
from codedoggy.tools.grok_build.web_fetch_config import (
    ACCEPT_HEADER,
    ACCEPT_LANGUAGE,
    MAX_REDIRECTS,
    USER_AGENT_STRING,
    WebFetchParams,
)
from codedoggy.tools.grok_build.web_fetch_content import (
    html_to_markdown,
    is_binary_content_type,
    is_html,
    is_image,
    is_pdf,
    is_same_host,
    is_video,
    media_extension,
    strip_base64_data_uris,
    upgrade_to_https,
    validate_media_magic_bytes,
    validate_url,
)
from codedoggy.tools.grok_build.web_fetch_domain import (
    DomainMatcher,
    domain_not_allowed_message,
)
from codedoggy.tools.grok_build.web_fetch_error import WebFetchError
from codedoggy.tools.grok_build.web_fetch_overflow import (
    OverflowResult,
    RecoveryTools,
    inline_budget,
    process_overflow,
)
from codedoggy.tools.util.ssrf import check_ssrf_url


@dataclass
class CrossHostRedirect:
    original_host: str
    redirect_url: str


def cross_host_redirect_message(original_host: str, redirect_url: str) -> str:
    """Grok ``WebFetchOutput::CrossHostRedirect`` prompt format."""
    return (
        f"Error: cross-host redirect from {original_host} to {redirect_url}. "
        "Make a new web_fetch call with the redirect URL if needed."
    )


class _RedirectSeen(Exception):
    """Internal: redirect Location captured without following."""

    def __init__(self, code: int, location: str) -> None:
        self.code = code
        self.location = location


class _CaptureRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        loc = headers.get("Location") or headers.get("location") or newurl
        raise _RedirectSeen(code, loc)


class WebFetchClient:
    """Shared fetch pipeline: validate → SSRF → HTTP → convert → overflow → cache."""

    def __init__(self, params: WebFetchParams | None = None) -> None:
        self.params = params or WebFetchParams()
        self._cache = FetchCache(
            self.params.get_cache_ttl_secs(),
            self.params.get_max_cache_entries(),
        )
        self._lock = threading.Lock()
        # Optional domain allowlist (None on params = no client-side filter).
        # Grok enforces DEFAULT_ALLOWED_DOMAINS via permission manager, not client.
        if self.params.allowed_domains is not None:
            self._domain_matcher: DomainMatcher | None = DomainMatcher(
                self.params.allowed_domains
            )
        else:
            self._domain_matcher = None
        self._ssl_ctx = ssl.create_default_context()
        self._opener = urllib.request.build_opener(
            _CaptureRedirect,
            urllib.request.HTTPSHandler(context=self._ssl_ctx),
        )

    def fetch(
        self,
        raw_url: str,
        session_folder: Path | None = None,
        read_tool_name: str | None = None,
        execute_tool_name: str | None = None,
    ) -> str:
        """Fetch URL; return model-facing prompt string (Grok to_prompt_format)."""
        url = validate_url(raw_url)
        url = upgrade_to_https(url)
        url_str = url

        with self._lock:
            cached = self._cache.get(url_str)
            if cached is not None:
                return cached

        if self._domain_matcher is not None:
            blocked = self._domain_matcher.check(url_str)
            if blocked is not None:
                return domain_not_allowed_message(blocked)

        try:
            check_ssrf_url(url_str)
        except ValueError as e:
            raise WebFetchError(str(e), code="ssrf_blocked") from e

        result = self._fetch_url(url_str)
        if isinstance(result, CrossHostRedirect):
            return cross_host_redirect_message(result.original_host, result.redirect_url)

        body, content_type, final_url, status_code = result

        if is_pdf(content_type):
            return self._save_media(
                session_folder,
                body,
                final_url,
                content_type,
                status_code,
                kind="pdf",
                read_tool_name=read_tool_name,
            )

        if is_image(content_type):
            if not validate_media_magic_bytes(content_type, body):
                raise WebFetchError.content_type_mismatch(content_type, final_url)
            return self._save_media(
                session_folder,
                body,
                final_url,
                content_type,
                status_code,
                kind="image",
                read_tool_name=read_tool_name,
            )

        if is_video(content_type):
            if not validate_media_magic_bytes(content_type, body):
                raise WebFetchError.content_type_mismatch(content_type, final_url)
            return self._save_media(
                session_folder,
                body,
                final_url,
                content_type,
                status_code,
                kind="video",
                read_tool_name=None,
            )

        if is_binary_content_type(content_type):
            raise WebFetchError.unsupported_content_type(content_type, final_url)

        processed = self._process_text(
            body,
            content_type,
            session_folder,
            RecoveryTools(read=read_tool_name, execute=execute_tool_name),
        )
        text_out = processed.content
        with self._lock:
            self._cache.insert_text(url_str, text_out, processed.was_truncated)
        return text_out

    def _process_text(
        self,
        body: bytes,
        content_type: str,
        session_folder: Path | None,
        tools: RecoveryTools,
    ) -> OverflowResult:
        raw_content = body.decode("utf-8", errors="replace")
        if is_html(content_type):
            content = html_to_markdown(raw_content)
            out_type = "markdown"
        else:
            content = raw_content
            out_type = content_type
        content = strip_base64_data_uris(content)
        budget = inline_budget(
            self.params.get_context_window_tokens(),
            self.params.get_max_markdown_length(),
        )
        return process_overflow(content, budget, session_folder, out_type, tools)

    def _fetch_url(self, url: str) -> tuple[bytes, str, str, int] | CrossHostRedirect:
        """Manual same-host redirect loop (Grok ``fetch_url``)."""
        current_url = url
        hops = 0
        max_len = self.params.get_max_content_length()
        timeout = self.params.get_timeout_secs()

        while True:
            req = urllib.request.Request(
                current_url,
                headers={
                    "User-Agent": USER_AGENT_STRING,
                    "Accept": ACCEPT_HEADER,
                    "Accept-Language": ACCEPT_LANGUAGE,
                },
                method="GET",
            )
            try:
                resp = self._opener.open(req, timeout=timeout)
            except _RedirectSeen as redir:
                hops += 1
                if hops > MAX_REDIRECTS:
                    raise WebFetchError.too_many_redirects(MAX_REDIRECTS) from redir
                try:
                    next_url = urljoin(current_url, redir.location)
                except Exception as ex:  # noqa: BLE001
                    raise WebFetchError.invalid_redirect(str(ex)) from ex
                if is_same_host(current_url, next_url):
                    current_url = next_url
                    continue
                host = urlparse(current_url).hostname or "unknown"
                return CrossHostRedirect(original_host=host, redirect_url=next_url)
            except urllib.error.HTTPError as e:
                # Non-redirect error (or redirect not handled)
                if e.code in {301, 302, 303, 307, 308}:
                    location = e.headers.get("Location") or e.headers.get("location")
                    if not location:
                        raise WebFetchError.http_request(
                            f"HTTP {e.code} without Location"
                        ) from e
                    hops += 1
                    if hops > MAX_REDIRECTS:
                        raise WebFetchError.too_many_redirects(MAX_REDIRECTS) from e
                    next_url = urljoin(current_url, location)
                    if is_same_host(current_url, next_url):
                        current_url = next_url
                        continue
                    host = urlparse(current_url).hostname or "unknown"
                    return CrossHostRedirect(original_host=host, redirect_url=next_url)
                body = e.read(max_len + 1)
                if len(body) > max_len:
                    raise WebFetchError.response_too_large(max_len) from e
                ctype = e.headers.get("Content-Type", "text/html")
                return body, ctype, e.geturl() or current_url, e.code
            except urllib.error.URLError as e:
                raise WebFetchError.http_request(str(e.reason)) from e
            except TimeoutError as e:
                raise WebFetchError.http_request(f"timeout: {e}") from e

            with resp:
                status = int(getattr(resp, "status", None) or resp.getcode() or 200)
                ctype = resp.headers.get("Content-Type", "text/html")
                final_url = resp.geturl() or current_url
                body = resp.read(max_len + 1)
                if len(body) > max_len:
                    raise WebFetchError.response_too_large(max_len)
                return body, ctype, final_url, status

    def _save_media(
        self,
        session_folder: Path | None,
        body: bytes,
        final_url: str,
        content_type: str,
        status_code: int,
        *,
        kind: str,
        read_tool_name: str | None,
    ) -> str:
        del final_url, status_code  # kept for signature parity with Grok
        if session_folder is None:
            raise WebFetchError.io_error("session folder is unavailable")
        if kind == "pdf":
            sub, ext = "downloads", "pdf"
        elif kind == "image":
            sub, ext = "images", media_extension(content_type)
        else:
            sub, ext = "videos", media_extension(content_type)
        dir_path = session_folder / sub
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            n = 1
            while (dir_path / f"{n}.{ext}").exists():
                n += 1
            path = dir_path / f"{n}.{ext}"
            path.write_bytes(body)
        except OSError as e:
            raise WebFetchError.io_error(str(e)) from e

        read_hint = (
            f" Use the {read_tool_name} tool to view its contents."
            if read_tool_name
            else ""
        )
        if kind == "pdf":
            return f"PDF downloaded ({len(body)} bytes) and saved to {path}.{read_hint}"
        if kind == "image":
            return (
                f"Image downloaded ({len(body)} bytes, {content_type}) "
                f"and saved to {path}.{read_hint}"
            )
        return (
            f"Video downloaded ({len(body)} bytes, {content_type}) and saved to {path}."
        )


_default_client: WebFetchClient | None = None
_default_lock = threading.Lock()


def get_default_client(params: WebFetchParams | None = None) -> WebFetchClient:
    global _default_client
    if params is not None:
        return WebFetchClient(params)
    with _default_lock:
        if _default_client is None:
            _default_client = WebFetchClient()
        return _default_client
