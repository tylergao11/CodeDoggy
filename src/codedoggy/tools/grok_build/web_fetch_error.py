"""Structured errors for web_fetch.

Ported from:
  grok-build/.../implementations/grok_build/web_fetch/error.rs
    WebFetchError Display messages (exact)

Subclasses ValueError so existing call-sites matching ValueError keep working.
"""

from __future__ import annotations


class WebFetchError(ValueError):
    """web_fetch failure; ``str(err)`` matches Grok ``thiserror`` Display text."""

    def __init__(self, message: str, *, code: str = "web_fetch") -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    @classmethod
    def url_too_long(cls, max_len: int) -> WebFetchError:
        return cls(
            f"URL exceeds maximum length of {max_len} characters",
            code="url_too_long",
        )

    @classmethod
    def unsupported_scheme(cls, scheme: str) -> WebFetchError:
        return cls(
            f"unsupported URL scheme: {scheme} (only http/https allowed)",
            code="unsupported_scheme",
        )

    @classmethod
    def credentials_in_url(cls) -> WebFetchError:
        return cls("URLs with embedded credentials are not allowed", code="credentials_in_url")

    @classmethod
    def single_label_host(cls, host: str) -> WebFetchError:
        return cls(
            f"hostname must have at least two dot-separated parts, got: {host}",
            code="single_label_host",
        )

    @classmethod
    def invalid_url(cls, detail: str) -> WebFetchError:
        return cls(f"invalid URL: {detail}", code="invalid_url")

    @classmethod
    def ssrf_blocked(cls, host: str, ip: object) -> WebFetchError:
        from codedoggy.tools.util.ssrf import format_ssrf_blocked

        return cls(format_ssrf_blocked(host, ip), code="ssrf_blocked")  # type: ignore[arg-type]

    @classmethod
    def dns_resolution(cls, host: str, source: object) -> WebFetchError:
        from codedoggy.tools.util.ssrf import format_dns_resolution_failed

        return cls(format_dns_resolution_failed(host, source), code="dns_resolution")

    @classmethod
    def dns_empty(cls, host: str) -> WebFetchError:
        from codedoggy.tools.util.ssrf import format_dns_empty

        return cls(format_dns_empty(host), code="dns_empty")

    @classmethod
    def too_many_redirects(cls, max_redirects: int) -> WebFetchError:
        return cls(f"too many redirects (max {max_redirects})", code="too_many_redirects")

    @classmethod
    def response_too_large(cls, max_bytes: int) -> WebFetchError:
        return cls(
            f"response body exceeds maximum size of {max_bytes} bytes",
            code="response_too_large",
        )

    @classmethod
    def invalid_redirect(cls, detail: str) -> WebFetchError:
        return cls(f"invalid redirect URL: {detail}", code="invalid_redirect")

    @classmethod
    def unsupported_content_type(cls, content_type: str, url: str) -> WebFetchError:
        return cls(
            f"unsupported content type {content_type} from {url}",
            code="unsupported_content_type",
        )

    @classmethod
    def content_type_mismatch(cls, content_type: str, url: str) -> WebFetchError:
        return cls(
            f"content body does not match claimed content type {content_type} from {url}",
            code="content_type_mismatch",
        )

    @classmethod
    def http_request(cls, detail: str) -> WebFetchError:
        return cls(f"HTTP request failed: {detail}", code="http_request")

    @classmethod
    def proxy_config(cls, detail: str) -> WebFetchError:
        return cls(f"invalid proxy configuration: {detail}", code="proxy_config")

    @classmethod
    def io_error(cls, detail: str) -> WebFetchError:
        return cls(f"failed to save downloaded file: {detail}", code="io_error")
