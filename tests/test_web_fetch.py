"""Focused tests for web_fetch S-port (SSRF, validate, domain, content, overflow)."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.tools.grok_build.web_fetch_cache import FetchCache
from codedoggy.tools.grok_build.web_fetch_config import (
    MAX_URL_LENGTH,
    USER_AGENT_STRING,
    WebFetchParams,
)
from codedoggy.tools.grok_build.web_fetch_content import (
    html_to_markdown,
    is_binary_content_type,
    is_html,
    is_pdf,
    is_same_host,
    strip_base64_data_uris,
    upgrade_to_https,
    validate_url,
)
from codedoggy.tools.grok_build.web_fetch_domain import (
    DomainMatcher,
    domain_from_url,
    domain_not_allowed_message,
    normalize_domain,
)
from codedoggy.tools.grok_build.web_fetch_error import WebFetchError
from codedoggy.tools.grok_build.web_fetch_overflow import (
    RecoveryTools,
    inline_budget,
    process_overflow,
    truncate_str,
)
from codedoggy.tools.util.ssrf import (
    check_ssrf_url,
    format_ssrf_blocked,
    is_blocked_ip,
    is_github_host,
)


# ── is_blocked_ip (ssrf.rs tests) ────────────────────────────────────


def test_blocks_rfc1918() -> None:
    assert is_blocked_ip("10.0.0.1")
    assert is_blocked_ip("10.255.255.255")
    assert is_blocked_ip("172.16.0.1")
    assert is_blocked_ip("172.31.255.255")
    assert not is_blocked_ip("172.15.0.1")
    assert not is_blocked_ip("172.32.0.1")
    assert is_blocked_ip("192.168.0.1")
    assert is_blocked_ip("192.168.255.255")


def test_blocks_link_local_and_cgnat() -> None:
    assert is_blocked_ip("169.254.0.1")
    assert is_blocked_ip("169.254.169.254")
    assert is_blocked_ip("100.64.0.1")
    assert is_blocked_ip("100.127.255.255")
    assert not is_blocked_ip("100.63.0.1")
    assert not is_blocked_ip("100.128.0.1")


def test_blocks_unspecified_allows_loopback_and_public() -> None:
    assert is_blocked_ip("0.0.0.0")
    assert is_blocked_ip("::")
    assert not is_blocked_ip("127.0.0.1")
    assert not is_blocked_ip("127.0.0.2")
    assert not is_blocked_ip("::1")
    assert not is_blocked_ip("1.1.1.1")
    assert not is_blocked_ip("8.8.8.8")


def test_ipv6_ula_and_mapped() -> None:
    assert is_blocked_ip("fe80::1")
    assert is_blocked_ip("fc00::1")
    assert is_blocked_ip("fd00::1")
    assert is_blocked_ip("::ffff:10.0.0.1")
    assert is_blocked_ip("::ffff:192.168.1.1")
    assert not is_blocked_ip("::ffff:8.8.8.8")


def test_ssrf_blocks_ip_literal_private() -> None:
    with pytest.raises(ValueError, match="private/internal IP"):
        check_ssrf_url("https://10.0.0.1/secret")
    with pytest.raises(ValueError, match="SSRF"):
        check_ssrf_url("http://192.168.0.5/secret")


def test_ssrf_allows_ip_literal_public() -> None:
    check_ssrf_url("https://1.1.1.1/")


def test_ssrf_message_exact_shape() -> None:
    msg = format_ssrf_blocked("10.0.0.5", "10.0.0.5")
    assert msg == "SSRF blocked: 10.0.0.5 resolves to private/internal IP 10.0.0.5"


def test_github_host_detection() -> None:
    assert is_github_host("github.com")
    assert is_github_host("api.github.com")
    assert is_github_host("github.ghe.example.com")
    assert not is_github_host("ghe.example.com")
    assert not is_github_host("gitlab.example.com")


# ── validate_url ─────────────────────────────────────────────────────


def test_validate_url_accepts_valid() -> None:
    assert validate_url("https://docs.rs/reqwest/latest")
    assert validate_url("https://github.com/seanmonstar/reqwest")
    assert validate_url("http://example.com/path?q=1#frag")


def test_validate_url_rejects_single_label() -> None:
    with pytest.raises(WebFetchError, match="two dot-separated parts"):
        validate_url("http://localhost:8080/foo")
    with pytest.raises(WebFetchError, match="two dot-separated parts"):
        validate_url("http://intranet/foo")


def test_validate_url_rejects_credentials() -> None:
    with pytest.raises(WebFetchError, match="embedded credentials"):
        validate_url("https://user:pass@example.com/foo")
    with pytest.raises(WebFetchError, match="embedded credentials"):
        validate_url("https://admin@example.com/foo")


def test_validate_url_rejects_long() -> None:
    long = f"https://example.com/{'a' * MAX_URL_LENGTH}"
    with pytest.raises(WebFetchError, match="maximum length"):
        validate_url(long)


def test_validate_url_rejects_schemes() -> None:
    with pytest.raises(WebFetchError, match="unsupported URL scheme: ftp"):
        validate_url("ftp://example.com/file.txt")
    with pytest.raises(WebFetchError, match="unsupported URL scheme: file"):
        validate_url("file:///etc/passwd")


def test_upgrade_http_to_https() -> None:
    assert upgrade_to_https("http://example.com/path").startswith("https://")
    assert upgrade_to_https("https://example.com/path").startswith("https://")


def test_same_host() -> None:
    assert is_same_host("https://example.com/a", "https://example.com/b")
    assert is_same_host("https://example.com/a", "https://www.example.com/a")
    assert not is_same_host("https://example.com/a", "https://other.com/a")


# ── domain matcher ───────────────────────────────────────────────────


def test_normalize_domain() -> None:
    assert normalize_domain("www.Example.COM.") == "example.com"
    assert normalize_domain("  docs.rs  ") == "docs.rs"


def test_domain_matcher_host_only() -> None:
    m = DomainMatcher(["docs.rs", "Example.Com"])
    assert m.check("https://docs.rs/reqwest/latest") is None
    assert m.check("https://example.com/page") is None
    assert m.check("https://evil.com/steal") == "evil.com"


def test_domain_matcher_empty_blocks_all() -> None:
    m = DomainMatcher([])
    assert m.check("https://docs.python.org/3/") is not None


def test_domain_matcher_path_scoped() -> None:
    m = DomainMatcher(["vercel.com/docs"])
    assert m.check("https://vercel.com/docs") is None
    assert m.check("https://vercel.com/docs/foo") is None
    assert m.check("https://vercel.com/api") == "vercel.com"
    assert m.check("https://vercel.com/docs-internal") == "vercel.com"
    assert m.check("https://vercel.com/docs/guide") is None


def test_domain_matcher_host_overrides_path() -> None:
    m = DomainMatcher(["github.com/docs", "github.com"])
    assert m.check("https://github.com/anything") is None


def test_domain_from_url() -> None:
    assert domain_from_url("https://docs.python.org/3/library/asyncio.html") == (
        "docs.python.org"
    )
    assert domain_from_url("https://www.React.Dev/learn") == "react.dev"
    assert domain_from_url("not a url") is None


def test_domain_not_allowed_message() -> None:
    assert domain_not_allowed_message("evil.com") == (
        "Error: domain evil.com is not in the allowed domains list"
    )


# ── content helpers ──────────────────────────────────────────────────


def test_is_html_pdf_binary() -> None:
    assert is_html("text/html; charset=utf-8")
    assert is_html("application/xhtml+xml")
    assert not is_html("text/plain")
    assert is_pdf("application/pdf")
    assert is_binary_content_type("image/png")
    assert is_binary_content_type("application/octet-stream")
    assert not is_binary_content_type("text/plain")
    assert not is_binary_content_type("application/json")
    assert not is_binary_content_type("application/yaml")


def test_strip_base64_output_format() -> None:
    result = strip_base64_data_uris(
        "Before ![logo](data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==) after"
    )
    assert result == "Before ![logo]([base64 image/png data removed]) after"


def test_strip_base64_rejects_invalid() -> None:
    cases = [
        "data: image/png ;base64,AAAA== end",
        "data:image/png;base64,AA= end",
        "data:image/png;base64, end",
        "data:image/png;base64 with no comma",
        "trailing data:",
        "metadata:foo;base64,AAAA==",
    ]
    for c in cases:
        assert strip_base64_data_uris(c) == c


def test_html_to_markdown_basic() -> None:
    md = html_to_markdown("<h1>Hello</h1><p>World</p>")
    assert "Hello" in md
    assert "World" in md
    md2 = html_to_markdown(
        '<h1>Title</h1><script>alert("hi")</script><style>body{}</style><p>Content</p>'
    )
    assert "Title" in md2 and "Content" in md2
    assert "alert" not in md2


# ── overflow / cache ─────────────────────────────────────────────────


def test_truncate_str_utf8() -> None:
    assert truncate_str("hello", 10) == "hello"
    assert truncate_str("hello", 0) == ""
    # multi-byte
    assert truncate_str("ééé", 5) in {"éé", "é"}  # 2 bytes each for é


def test_inline_budget() -> None:
    b = inline_budget(1_000_000, 100_000)
    assert b.preview_bytes == 100_000
    assert b.output_bytes == 100_000
    assert inline_budget(1_000_000, 20_000).preview_bytes == 20_000


def test_overflow_exact_limit_and_one_over(tmp_path: Path) -> None:
    from codedoggy.tools.grok_build.web_fetch_overflow import InlineBudget

    budget = InlineBudget(preview_bytes=100, output_bytes=512)
    exact = process_overflow(
        "a" * 100,
        budget,
        tmp_path,
        "text/plain",
        RecoveryTools(read="ReadAsset", execute="ExecuteAsset"),
    )
    assert not exact.was_truncated
    assert exact.content == "a" * 100

    one_over = process_overflow(
        "b" * 101,
        budget,
        tmp_path,
        "text/plain",
        RecoveryTools(read="ReadAsset", execute="ExecuteAsset"),
    )
    assert one_over.was_truncated
    assert "web_fetch content truncated" in one_over.content
    assert (tmp_path / "web_fetch" / "1.txt").read_text(encoding="utf-8") == "b" * 101


def test_cache_skips_truncated() -> None:
    cache = FetchCache(60, 10)
    cache.insert_text("https://example.com/", "path/to/artifact", True)
    assert cache.get("https://example.com/") is None
    cache.insert_text("https://example.com/", "fully inline", False)
    assert cache.get("https://example.com/") == "fully inline"


def test_user_agent_string() -> None:
    assert "grok-agent/1.0" in USER_AGENT_STRING


def test_params_defaults() -> None:
    p = WebFetchParams()
    assert p.get_max_content_length() == 10 * 1024 * 1024
    assert p.get_max_markdown_length() == 100_000
    assert p.get_timeout_secs() == 60.0


def test_error_messages_exact() -> None:
    assert str(WebFetchError.url_too_long(2000)) == (
        "URL exceeds maximum length of 2000 characters"
    )
    assert str(WebFetchError.credentials_in_url()) == (
        "URLs with embedded credentials are not allowed"
    )
    assert str(WebFetchError.too_many_redirects(10)) == "too many redirects (max 10)"
    assert str(WebFetchError.response_too_large(100)) == (
        "response body exceeds maximum size of 100 bytes"
    )
