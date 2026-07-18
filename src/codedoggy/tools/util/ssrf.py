"""SSRF guards for web_fetch.

Ported from:
  grok-build/.../implementations/grok_build/web_fetch/ssrf.rs
    is_blocked_ip, check_ssrf
  grok-build/.../implementations/grok_build/web_fetch/error.rs
    SsrfBlocked / DnsResolution / DnsEmpty / SingleLabelHost messages
    ssrf_recovery_hint, is_github_host, gh_available
"""

from __future__ import annotations

import ipaddress
import shutil
import socket
from urllib.parse import urlparse

# Exact Display strings from error.rs (built in helpers below).


def is_blocked_ip(ip: str | ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True if IP is private/link-local/CGNAT/metadata. Loopback allowed (local dev).

    Grok ``is_blocked_ip`` — octet/segment checks only (not Python ``is_private``).
    """
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv4Address):
        octets = addr.packed
        # Loopback (127.0.0.0/8) — allowed for local dev servers.
        if octets[0] == 127:
            return False
        # RFC 1918: 10.0.0.0/8
        if octets[0] == 10:
            return True
        # RFC 1918: 172.16.0.0/12
        if octets[0] == 172 and 16 <= octets[1] <= 31:
            return True
        # RFC 1918: 192.168.0.0/16
        if octets[0] == 192 and octets[1] == 168:
            return True
        # RFC 3927: 169.254.0.0/16 — link-local / cloud metadata
        if octets[0] == 169 and octets[1] == 254:
            return True
        # RFC 6598: 100.64.0.0/10 — CGNAT
        if octets[0] == 100 and 64 <= octets[1] <= 127:
            return True
        # 0.0.0.0 — unspecified
        if addr.is_unspecified:
            return True
        return False

    # IPv6
    if addr.is_loopback:
        return False
    if addr.is_unspecified:
        return True
    # IPv4-mapped IPv6 (::ffff:x.x.x.x) — delegate to v4 checks.
    mapped = addr.ipv4_mapped
    if mapped is not None:
        return is_blocked_ip(mapped)
    segments0 = (addr.packed[0] << 8) | addr.packed[1]
    # RFC 4291: fe80::/10 — link-local unicast.
    if segments0 & 0xFFC0 == 0xFE80:
        return True
    # RFC 4193: fc00::/7 — unique local address (ULA).
    if segments0 & 0xFE00 == 0xFC00:
        return True
    return False


def is_github_host(host: str) -> bool:
    """Grok ``is_github_host`` — GitHub / GHE-style hostnames."""
    h = host.lower()
    return h == "github.com" or h.endswith(".github.com") or "github" in h


def gh_available() -> bool:
    """Grok ``gh_available`` — ``which::which("gh")``."""
    return shutil.which("gh") is not None


def ssrf_recovery_hint(host: str) -> str:
    """Grok ``ssrf_recovery_hint`` — exact trailing text or empty."""
    if is_github_host(host) and gh_available():
        return ". Use the `gh` CLI instead (e.g. `gh pr view` or `gh api`)."
    return ""


def format_ssrf_blocked(host: str, ip: str | ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    """Grok ``WebFetchError::SsrfBlocked`` Display."""
    return f"SSRF blocked: {host} resolves to private/internal IP {ip}{ssrf_recovery_hint(host)}"


def format_dns_resolution_failed(host: str, source: object) -> str:
    """Grok ``WebFetchError::DnsResolution`` Display."""
    return f"DNS resolution failed for {host}: {source}"


def format_dns_empty(host: str) -> str:
    """Grok ``WebFetchError::DnsEmpty`` Display."""
    return f"DNS resolution returned no addresses for {host}"


def check_ssrf_url(url: str) -> None:
    """Resolve host and raise ``ValueError`` with Grok SSRF/DNS messages if blocked.

    Mirrors ``check_ssrf(&Url)``. Call after scheme/host validation when possible.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if host is None or host == "":
        # Grok: SingleLabelHost with empty host when host_str missing
        raise ValueError("hostname must have at least two dot-separated parts, got: ")

    # Literal IP — check directly (Grok host.parse::<IpAddr>()).
    try:
        ip_obj = ipaddress.ip_address(host)
    except ValueError:
        ip_obj = None

    if ip_obj is not None:
        if is_blocked_ip(ip_obj):
            raise ValueError(format_ssrf_blocked(host, ip_obj))
        return

    port = parsed.port
    if port is None:
        port = 443 if (parsed.scheme or "https") == "https" else 80

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise ValueError(format_dns_resolution_failed(host, e)) from e

    if not infos:
        raise ValueError(format_dns_empty(host))

    for info in infos:
        sockaddr = info[4]
        ip = sockaddr[0]
        # Strip zone id if present (e.g. fe80::1%eth0)
        if isinstance(ip, str) and "%" in ip:
            ip = ip.split("%", 1)[0]
        if is_blocked_ip(ip):
            raise ValueError(format_ssrf_blocked(host, ip))
