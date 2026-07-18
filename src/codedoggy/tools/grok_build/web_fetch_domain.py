"""Domain allowlist matching for web_fetch.

Ported from:
  grok-build/.../implementations/grok_build/web_fetch/domain.rs
    normalize_domain, DomainMatcher, domain_from_url
  types/output.rs WebFetchOutput::DomainNotAllowed prompt format
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from urllib.parse import urlparse


def normalize_domain(raw: str) -> str:
    """Canonical form: trim, strip trailing slashes/dots, remove ``www.``, lower.

    Grok: strip_prefix(\"www.\") then to_lowercase (prefix match is case-sensitive).
    """
    s = raw.strip().rstrip("/").rstrip(".")
    if s.startswith("www."):
        s = s[4:]
    return s.lower()


class _HostKind(Enum):
    AnyPath = auto()
    PathPrefixes = auto()


@dataclass
class _HostEntry:
    kind: _HostKind
    prefixes: list[str] = field(default_factory=list)


class DomainMatcher:
    """Precomputed domain allowlist. O(1) host lookup + path-prefix scan."""

    def __init__(self, raw_entries: list[str] | tuple[str, ...] | None = None) -> None:
        self.entries: dict[str, _HostEntry] = {}
        for raw in raw_entries or []:
            normalized = normalize_domain(raw)
            if not normalized:
                continue

            # Split on first '/' to separate host from optional path.
            slash = normalized.find("/")
            if slash < 0:
                host, raw_path = normalized, None
            else:
                host, raw_path = normalized[:slash], normalized[slash:]

            if raw_path is None:
                # Host-only → any path. Overrides any existing prefixes.
                self.entries[host] = _HostEntry(_HostKind.AnyPath)
                continue

            # Don't downgrade AnyPath to PathPrefixes.
            existing = self.entries.get(host)
            if existing is not None and existing.kind is _HostKind.AnyPath:
                continue

            # Normalize path: ensure leading '/', strip trailing '/'.
            prefix = raw_path.rstrip("/")
            if not prefix or prefix == "/":
                self.entries[host] = _HostEntry(_HostKind.AnyPath)
                continue
            if not prefix.startswith("/"):
                prefix = "/" + prefix

            if existing is None:
                self.entries[host] = _HostEntry(_HostKind.PathPrefixes, [prefix])
            elif existing.kind is _HostKind.PathPrefixes and prefix not in existing.prefixes:
                existing.prefixes.append(prefix)

    def check(self, url: str) -> str | None:
        """Return blocked domain string if not allowed, else ``None``.

        Empty matcher entries → all blocked (Grok).
        """
        parsed = urlparse(url)
        raw_host = parsed.hostname
        if raw_host is None:
            return ""
        host = normalize_domain(raw_host)
        entry = self.entries.get(host)
        if entry is None:
            return host
        if entry.kind is _HostKind.AnyPath:
            return None
        url_path = (parsed.path or "/").lower()
        for prefix in entry.prefixes:
            if url_path == prefix:
                return None
            # Child path: starts with prefix and next char is '/'
            if (
                url_path.startswith(prefix)
                and len(url_path) > len(prefix)
                and url_path[len(prefix)] == "/"
            ):
                return None
        return host


def domain_from_url(raw_url: str) -> str | None:
    """Extract and normalize domain; ``None`` if unparseable / no host."""
    try:
        parsed = urlparse(raw_url)
    except Exception:  # noqa: BLE001
        return None
    if not parsed.scheme or not parsed.hostname:
        return None
    return normalize_domain(parsed.hostname)


def domain_not_allowed_message(domain: str) -> str:
    """Grok ``WebFetchOutput::DomainNotAllowed`` prompt format."""
    return f"Error: domain {domain} is not in the allowed domains list"
