"""Auth primitives: credential, status, login contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

AuthKind = Literal["oauth", "api_key"]

AUTH_OAUTH: AuthKind = "oauth"
AUTH_API_KEY: AuthKind = "api_key"


@dataclass(slots=True)
class AuthCredential:
    """Resolved credential ready for HTTP headers / client construction."""

    provider: str
    kind: AuthKind
    # Bearer / x-api-key material (never log this)
    token: str
    # Optional refresh material (OAuth)
    refresh_token: str | None = None
    # Where it came from (for UI / doctor)
    source: str = ""
    # Extra headers the transport should merge (e.g. anthropic-version)
    headers: dict[str, str] = field(default_factory=dict)
    # Opaque metadata (email, expires_at, …) — no secrets beyond token fields
    meta: dict[str, Any] = field(default_factory=dict)

    def redacted(self) -> dict[str, Any]:
        t = self.token
        tip = f"…{t[-4:]}" if len(t) >= 4 else "****"
        return {
            "provider": self.provider,
            "kind": self.kind,
            "source": self.source,
            "token": tip,
            "has_refresh": bool(self.refresh_token),
            "meta": {k: v for k, v in self.meta.items() if k not in {"token", "key"}},
        }


@dataclass(slots=True)
class AuthStatus:
    """Non-secret snapshot for UI: logged in or not."""

    provider: str
    kind: AuthKind
    logged_in: bool
    source: str = ""
    detail: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class LoginRequired(Exception):
    """OAuth provider needs interactive login before API calls."""

    def __init__(self, provider: str, message: str = "") -> None:
        self.provider = provider
        super().__init__(message or f"login required for provider {provider!r}")


@runtime_checkable
class AuthProvider(Protocol):
    """One auth strategy (oauth imperial or api_key)."""

    @property
    def name(self) -> str:
        ...

    @property
    def kind(self) -> AuthKind:
        ...

    def status(self) -> AuthStatus:
        """Check local credentials without network (best-effort)."""
        ...

    def resolve(self, *, explicit_token: str | None = None) -> AuthCredential | None:
        """Return a usable credential or None if login/key missing."""
        ...

    def begin_login(self) -> AuthStatus:
        """Start interactive login (open browser / print device code).

        Phase-1 implementations may only document the official CLI to run
        (``grok login``, ``claude /login``, ``codex login``) and re-read
        local stores. Full in-process OIDC can replace this later.
        """
        ...
