"""RFC 8628 device-code OAuth — open a webpage, poll until approved."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"

# Hosts allowed in verification_uri for xAI (and tests can pass custom).
DEFAULT_ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "auth.x.ai",
        "accounts.x.ai",
        "x.ai",
    }
)


@dataclass(slots=True)
class DeviceCodeSession:
    verification_uri: str
    user_code: str
    device_code: str
    interval: int
    expires_in: int
    verification_uri_complete: str | None = None

    @property
    def open_url(self) -> str:
        return self.verification_uri_complete or self.verification_uri


@dataclass(slots=True)
class TokenBundle:
    access_token: str
    refresh_token: str | None = None
    expires_in: int | None = None
    id_token: str | None = None
    scope: str | None = None
    raw: dict[str, Any] | None = None


class DeviceFlowError(Exception):
    pass


def validate_verification_uri(
    uri: str,
    *,
    allowed_hosts: frozenset[str] | None = None,
) -> None:
    """Reject non-HTTPS or off-allowlist verification pages (phishing guard)."""
    hosts = allowed_hosts if allowed_hosts is not None else DEFAULT_ALLOWED_HOSTS
    if not uri or not str(uri).strip():
        raise DeviceFlowError("empty verification_uri")
    parsed = urlparse(str(uri).strip())
    if parsed.scheme != "https":
        raise DeviceFlowError(f"verification_uri must be https, got {parsed.scheme!r}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise DeviceFlowError("verification_uri missing host")
    # Allow exact host or subdomain of an allowed registrable host.
    ok = host in hosts or any(host.endswith("." + h) for h in hosts)
    if not ok:
        raise DeviceFlowError(f"verification_uri host not allowed: {host}")


def request_device_code(
    *,
    issuer: str,
    client_id: str,
    scopes: list[str],
    referrer: str = "codedoggy",
    extra_headers: dict[str, str] | None = None,
    timeout_s: float = 30.0,
    allowed_hosts: frozenset[str] | None = None,
) -> DeviceCodeSession:
    url = f"{issuer.rstrip('/')}/oauth2/device/code"
    form = {
        "client_id": client_id,
        "scope": " ".join(scopes),
        "referrer": referrer,
    }
    data = urllib.parse.urlencode(form).encode("utf-8")
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        **(extra_headers or {}),
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        if e.code == 404:
            raise DeviceFlowError("device-code login not enabled for this issuer") from e
        raise DeviceFlowError(f"device code request failed HTTP {e.code}: {body[:300]}") from e
    except urllib.error.URLError as e:
        raise DeviceFlowError(f"device code request failed: {e.reason}") from e

    user_code = str(payload.get("user_code") or "")
    if not user_code:
        raise DeviceFlowError(f"no user_code in response: {payload!r}"[:200])
    # Basic format check (alphanumeric + hyphen)
    if not all(c.isalnum() or c == "-" for c in user_code):
        raise DeviceFlowError("server returned invalid user_code format")

    vuri = str(payload.get("verification_uri") or "")
    vcomplete = (
        str(payload["verification_uri_complete"])
        if payload.get("verification_uri_complete")
        else None
    )
    validate_verification_uri(vuri, allowed_hosts=allowed_hosts)
    if vcomplete:
        validate_verification_uri(vcomplete, allowed_hosts=allowed_hosts)

    device_code = str(payload.get("device_code") or "")
    if not device_code:
        raise DeviceFlowError("no device_code in response")

    return DeviceCodeSession(
        verification_uri=vuri,
        verification_uri_complete=vcomplete,
        user_code=user_code,
        device_code=device_code,
        interval=int(payload.get("interval") or 5),
        expires_in=int(payload.get("expires_in") or 600),
    )


def open_verification_page(session: DeviceCodeSession) -> bool:
    url = session.open_url
    if not url:
        return False
    try:
        return bool(webbrowser.open(url))
    except Exception as exc:  # noqa: BLE001
        logger.warning("webbrowser.open failed: %s", exc)
        return False


def poll_device_token(
    *,
    issuer: str,
    client_id: str,
    session: DeviceCodeSession,
    extra_headers: dict[str, str] | None = None,
    on_pending: Callable[[], None] | None = None,
    timeout_s: float | None = None,
) -> TokenBundle:
    token_url = f"{issuer.rstrip('/')}/oauth2/token"
    interval = max(1, int(session.interval))
    deadline = time.monotonic() + float(
        timeout_s if timeout_s is not None else max(session.expires_in, 60)
    )
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        **(extra_headers or {}),
    }

    while True:
        time.sleep(interval)
        if time.monotonic() > deadline:
            raise DeviceFlowError("device code expired — run login again")

        form = {
            "grant_type": DEVICE_GRANT_TYPE,
            "device_code": session.device_code,
            "client_id": client_id,
        }
        data = urllib.parse.urlencode(form).encode("utf-8")
        req = urllib.request.Request(token_url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                access = str(payload.get("access_token") or "")
                if not access:
                    raise DeviceFlowError("token response missing access_token")
                return TokenBundle(
                    access_token=access,
                    refresh_token=(
                        str(payload["refresh_token"])
                        if payload.get("refresh_token")
                        else None
                    ),
                    expires_in=(
                        int(payload["expires_in"])
                        if payload.get("expires_in") is not None
                        else None
                    ),
                    id_token=(str(payload["id_token"]) if payload.get("id_token") else None),
                    scope=(str(payload["scope"]) if payload.get("scope") else None),
                    raw=payload,
                )
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            try:
                err = json.loads(body) if body else {}
            except json.JSONDecodeError:
                err = {}
            code = str(err.get("error") or "")
            if code == "authorization_pending":
                if on_pending:
                    on_pending()
                continue
            if code == "slow_down":
                interval += 5
                continue
            if code == "access_denied":
                raise DeviceFlowError("authorization denied in browser") from e
            if code == "expired_token":
                raise DeviceFlowError("device code expired — run login again") from e
            raise DeviceFlowError(
                f"token poll failed HTTP {e.code}: "
                f"{err.get('error_description') or body[:200]}"
            ) from e
        except urllib.error.URLError as e:
            raise DeviceFlowError(f"token poll network error: {e.reason}") from e


def refresh_access_token(
    *,
    issuer: str,
    client_id: str,
    refresh_token: str,
    timeout_s: float = 30.0,
) -> TokenBundle:
    """OAuth2 refresh_token grant against ``{issuer}/oauth2/token``."""
    token_url = f"{issuer.rstrip('/')}/oauth2/token"
    form = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    data = urllib.parse.urlencode(form).encode("utf-8")
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    req = urllib.request.Request(token_url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            err = json.loads(body) if body else {}
        except json.JSONDecodeError:
            err = {}
        code = str(err.get("error") or "")
        if code in {"invalid_grant", "invalid_client"}:
            raise DeviceFlowError(
                f"refresh rejected ({code}) — re-login required"
            ) from e
        raise DeviceFlowError(
            f"refresh failed HTTP {e.code}: {err.get('error_description') or body[:200]}"
        ) from e
    except urllib.error.URLError as e:
        raise DeviceFlowError(f"refresh network error: {e.reason}") from e

    access = str(payload.get("access_token") or "")
    if not access:
        raise DeviceFlowError("refresh response missing access_token")
    # Rotation: new refresh_token may replace old
    new_rt = payload.get("refresh_token")
    return TokenBundle(
        access_token=access,
        refresh_token=str(new_rt) if new_rt else refresh_token,
        expires_in=(
            int(payload["expires_in"]) if payload.get("expires_in") is not None else None
        ),
        id_token=(str(payload["id_token"]) if payload.get("id_token") else None),
        scope=(str(payload["scope"]) if payload.get("scope") else None),
        raw=payload,
    )
