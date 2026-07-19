"""Grok (xAI) OAuth — browser device-code + session file.

Priority (product rule: subscription first, pay-as-you-go last):
  1. explicit token argument (caller override)
  2. OAuth session in ~/.grok/auth.json (refresh if near expiry)
  3. XAI_API_KEY only if no usable session
  4. CODEDOGGY_FORCE_API_KEY=1 forces step 3 before 2
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from codedoggy.model.auth.base import (
    AUTH_API_KEY,
    AUTH_OAUTH,
    AuthCredential,
    AuthStatus,
    LoginRequired,
)
from codedoggy.model.auth.device_flow import (
    DeviceFlowError,
    open_verification_page,
    poll_device_token,
    refresh_access_token,
    request_device_code,
)
from codedoggy.model.auth.secure_store import atomic_write_json

logger = logging.getLogger(__name__)

XAI_OAUTH2_ISSUER = "https://auth.x.ai"
XAI_OAUTH2_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_OAUTH2_SCOPES = [
    "openid",
    "profile",
    "email",
    "offline_access",
    "grok-cli:access",
    "api:access",
    "conversations:read",
    "conversations:write",
    "workspaces:read",
    "workspaces:write",
]
_API_KEY_SCOPE = "xai::api_key"
_SCOPE_KEY = f"{XAI_OAUTH2_ISSUER}::{XAI_OAUTH2_CLIENT_ID}"
# Refresh this many seconds before expires_at
_REFRESH_SKEW_S = 120


def grok_home() -> Path:
    raw = os.environ.get("GROK_HOME") or os.environ.get("XAI_GROK_HOME")
    if raw and str(raw).strip():
        return Path(raw).expanduser()
    return Path.home() / ".grok"


def auth_json_path() -> Path:
    return grok_home() / "auth.json"


def _force_api_key() -> bool:
    return (os.environ.get("CODEDOGGY_FORCE_API_KEY") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class GrokOAuthAuth:
    name = "grok"
    kind = AUTH_OAUTH

    def status(self) -> AuthStatus:
        try:
            cred = self.resolve()
        except LoginRequired as exc:
            return AuthStatus(
                provider=self.name,
                kind=AUTH_OAUTH,
                logged_in=False,
                detail=str(exc),
            )
        if cred is None:
            return AuthStatus(
                provider=self.name,
                kind=AUTH_OAUTH,
                logged_in=False,
                detail="not logged in — call begin_login() to open the browser",
            )
        return AuthStatus(
            provider=self.name,
            kind=cred.kind,
            logged_in=True,
            source=cred.source,
            detail=str(cred.meta.get("email") or cred.meta.get("auth_mode") or ""),
            meta={k: v for k, v in cred.meta.items() if k not in {"key", "token"}},
        )

    def resolve(self, *, explicit_token: str | None = None) -> AuthCredential | None:
        if explicit_token is not None and str(explicit_token).strip():
            return AuthCredential(
                provider=self.name,
                kind=AUTH_API_KEY,
                token=str(explicit_token).strip(),
                source="explicit",
            )

        env_key = (os.environ.get("XAI_API_KEY") or "").strip()
        if _force_api_key() and env_key:
            return AuthCredential(
                provider=self.name,
                kind=AUTH_API_KEY,
                token=env_key,
                source="env:XAI_API_KEY(forced)",
            )

        # Subscription session first
        session = self._resolve_session_file()
        if session is not None:
            return session

        if env_key:
            return AuthCredential(
                provider=self.name,
                kind=AUTH_API_KEY,
                token=env_key,
                source="env:XAI_API_KEY",
            )
        return None

    def begin_login(self) -> AuthStatus:
        """Open browser (device-code); block until approved; persist session."""
        try:
            session = request_device_code(
                issuer=XAI_OAUTH2_ISSUER,
                client_id=XAI_OAUTH2_CLIENT_ID,
                scopes=list(XAI_OAUTH2_SCOPES),
                referrer="codedoggy",
            )
        except DeviceFlowError as exc:
            return AuthStatus(
                provider=self.name,
                kind=AUTH_OAUTH,
                logged_in=False,
                detail=f"could not start browser login: {exc}",
            )

        opened = open_verification_page(session)
        hint = (
            f"browser opened ({session.open_url})"
            if opened
            else f"open {session.open_url} and enter code {session.user_code}"
        )
        logger.info("Grok login user_code=%s", session.user_code)

        try:
            tokens = poll_device_token(
                issuer=XAI_OAUTH2_ISSUER,
                client_id=XAI_OAUTH2_CLIENT_ID,
                session=session,
            )
        except DeviceFlowError as exc:
            return AuthStatus(
                provider=self.name,
                kind=AUTH_OAUTH,
                logged_in=False,
                detail=f"{hint}; login incomplete: {exc}",
                meta={"user_code": session.user_code, "url": session.open_url},
            )

        path = _persist_session(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            expires_in=tokens.expires_in,
        )
        return AuthStatus(
            provider=self.name,
            kind=AUTH_OAUTH,
            logged_in=True,
            source=f"file:{path}",
            detail=f"signed in via browser — {hint}",
            meta={"user_code": session.user_code},
        )

    def _resolve_session_file(self) -> AuthCredential | None:
        path = auth_json_path()
        entry = _load_best_oauth_entry(path)
        if entry is None:
            return None

        key = str(entry.get("key") or "").strip()
        if not key:
            return None

        expires_at = _parse_expires_at(entry.get("expires_at"))
        refresh = (
            str(entry["refresh_token"]).strip()
            if isinstance(entry.get("refresh_token"), str)
            else None
        )
        issuer = str(entry.get("oidc_issuer") or XAI_OAUTH2_ISSUER).strip()
        client_id = str(entry.get("oidc_client_id") or XAI_OAUTH2_CLIENT_ID).strip()

        if expires_at is not None and _needs_refresh(expires_at):
            if not refresh:
                raise LoginRequired(
                    self.name,
                    "Grok session expired and no refresh_token — call begin_login()",
                )
            try:
                bundle = refresh_access_token(
                    issuer=issuer,
                    client_id=client_id,
                    refresh_token=refresh,
                )
            except DeviceFlowError as exc:
                raise LoginRequired(
                    self.name,
                    f"Grok token refresh failed ({exc}) — call begin_login()",
                ) from exc
            path = _persist_session(
                access_token=bundle.access_token,
                refresh_token=bundle.refresh_token,
                expires_in=bundle.expires_in,
                email=entry.get("email"),
                user_id=entry.get("user_id"),
            )
            key = bundle.access_token
            refresh = bundle.refresh_token
            entry = _load_best_oauth_entry(path) or entry

        return AuthCredential(
            provider=self.name,
            kind=AUTH_OAUTH,
            token=key,
            refresh_token=refresh,
            source=f"file:{path}",
            meta={
                "auth_mode": str(entry.get("auth_mode") or "oidc"),
                "email": entry.get("email"),
                "user_id": entry.get("user_id"),
                "expires_at": entry.get("expires_at"),
                "oidc_issuer": issuer,
            },
        )


def _needs_refresh(expires_at: datetime) -> bool:
    now = datetime.now(timezone.utc)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return now >= (expires_at - timedelta(seconds=_REFRESH_SKEW_S))


def _parse_expires_at(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    s = str(raw).strip()
    if not s:
        return None
    # Support trailing Z
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _persist_session(
    *,
    access_token: str,
    refresh_token: str | None,
    expires_in: int | None,
    email: Any = None,
    user_id: Any = None,
) -> Path:
    path = auth_json_path()
    store: dict[str, Any] = {}
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise LoginRequired(
                    "grok",
                    f"corrupt auth store {path} — fix or delete the file, then begin_login()",
                )
            store = raw
        except json.JSONDecodeError as exc:
            raise LoginRequired(
                "grok",
                f"corrupt auth store {path}: {exc} — fix or delete, then begin_login()",
            ) from exc
        except OSError as exc:
            raise LoginRequired("grok", f"cannot read {path}: {exc}") from exc

    now = datetime.now(timezone.utc)
    entry: dict[str, Any] = {
        "key": access_token,
        "auth_mode": "oidc",
        "create_time": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "user_id": user_id if user_id is not None else "",
        "email": email,
        "oidc_issuer": XAI_OAUTH2_ISSUER,
        "oidc_client_id": XAI_OAUTH2_CLIENT_ID,
    }
    if refresh_token:
        entry["refresh_token"] = refresh_token
    if expires_in is not None:
        entry["expires_at"] = (
            now + timedelta(seconds=int(expires_in))
        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    store[_SCOPE_KEY] = entry
    atomic_write_json(path, store)
    return path


def _load_best_oauth_entry(path: Path) -> dict[str, Any] | None:
    """Prefer OIDC/session scopes; never pick xai::api_key as 'session'."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("failed to read %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or not data:
        return None

    oauth: list[dict[str, Any]] = []
    for scope, entry in data.items():
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if not isinstance(key, str) or not key.strip():
            continue
        scope_s = str(scope)
        mode = str(entry.get("auth_mode") or "").lower()
        if scope_s == _API_KEY_SCOPE or mode == "api_key":
            continue
        oauth.append(entry)

    if not oauth:
        return None

    def sort_key(e: dict[str, Any]) -> tuple:
        return (
            1 if e.get("refresh_token") else 0,
            str(e.get("create_time") or ""),
        )

    oauth.sort(key=sort_key, reverse=True)
    return oauth[0]
