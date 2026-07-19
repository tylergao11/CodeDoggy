"""Codex auth — honest contract.

Resolves ``~/.codex/auth.json`` or OPENAI_API_KEY.
``begin_login()`` opens ChatGPT in the browser but does **not** claim a
closed device-code loop until a public OAuth grant is wired (same as Claude).
"""

from __future__ import annotations

import json
import logging
import os
import webbrowser
from pathlib import Path
from typing import Any

from codedoggy.model.auth.base import (
    AUTH_API_KEY,
    AUTH_OAUTH,
    AuthCredential,
    AuthStatus,
)

logger = logging.getLogger(__name__)

CODEX_LOGIN_URL = "https://chatgpt.com"


def codex_home() -> Path:
    raw = os.environ.get("CODEX_HOME")
    if raw and str(raw).strip():
        return Path(raw).expanduser()
    return Path.home() / ".codex"


class CodexOAuthAuth:
    name = "codex"
    kind = AUTH_OAUTH

    def status(self) -> AuthStatus:
        cred = self.resolve()
        if cred is None:
            return AuthStatus(
                provider=self.name,
                kind=AUTH_OAUTH,
                logged_in=False,
                detail=(
                    "no credentials — browser alone is not enough; "
                    "need ~/.codex/auth.json or OPENAI_API_KEY"
                ),
            )
        return AuthStatus(
            provider=self.name,
            kind=cred.kind,
            logged_in=True,
            source=cred.source,
            detail=str(cred.meta.get("auth_mode") or ""),
        )

    def resolve(self, *, explicit_token: str | None = None) -> AuthCredential | None:
        if explicit_token is not None and str(explicit_token).strip():
            return AuthCredential(
                provider=self.name,
                kind=AUTH_API_KEY,
                token=str(explicit_token).strip(),
                source="explicit",
            )

        # Prefer session file over generic OPENAI_API_KEY (subscription path)
        path = codex_home() / "auth.json"
        entry = _load_codex_auth(path)
        if entry and entry.get("token"):
            tok = str(entry["token"])
            kind = AUTH_API_KEY if entry.get("kind") == "api_key" else AUTH_OAUTH
            return AuthCredential(
                provider=self.name,
                kind=kind,
                token=tok,
                refresh_token=entry.get("refresh_token"),
                source=f"file:{path}",
                meta={
                    k: v
                    for k, v in entry.items()
                    if k not in {"token", "refresh_token"}
                },
            )

        env_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if env_key:
            return AuthCredential(
                provider=self.name,
                kind=AUTH_API_KEY,
                token=env_key,
                source="env:OPENAI_API_KEY",
            )
        return None

    def begin_login(self) -> AuthStatus:
        opened = False
        try:
            opened = bool(webbrowser.open(CODEX_LOGIN_URL))
        except Exception as exc:  # noqa: BLE001
            logger.debug("webbrowser open failed: %s", exc)

        cred = self.resolve()
        if cred is not None:
            return AuthStatus(
                provider=self.name,
                kind=cred.kind,
                logged_in=True,
                source=cred.source,
                detail="credentials already present",
            )

        detail = (
            f"{'opened' if opened else 'open'} {CODEX_LOGIN_URL}. "
            "No public device-code loop yet: after sign-in, credentials must "
            "appear in ~/.codex/auth.json (e.g. via Codex tooling) or set OPENAI_API_KEY."
        )
        return AuthStatus(
            provider=self.name,
            kind=AUTH_OAUTH,
            logged_in=False,
            detail=detail,
            meta={
                "url": CODEX_LOGIN_URL,
                "closed_loop": False,
                "requires": ["~/.codex/auth.json", "OPENAI_API_KEY"],
            },
        )


def _load_codex_auth(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("failed to read %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        return None

    for key in (
        "access_token",
        "accessToken",
        "token",
        "id_token",
        "OPENAI_API_KEY",
        "api_key",
        "apiKey",
    ):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            kind = "api_key" if key in {"OPENAI_API_KEY", "api_key", "apiKey"} else "oauth"
            return {
                "token": val.strip(),
                "kind": kind,
                "refresh_token": _as_str(data.get("refresh_token") or data.get("refreshToken")),
                "auth_mode": key,
            }

    tokens = data.get("tokens")
    if isinstance(tokens, dict):
        for key in ("access_token", "accessToken", "id_token", "token"):
            val = tokens.get(key)
            if isinstance(val, str) and val.strip():
                return {
                    "token": val.strip(),
                    "kind": "oauth",
                    "refresh_token": _as_str(
                        tokens.get("refresh_token") or tokens.get("refreshToken")
                    ),
                    "auth_mode": f"tokens.{key}",
                }

    for _, entry in data.items():
        if not isinstance(entry, dict):
            continue
        for key in ("access_token", "accessToken", "token", "api_key", "key"):
            val = entry.get(key)
            if isinstance(val, str) and val.strip():
                return {
                    "token": val.strip(),
                    "kind": "oauth" if "api" not in key else "api_key",
                    "refresh_token": _as_str(
                        entry.get("refresh_token") or entry.get("refreshToken")
                    ),
                    "auth_mode": key,
                }
    return None


def _as_str(v: Any) -> str | None:
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None
