"""Claude auth — honest contract.

Anthropic does not publish a third-party device-code grant comparable to xAI.
This provider therefore:

* resolves existing credentials (env / Claude Code files)
* ``begin_login()`` opens the browser **and states that token import is still
  required** unless credentials already exist

Do not claim closed-loop browser OAuth until a public grant is wired.
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

CLAUDE_LOGIN_URL = "https://claude.ai/login"

_LOGIN_HELP = (
    f"opened {CLAUDE_LOGIN_URL}. Claude has no public device-code for third-party "
    "apps: after browser sign-in set ANTHROPIC_TOKEN / CLAUDE_CODE_OAUTH_TOKEN, "
    "or use Claude Code so ~/.claude credentials appear, or set ANTHROPIC_API_KEY."
)


def _claude_paths() -> list[Path]:
    home = Path.home()
    return [
        home / ".claude" / ".credentials.json",
        home / ".claude.json",
        home / ".config" / "claude" / ".credentials.json",
    ]


class ClaudeOAuthAuth:
    name = "claude"
    kind = AUTH_OAUTH

    def status(self) -> AuthStatus:
        cred = self.resolve()
        if cred is None:
            return AuthStatus(
                provider=self.name,
                kind=AUTH_OAUTH,
                logged_in=False,
                detail=(
                    "no credentials — browser login alone is not enough; "
                    "set ANTHROPIC_TOKEN / CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_API_KEY "
                    "or use Claude Code credential files"
                ),
            )
        return AuthStatus(
            provider=self.name,
            kind=cred.kind,
            logged_in=True,
            source=cred.source,
            detail=str(cred.meta.get("auth_type") or ""),
        )

    def resolve(self, *, explicit_token: str | None = None) -> AuthCredential | None:
        if explicit_token is not None and str(explicit_token).strip():
            tok = str(explicit_token).strip()
            return AuthCredential(
                provider=self.name,
                kind=_classify_anthropic_token(tok),
                token=tok,
                source="explicit",
                headers=_headers_for_token(),
            )

        # Prefer OAuth-shaped env over API key
        for env_name in ("ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
            raw = (os.environ.get(env_name) or "").strip()
            if raw:
                return AuthCredential(
                    provider=self.name,
                    kind=_classify_anthropic_token(raw),
                    token=raw,
                    source=f"env:{env_name}",
                    headers=_headers_for_token(),
                )

        for path in _claude_paths():
            token, meta = _read_claude_file(path)
            if token:
                return AuthCredential(
                    provider=self.name,
                    kind=_classify_anthropic_token(token),
                    token=token,
                    source=f"file:{path}",
                    headers=_headers_for_token(),
                    meta=meta,
                )

        api = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if api:
            return AuthCredential(
                provider=self.name,
                kind=AUTH_API_KEY,
                token=api,
                source="env:ANTHROPIC_API_KEY",
                headers=_headers_for_token(),
            )
        return None

    def begin_login(self) -> AuthStatus:
        opened = False
        try:
            opened = bool(webbrowser.open(CLAUDE_LOGIN_URL))
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

        return AuthStatus(
            provider=self.name,
            kind=AUTH_OAUTH,
            logged_in=False,
            detail=("opened " if opened else "open ") + _LOGIN_HELP.removeprefix("opened "),
            meta={
                "url": CLAUDE_LOGIN_URL,
                "closed_loop": False,
                "requires": [
                    "ANTHROPIC_TOKEN",
                    "CLAUDE_CODE_OAUTH_TOKEN",
                    "ANTHROPIC_API_KEY",
                    "~/.claude credentials",
                ],
            },
        )


def _classify_anthropic_token(token: str) -> str:
    if token.startswith("sk-ant-api"):
        return AUTH_API_KEY
    if token.startswith(("sk-ant-", "eyJ", "cc-")):
        return AUTH_OAUTH
    return AUTH_API_KEY


def _headers_for_token() -> dict[str, str]:
    return {"anthropic-version": "2023-06-01"}


def _read_claude_file(path: Path) -> tuple[str | None, dict[str, Any]]:
    if not path.is_file():
        return None, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, {}
    if not isinstance(data, dict):
        return None, {}

    for key in (
        "claudeAiOauth",
        "claude_ai_oauth",
        "oauth",
        "accessToken",
        "access_token",
        "token",
        "primaryApiKey",
        "apiKey",
        "api_key",
    ):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip(), {"auth_type": key, "path": str(path)}
        if isinstance(val, dict):
            for nested in ("accessToken", "access_token", "token", "apiKey", "key"):
                n = val.get(nested)
                if isinstance(n, str) and n.strip():
                    return n.strip(), {"auth_type": f"{key}.{nested}", "path": str(path)}

    accounts = data.get("accounts") or data.get("credentials")
    if isinstance(accounts, dict):
        for _, entry in accounts.items():
            if not isinstance(entry, dict):
                continue
            for nested in ("accessToken", "access_token", "apiKey", "key", "token"):
                n = entry.get(nested)
                if isinstance(n, str) and n.strip():
                    return n.strip(), {"auth_type": nested, "path": str(path)}
    return None, {}
