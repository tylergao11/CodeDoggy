"""API-key auth for non-imperial providers (DeepSeek, Ollama, custom, …)."""

from __future__ import annotations

import os
from typing import Any

from codedoggy.model.auth.base import (
    AUTH_API_KEY,
    AuthCredential,
    AuthStatus,
)


class ApiKeyAuth:
    """Resolve Bearer token from explicit value or env vars."""

    def __init__(
        self,
        name: str,
        *,
        env_vars: tuple[str, ...] = (),
        display_name: str = "",
    ) -> None:
        self._name = name
        self._env_vars = env_vars
        self._display = display_name or name

    @property
    def name(self) -> str:
        return self._name

    @property
    def kind(self) -> str:
        return AUTH_API_KEY

    def status(self) -> AuthStatus:
        token = self._pick(None)
        return AuthStatus(
            provider=self._name,
            kind=AUTH_API_KEY,
            logged_in=bool(token),
            source="env" if token else "",
            detail=self._display if token else f"set one of {list(self._env_vars)}",
        )

    def resolve(self, *, explicit_token: str | None = None) -> AuthCredential | None:
        token = self._pick(explicit_token)
        if not token:
            return None
        source = "explicit" if explicit_token else "env"
        return AuthCredential(
            provider=self._name,
            kind=AUTH_API_KEY,
            token=token,
            source=source,
        )

    def begin_login(self) -> AuthStatus:
        # No browser flow — tell user which env to set.
        return AuthStatus(
            provider=self._name,
            kind=AUTH_API_KEY,
            logged_in=False,
            detail=(
                f"{self._display}: set API key via env "
                f"{' / '.join(self._env_vars) or 'CODEDOGGY_API_KEY'}"
            ),
        )

    def _pick(self, explicit: str | None) -> str | None:
        if explicit is not None and str(explicit).strip():
            return str(explicit).strip()
        for name in self._env_vars:
            raw = os.environ.get(name)
            if raw is not None and str(raw).strip():
                return str(raw).strip()
        return None
