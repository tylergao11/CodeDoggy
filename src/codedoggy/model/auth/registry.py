"""Registry of AuthProvider implementations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codedoggy.model.auth.base import AuthProvider

logger = logging.getLogger(__name__)

_AUTH: dict[str, AuthProvider] = {}
_ALIASES: dict[str, str] = {}


def register_auth_provider(
    provider: AuthProvider,
    *,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
) -> None:
    key = provider.name.strip().lower()
    if not key:
        raise ValueError("auth provider name empty")
    if key in _AUTH and not replace:
        raise ValueError(f"auth provider already registered: {key}")
    _AUTH[key] = provider
    _ALIASES[key] = key
    for a in aliases:
        aa = a.strip().lower()
        if aa:
            _ALIASES[aa] = key
    logger.debug("registered auth provider %s", key)


def get_auth_provider(name: str | None) -> AuthProvider | None:
    if not name:
        return None
    key = name.strip().lower()
    canon = _ALIASES.get(key, key)
    return _AUTH.get(canon)


def list_auth_providers() -> list[str]:
    return sorted(_AUTH.keys())


def resolve_auth_name(name: str | None) -> str | None:
    if not name:
        return None
    key = name.strip().lower()
    return _ALIASES.get(key, key)
