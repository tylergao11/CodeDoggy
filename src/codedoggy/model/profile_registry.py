"""Registry of ProviderProfile objects (name + aliases)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codedoggy.model.profile import ProviderProfile

logger = logging.getLogger(__name__)

_PROFILES: dict[str, ProviderProfile] = {}
_ALIASES: dict[str, str] = {}  # alias → canonical name


def register_profile(profile: ProviderProfile, *, replace: bool = False) -> None:
    key = profile.name.strip().lower()
    if not key:
        raise ValueError("profile name must be non-empty")
    if key in _PROFILES and not replace:
        raise ValueError(f"provider profile already registered: {key}")
    _PROFILES[key] = profile
    for alias in profile.aliases:
        a = alias.strip().lower()
        if a:
            _ALIASES[a] = key
    # name maps to itself
    _ALIASES[key] = key
    logger.debug("registered provider profile %s", key)


def unregister_profile(name: str) -> None:
    key = name.strip().lower()
    _PROFILES.pop(key, None)
    dead = [a for a, n in _ALIASES.items() if n == key or a == key]
    for a in dead:
        _ALIASES.pop(a, None)


def get_profile(name: str | None) -> ProviderProfile | None:
    if not name:
        return None
    key = name.strip().lower()
    canon = _ALIASES.get(key, key)
    return _PROFILES.get(canon)


def list_profiles() -> list[str]:
    return sorted(_PROFILES.keys())


def resolve_profile_name(name: str | None) -> str | None:
    if not name:
        return None
    key = name.strip().lower()
    return _ALIASES.get(key, key)
