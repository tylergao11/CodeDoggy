"""Persist / resolve the active chat provider across launches.

OAuth success only writes credentials. Without a remembered provider (or a
logged-in imperial probe), cold start falls back to ollama — so the TUI can
show ✓grok while MAIN still hits 127.0.0.1:11434. This module closes that gap.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PREFERRED_NAME = "active_provider"
_IMPERIAL_ORDER = ("grok", "claude", "codex")


def preferred_provider_path(home: Path | None = None) -> Path:
    from codedoggy.memory.paths import default_memory_home

    root = home if home is not None else default_memory_home()
    return Path(root).expanduser().resolve() / _PREFERRED_NAME


def load_preferred_provider(home: Path | None = None) -> str | None:
    path = preferred_provider_path(home)
    try:
        raw = path.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return None
    if not raw or "\n" in raw or len(raw) > 64:
        return None
    return raw


def save_preferred_provider(provider: str, home: Path | None = None) -> None:
    name = (provider or "").strip().lower()
    if not name:
        return
    path = preferred_provider_path(home)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(name + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        logger.debug("save_preferred_provider failed", exc_info=True)


def provider_usable(provider: str) -> bool:
    """True when this provider can sample without another login dance."""
    name = (provider or "").strip().lower()
    if not name:
        return False
    if name == "ollama":
        return True
    try:
        from codedoggy.model.auth import auth_status

        return bool(auth_status(name).logged_in)
    except Exception:  # noqa: BLE001
        return False


def resolve_startup_provider() -> str:
    """Pick MAIN provider when ``CODEDOGGY_PROVIDER`` is unset.

    Order: remembered preferred (if still usable) → first logged-in imperial
    (grok, claude, codex) → ollama.
    """
    preferred = load_preferred_provider()
    if preferred and provider_usable(preferred):
        return preferred

    for name in _IMPERIAL_ORDER:
        if provider_usable(name):
            return name

    return "ollama"
