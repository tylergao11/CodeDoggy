"""Persist / resolve the active chat provider across launches.

Truth rules (product):
  - OAuth tokens ≠ ActiveConnection.
  - Never silently default to ollama / qwen3:8b.
  - Sticky preference is only for providers the user applied in the panel.
  - No preference + no logged-in imperial → unconfigured (must choose).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PREFERRED_NAME = "active_provider"
_IMPERIAL_ORDER = ("grok", "claude", "codex")
UNCONFIGURED_PROVIDER = "unconfigured"


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
    # Bootstrap used to poison this file with "ollama". Local default is never
    # a sticky choice — clear it so imperial login / panel apply can win.
    if raw == "ollama":
        try:
            path.unlink()
        except OSError:
            pass
        return None
    return raw


def save_preferred_provider(provider: str, home: Path | None = None) -> None:
    name = (provider or "").strip().lower()
    path = preferred_provider_path(home)
    if not name or name in {UNCONFIGURED_PROVIDER, "ollama"}:
        # Do not persist silent local default as "user chose this".
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass
        return
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
    if not name or name == UNCONFIGURED_PROVIDER:
        return False
    if name == "ollama":
        return True
    try:
        from codedoggy.model.auth import auth_status

        return bool(auth_status(name).logged_in)
    except Exception:  # noqa: BLE001
        return False


def resolve_startup_provider() -> str | None:
    """Pick MAIN provider when ``CODEDOGGY_PROVIDER`` is unset.

    Returns ``None`` when nothing was chosen — caller must not invent ollama.
    Order: sticky imperial preference → first logged-in imperial → None.
    """
    preferred = load_preferred_provider()
    if preferred and preferred != "ollama" and provider_usable(preferred):
        return preferred

    for name in _IMPERIAL_ORDER:
        if provider_usable(name):
            return name

    return None
