"""Memory provider plugin discovery — hermes-agent/plugins/memory/__init__.py.

Scans:
  1. Bundled: ``<package>/memory/plugins/<name>/`` (optional ship)
  2. User: ``$CODEDOGGY_HOME/plugins/memory/<name>/`` or ``.../plugins/<name>/``
     with MemoryProvider / register_memory_provider in ``__init__.py``

Only ONE external provider is active (MemoryManager.add_provider enforces).
Selection: ``CODEDOGGY_MEMORY_PROVIDER`` env or ``load_memory_provider(name)``.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from codedoggy.memory.paths import default_memory_home

logger = logging.getLogger(__name__)

_BUNDLED_DIR = Path(__file__).resolve().parent / "plugins"
_USER_NAMESPACE = "_codedoggy_user_memory"


def _register_synthetic_package(name: str, search_locations: list[str]) -> None:
    if name in sys.modules:
        return
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = search_locations
    sys.modules[name] = importlib.util.module_from_spec(spec)


def _get_user_plugins_roots() -> list[Path]:
    home = default_memory_home()
    candidates = [
        home / "plugins" / "memory",
        home / "plugins",
    ]
    return [p for p in candidates if p.is_dir()]


def _is_memory_provider_dir(path: Path) -> bool:
    init_file = path / "__init__.py"
    if not init_file.exists():
        return False
    try:
        source = init_file.read_text(encoding="utf-8", errors="replace")[:8192]
        return (
            "register_memory_provider" in source
            or "MemoryProvider" in source
            or "BaseMemoryProvider" in source
        )
    except Exception:  # noqa: BLE001
        return False


def _iter_provider_dirs() -> list[tuple[str, Path]]:
    seen: set[str] = set()
    dirs: list[tuple[str, Path]] = []
    # Bundled
    if _BUNDLED_DIR.is_dir():
        for child in sorted(_BUNDLED_DIR.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            if not (child / "__init__.py").exists():
                continue
            seen.add(child.name)
            dirs.append((child.name, child))
    # User
    for user_root in _get_user_plugins_roots():
        for child in sorted(user_root.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            if child.name in seen:
                continue
            if user_root.name == "plugins" and child.name == "memory":
                # nested memory/ already handled via plugins/memory root
                continue
            if not _is_memory_provider_dir(child):
                continue
            seen.add(child.name)
            dirs.append((child.name, child))
    return dirs


def list_memory_provider_names() -> list[str]:
    """Directory scan only — no imports (Hermes list_memory_provider_names)."""
    return sorted({name for name, _ in _iter_provider_dirs()})


def discover_memory_providers() -> list[tuple[str, str, bool]]:
    """Return (name, description, is_available) — Hermes discover_memory_providers."""
    results: list[tuple[str, str, bool]] = []
    for name, child in _iter_provider_dirs():
        desc = ""
        yaml_file = child / "plugin.yaml"
        if yaml_file.exists():
            try:
                text = yaml_file.read_text(encoding="utf-8")
                for line in text.splitlines():
                    if line.strip().startswith("description:"):
                        desc = line.split(":", 1)[1].strip().strip("\"'")
                        break
            except Exception:  # noqa: BLE001
                pass
        available = True
        try:
            provider = _load_provider_from_dir(child)
            if provider is None:
                available = False
            else:
                avail = getattr(provider, "is_available", None)
                available = bool(avail()) if callable(avail) else True
        except Exception:  # noqa: BLE001
            available = False
        results.append((name, desc, available))
    return results


def find_provider_dir(name: str) -> Optional[Path]:
    for n, path in _iter_provider_dirs():
        if n == name:
            return path
    return None


def load_memory_provider(name: str) -> Any | None:
    """Load MemoryProvider instance by name (Hermes load_memory_provider)."""
    provider_dir = find_provider_dir(name)
    if not provider_dir:
        logger.debug("Memory provider %r not found", name)
        return None
    try:
        provider = _load_provider_from_dir(provider_dir)
        if provider is None:
            logger.warning("Memory provider %r loaded but no instance", name)
        return provider
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to load memory provider %r: %s", name, e)
        return None


def load_memory_provider_from_env() -> Any | None:
    """``CODEDOGGY_MEMORY_PROVIDER`` → instance or None."""
    name = (os.environ.get("CODEDOGGY_MEMORY_PROVIDER") or "").strip()
    if not name:
        return None
    return load_memory_provider(name)


def _load_provider_from_dir(provider_dir: Path) -> Any | None:
    """Import provider package and call register_memory_provider or find class."""
    name = provider_dir.name
    # Prefer bundled package path if under our memory/plugins
    if _BUNDLED_DIR in provider_dir.parents or provider_dir.parent == _BUNDLED_DIR:
        mod_name = f"codedoggy.memory.plugins.{name}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            mod = _load_as_path(provider_dir, f"codedoggy.memory.plugins.{name}")
    else:
        _register_synthetic_package(_USER_NAMESPACE, [str(provider_dir.parent)])
        mod_name = f"{_USER_NAMESPACE}.{name}"
        mod = _load_as_path(provider_dir, mod_name)

    if mod is None:
        return None

    reg = getattr(mod, "register_memory_provider", None)
    if callable(reg):
        inst = reg()
        return inst

    # Find first class with name attr and is_available / prefetch
    for attr in dir(mod):
        obj = getattr(mod, attr, None)
        if not isinstance(obj, type) or attr.startswith("_"):
            continue
        if attr in {"BaseMemoryProvider", "MemoryProvider"}:
            continue
        if hasattr(obj, "prefetch") and hasattr(obj, "name"):
            try:
                return obj()
            except Exception:  # noqa: BLE001
                continue
    return None


def _load_as_path(provider_dir: Path, mod_name: str) -> Any | None:
    init_file = provider_dir / "__init__.py"
    if not init_file.exists():
        return None
    # Ensure parent packages exist for relative imports
    parent = mod_name.rsplit(".", 1)[0] if "." in mod_name else None
    if parent and parent not in sys.modules:
        _register_synthetic_package(parent, [str(provider_dir.parent)])
    spec = importlib.util.spec_from_file_location(
        mod_name,
        init_file,
        submodule_search_locations=[str(provider_dir)],
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        logger.warning("import %s failed: %s", mod_name, e)
        sys.modules.pop(mod_name, None)
        return None
    return mod
