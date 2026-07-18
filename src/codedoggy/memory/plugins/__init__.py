"""Bundled memory provider plugins + discovery API.

Hermes layout: plugins/memory/<name>/. Discovery lives here so the package
``codedoggy.memory.plugins`` is the single import surface (no plugins.py clash).
"""

from codedoggy.memory._plugin_discovery import (
    discover_memory_providers,
    find_provider_dir,
    list_memory_provider_names,
    load_memory_provider,
    load_memory_provider_from_env,
)

__all__ = [
    "discover_memory_providers",
    "find_provider_dir",
    "list_memory_provider_names",
    "load_memory_provider",
    "load_memory_provider_from_env",
]
