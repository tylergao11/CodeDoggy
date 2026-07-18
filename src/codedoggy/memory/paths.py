"""Resolve the on-disk memories directory."""

from __future__ import annotations

import os
from pathlib import Path


def default_memory_home() -> Path:
    """Profile/home root for CodeDoggy state (override with CODEDOGGY_HOME)."""
    env = os.environ.get("CODEDOGGY_HOME", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".codedoggy").resolve()


def get_memory_dir(home: Path | None = None) -> Path:
    """Directory that holds MEMORY.md and USER.md."""
    root = home if home is not None else default_memory_home()
    return Path(root).expanduser().resolve() / "memories"
