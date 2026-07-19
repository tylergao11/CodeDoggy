"""Atomic credential file writes with best-effort owner-only perms."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def atomic_write_json(path: Path, data: dict[str, Any], *, indent: int = 2) -> None:
    """Write JSON via temp file + replace. Never truncate then fail mid-write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=indent, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    _owner_only(path)


def _owner_only(path: Path) -> None:
    """Best-effort restrict credentials to the current user."""
    try:
        if os.name == "nt":
            # Windows: ACL via icacls is heavy; at least clear world-writable bit.
            os.chmod(path, 0o600)
        else:
            os.chmod(path, 0o600)
    except OSError as exc:
        logger.debug("could not chmod %s: %s", path, exc)
