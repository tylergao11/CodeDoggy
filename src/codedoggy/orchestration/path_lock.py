"""Write-path locks for tool batches (Grok tool_dispatch::lock_path_for_args).

Same write path → share a mutex → concurrent workers serialize (acquisition
order is scheduler-dependent; **writeback** to the model is still emission order).
No path / read-only → fully concurrent with everything else.
``target_directory`` deliberately omitted (list_dir is not an edit).
Shell / apply_patch often have no single path key → no lock (GLUE limit).
"""

from __future__ import annotations

import threading
from typing import Any


def lock_path_for_args(args: dict[str, Any] | None) -> str | None:
    """Return the path key that should serialize concurrent edits, or None."""
    if not isinstance(args, dict):
        return None
    for key in ("file_path", "path", "target_file"):
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            # Normalize separators for lock key stability
            return v.strip().replace("\\", "/")
    return None


class PathLockTable:
    """Per-batch path mutex table (sync stand-in for tokio Mutex map)."""

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._meta = threading.Lock()

    def lock_for(self, path: str | None) -> threading.Lock | None:
        if not path:
            return None
        with self._meta:
            if path not in self._locks:
                self._locks[path] = threading.Lock()
            return self._locks[path]
