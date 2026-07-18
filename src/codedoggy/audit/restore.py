"""Shadow soft restore — best-effort undo of a single mutation on P0.

Not a full OS transaction system. Fail-soft; never raise into the loop.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def shadow_restore_enabled() -> bool:
    """``CODEDOGGY_SHADOW_RESTORE`` default ON; ``0`` / false / off disables."""
    raw = os.environ.get("CODEDOGGY_SHADOW_RESTORE", "1")
    v = str(raw).strip().lower()
    return v not in ("0", "false", "no", "off", "")


def _resolve_under_cwd(cwd: Path, path: str) -> Path | None:
    """Return resolved path if it stays under session cwd; else None."""
    try:
        base = Path(cwd).resolve()
        p = Path(path)
        target = p.resolve() if p.is_absolute() else (base / p).resolve()
        target.relative_to(base)
        return target
    except (OSError, ValueError, RuntimeError):
        return None


def restore_mutation_before(
    cwd: Path,
    mutation: Any,
) -> dict[str, Any]:
    """Soft-restore one mutation to its pre-write state.

    Accepts :class:`~codedoggy.turn.types.FileMutation` or
    :class:`~codedoggy.audit.types.MutationEvent`.

    Rules (best-effort, never raises):
    - Path must resolve under ``cwd``
    - ``is_create`` → delete the file if it exists
    - ``is_delete`` with ``before`` → rewrite file from ``before``
    - otherwise restore only when ``before is not None`` and not create
    - returns ``{ok, path, reason}``
    """
    path_s = str(getattr(mutation, "path", "") or "")
    result: dict[str, Any] = {"ok": False, "path": path_s, "reason": ""}
    try:
        if not path_s:
            result["reason"] = "empty_path"
            return result
        if cwd is None:
            result["reason"] = "no_cwd"
            return result

        target = _resolve_under_cwd(Path(cwd), path_s)
        if target is None:
            result["reason"] = "path_outside_cwd"
            return result

        is_create = bool(getattr(mutation, "is_create", False))
        is_delete = bool(getattr(mutation, "is_delete", False))
        before = getattr(mutation, "before", None)

        if is_create:
            if target.is_file() or target.is_symlink():
                try:
                    target.unlink()
                except OSError as e:
                    result["reason"] = f"unlink_failed: {e}"
                    return result
            # Already gone counts as restored
            result["ok"] = True
            result["reason"] = "create_undone"
            return result

        if before is None:
            result["reason"] = "no_before"
            return result

        # is_delete with before, or normal edit with before → rewrite.
        # Bytes write preserves exact content (text mode rewrites \n on Windows).
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(before, bytes):
                target.write_bytes(before)
            else:
                target.write_bytes(str(before).encode("utf-8"))
        except OSError as e:
            result["reason"] = f"write_failed: {e}"
            return result

        result["ok"] = True
        result["reason"] = "delete_restored" if is_delete else "before_restored"
        return result
    except Exception as e:  # noqa: BLE001 — never raise into the loop
        result["reason"] = f"unexpected: {type(e).__name__}: {e}"
        return result
