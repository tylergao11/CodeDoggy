"""Bounded curated memory: MEMORY.md + USER.md with frozen system-prompt snapshot."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from codedoggy.memory.defaults import (
    ENTRY_DELIMITER,
    MAX_CONSOLIDATION_FAILURES_PER_TURN,
    MEMORY_CHAR_LIMIT,
    USER_CHAR_LIMIT,
)
from codedoggy.memory.paths import get_memory_dir
from codedoggy.memory.scan import first_threat_message, threat_ids

logger = logging.getLogger(__name__)

# Optional platform locks
try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt
except ImportError:
    msvcrt = None  # type: ignore[assignment]


def load_on_disk_store(
    memory_dir: Path | str | None = None,
    *,
    memory_char_limit: int | None = None,
    user_char_limit: int | None = None,
) -> "MemoryStore":
    """Hermes ``load_on_disk_store`` — fresh store for CLI/gateway without live agent.

    Source: hermes-agent/tools/memory_tool.py. Honors env overrides:
      CODEDOGGY_MEMORY_CHAR_LIMIT / CODEDOGGY_USER_CHAR_LIMIT
    """
    mem_lim = memory_char_limit
    user_lim = user_char_limit
    if mem_lim is None:
        raw = os.environ.get("CODEDOGGY_MEMORY_CHAR_LIMIT", "").strip()
        try:
            mem_lim = int(raw) if raw else MEMORY_CHAR_LIMIT
        except ValueError:
            mem_lim = MEMORY_CHAR_LIMIT
    if user_lim is None:
        raw = os.environ.get("CODEDOGGY_USER_CHAR_LIMIT", "").strip()
        try:
            user_lim = int(raw) if raw else USER_CHAR_LIMIT
        except ValueError:
            user_lim = USER_CHAR_LIMIT
    store = MemoryStore(
        memory_dir=memory_dir,
        memory_char_limit=int(mem_lim),
        user_char_limit=int(user_lim),
    )
    store.load_from_disk()
    return store


class MemoryStore:
    """
    File-backed curated memory with two stores (Hermes tools/memory_tool.py):

    - ``memory`` → MEMORY.md — agent notes (env, conventions, lessons)
    - ``user`` → USER.md — user profile (prefs, style, habits)

    Parallel state:
    - ``_system_prompt_snapshot``: frozen at ``load_from_disk()`` for prompt injection
      (stable prefix cache; mid-session writes do not change it)
    - ``memory_entries`` / ``user_entries``: live state mutated by the memory tool
    """

    def __init__(
        self,
        memory_dir: Path | str | None = None,
        *,
        memory_char_limit: int = MEMORY_CHAR_LIMIT,
        user_char_limit: int = USER_CHAR_LIMIT,
    ) -> None:
        self.memory_dir = (
            Path(memory_dir).resolve()
            if memory_dir is not None
            else get_memory_dir()
        )
        self.memory_entries: list[str] = []
        self.user_entries: list[str] = []
        self.memory_char_limit = int(memory_char_limit)
        self.user_char_limit = int(user_char_limit)
        self._system_prompt_snapshot: dict[str, str] = {"memory": "", "user": ""}
        self._consolidation_failures = 0

    # -- lifecycle ------------------------------------------------------------

    def reset_consolidation_failures(self) -> None:
        self._consolidation_failures = 0

    def load_from_disk(self) -> None:
        """Load entries and capture the frozen system-prompt snapshot."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_entries = self._read_file(self.memory_dir / "MEMORY.md")
        self.user_entries = self._read_file(self.memory_dir / "USER.md")
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))
        self.refresh_system_prompt_snapshot()

    def refresh_system_prompt_snapshot(self) -> None:
        """Re-freeze snapshot from current live entries (no disk re-read).

        Called after mid-turn memory_flush so the system-prompt view and
        HermesMemorySelector (prefer_frozen) see newly durable facts without
        waiting for the next session load.
        """
        sanitized_memory = self._sanitize_for_snapshot(self.memory_entries, "MEMORY.md")
        sanitized_user = self._sanitize_for_snapshot(self.user_entries, "USER.md")
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", sanitized_memory),
            "user": self._render_block("user", sanitized_user),
        }

    def format_for_system_prompt(self, target: str) -> str | None:
        """Frozen snapshot block for system prompt; None if empty at load time."""
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    def format_live_block(self, target: str) -> str | None:
        """Render *live* entries (post mid-session writes) without using freeze."""
        entries = self._entries_for(target)
        if not entries:
            return None
        sanitized = self._sanitize_for_snapshot(
            list(entries),
            "USER.md" if target == "user" else "MEMORY.md",
        )
        block = self._render_block(target, sanitized)
        return block if block else None

    def live_system_prompt_blocks(self) -> str:
        """Concatenate non-empty *live* memory + user blocks (not frozen)."""
        parts: list[str] = []
        for key in ("user", "memory"):
            block = self.format_live_block(key)
            if block:
                parts.append(block)
        return "\n\n".join(parts)

    def system_prompt_blocks(self) -> str:
        """Concatenate non-empty frozen memory + user blocks."""
        parts: list[str] = []
        for key in ("user", "memory"):
            block = self.format_for_system_prompt(key)
            if block:
                parts.append(block)
        hint = self.consolidation_hint()
        if hint:
            parts.append(hint)
        return "\n\n".join(parts)

    def usage_ratio(self, target: str = "memory") -> float:
        limit = self._char_limit(target)
        if limit <= 0:
            return 0.0
        return min(1.0, self._char_count(target) / float(limit))

    def consolidation_hint(self, *, warn_ratio: float = 0.80) -> str | None:
        """When curated memory is nearly full, tell the model to consolidate."""
        notes: list[str] = []
        for target, label in (("memory", "MEMORY.md"), ("user", "USER.md")):
            ratio = self.usage_ratio(target)
            if ratio >= warn_ratio:
                pct = int(ratio * 100)
                cur = self._char_count(target)
                lim = self._char_limit(target)
                notes.append(
                    f"- {label} at {pct}% ({cur:,}/{lim:,} chars). "
                    f"Before adding more: memory(action=replace|remove|batch) to "
                    f"merge/drop stale entries, then add."
                )
        if not notes:
            return None
        return (
            "── memory capacity ──\n"
            "Curated memory nearing limit — consolidate, do not force-add:\n"
            + "\n".join(notes)
            + "\n── end capacity ──"
        )

    # -- mutations ------------------------------------------------------------

    def add(self, target: str, content: str) -> dict[str, Any]:
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}
        delim_err = self._reject_delimiter(content)
        if delim_err:
            return {"success": False, "error": delim_err}
        scan = first_threat_message(content)
        if scan:
            return {"success": False, "error": scan}

        with self._file_lock(self._path_for(target)):
            self._reload_target(target, skip_drift=True)
            entries = self._entries_for(target)
            limit = self._char_limit(target)

            if content in entries:
                return self._success_response(target, "Entry already exists (no duplicate added).")

            new_entries = entries + [content]
            new_total = len(ENTRY_DELIMITER.join(new_entries))
            if new_total > limit:
                current = self._char_count(target)
                return self._consolidation_failure(
                    {
                        "success": False,
                        "error": (
                            f"Memory at {current:,}/{limit:,} chars. "
                            f"Adding this entry ({len(content)} chars) would exceed the limit. "
                            f"Consolidate: use 'replace' to merge entries or 'remove' stale ones "
                            f"(see current_entries), then retry add in this turn."
                        ),
                        "current_entries": list(entries),
                        "usage": f"{current:,}/{limit:,}",
                    }
                )

            entries.append(content)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry added.")

    def replace(self, target: str, old_text: str, new_content: str) -> dict[str, Any]:
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {
                "success": False,
                "error": "new_content cannot be empty. Use 'remove' to delete entries.",
            }
        delim_err = self._reject_delimiter(new_content)
        if delim_err:
            return {"success": False, "error": delim_err}
        scan = first_threat_message(new_content)
        if scan:
            return {"success": False, "error": scan}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak:
                return self._drift_error(self._path_for(target), bak)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]
            if not matches:
                return {
                    "success": False,
                    "error": (
                        f"No entry matched {old_text!r}. "
                        "Check current_entries and retry with exact text."
                    ),
                    "current_entries": list(entries),
                }
            if len(matches) > 1:
                unique = {e for _, e in matches}
                if len(unique) > 1:
                    return {
                        "success": False,
                        "error": f"Multiple entries matched {old_text!r}. Be more specific.",
                        "matches": self._previews([e for _, e in matches]),
                    }

            idx = matches[0][0]
            limit = self._char_limit(target)
            test = entries.copy()
            test[idx] = new_content
            new_total = len(ENTRY_DELIMITER.join(test))
            if new_total > limit:
                current = self._char_count(target)
                return self._consolidation_failure(
                    {
                        "success": False,
                        "error": (
                            f"Replacement would put memory at {new_total:,}/{limit:,} chars. "
                            "Shorten content or remove other entries first."
                        ),
                        "current_entries": list(entries),
                        "usage": f"{current:,}/{limit:,}",
                    }
                )

            entries[idx] = new_content
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> dict[str, Any]:
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak:
                return self._drift_error(self._path_for(target), bak)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]
            if not matches:
                return {
                    "success": False,
                    "error": (
                        f"No entry matched {old_text!r}. "
                        "Check current_entries and retry."
                    ),
                    "current_entries": list(entries),
                }
            if len(matches) > 1:
                unique = {e for _, e in matches}
                if len(unique) > 1:
                    return {
                        "success": False,
                        "error": f"Multiple entries matched {old_text!r}. Be more specific.",
                        "matches": self._previews([e for _, e in matches]),
                    }

            entries.pop(matches[0][0])
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry removed.")

    def apply_batch(self, target: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply add/replace/remove ops atomically (all-or-nothing)."""
        if not operations:
            return {"success": False, "error": "operations list is empty."}

        for i, op in enumerate(operations):
            act = (op or {}).get("action")
            new_content = (op or {}).get("content")
            if act in {"add", "replace"} and new_content:
                text = str(new_content)
                delim_err = self._reject_delimiter(text)
                if delim_err:
                    return {"success": False, "error": f"Operation {i + 1}: {delim_err}"}
                scan = first_threat_message(text)
                if scan:
                    return {"success": False, "error": f"Operation {i + 1}: {scan}"}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak:
                return self._drift_error(self._path_for(target), bak)

            working = list(self._entries_for(target))
            limit = self._char_limit(target)

            for i, op in enumerate(operations):
                op = op or {}
                act = op.get("action")
                content = (op.get("content") or "").strip()
                old_text = (op.get("old_text") or "").strip()
                pos = f"Operation {i + 1} ({act or 'unknown'})"

                if act == "add":
                    if not content:
                        return self._batch_error(target, f"{pos}: content is required.")
                    if content in working:
                        continue
                    working.append(content)
                elif act == "replace":
                    if not old_text:
                        return self._batch_error(target, f"{pos}: old_text is required.")
                    if not content:
                        return self._batch_error(
                            target,
                            f"{pos}: content is required (use action='remove' to delete).",
                        )
                    matches = [j for j, e in enumerate(working) if old_text in e]
                    if not matches:
                        return self._batch_error(
                            target, f"{pos}: no entry matched {old_text!r}."
                        )
                    if len({working[j] for j in matches}) > 1:
                        return self._batch_error(
                            target,
                            f"{pos}: {old_text!r} matched multiple distinct entries.",
                        )
                    working[matches[0]] = content
                elif act == "remove":
                    if not old_text:
                        return self._batch_error(target, f"{pos}: old_text is required.")
                    matches = [j for j, e in enumerate(working) if old_text in e]
                    if not matches:
                        return self._batch_error(
                            target, f"{pos}: no entry matched {old_text!r}."
                        )
                    if len({working[j] for j in matches}) > 1:
                        return self._batch_error(
                            target,
                            f"{pos}: {old_text!r} matched multiple distinct entries.",
                        )
                    working.pop(matches[0])
                else:
                    return self._batch_error(
                        target, f"{pos}: unknown action. Use add, replace, or remove."
                    )

            new_total = len(ENTRY_DELIMITER.join(working)) if working else 0
            if new_total > limit:
                current = self._char_count(target)
                return self._consolidation_failure(
                    {
                        "success": False,
                        "error": (
                            f"After {len(operations)} operation(s), memory would be at "
                            f"{new_total:,}/{limit:,} chars. Remove or shorten more, then retry."
                        ),
                        "current_entries": list(self._entries_for(target)),
                        "usage": f"{current:,}/{limit:,}",
                    }
                )

            self._set_entries(target, working)
            self.save_to_disk(target)

        return self._success_response(
            target, f"Applied {len(operations)} operation(s)."
        )

    def save_to_disk(self, target: str) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    # -- internals ------------------------------------------------------------

    def _path_for(self, target: str) -> Path:
        if target == "user":
            return self.memory_dir / "USER.md"
        if target == "memory":
            return self.memory_dir / "MEMORY.md"
        raise ValueError(f"unknown memory target: {target!r}")

    def _entries_for(self, target: str) -> list[str]:
        return self.user_entries if target == "user" else self.memory_entries

    def _set_entries(self, target: str, entries: list[str]) -> None:
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        return self.user_char_limit if target == "user" else self.memory_char_limit

    def _reload_target(self, target: str, *, skip_drift: bool = False) -> str | None:
        bak = None if skip_drift else self._detect_external_drift(target)
        fresh = self._read_file(self._path_for(target))
        self._set_entries(target, list(dict.fromkeys(fresh)))
        return bak

    def _consolidation_failure(self, response: dict[str, Any]) -> dict[str, Any]:
        self._consolidation_failures += 1
        if self._consolidation_failures <= MAX_CONSOLIDATION_FAILURES_PER_TURN:
            return response
        return {
            "success": False,
            "done": True,
            "error": (
                f"Memory consolidation failed {self._consolidation_failures} times "
                "this turn. Stop retrying memory calls — leave memory unchanged and "
                "continue your reply. Save later if needed."
            ),
        }

    def _batch_error(self, target: str, message: str) -> dict[str, Any]:
        """Validation failure (not capacity): do not burn consolidation budget."""
        current = self._char_count(target)
        limit = self._char_limit(target)
        return {
            "success": False,
            "error": message + " No operations were applied (batch is all-or-nothing).",
            "current_entries": list(self._entries_for(target)),
            "usage": f"{current:,}/{limit:,}",
        }

    def _success_response(self, target: str, message: str | None = None) -> dict[str, Any]:
        self._consolidation_failures = 0
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        resp: dict[str, Any] = {
            "success": True,
            "done": True,
            "target": target,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
            "note": "Write saved. This update is complete — do not repeat it.",
        }
        if message:
            resp["message"] = message
        return resp

    def _render_block(self, target: str, entries: list[str]) -> str:
        if not entries:
            return ""
        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"
        sep = "═" * 46
        return f"{sep}\n{header}\n{sep}\n{content}"

    @staticmethod
    def _sanitize_for_snapshot(entries: list[str], filename: str) -> list[str]:
        out: list[str] = []
        for entry in entries:
            if not entry or entry.startswith("[BLOCKED:"):
                out.append(entry)
                continue
            hits = threat_ids(entry)
            if hits:
                logger.warning("Memory entry from %s blocked at load: %s", filename, hits)
                out.append(
                    f"[BLOCKED: {filename} entry contained threat pattern(s): "
                    f"{', '.join(hits)}. Removed from system prompt; "
                    f"use memory(action=remove) to delete the original.]"
                )
            else:
                out.append(entry)
        return out

    @staticmethod
    def _previews(entries: list[str], width: int = 80) -> list[str]:
        return [e[:width] + ("..." if len(e) > width else "") for e in entries]

    @staticmethod
    def _drift_error(path: Path, bak_path: str) -> dict[str, Any]:
        """Hermes tools/memory_tool.py ``_drift_error`` (issue #26045 wording)."""
        return {
            "success": False,
            "error": (
                f"Refusing to write {path.name}: file on disk has content that "
                f"wouldn't round-trip through the memory tool (likely added by "
                f"a patch tool, shell append, manual edit, or concurrent session). "
                f"A snapshot was saved to {bak_path}. "
                f"Resolve the drift first — either rewrite the file as a clean "
                f"§-delimited list of entries, or move the extra content out — "
                f"then retry. This guard exists to prevent silent data loss."
            ),
            "drift_backup": bak_path,
            "remediation": (
                "Open the .bak file, integrate missing entries via "
                "memory(action=add), then rewrite the original to a clean "
                "§-delimited state."
            ),
        }

    def _detect_external_drift(self, target: str) -> str | None:
        """Hermes MemoryStore._detect_external_drift.

        Two signals (hermes-agent tools/memory_tool.py):
          1. Round-trip mismatch through § parse/join
          2. Any single entry larger than the *store* char limit (external
             free-form append treated as one entry — issue #26045)
        """
        path = self._path_for(target)
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        if not raw.strip():
            return None
        parsed = [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]
        roundtrip = ENTRY_DELIMITER.join(parsed)
        char_limit = self._char_limit(target)
        max_entry_len = max((len(e) for e in parsed), default=0)
        drift = (raw.strip() != roundtrip) or (max_entry_len > char_limit)
        if not drift:
            return None
        ts = int(time.time())
        bak = path.with_suffix(path.suffix + f".bak.{ts}")
        try:
            bak.write_text(raw, encoding="utf-8")
        except OSError:
            return str(bak) + " (BACKUP FAILED — file unchanged on disk)"
        return str(bak)

    @staticmethod
    def _reject_delimiter(content: str) -> str | None:
        if ENTRY_DELIMITER in content or content.strip() == "§":
            return (
                "Content must not contain the entry delimiter "
                f"({ENTRY_DELIMITER!r}). Split into separate entries instead."
            )
        return None

    @staticmethod
    def _read_file(path: Path) -> list[str]:
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return []
        if not raw.strip():
            return []
        return [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]

    @staticmethod
    def _write_file(path: Path, entries: list[str]) -> None:
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".mem_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    @contextmanager
    def _file_lock(path: Path) -> Iterator[None]:
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if fcntl is None and msvcrt is None:
            yield
            return
        fd = open(lock_path, "a+", encoding="utf-8")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                assert msvcrt is not None
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            try:
                if fcntl:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                elif msvcrt:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            fd.close()
