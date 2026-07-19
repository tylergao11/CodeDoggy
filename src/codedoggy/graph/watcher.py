"""Workspace file watcher — FSNotify spirit (watchdog OS notify).

  watchdog (debounced via EventDebouncer) → FileEvent → IndexManager

Hard dependency: watchdog (pyproject).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from codedoggy.graph.index_manager import FileEvent, FileEventKind
from codedoggy.graph.languages import LanguageRegistry

logger = logging.getLogger(__name__)

DEFAULT_DEBOUNCE_SECS = 0.35


class EventSink(Protocol):
    """Actor/manager surface consumed by the filesystem watcher."""

    def send_events(self, events: list[FileEvent]) -> None: ...


_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".codedoggy",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        ".venv",
        "venv",
        ".idea",
        ".vscode",
    }
)


class EventDebouncer:
    """Coalesce FileEvents by path until debounce_secs of quiet time."""

    def __init__(
        self,
        on_flush: Callable[[list[FileEvent]], None],
        *,
        debounce_secs: float = DEFAULT_DEBOUNCE_SECS,
    ) -> None:
        self._on_flush = on_flush
        self._debounce_secs = max(0.05, float(debounce_secs))
        self._lock = threading.Lock()
        self._pending: dict[str, FileEvent] = {}
        self._timer: threading.Timer | None = None

    def push(self, event: FileEvent) -> None:
        with self._lock:
            if event.kind is FileEventKind.Renamed and len(event.paths) >= 2:
                key = f"rename:{event.paths[0]}→{event.paths[1]}"
                self._pending[key] = event
            else:
                for p in event.paths:
                    try:
                        key = str(Path(p).resolve())
                    except OSError:
                        key = str(p)
                    prev = self._pending.get(key)
                    # Atomic save: Removed then Created → net Modified/Created
                    # (never keep Removed if a later create/modify arrives).
                    if event.kind is FileEventKind.Removed:
                        self._pending[key] = FileEvent.removed(p)
                    elif event.kind is FileEventKind.Created:
                        if prev is not None and prev.kind is FileEventKind.Removed:
                            # Atomic rewrite: treat as modified
                            self._pending[key] = FileEvent.modified(p)
                        else:
                            self._pending[key] = FileEvent.created(p)
                    else:
                        if prev is not None and prev.kind is FileEventKind.Removed:
                            self._pending[key] = FileEvent.modified(p)
                        elif prev is not None and prev.kind is FileEventKind.Created:
                            self._pending[key] = FileEvent.created(p)
                        else:
                            self._pending[key] = FileEvent.modified(p)
            self._arm_timer_unlocked()

    def flush_now(self) -> None:
        with self._lock:
            self._cancel_timer_unlocked()
            batch = list(self._pending.values())
            self._pending.clear()
        if batch:
            self._on_flush(batch)

    def close(self) -> None:
        self.flush_now()

    def _arm_timer_unlocked(self) -> None:
        self._cancel_timer_unlocked()
        t = threading.Timer(self._debounce_secs, self._fire)
        t.daemon = True
        self._timer = t
        t.start()

    def _cancel_timer_unlocked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _fire(self) -> None:
        with self._lock:
            batch = list(self._pending.values())
            self._pending.clear()
            self._timer = None
        if batch:
            try:
                self._on_flush(batch)
            except Exception:  # noqa: BLE001
                logger.exception("EventDebouncer flush failed")


def _is_skipped(path: Path, root: Path) -> bool:
    try:
        rel = path.resolve().relative_to(root)
    except ValueError:
        return True
    return any(part in _SKIP_DIR_NAMES or part.startswith(".") for part in rel.parts)


class WorkspaceWatcher:
    """OS notify via watchdog → debounced FileEvents → IndexManager."""

    def __init__(
        self,
        root: Path | str,
        manager: EventSink,
        *,
        registry: LanguageRegistry | None = None,
        debounce_secs: float = DEFAULT_DEBOUNCE_SECS,
        on_events_applied: Callable[[], None] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self._sink = manager
        # Compatibility/debug pointer. Delivery always goes through _sink so a
        # shared runtime can swap its internal IndexManager on rebuild.
        self.manager = manager
        self._manager_lock = threading.Lock()
        self.registry = registry or LanguageRegistry()
        self._on_events_applied = on_events_applied
        self._debouncer = EventDebouncer(self._deliver, debounce_secs=debounce_secs)
        self._observer: Observer | None = None

    def set_manager(self, manager: EventSink) -> None:
        """Point deliveries at a new IndexManager after reindex (no restart)."""
        with self._manager_lock:
            self._sink = manager
            self.manager = manager

    def set_display_manager(self, manager: object) -> None:
        """Refresh the compatibility pointer without changing event routing."""
        with self._manager_lock:
            self.manager = manager

    def start(self) -> None:
        if self.is_running():
            return
        root = self.root
        registry = self.registry
        debouncer = self._debouncer

        class _Handler(FileSystemEventHandler):
            def on_created(self, event) -> None:  # noqa: ANN001
                if event.is_directory:
                    return
                p = Path(event.src_path)
                if _is_skipped(p, root) or not registry.is_supported(p):
                    return
                debouncer.push(FileEvent.created(p))

            def on_modified(self, event) -> None:  # noqa: ANN001
                if event.is_directory:
                    return
                p = Path(event.src_path)
                if _is_skipped(p, root) or not registry.is_supported(p):
                    return
                debouncer.push(FileEvent.modified(p))

            def on_deleted(self, event) -> None:  # noqa: ANN001
                if event.is_directory:
                    return
                p = Path(event.src_path)
                if _is_skipped(p, root):
                    return
                debouncer.push(FileEvent.removed(p))

            def on_moved(self, event) -> None:  # noqa: ANN001
                if event.is_directory:
                    return
                src = Path(event.src_path)
                dest = Path(getattr(event, "dest_path", event.src_path))
                if _is_skipped(src, root) and _is_skipped(dest, root):
                    return
                debouncer.push(FileEvent.renamed(src, dest))

        observer = Observer()
        observer.schedule(_Handler(), str(root), recursive=True)
        observer.daemon = True
        observer.start()
        self._observer = observer
        logger.info("WorkspaceWatcher started on %s", root)

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._observer = None
        self._debouncer.close()

    def is_running(self) -> bool:
        return self._observer is not None and self._observer.is_alive()

    def _deliver(self, events: list[FileEvent]) -> None:
        filtered: list[FileEvent] = []
        for e in events:
            if e.kind is FileEventKind.Removed or e.kind is FileEventKind.Renamed:
                filtered.append(e)
                continue
            keep = [p for p in e.paths if self.registry.is_supported(p) or not p.exists()]
            if keep:
                filtered.append(FileEvent(keep, e.kind))
        if filtered:
            with self._manager_lock:
                sink = self._sink
            sink.send_events(filtered)
            if self._on_events_applied is not None:
                try:
                    self._on_events_applied()
                except Exception:  # noqa: BLE001
                    logger.exception("on_events_applied failed")
