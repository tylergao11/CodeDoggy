"""Process-wide codebase graph runtime and session watch leases.

This is the Python glue analogue of two Grok layers:

* ``xai-codebase-graph::IndexManagerHandle`` — commands are submitted to one
  single-owner worker per canonical workspace.
* ``xai-grok-workspace::CodebaseIndexManager`` — process-wide weak handle
  reuse, so sessions do not build competing indexes for the same root.

The symbol extraction/index algorithms remain in ``IndexBuilder`` and
``IndexManager``.  This module only owns lifecycle, ordering, cache ownership,
and filesystem-watch leases.
"""

from __future__ import annotations

import logging
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from codedoggy.graph.builder import IndexBuilder
from codedoggy.graph.cache import get_cache_path, load_index, save_index
from codedoggy.graph.index import ScopeGraphIndex
from codedoggy.graph.index_manager import FileEvent, IndexManager
from codedoggy.graph.languages import LanguageRegistry
from codedoggy.graph.types import IndexStats
from codedoggy.graph.watcher import DEFAULT_DEBOUNCE_SECS, WorkspaceWatcher

logger = logging.getLogger(__name__)


def _stats_dict(index: ScopeGraphIndex) -> dict[str, Any]:
    stats = index.stats()
    return {
        "files": stats.files,
        "definitions": stats.definitions,
        "references": stats.references,
        "query_version": index.query_version.to_wire(),
    }


class WorkspaceGraphRuntime:
    """Single-owner command runtime for one canonical workspace.

    File events are fire-and-forget commands.  If the index has not been used
    yet, events are retained without triggering an O(repository) build.  The
    first query/reindex is a fence: it builds or loads once, replays earlier
    events, and then returns a stable manager handle.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.registry = LanguageRegistry()
        self._builder = IndexBuilder(registry=self.registry)
        thread_tag = f"{abs(hash(str(self.root))) & 0xFFFF_FFFF:08x}"
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"codegraph-{thread_tag}",
        )
        self._submit_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._manager: IndexManager | None = None
        self._pending_events: list[FileEvent] = []
        self._dirty = False
        self._rebuild_requested = False
        self._rebuild_future: Future[dict[str, Any]] | None = None

        self._watch_lock = threading.RLock()
        self._watcher: WorkspaceWatcher | None = None
        self._watch_leases: set[object] = set()

    # ----- actor command submission -----

    def _submit(self, fn, /, *args):  # noqa: ANN001, ANN202
        # A submission mutex gives commands from competing session threads one
        # definite queue order, matching a multi-producer actor channel.
        with self._submit_lock:
            return self._executor.submit(fn, *args)

    @staticmethod
    def _log_background_failure(future: Future[Any]) -> None:
        try:
            exc = future.exception()
        except BaseException:  # cancelled/interpreter shutdown
            return
        if exc is not None:
            logger.error(
                "code graph actor command failed: %s",
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    def _submit_background(self, fn, /, *args) -> None:  # noqa: ANN001
        future = self._submit(fn, *args)
        future.add_done_callback(self._log_background_failure)

    def fence(self) -> None:
        """Wait until every command submitted before this call is applied."""
        self._submit(lambda: None).result()

    # ----- lazy build / snapshots -----

    def ensure_indexed(
        self,
        *,
        use_cache: bool = True,
        allow_cache_write: bool = True,
    ) -> IndexManager:
        return self._submit(
            self._ensure_worker,
            bool(use_cache),
            bool(allow_cache_write),
        ).result()

    def _ensure_worker(
        self,
        use_cache: bool,
        allow_cache_write: bool,
    ) -> IndexManager:
        with self._state_lock:
            manager = self._manager
            rebuild_requested = self._rebuild_requested
        if manager is not None and not rebuild_requested:
            return manager

        loaded_from_cache = False
        candidate: ScopeGraphIndex | None = None
        cache_path = get_cache_path(self.root)
        if manager is None and use_cache and cache_path.is_file():
            try:
                loaded = load_index(cache_path)
                if self.cache_fresh(loaded):
                    candidate = loaded
                    loaded_from_cache = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("graph cache load failed path=%s: %s", cache_path, exc)

        if candidate is None:
            candidate = self._builder.build(self.root)

        manager = IndexManager(
            self.root,
            index=candidate,
            registry=self.registry,
        )

        with self._state_lock:
            pending = self._pending_events
            self._pending_events = []
        if pending:
            manager.send_events(pending)

        with self._state_lock:
            self._manager = manager
            self._rebuild_requested = False
            self._dirty = (not loaded_from_cache) or bool(pending)
        self._refresh_watcher_display(manager)

        if use_cache and allow_cache_write and self.is_dirty:
            self._persist_worker(True, True)
        return manager

    def cache_fresh(self, index: ScopeGraphIndex) -> bool:
        """Validate query version, metadata, and the full supported-file set."""
        current_qv = self.registry.compute_query_hash()
        if index.needs_query_rebuild(current_qv) or not index.file_meta:
            return False
        for rel, meta in index.file_meta.items():
            if meta.is_stale(self.root / rel):
                return False
        try:
            on_disk = {
                path.resolve().relative_to(self.root).as_posix()
                for path in self._builder._collect_files(self.root)
            }
        except Exception:  # noqa: BLE001
            logger.debug("graph cache disk scan failed", exc_info=True)
            return False
        cached = {str(path).replace("\\", "/") for path in index.file_meta}
        return on_disk == cached

    # ----- actor mutation commands -----

    def send_event(self, event: FileEvent) -> None:
        """Enqueue one event; never force the lazy full index to build."""
        self._submit_background(self._apply_events_worker, [event])

    def send_events(self, events: list[FileEvent]) -> None:
        """Enqueue an already-debounced event batch for sequential handling."""
        if events:
            self._submit_background(self._apply_events_worker, list(events))

    def _apply_events_worker(self, events: list[FileEvent]) -> None:
        with self._state_lock:
            manager = self._manager
            if manager is None:
                self._pending_events.extend(events)
                self._dirty = True
                return
        manager.send_events(events)
        with self._state_lock:
            self._dirty = True

    def request_rebuild(self) -> None:
        """Invalidate lazily; the next query/reindex performs the full build."""
        self._submit_background(self._request_rebuild_worker)

    def _request_rebuild_worker(self) -> None:
        with self._state_lock:
            self._rebuild_requested = True
            self._dirty = True

    def rebuild(
        self,
        *,
        use_cache: bool = True,
        allow_cache_write: bool = True,
    ) -> dict[str, Any]:
        """Run one explicit rebuild; concurrent callers share the in-flight work."""
        with self._submit_lock:
            future = self._rebuild_future
            if future is None or future.done():
                future = self._executor.submit(self._rebuild_worker)
                self._rebuild_future = future
        result = future.result()
        # Cache policy belongs to each Session lease, not to whichever caller
        # happened to win the shared rebuild submission race.
        if use_cache and allow_cache_write:
            self.persist(use_cache=True, allow_cache_write=True)
        return result

    def _rebuild_worker(self) -> dict[str, Any]:
        candidate = self._builder.build(self.root)
        with self._state_lock:
            manager = self._manager
            pending = self._pending_events
            self._pending_events = []
        manager = IndexManager(
            self.root,
            index=candidate,
            registry=self.registry,
        )
        if pending:
            manager.send_events(pending)
        with self._state_lock:
            self._manager = manager
            self._rebuild_requested = False
            self._dirty = True
        self._refresh_watcher_display(manager)
        return manager.read_index(_stats_dict)

    def _refresh_watcher_display(self, manager: IndexManager) -> None:
        with self._watch_lock:
            watcher = self._watcher
            if watcher is not None:
                watcher.set_display_manager(manager)

    # ----- persistence -----

    @property
    def is_dirty(self) -> bool:
        with self._state_lock:
            return self._dirty

    def persist(
        self,
        *,
        use_cache: bool = True,
        allow_cache_write: bool = True,
    ) -> bool:
        return self._submit(
            self._persist_worker,
            bool(use_cache),
            bool(allow_cache_write),
        ).result()

    def repair_cache_if_invalid(self, *, allow_cache_write: bool = True) -> bool:
        """Rewrite an unreadable/incompatible cache from the live actor state."""
        return self._submit(
            self._repair_cache_worker,
            bool(allow_cache_write),
        ).result()

    def _repair_cache_worker(self, allow_cache_write: bool) -> bool:
        if not allow_cache_write:
            return False
        with self._state_lock:
            manager = self._manager
        if manager is None:
            return False
        cache_path = get_cache_path(self.root)
        try:
            load_index(cache_path)
            return False
        except Exception:  # noqa: BLE001
            save_index(cache_path, manager.get_snapshot())
            with self._state_lock:
                self._dirty = False
            return True

    def _persist_worker(
        self,
        use_cache: bool,
        allow_cache_write: bool,
    ) -> bool:
        if not use_cache or not allow_cache_write:
            return False
        with self._state_lock:
            manager = self._manager
            dirty = self._dirty
        if manager is None or not dirty:
            return False
        save_index(get_cache_path(self.root), manager.get_snapshot())
        with self._state_lock:
            self._dirty = False
        return True

    def stats(
        self,
        *,
        use_cache: bool = True,
        allow_cache_write: bool = True,
    ) -> IndexStats:
        manager = self.ensure_indexed(
            use_cache=use_cache,
            allow_cache_write=allow_cache_write,
        )
        return manager.stats()

    @property
    def manager_if_ready(self) -> IndexManager | None:
        with self._state_lock:
            return self._manager

    # ----- shared filesystem-watch leases -----

    def acquire_watch(
        self,
        lease: object,
        *,
        debounce_secs: float = DEFAULT_DEBOUNCE_SECS,
    ) -> WorkspaceWatcher:
        """Acquire a session lease without forcing the lazy index to build."""
        with self._watch_lock:
            if lease in self._watch_leases and self._watcher is not None:
                display = self.manager_if_ready
                if display is not None:
                    self._watcher.set_display_manager(display)
                return self._watcher
            watcher = self._watcher
            if watcher is None or not watcher.is_running():
                watcher = WorkspaceWatcher(
                    self.root,
                    self,
                    registry=self.registry,
                    debounce_secs=debounce_secs,
                )
                watcher.start()
                self._watcher = watcher
            display = self.manager_if_ready
            if display is not None:
                watcher.set_display_manager(display)
            self._watch_leases.add(lease)
            return watcher

    def release_watch(self, lease: object) -> bool:
        """Release one lease; stop/flush only after the final lease leaves."""
        with self._watch_lock:
            self._watch_leases.discard(lease)
            if self._watch_leases or self._watcher is None:
                return False
            watcher = self._watcher
            self._watcher = None
            # Keep the lease lock while stopping so another session cannot start
            # a second observer before the old observer has flushed and exited.
            watcher.stop()
            return True

    @property
    def watcher(self) -> WorkspaceWatcher | None:
        with self._watch_lock:
            return self._watcher

    @property
    def watch_lease_count(self) -> int:
        with self._watch_lock:
            return len(self._watch_leases)


class CodebaseIndexManager:
    """Process-wide weak registry of workspace graph runtimes.

    Sessions own the strong references through ``CodebaseGraph``.  Once the
    last session handle and watch lease disappear, the weak entry is reaped.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runtimes: weakref.WeakValueDictionary[
            Path, WorkspaceGraphRuntime
        ] = weakref.WeakValueDictionary()

    def get_or_create(
        self, root: Path | str
    ) -> tuple[WorkspaceGraphRuntime, bool]:
        canonical = Path(root).expanduser().resolve()
        with self._lock:
            runtime = self._runtimes.get(canonical)
            if runtime is not None:
                return runtime, False
            runtime = WorkspaceGraphRuntime(canonical)
            self._runtimes[canonical] = runtime
            return runtime, True

    def get(self, root: Path | str) -> WorkspaceGraphRuntime | None:
        canonical = Path(root).expanduser().resolve()
        with self._lock:
            return self._runtimes.get(canonical)

    def active_count(self) -> int:
        with self._lock:
            return len(self._runtimes)


_CODEBASE_INDEX_MANAGER = CodebaseIndexManager()


def get_codebase_index_manager() -> CodebaseIndexManager:
    """Return the process-wide workspace graph runtime registry."""
    return _CODEBASE_INDEX_MANAGER
