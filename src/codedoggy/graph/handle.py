"""Session-facing graph handle — IndexBuilder + IndexManager + Navigator.

  IndexBuilder.build / cache load (query_version + mtime)
  IndexManager FileEvent for incremental updates
  WorkspaceWatcher (watchdog) on start_watch()
  Navigator for goto_* queries
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from codedoggy.graph.builder import IndexBuilder
from codedoggy.graph.cache import get_cache_path, load_index, save_index
from codedoggy.graph.index import ScopeGraphIndex
from codedoggy.graph.index_manager import FileEvent, IndexManager
from codedoggy.graph.languages import LanguageRegistry
from codedoggy.graph.navigation import Navigator
from codedoggy.graph.types import IndexStats
from codedoggy.graph.watcher import WorkspaceWatcher

logger = logging.getLogger(__name__)


class CodebaseGraph:
    """Workspace code graph for Session.extensions.graph."""

    def __init__(self, root: Path | str, *, use_cache: bool = True) -> None:
        self.root = Path(root).resolve()
        self.use_cache = use_cache
        self._registry = LanguageRegistry()
        self._index: ScopeGraphIndex | None = None
        self._manager: IndexManager | None = None
        self._navigator: Navigator | None = None
        self._watcher: WorkspaceWatcher | None = None

    def ensure_indexed(self) -> ScopeGraphIndex:
        if self._index is not None and self._manager is not None:
            return self._manager.index
        cache = get_cache_path(self.root)
        if self.use_cache and cache.is_file():
            try:
                loaded = load_index(cache)
                if self._cache_fresh(loaded):
                    self._bind_index(loaded)
                    return self._index  # type: ignore[return-value]
            except Exception as e:  # noqa: BLE001
                logger.warning("graph cache load failed: %s", e)
        self.reindex()
        assert self._index is not None
        return self._index

    def _bind_index(self, index: ScopeGraphIndex) -> None:
        self._index = index
        self._manager = IndexManager(self.root, index=index, registry=self._registry)
        self._navigator = Navigator(index, root=self.root)

    def _cache_fresh(self, index: ScopeGraphIndex) -> bool:
        current_qv = self._registry.compute_query_hash()
        if index.needs_query_rebuild(current_qv):
            logger.info(
                "graph cache query_version mismatch (cached=%s current=%s), rebuild",
                index.query_version,
                current_qv,
            )
            return False
        if not index.file_meta:
            return False
        checked = 0
        for rel, meta in index.file_meta.items():
            if meta.is_stale(self.root / rel):
                return False
            checked += 1
            if checked >= 50:
                break
        return True

    def mark_dirty(self, path: str | None = None) -> None:
        self.ensure_indexed()
        assert self._manager is not None
        if path:
            self._manager.send_event(FileEvent.modified(path))
            self._index = self._manager.index
            self._navigator = Navigator(self._index, root=self.root)
        else:
            self.reindex()

    def reindex(self) -> dict[str, Any]:
        index = IndexBuilder(registry=self._registry).build(self.root)
        self._bind_index(index)
        if self.use_cache:
            try:
                save_index(get_cache_path(self.root), index)
            except Exception as e:  # noqa: BLE001
                logger.warning("graph cache save failed: %s", e)
        s = index.stats()
        return {
            "files": s.files,
            "definitions": s.definitions,
            "references": s.references,
            "query_version": index.query_version.to_wire(),
        }

    def send_event(self, event: FileEvent) -> None:
        self.ensure_indexed()
        assert self._manager is not None
        self._manager.send_event(event)
        self._index = self._manager.index
        self._navigator = Navigator(self._index, root=self.root)

    def start_watch(self, *, debounce_secs: float = 0.35) -> None:
        """Start OS file watch (watchdog)."""
        self.ensure_indexed()
        assert self._manager is not None
        if self._watcher is not None and self._watcher.is_running():
            return
        self._watcher = WorkspaceWatcher(
            self.root,
            self._manager,
            registry=self._registry,
            debounce_secs=debounce_secs,
        )
        self._watcher.start()

    def stop_watch(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None
            if self._manager is not None:
                self._index = self._manager.index
                self._navigator = Navigator(self._index, root=self.root)

    @property
    def navigator(self) -> Navigator:
        self.ensure_indexed()
        assert self._navigator is not None
        return self._navigator

    def get_navigator(self) -> Navigator:
        return self.navigator

    def stats(self) -> IndexStats:
        return self.ensure_indexed().stats()

    @property
    def manager(self) -> IndexManager | None:
        self.ensure_indexed()
        return self._manager

    @property
    def watcher(self) -> WorkspaceWatcher | None:
        return self._watcher
