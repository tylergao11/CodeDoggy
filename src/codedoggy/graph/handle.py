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
from codedoggy.graph.cache import CACHE_FILE_NAME, get_cache_path, load_index, save_index
from codedoggy.graph.index import ScopeGraphIndex
from codedoggy.graph.index_manager import FileEvent, IndexManager
from codedoggy.graph.languages import LanguageRegistry
from codedoggy.graph.navigation import Navigator
from codedoggy.graph.types import IndexStats
from codedoggy.graph.watcher import WorkspaceWatcher

logger = logging.getLogger(__name__)


class CodebaseGraph:
    """Workspace code graph for Session.extensions.graph."""

    def __init__(
        self,
        root: Path | str,
        *,
        use_cache: bool = True,
        policy: Any = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.use_cache = use_cache
        self._policy = policy
        self._registry = LanguageRegistry()
        self._index: ScopeGraphIndex | None = None
        self._manager: IndexManager | None = None
        self._navigator: Navigator | None = None
        self._watcher: WorkspaceWatcher | None = None
        self._dirty: bool = False

    def set_policy(self, policy: Any) -> None:
        """Attach workspace policy used to gate on-disk cache writes."""
        self._policy = policy

    @property
    def policy(self) -> Any:
        return self._policy

    def _cache_writes_allowed(self, allow_write: bool | None = None) -> bool:
        """Whether save_index / persist may touch CACHE_FILE_NAME.

        * ``allow_write=True`` / ``False`` forces the decision for one call.
        * ``allow_write=None`` consults attached policy (``check_write`` /
          ``allow_writes``); no policy ⇒ allow (backward compatible).
        """
        if allow_write is False:
            return False
        if allow_write is True:
            return True
        policy = self._policy
        if policy is None:
            return True
        check_w = getattr(policy, "check_write", None)
        if callable(check_w):
            wd = check_w(CACHE_FILE_NAME)
            if wd is not None and not getattr(wd, "allowed", True):
                return False
            return True
        if hasattr(policy, "allow_writes") and not bool(
            getattr(policy, "allow_writes", True)
        ):
            return False
        return True

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
        """Full manifest check: every cached file + set equality vs disk scan."""
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
        # All cached files must exist and match mtime/size
        for rel, meta in index.file_meta.items():
            if meta.is_stale(self.root / rel):
                return False
        # Disk set of supported files must match cache keys (add/delete detection)
        try:
            from codedoggy.graph.builder import IndexBuilder

            on_disk = {
                p.resolve().relative_to(self.root).as_posix()
                for p in IndexBuilder(registry=self._registry)._collect_files(self.root)
            }
            cached = {str(k).replace("\\", "/") for k in index.file_meta.keys()}
            if on_disk != cached:
                logger.info(
                    "graph cache file set mismatch disk=%s cache=%s, rebuild",
                    len(on_disk),
                    len(cached),
                )
                return False
        except Exception:  # noqa: BLE001
            logger.debug("graph cache disk scan failed", exc_info=True)
            return False
        return True

    def persist_if_dirty(self, *, allow_write: bool | None = None) -> None:
        """Save index after incremental updates (session close / watcher).

        In-memory index stays current either way; when policy denies cache
        writes (or ``allow_write=False``), skip disk and leave ``_dirty`` set.
        """
        if self._index is None or not self.use_cache or not self._dirty:
            return
        if not self._cache_writes_allowed(allow_write):
            logger.info(
                "graph persist skipped (cache write denied for %s)",
                CACHE_FILE_NAME,
            )
            return
        try:
            save_index(get_cache_path(self.root), self._index)
            self._dirty = False
        except Exception as e:  # noqa: BLE001
            logger.warning("graph persist failed: %s", e)

    def mark_dirty(self, path: str | None = None) -> None:
        self.ensure_indexed()
        assert self._manager is not None
        if path:
            self._manager.send_event(FileEvent.modified(path))
            self._index = self._manager.index
            self._navigator = Navigator(self._index, root=self.root)
            self._dirty = True
        else:
            self.reindex()

    def reindex(self, *, allow_write: bool | None = None) -> dict[str, Any]:
        """Full rebuild of the in-memory index.

        Disk cache (``.goto_index.json``) is written only when ``use_cache``
        and cache writes are allowed (explicit ``allow_write`` or policy).
        Tool path is additionally gated in ``code_nav``; this method still
        rebuilds memory when writes are denied (fail-open for queries).
        """
        index = IndexBuilder(registry=self._registry).build(self.root)
        self._bind_index(index)
        self._dirty = False
        # Re-point watcher at the new manager (do not leave events on dead index)
        if self._watcher is not None and self._manager is not None:
            self._watcher.set_manager(self._manager)
        if self.use_cache:
            if self._cache_writes_allowed(allow_write):
                try:
                    save_index(get_cache_path(self.root), index)
                except Exception as e:  # noqa: BLE001
                    logger.warning("graph cache save failed: %s", e)
            else:
                logger.info(
                    "graph cache save skipped after reindex (write denied for %s)",
                    CACHE_FILE_NAME,
                )
        s = index.stats()
        return {
            "files": s.files,
            "definitions": s.definitions,
            "references": s.references,
            "query_version": index.query_version.to_wire(),
        }

    def send_event(self, event: FileEvent) -> None:
        """Apply an incremental FileEvent and mark cache dirty for persist."""
        self.ensure_indexed()
        assert self._manager is not None
        self._manager.send_event(event)
        self._index = self._manager.index
        self._navigator = Navigator(self._index, root=self.root)
        self._dirty = True

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
            on_events_applied=self._on_watcher_events,
        )
        self._watcher.start()

    def _on_watcher_events(self) -> None:
        """Watcher applied incremental events — keep handle in sync + dirty for persist."""
        if self._manager is not None:
            self._index = self._manager.index
            self._navigator = Navigator(self._index, root=self.root)
        self._dirty = True

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
