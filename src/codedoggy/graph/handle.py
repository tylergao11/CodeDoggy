"""Session-facing lease over the process-wide codebase graph runtime.

``CodebaseGraph`` preserves the original Session/extensions API while sharing
one single-owner runtime per canonical workspace.  It is intentionally a thin
lease: extraction and incremental indexing remain owned by ``IndexBuilder`` and
``IndexManager``.
"""

from __future__ import annotations

import logging
import weakref
from pathlib import Path
from typing import Any

from codedoggy.graph.cache import CACHE_FILE_NAME
from codedoggy.graph.index import ScopeGraphIndex
from codedoggy.graph.index_manager import FileEvent, IndexManager
from codedoggy.graph.navigation import Navigator
from codedoggy.graph.runtime import (
    WorkspaceGraphRuntime,
    get_codebase_index_manager,
)
from codedoggy.graph.types import IndexStats
from codedoggy.graph.watcher import WorkspaceWatcher

logger = logging.getLogger(__name__)


def _release_graph_finalizer(
    runtime_ref: weakref.ReferenceType[WorkspaceGraphRuntime],
    lease: object,
) -> None:
    """Best-effort leak shield when a host drops a Session without close()."""
    runtime = runtime_ref()
    if runtime is None:
        return
    try:
        runtime.release_watch(lease)
    except Exception:  # noqa: BLE001
        logger.debug("graph watch finalizer failed", exc_info=True)
    try:
        runtime.release_handle(lease)
    except Exception:  # noqa: BLE001
        logger.debug("graph handle finalizer failed", exc_info=True)


class CodebaseGraph:
    """Session graph handle backed by one shared workspace runtime.

    Constructing a handle is cheap and does not build the index.  ``start_watch``
    is also lazy with respect to indexing: it starts OS observation immediately,
    and mutations queue on the runtime actor until the first query/reindex.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        use_cache: bool = True,
        policy: Any = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.use_cache = use_cache
        self._policy = policy
        self._runtime, self._runtime_was_new = (
            get_codebase_index_manager().get_or_create(self.root)
        )
        self._registry = self._runtime.registry
        self._lease_token = object()
        try:
            self._runtime.acquire_handle(self._lease_token)
        except RuntimeError:
            # The last prior Session may have closed between registry lookup
            # and lease acquisition.  Resolve the fresh generation once.
            self._runtime, self._runtime_was_new = (
                get_codebase_index_manager().get_or_create(self.root)
            )
            self._registry = self._runtime.registry
            self._runtime.acquire_handle(self._lease_token)
        self._finalizer = weakref.finalize(
            self,
            _release_graph_finalizer,
            weakref.ref(self._runtime),
            self._lease_token,
        )

        # Compatibility mirrors for callers that inspected the old handle.
        # Runtime truth remains authoritative and is refreshed at API fences.
        self._index: ScopeGraphIndex | None = None
        self._manager: IndexManager | None = self._runtime.manager_if_ready
        self._navigator: Navigator | None = None
        self._watcher: WorkspaceWatcher | None = self._runtime.watcher
        self._dirty = self._runtime.is_dirty
        self._watch_acquired = False
        self._closed = False

    @property
    def runtime(self) -> WorkspaceGraphRuntime:
        """Shared runtime/lease target, exposed for host integration probes."""
        return self._runtime

    @property
    def runtime_was_new(self) -> bool:
        """Whether this Session created the process workspace runtime."""
        return self._runtime_was_new

    def set_policy(self, policy: Any) -> None:
        """Attach the Session policy used to gate cache persistence."""
        self._policy = policy

    @property
    def policy(self) -> Any:
        return self._policy

    def _cache_writes_allowed(self, allow_write: bool | None = None) -> bool:
        """Preserve the previous explicit/policy cache-write contract.

        The cache now lives under ``CODEDOGGY_HOME`` rather than the workspace,
        but callers that explicitly disable writes still receive a no-write
        execution path.
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
            decision = check_w(CACHE_FILE_NAME)
            if decision is not None and not getattr(decision, "allowed", True):
                return False
            return True
        if hasattr(policy, "allow_writes"):
            return bool(getattr(policy, "allow_writes", True))
        return True

    def _ensure_manager(self) -> IndexManager:
        if self._closed:
            raise RuntimeError("CodebaseGraph is closed")
        manager = self._runtime.ensure_indexed(
            use_cache=self.use_cache,
            allow_cache_write=self._cache_writes_allowed(),
        )
        self._sync_refs(manager)
        return manager

    def _sync_refs(self, manager: IndexManager | None = None) -> None:
        manager = manager or self._runtime.manager_if_ready
        self._manager = manager
        self._watcher = self._runtime.watcher
        self._dirty = self._runtime.is_dirty
        if manager is None:
            return
        self._index = manager.index
        if self._navigator is None or self._navigator._manager is not manager:
            self._navigator = Navigator(
                manager.index,
                manager=manager,
                before_read=self._runtime.fence,
                registry=self._registry,
                root=self.root,
            )

    def ensure_indexed(self) -> ScopeGraphIndex:
        """Fence prior actor commands and return an isolated index snapshot."""
        manager = self._ensure_manager()
        if self.use_cache:
            self._runtime.repair_cache_if_invalid(
                allow_cache_write=self._cache_writes_allowed(),
            )
        snapshot = manager.get_snapshot()
        self._index = snapshot
        return snapshot

    def _cache_fresh(self, index: ScopeGraphIndex) -> bool:
        """Compatibility delegate for the previous private helper."""
        return self._runtime.cache_fresh(index)

    def persist_if_dirty(self, *, allow_write: bool | None = None) -> None:
        """Fence prior events and atomically persist the shared index if dirty."""
        allowed = self._cache_writes_allowed(allow_write)
        if not allowed:
            logger.info("graph cache persist skipped by policy")
            return
        try:
            self._runtime.persist(
                use_cache=self.use_cache,
                allow_cache_write=allowed,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("graph persist failed: %s", exc)
        finally:
            self._sync_refs()

    def mark_dirty(self, path: str | None = None) -> None:
        """Enqueue mutation without synchronously building the repository.

        A known path becomes an incremental ``Modified`` event.  An unknown
        mutation invalidates the runtime so the next query performs one full
        rebuild.  Both operations return after actor submission.
        """
        if self._closed:
            raise RuntimeError("CodebaseGraph is closed")
        if path:
            self._runtime.send_event(FileEvent.modified(path))
        else:
            self._runtime.request_rebuild()
        self._dirty = True

    def reindex(self, *, allow_write: bool | None = None) -> dict[str, Any]:
        """Explicit full rebuild shared by every Session on this workspace."""
        if self._closed:
            raise RuntimeError("CodebaseGraph is closed")
        allowed = self._cache_writes_allowed(allow_write)
        stats = self._runtime.rebuild(
            use_cache=self.use_cache,
            allow_cache_write=allowed,
        )
        self._sync_refs()
        return stats

    def send_event(self, event: FileEvent) -> None:
        """Submit an incremental event; a subsequent query is the read fence."""
        if self._closed:
            raise RuntimeError("CodebaseGraph is closed")
        self._runtime.send_event(event)
        self._dirty = True

    def start_watch(self, *, debounce_secs: float = 0.35) -> None:
        """Acquire this Session's shared watcher lease without building index."""
        if self._closed:
            raise RuntimeError("CodebaseGraph is closed")
        self._watcher = self._runtime.acquire_watch(
            self._lease_token,
            debounce_secs=debounce_secs,
        )
        self._watch_acquired = True

    def _on_watcher_events(self) -> None:
        """Compatibility hook; the shared runtime now owns watcher delivery."""
        self._dirty = True
        self._sync_refs()

    def stop_watch(self) -> None:
        """Release only this Session lease; final lease stops and flushes watch."""
        if not self._watch_acquired:
            self._sync_refs()
            return
        try:
            self._runtime.release_watch(self._lease_token)
            self._watch_acquired = False
            # watcher.stop() flushes into the actor queue.  Persist again after
            # that fence because RuntimeKernel currently calls persist *before*
            # stop_watch during teardown.
            self._runtime.persist(
                use_cache=self.use_cache,
                allow_cache_write=self._cache_writes_allowed(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("graph watch release failed: %s", exc)
        finally:
            self._sync_refs()

    def close(self) -> None:
        """Persist and release this Session lease; safe to call repeatedly."""
        if self._closed:
            return
        self.persist_if_dirty()
        self.stop_watch()
        self._closed = True
        self._finalizer.detach()
        self._runtime.release_handle(self._lease_token)

    @property
    def navigator(self) -> Navigator:
        self._ensure_manager()
        assert self._navigator is not None
        return self._navigator

    def get_navigator(self) -> Navigator:
        return self.navigator

    def stats(self) -> IndexStats:
        if self._closed:
            raise RuntimeError("CodebaseGraph is closed")
        stats = self._runtime.stats(
            use_cache=self.use_cache,
            allow_cache_write=self._cache_writes_allowed(),
        )
        self._sync_refs()
        return stats

    @property
    def manager(self) -> IndexManager | None:
        return self._ensure_manager()

    @property
    def watcher(self) -> WorkspaceWatcher | None:
        self._watcher = self._runtime.watcher
        return self._watcher

    @property
    def watch_lease_count(self) -> int:
        return self._runtime.watch_lease_count
