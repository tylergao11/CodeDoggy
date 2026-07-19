"""IndexManager — incremental reindex from FileEvents.

Source: xai-codebase-graph/src/index_manager.rs

Architecture (same idea, sync process loop instead of tokio channel):
  FileEvent(Created|Modified|Removed|Renamed)
    → remove_file(old) if needed
    → reindex_file(path)
  Rebuild → IndexBuilder.build

Owns the ScopeGraphIndex; Navigator should use get_snapshot / current index.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, TypeVar

from codedoggy.graph.builder import MAX_INDEXABLE_FILE_SIZE, IndexBuilder
from codedoggy.graph.index import ScopeGraphIndex
from codedoggy.graph.languages import LanguageRegistry
from codedoggy.graph.types import FileMeta, IndexStats

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class FileEventKind(str, Enum):
    Created = "Created"
    Modified = "Modified"
    Removed = "Removed"
    Renamed = "Renamed"


@dataclass
class FileEvent:
    paths: list[Path]
    kind: FileEventKind

    @classmethod
    def created(cls, path: Path | str) -> FileEvent:
        return cls([Path(path)], FileEventKind.Created)

    @classmethod
    def modified(cls, path: Path | str) -> FileEvent:
        return cls([Path(path)], FileEventKind.Modified)

    @classmethod
    def removed(cls, path: Path | str) -> FileEvent:
        return cls([Path(path)], FileEventKind.Removed)

    @classmethod
    def renamed(cls, old: Path | str, new: Path | str) -> FileEvent:
        return cls([Path(old), Path(new)], FileEventKind.Renamed)


class IndexManager:
    """Process FileEvents against one workspace index (sequential mutations)."""

    def __init__(
        self,
        root: Path | str,
        *,
        index: ScopeGraphIndex | None = None,
        registry: LanguageRegistry | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.registry = registry or LanguageRegistry()
        self._lock = threading.RLock()
        self._index = index if index is not None else ScopeGraphIndex()
        self._builder = IndexBuilder(registry=self.registry)

    @property
    def index(self) -> ScopeGraphIndex:
        with self._lock:
            return self._index

    def get_snapshot(self) -> ScopeGraphIndex:
        """Return an isolated snapshot of the current index.

        Grok returns an ``Arc`` snapshot and uses copy-on-write for subsequent
        mutations.  Python has no equivalent cheap COW container here, so take
        the copy while holding the owner lock.  This path is reserved for cache
        persistence and the compatibility ``ensure_indexed`` API; navigation
        uses :meth:`read_index` and does not clone the repository index.
        """
        import copy

        with self._lock:
            return copy.deepcopy(self._index)

    def read_index(self, reader: Callable[[ScopeGraphIndex], _T]) -> _T:
        """Run a lightweight query while the owned index is stable."""
        with self._lock:
            return reader(self._index)

    def stats(self) -> IndexStats:
        """Return stats without cloning the whole index."""
        with self._lock:
            return self._index.stats()

    def replace_index(self, index: ScopeGraphIndex) -> ScopeGraphIndex:
        """Atomically replace the owned index while keeping this handle stable."""
        with self._lock:
            self._index = index
            return self._index

    def rebuild(self) -> ScopeGraphIndex:
        with self._lock:
            self._index = self._builder.build(self.root)
            return self._index

    def send_event(self, event: FileEvent) -> None:
        """index_manager send_event — process immediately.

        Debouncing is external (EventDebouncer / WorkspaceWatcher), matching
        index_manager.rs: events arriving here are already debounced.
        """
        self.apply_event(event)

    def send_events(self, events: list[FileEvent]) -> None:
        """Batch of already-debounced events (coalesced by EventDebouncer)."""
        for e in events:
            self.apply_event(e)

    def apply_event(self, event: FileEvent) -> None:
        with self._lock:
            kind = event.kind
            if kind in (FileEventKind.Created, FileEventKind.Modified):
                for p in event.paths:
                    self._reindex_file(p)
            elif kind is FileEventKind.Removed:
                for p in event.paths:
                    self._remove_file(p)
            elif kind is FileEventKind.Renamed:
                if len(event.paths) >= 1:
                    self._remove_file(event.paths[0])
                if len(event.paths) >= 2:
                    self._reindex_file(event.paths[1])

    def _rel(self, path: Path) -> str:
        p = path if path.is_absolute() else (self.root / path)
        try:
            return p.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix().replace("\\", "/")

    def _remove_file(self, path: Path) -> None:
        rel = self._rel(path)
        self._index.remove_file(rel)

    def _reindex_file(self, path: Path) -> None:
        """reindex_file: extract first, then swap — never leave a hole on extract fail."""
        abs_path = path if path.is_absolute() else (self.root / path)
        abs_path = abs_path.resolve()
        rel = self._rel(abs_path)

        if not abs_path.is_file():
            # File gone — drop index entries only
            self._index.remove_file(rel)
            return
        if not self.registry.is_supported(abs_path):
            self._index.remove_file(rel)
            return
        try:
            st = abs_path.stat()
        except OSError:
            return
        if st.st_size == 0 or st.st_size > MAX_INDEXABLE_FILE_SIZE:
            self._index.remove_file(rel)
            return
        try:
            with abs_path.open("rb") as f:
                head = f.read(8000)
                if b"\x00" in head:
                    return  # keep prior index; binary/corrupt skip without wipe
                raw = head + f.read()
            source = raw.decode("utf-8", errors="replace")
            extracted = self.registry.extract(abs_path, source)
        except OSError:
            return
        except Exception:  # noqa: BLE001
            # Extract failed — leave previous symbols for this path intact
            logger.debug("reindex extract failed path=%s", rel, exc_info=True)
            return

        # Successful extract → atomic-ish swap
        self._index.remove_file(rel)
        self._index.add_definitions(rel, extracted.definitions)
        self._index.add_references(rel, extracted.references)
        self._index.add_aliases(rel, extracted.aliases)
        self._index.set_file_meta(rel, FileMeta.from_path(abs_path))
