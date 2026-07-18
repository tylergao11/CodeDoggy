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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from codedoggy.graph.builder import MAX_INDEXABLE_FILE_SIZE, IndexBuilder
from codedoggy.graph.index import ScopeGraphIndex
from codedoggy.graph.languages import LanguageRegistry
from codedoggy.graph.types import FileMeta

logger = logging.getLogger(__name__)


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
        """Return current index (caller must not mutate without lock)."""
        return self.index

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
        """reindex_file: remove then extract single file (index_manager.rs)."""
        abs_path = path if path.is_absolute() else (self.root / path)
        abs_path = abs_path.resolve()
        rel = self._rel(abs_path)
        self._index.remove_file(rel)

        if not abs_path.is_file():
            return
        if not self.registry.is_supported(abs_path):
            return
        try:
            st = abs_path.stat()
        except OSError:
            return
        if st.st_size == 0 or st.st_size > MAX_INDEXABLE_FILE_SIZE:
            return
        try:
            with abs_path.open("rb") as f:
                head = f.read(8000)
                if b"\x00" in head:
                    return
                raw = head + f.read()
            source = raw.decode("utf-8", errors="replace")
        except OSError:
            return

        extracted = self.registry.extract(abs_path, source)
        self._index.add_definitions(rel, extracted.definitions)
        self._index.add_references(rel, extracted.references)
        self._index.add_aliases(rel, extracted.aliases)
        self._index.set_file_meta(rel, FileMeta.from_path(abs_path))
