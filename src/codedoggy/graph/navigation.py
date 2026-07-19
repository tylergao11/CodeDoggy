"""Navigator — mirror ``xai-codebase-graph::navigation::Navigator``.

APIs (1-indexed row/col):
  - get_symbol_at_position
  - goto_definition / goto_references
  - goto_definition_by_name / goto_references_by_name
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from codedoggy.graph.index import ScopeGraphIndex
from codedoggy.graph.index_manager import IndexManager
from codedoggy.graph.languages import LanguageRegistry
from codedoggy.graph.types import Location, NavigationError, NavigationResult

_T = TypeVar("_T")


class Navigator:
    """Location-based code navigation over a ScopeGraphIndex."""

    def __init__(
        self,
        index: ScopeGraphIndex,
        *,
        manager: IndexManager | None = None,
        before_read: Callable[[], None] | None = None,
        registry: LanguageRegistry | None = None,
        root: Path | str | None = None,
    ) -> None:
        self.index = index
        self._manager = manager
        self._before_read = before_read
        self.registry = registry or LanguageRegistry()
        self.root = Path(root).resolve() if root is not None else None

    def _read_index(self, reader: Callable[[ScopeGraphIndex], _T]) -> _T:
        """Query the shared manager under its read lock when one is attached."""
        if self._before_read is not None:
            self._before_read()
        if self._manager is not None:
            return self._manager.read_index(reader)
        return reader(self.index)

    def get_symbol_at_position(
        self, file_path: Path | str, row: int, col: int
    ) -> str:
        if row == 0 or col == 0:
            raise NavigationError.position_out_of_bounds(row, col)
        abs_path = self._resolve_read_path(file_path)
        if not abs_path.is_file():
            raise NavigationError.file_not_found(abs_path)
        if not self.registry.is_supported(abs_path):
            ext = abs_path.suffix.lstrip(".") or "unknown"
            raise NavigationError.unsupported_language(ext)
        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise NavigationError.file_not_found(abs_path) from e
        name = self.registry.identifier_at(abs_path, source, row, col)
        if not name:
            raise NavigationError.no_symbol_at_position(row, col)
        return name

    def goto_definition(
        self, file_path: Path | str, row: int, col: int
    ) -> NavigationResult:
        symbol = self.get_symbol_at_position(file_path, row, col)
        return self.goto_definition_by_name(symbol, context_file=file_path)

    def goto_references(
        self,
        file_path: Path | str,
        row: int,
        col: int,
        include_definition: bool = True,
    ) -> NavigationResult:
        symbol = self.get_symbol_at_position(file_path, row, col)
        return self.goto_references_by_name(
            symbol, context_file=file_path, include_definition=include_definition
        )

    def goto_definition_by_name(
        self,
        symbol: str,
        context_file: Path | str | None = None,
    ) -> NavigationResult:
        defs = self._read_index(
            lambda index: index.find_definitions_smart(
                symbol, context_file, self.registry
            )
        )
        locations = [Location(path=p, line=line) for p, line in defs]
        return NavigationResult(symbol=symbol, locations=locations)

    def goto_references_by_name(
        self,
        symbol: str,
        context_file: Path | str | None = None,
        include_definition: bool = True,
    ) -> NavigationResult:
        def _query(
            index: ScopeGraphIndex,
        ) -> tuple[list[tuple[str, str, int]], list[tuple[str, int]]]:
            refs = index.find_references_smart(
                symbol, context_file, self.registry
            )
            defs = (
                index.find_definitions_smart(symbol, context_file, self.registry)
                if include_definition
                else []
            )
            return refs, defs

        refs, defs = self._read_index(_query)
        locations = [
            Location(path=p, line=line, symbol=sym) for sym, p, line in refs
        ]
        if include_definition:
            for p, line in defs:
                if not any(l.path == p and l.line == line for l in locations):
                    locations.insert(0, Location(path=p, line=line))
        return NavigationResult(symbol=symbol, locations=locations)

    def _resolve_read_path(self, file_path: Path | str) -> Path:
        p = Path(file_path)
        if p.is_absolute():
            return p
        if self.root is not None:
            return (self.root / p).resolve()
        return p.resolve()
