"""Core types — mirror ``xai-codebase-graph`` types + navigation Location.

Source: crates/codegen/xai-codebase-graph/src/{types,navigation}.rs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class QueryVersion:
    """scope_graph::QueryVersion — invalidate cache when extract queries change.

    Legacy = pre-version caches (always rebuild).
    Version(hash) = LanguageRegistry.compute_query_hash() at build time.
    """

    __slots__ = ("_kind", "hash")

    def __init__(self, kind: str, hash: int | None = None) -> None:
        self._kind = kind  # "legacy" | "version"
        self.hash = hash

    @classmethod
    def legacy(cls) -> QueryVersion:
        return cls("legacy", None)

    @classmethod
    def version(cls, h: int) -> QueryVersion:
        return cls("version", int(h) & 0xFFFFFFFFFFFFFFFF)

    @property
    def is_legacy(self) -> bool:
        return self._kind == "legacy"

    def needs_rebuild(self, current_version: int) -> bool:
        if self._kind == "legacy" or self.hash is None:
            return True
        return self.hash != (int(current_version) & 0xFFFFFFFFFFFFFFFF)

    def to_wire(self) -> dict[str, Any]:
        if self._kind == "legacy" or self.hash is None:
            return {"kind": "legacy"}
        return {"kind": "version", "hash": self.hash}

    @classmethod
    def from_wire(cls, raw: Any) -> QueryVersion:
        if raw is None:
            return cls.legacy()
        if isinstance(raw, int):
            return cls.version(raw)
        if isinstance(raw, dict):
            if raw.get("kind") == "version" and raw.get("hash") is not None:
                return cls.version(int(raw["hash"]))
            return cls.legacy()
        return cls.legacy()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QueryVersion):
            return NotImplemented
        return self._kind == other._kind and self.hash == other.hash

    def __repr__(self) -> str:
        if self._kind == "legacy":
            return "QueryVersion.Legacy"
        return f"QueryVersion.Version({self.hash})"


@dataclass(slots=True)
class Position:
    """0-indexed row/col (tree-sitter style). Navigation API uses 1-indexed."""

    row: int
    col: int


@dataclass(slots=True)
class Range:
    """Byte + line range for a symbol occurrence."""

    start_byte: int
    end_byte: int
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed

    def contains_line(self, line: int) -> bool:
        return self.start_line <= line <= self.end_line


@dataclass(slots=True)
class Location:
    """Navigation hit — mirror ``navigation::Location``.

    path is relative to repo root when from index; line is 1-indexed.
    """

    path: str
    line: int
    symbol: str | None = None

    def as_path(self) -> Path:
        return Path(self.path)


@dataclass(slots=True)
class NavigationResult:
    """Mirror ``navigation::NavigationResult``."""

    symbol: str
    locations: list[Location]


@dataclass(slots=True)
class SymbolOccurrence:
    """Mirror ``types::SymbolOccurrence`` — name + 1-indexed line."""

    name: str
    line: int
    kind: str = ""  # function | class | variable | module | call | import


@dataclass(slots=True)
class SymbolAlias:
    """Mirror ``types::SymbolAlias`` — alias → original."""

    alias: str
    original: str


@dataclass(slots=True)
class FileMeta:
    """Mirror ``types::FileMeta`` — staleness via size + mtime."""

    size: int
    mtime_secs: float

    @classmethod
    def from_path(cls, path: Path) -> FileMeta:
        st = path.stat()
        return cls(size=int(st.st_size), mtime_secs=float(st.st_mtime))

    def is_stale(self, path: Path) -> bool:
        try:
            cur = FileMeta.from_path(path)
        except OSError:
            return True
        return cur.size != self.size or abs(cur.mtime_secs - self.mtime_secs) > 1e-6


@dataclass(slots=True)
class IndexStats:
    """Mirror ``types::IndexStats``."""

    files: int
    definitions: int
    references: int


class NavigationError(Exception):
    """Mirror ``navigation::NavigationError``."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind

    @classmethod
    def file_not_found(cls, path: Path | str) -> NavigationError:
        return cls("file_not_found", f"File not found: {path}")

    @classmethod
    def position_out_of_bounds(cls, row: int, col: int) -> NavigationError:
        return cls("position_out_of_bounds", f"Position out of bounds: {row}:{col}")

    @classmethod
    def no_symbol_at_position(cls, row: int, col: int) -> NavigationError:
        return cls("no_symbol_at_position", f"No symbol found at position {row}:{col}")

    @classmethod
    def unsupported_language(cls, ext: str) -> NavigationError:
        return cls("unsupported_language", f"Unsupported language: {ext}")

    @classmethod
    def parse_error(cls, msg: str) -> NavigationError:
        return cls("parse_error", f"Parse error: {msg}")


def location_to_dict(loc: Location) -> dict[str, Any]:
    d: dict[str, Any] = {"path": loc.path, "line": loc.line}
    if loc.symbol:
        d["symbol"] = loc.symbol
    return d


def navigation_result_to_dict(result: NavigationResult) -> dict[str, Any]:
    return {
        "symbol": result.symbol,
        "locations": [location_to_dict(l) for l in result.locations],
    }
