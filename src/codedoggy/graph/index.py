"""ScopeGraphIndex — mirror ``scope_graph::ScopeGraphIndex`` query maps.

Source: xai-codebase-graph/src/scope_graph/graph.rs

Navigator uses definitions / references / aliases + smart ranking on this index.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from codedoggy.graph.languages import LanguageRegistry
from codedoggy.graph.types import (
    FileMeta,
    IndexStats,
    QueryVersion,
    SymbolAlias,
    SymbolOccurrence,
)


class ScopeGraphIndex:
    """Global symbol index: def/ref maps + aliases + file meta + query_version."""

    def __init__(self) -> None:
        # symbol -> list[(rel_path, line)]
        self.definitions: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.references: dict[str, list[tuple[str, int]]] = defaultdict(list)
        # alias -> original
        self.aliases: dict[str, str] = {}
        # original -> set of aliases
        self.reverse_aliases: dict[str, set[str]] = defaultdict(set)
        # rel_path -> FileMeta
        self.file_meta: dict[str, FileMeta] = {}
        self._files: set[str] = set()
        # QueryVersion — Legacy until IndexBuilder sets hash
        self.query_version: QueryVersion = QueryVersion.legacy()

    def set_query_version(self, version: int) -> None:
        """builder.rs / ScopeGraphIndex::set_query_version."""
        self.query_version = QueryVersion.version(version)

    def needs_query_rebuild(self, current_version: int) -> bool:
        """True if cache was built with different extract queries."""
        return self.query_version.needs_rebuild(current_version)

    def add_definitions(self, path: str, occs: list[SymbolOccurrence]) -> None:
        self._files.add(path)
        for o in occs:
            self.definitions[o.name].append((path, o.line))

    def add_references(self, path: str, occs: list[SymbolOccurrence]) -> None:
        self._files.add(path)
        for o in occs:
            self.references[o.name].append((path, o.line))

    def add_alias(self, alias_name: str, original_name: str, path: str = "") -> None:
        """Alias with file provenance — multi-file same alias keeps last-wins
        but reverse_aliases and per-file cleanup stay correct.
        """
        # Store as "path::alias" when path known to avoid cross-file clobber
        key = alias_name
        if path:
            # Prefer path-scoped alias map entry for reverse cleanup
            self.aliases[f"{path}::{alias_name}"] = original_name
            # Also expose bare name → original for lookup (last writer wins bare)
            self.aliases[alias_name] = original_name
        else:
            self.aliases[alias_name] = original_name
        self.reverse_aliases[original_name].add(alias_name)

    def add_aliases(self, path: str, aliases: list[SymbolAlias]) -> None:
        for a in aliases:
            self.add_alias(a.alias, a.original, path=path)

    def set_file_meta(self, path: str, meta: FileMeta) -> None:
        self.file_meta[path] = meta
        self._files.add(path)

    def remove_file(self, path: str | Path) -> None:
        """Drop all symbols and path-scoped aliases for path.

        Scans def/ref maps by name (O(names)); acceptable at current scale.
        Path-scoped aliases (``rel::alias``) drive reverse_alias cleanup.
        """
        rel = str(path).replace("\\", "/")
        candidates = {rel}
        if rel.startswith("./"):
            candidates.add(rel[2:])

        def _keep(entries: list[tuple[str, int]]) -> list[tuple[str, int]]:
            return [
                (p, ln)
                for p, ln in entries
                if p.replace("\\", "/") not in candidates
            ]

        for name in list(self.definitions.keys()):
            kept = _keep(self.definitions[name])
            if kept:
                self.definitions[name] = kept
            else:
                del self.definitions[name]
        for name in list(self.references.keys()):
            kept = _keep(self.references[name])
            if kept:
                self.references[name] = kept
            else:
                del self.references[name]
        # Clear path-scoped aliases + bare names only owned by this file
        for c in candidates:
            prefix = f"{c}::"
            for key in list(self.aliases.keys()):
                if not key.startswith(prefix):
                    continue
                bare = key.split("::", 1)[-1]
                orig = self.aliases.pop(key, None)
                still = any(
                    k.endswith(f"::{bare}") and not k.startswith(prefix)
                    for k in self.aliases
                )
                if not still:
                    if bare in self.aliases and self.aliases[bare] == orig:
                        self.aliases.pop(bare, None)
                    if orig is not None:
                        self.reverse_aliases[orig].discard(bare)
                        if not self.reverse_aliases[orig]:
                            del self.reverse_aliases[orig]
        for c in candidates:
            self.file_meta.pop(c, None)
            self._files.discard(c)

    def stats(self) -> IndexStats:
        n_defs = sum(len(v) for v in self.definitions.values())
        n_refs = sum(len(v) for v in self.references.values())
        return IndexStats(files=len(self._files), definitions=n_defs, references=n_refs)

    def find_definitions_smart(
        self,
        symbol: str,
        context_file: Path | str | None = None,
        language_registry: LanguageRegistry | None = None,
    ) -> list[tuple[str, int]]:
        """Mirror ``ScopeGraphIndex::find_definitions_smart``."""
        results: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()

        def _add(name: str) -> None:
            for path, line in self.definitions.get(name, []):
                key = (path, line)
                if key not in seen:
                    seen.add(key)
                    results.append(key)

        _add(symbol)
        # Prefer path-scoped alias when context_file known; bare is last-wins fallback
        original = self._resolve_alias(symbol, context_file)
        if original:
            _add(original)

        context_ext = None
        if context_file is not None:
            context_ext = Path(context_file).suffix.lstrip(".").lower() or None

        if context_ext and language_registry is not None:
            reg = language_registry

            def sort_key(item: tuple[str, int]) -> tuple[int, str]:
                ext = Path(item[0]).suffix.lstrip(".").lower()
                match = 0 if reg.extensions_same_language(ext, context_ext) else 1
                return (match, item[0])

            results.sort(key=sort_key)
        else:
            results.sort(key=lambda x: x[0])
        return results

    def _resolve_alias(
        self, symbol: str, context_file: Path | str | None
    ) -> str | None:
        """Resolve alias → original, preferring ``path::alias`` when context known."""
        if context_file is not None:
            ctx = str(context_file).replace("\\", "/")
            if ctx.startswith("./"):
                ctx = ctx[2:]
            scoped = self.aliases.get(f"{ctx}::{symbol}")
            if scoped:
                return scoped
        return self.aliases.get(symbol)

    def find_references_smart(
        self,
        symbol: str,
        context_file: Path | str | None = None,
        language_registry: LanguageRegistry | None = None,
    ) -> list[tuple[str, str, int]]:
        """Mirror ``find_references_smart`` → (sym_name, path, line)."""
        results: list[tuple[str, str, int]] = []
        seen: set[tuple[str, str, int]] = set()

        def _add(name: str) -> None:
            for path, line in self.references.get(name, []):
                key = (name, path, line)
                if key not in seen:
                    seen.add(key)
                    results.append(key)

        _add(symbol)
        # Aliases of this symbol also count as references to look up
        for alias in self.reverse_aliases.get(symbol, ()):
            _add(alias)
        original = self._resolve_alias(symbol, context_file)
        if original:
            _add(original)
            for alias in self.reverse_aliases.get(original, ()):
                _add(alias)

        context_ext = None
        if context_file is not None:
            context_ext = Path(context_file).suffix.lstrip(".").lower() or None

        if context_ext and language_registry is not None:
            reg = language_registry

            def sort_key(item: tuple[str, str, int]) -> tuple[int, str]:
                ext = Path(item[1]).suffix.lstrip(".").lower()
                match = 0 if reg.extensions_same_language(ext, context_ext) else 1
                return (match, item[1])

            results.sort(key=sort_key)
        else:
            results.sort(key=lambda x: x[1])
        return results
