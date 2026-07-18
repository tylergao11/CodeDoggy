"""Tree-sitter symbol extract — builder.rs extract_symbols_fast_inline.

Hard dependency: tree-sitter + language packs (declared in pyproject).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from tree_sitter import Language, Parser, Query, QueryCursor

from codedoggy.graph.languages.queries import QUERIES_BY_LANG
from codedoggy.graph.types import SymbolAlias, SymbolOccurrence

logger = logging.getLogger(__name__)

_tls = threading.local()

# Import grammars at module load — fail install/misconfig immediately.
import tree_sitter_go as _ts_go  # noqa: E402
import tree_sitter_javascript as _ts_js  # noqa: E402
import tree_sitter_python as _ts_py  # noqa: E402
import tree_sitter_rust as _ts_rust  # noqa: E402
import tree_sitter_typescript as _ts_ts  # noqa: E402

_LANGUAGES: dict[str, Language] = {
    "python": Language(_ts_py.language()),
    "javascript": Language(_ts_js.language()),
    "typescript": Language(_ts_ts.language_tsx()),
    "rust": Language(_ts_rust.language()),
    "golang": Language(_ts_go.language()),
}


@dataclass
class FileExtract:
    """Per-file extract merged into ScopeGraphIndex (FileSymbols spirit)."""

    definitions: list[SymbolOccurrence] = field(default_factory=list)
    references: list[SymbolOccurrence] = field(default_factory=list)
    aliases: list[SymbolAlias] = field(default_factory=list)


def _parser_for(lang_id: str) -> Parser:
    cache = getattr(_tls, "parsers", None)
    if cache is None:
        cache = {}
        _tls.parsers = cache
    if lang_id not in cache:
        language = _LANGUAGES[lang_id]
        cache[lang_id] = Parser(language)
    return cache[lang_id]


def _query_for(lang_id: str) -> Query:
    cache = getattr(_tls, "queries", None)
    if cache is None:
        cache = {}
        _tls.queries = cache
    if lang_id not in cache:
        cache[lang_id] = Query(_LANGUAGES[lang_id], QUERIES_BY_LANG[lang_id])
    return cache[lang_id]


def _kind_from_capture(name: str, prefix: str) -> str:
    rest = name[len(prefix) :] if name.startswith(prefix) else name
    return rest or "symbol"


def extract_symbols(lang_id: str, source: str) -> FileExtract:
    """Extract definitions / references / aliases for one language id."""
    if lang_id not in _LANGUAGES:
        return FileExtract()
    parser = _parser_for(lang_id)
    query = _query_for(lang_id)
    raw = source.encode("utf-8")
    try:
        tree = parser.parse(raw)
    except Exception as e:  # noqa: BLE001
        logger.debug("tree-sitter parse failed %s: %s", lang_id, e)
        return FileExtract()
    return _extract_symbols(query, tree.root_node, raw)


def _extract_symbols(query: Query, root_node: Any, src: bytes) -> FileExtract:
    out = FileExtract()
    cursor = QueryCursor(query)
    try:
        matches = cursor.matches(root_node)
    except Exception as e:  # noqa: BLE001
        logger.debug("tree-sitter query matches failed: %s", e)
        return out

    for _pattern_idx, cap_map in matches:
        alias_original: str | None = None
        alias_name: str | None = None
        for cap_name, nodes in cap_map.items():
            if not nodes:
                continue
            if cap_name.startswith("name.definition."):
                kind = _kind_from_capture(cap_name, "name.definition.")
                for node in nodes:
                    text = src[node.start_byte : node.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    out.definitions.append(
                        SymbolOccurrence(
                            name=text, line=node.start_point[0] + 1, kind=kind
                        )
                    )
            elif cap_name.startswith("name.reference."):
                kind = _kind_from_capture(cap_name, "name.reference.")
                for node in nodes:
                    text = src[node.start_byte : node.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    out.references.append(
                        SymbolOccurrence(
                            name=text, line=node.start_point[0] + 1, kind=kind
                        )
                    )
            elif cap_name == "alias.original":
                node = nodes[0]
                alias_original = src[node.start_byte : node.end_byte].decode(
                    "utf-8", errors="replace"
                )
                if (
                    len(alias_original) >= 2
                    and alias_original[0] == '"'
                    and alias_original[-1] == '"'
                ):
                    alias_original = alias_original[1:-1]
            elif cap_name == "alias.name":
                node = nodes[0]
                alias_name = src[node.start_byte : node.end_byte].decode(
                    "utf-8", errors="replace"
                )
        if alias_original and alias_name:
            out.aliases.append(SymbolAlias(alias=alias_name, original=alias_original))
    return out


def query_fingerprint() -> str:
    """Stable fingerprint of all query strings (for cache invalidation)."""
    import hashlib

    parts = ["ts-v1"]
    for lid in sorted(QUERIES_BY_LANG.keys()):
        q = QUERIES_BY_LANG[lid]
        parts.append(f"{lid}:{hashlib.sha256(q.encode()).hexdigest()[:16]}")
    return "|".join(parts)


def identifier_at_position(source: str, row: int, col: int) -> str | None:
    """Word under 1-indexed (row, col) — for at_position navigation."""
    if row < 1 or col < 1:
        return None
    lines = source.splitlines()
    if row > len(lines):
        return None
    line = lines[row - 1]
    i = col - 1
    if i >= len(line):
        i = len(line) - 1
    if i < 0:
        return None
    if not (line[i].isalnum() or line[i] == "_"):
        if i > 0 and (line[i - 1].isalnum() or line[i - 1] == "_"):
            i -= 1
        else:
            return None
    start = i
    while start > 0 and (line[start - 1].isalnum() or line[start - 1] == "_"):
        start -= 1
    end = i + 1
    while end < len(line) and (line[end].isalnum() or line[end] == "_"):
        end += 1
    name = line[start:end]
    return name if name and not name[0].isdigit() else None
