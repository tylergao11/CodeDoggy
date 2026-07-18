"""Language registry — languages/mod.rs LanguageRegistry.

Symbol extract: tree-sitter queries only (xai-codebase-graph path).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from codedoggy.graph.languages.extract import (
    FileExtract,
    extract_symbols,
    identifier_at_position,
    query_fingerprint,
)
from codedoggy.graph.languages.queries import QUERIES_BY_LANG


class LanguageRegistry:
    """Extension / id lookup, extract, same-language ranking, query hash."""

    def __init__(self) -> None:
        self._ext_to_lang: dict[str, str] = {
            "py": "python",
            "pyi": "python",
            "js": "javascript",
            "jsx": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
            "rs": "rust",
            "go": "golang",
        }
        self._lang_exts: dict[str, set[str]] = {
            "python": {"py", "pyi"},
            "javascript": {"js", "jsx"},
            "typescript": {"ts", "tsx", "js", "jsx"},
            "rust": {"rs"},
            "golang": {"go"},
        }

    def is_supported(self, path: str | Path) -> bool:
        ext = Path(path).suffix.lstrip(".").lower()
        return ext in self._ext_to_lang

    def language_for(self, path: str | Path) -> str | None:
        ext = Path(path).suffix.lstrip(".").lower()
        return self._ext_to_lang.get(ext)

    def for_file_path(self, path: str | Path) -> str | None:
        return self.language_for(path)

    def extensions_same_language(self, ext1: str, ext2: str) -> bool:
        e1, e2 = ext1.lower().lstrip("."), ext2.lower().lstrip(".")
        if e1 == e2:
            return True
        for group in self._lang_exts.values():
            if e1 in group and e2 in group:
                return True
        return False

    def extract(self, path: str | Path, source: str) -> FileExtract:
        lang = self.language_for(path)
        if lang is None:
            return FileExtract()
        return extract_symbols(lang, source)

    def identifier_at(
        self, path: str | Path, source: str, row: int, col: int
    ) -> str | None:
        return identifier_at_position(source, row, col)

    def supported_extensions(self) -> list[str]:
        return sorted(self._ext_to_lang.keys())

    def compute_query_hash(self) -> int:
        """languages/mod.rs compute_query_hash — u64 for cache invalidation."""
        parts: list[str] = [query_fingerprint()]
        for lang_id in sorted(QUERIES_BY_LANG.keys()):
            parts.append(lang_id)
            parts.append(QUERIES_BY_LANG[lang_id])
        digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little", signed=False)


__all__ = [
    "FileExtract",
    "LanguageRegistry",
    "identifier_at_position",
]
