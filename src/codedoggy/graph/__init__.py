"""Codebase graph — ported API surface of ``xai-codebase-graph``.

Quick start::

    from codedoggy.graph import IndexBuilder, Navigator, get_cache_path, load_index, save_index

    root = Path(".")
    cache = get_cache_path(root)
    try:
        index = load_index(cache)
    except Exception:
        index = IndexBuilder().build(root)
        save_index(cache, index)
    nav = Navigator(index, root=root)
    result = nav.goto_definition_by_name("MyClass")
"""

from codedoggy.graph.builder import (
    MAX_INDEXABLE_FILE_SIZE,
    FileSymbols,
    IndexBuilder,
    IndexError,
)
from codedoggy.graph.cache import (
    CACHE_FILE_NAME,
    get_cache_path,
    load_index,
    save_index,
)
from codedoggy.graph.handle import CodebaseGraph
from codedoggy.graph.index import ScopeGraphIndex
from codedoggy.graph.index_manager import FileEvent, FileEventKind, IndexManager
from codedoggy.graph.languages import LanguageRegistry
from codedoggy.graph.navigation import Navigator
from codedoggy.graph.types import (
    FileMeta,
    IndexStats,
    Location,
    NavigationError,
    NavigationResult,
    Position,
    QueryVersion,
    Range,
    SymbolAlias,
    SymbolOccurrence,
    location_to_dict,
    navigation_result_to_dict,
)
from codedoggy.graph.watcher import EventDebouncer, WorkspaceWatcher

__all__ = [
    "CACHE_FILE_NAME",
    "MAX_INDEXABLE_FILE_SIZE",
    "CodebaseGraph",
    "EventDebouncer",
    "FileEvent",
    "FileEventKind",
    "FileMeta",
    "FileSymbols",
    "IndexBuilder",
    "IndexError",
    "IndexManager",
    "IndexStats",
    "LanguageRegistry",
    "Location",
    "NavigationError",
    "NavigationResult",
    "Navigator",
    "Position",
    "QueryVersion",
    "Range",
    "ScopeGraphIndex",
    "SymbolAlias",
    "SymbolOccurrence",
    "WorkspaceWatcher",
    "get_cache_path",
    "load_index",
    "location_to_dict",
    "navigation_result_to_dict",
    "save_index",
]
