"""Hermes-style memory: curated notes + session FTS + MemoryManager."""

from codedoggy.memory.defaults import (
    ENTRY_DELIMITER,
    MEMORY_CHAR_LIMIT,
    USER_CHAR_LIMIT,
)
from codedoggy.memory.hermes_select import HermesMemorySelector
from codedoggy.memory.manager import MemoryManager
from codedoggy.memory.paths import default_memory_home, get_memory_dir
from codedoggy.memory.provider import (
    BaseMemoryProvider,
    CuratedMemoryProvider,
    SessionFtsProvider,
)
from codedoggy.memory.session_store import SessionStore, default_session_db_path
from codedoggy.memory.store import MemoryStore

__all__ = [
    "BaseMemoryProvider",
    "CuratedMemoryProvider",
    "ENTRY_DELIMITER",
    "HermesMemorySelector",
    "MEMORY_CHAR_LIMIT",
    "MemoryManager",
    "MemoryStore",
    "SessionFtsProvider",
    "SessionStore",
    "USER_CHAR_LIMIT",
    "default_memory_home",
    "default_session_db_path",
    "get_memory_dir",
]
