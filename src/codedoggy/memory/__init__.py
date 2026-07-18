"""Hermes memory adapter (ported surfaces from hermes-agent).

Lifecycle against Grok runtime goes through ``hermes_seam`` — not ad-hoc
manager calls from runner/kernel/compactor.
"""

from codedoggy.memory.context_fence import (
    build_memory_context_block,
    messages_with_ephemeral_memory,
    sanitize_context,
)
from codedoggy.memory.defaults import (
    ENTRY_DELIMITER,
    MEMORY_CHAR_LIMIT,
    USER_CHAR_LIMIT,
)
from codedoggy.memory.hermes_seam import (
    bind_session,
    build_system_memory_block,
    commit_session_boundary,
    notify_curated_write,
    on_pre_compress,
    on_session_close,
    on_transcript_rewound,
    on_turn_begin,
    on_turn_end,
    prefetch_fenced,
    sample_messages_with_memory,
)
from codedoggy.memory.hermes_select import HermesMemorySelector
from codedoggy.memory.manager import MemoryManager
from codedoggy.memory.paths import default_memory_home, get_memory_dir
from codedoggy.memory.plugins import (
    discover_memory_providers,
    list_memory_provider_names,
    load_memory_provider,
    load_memory_provider_from_env,
)
from codedoggy.memory.provider import (
    BaseMemoryProvider,
    CuratedMemoryProvider,
    SessionFtsProvider,
)
from codedoggy.memory.redact import redact_secrets
from codedoggy.memory.session_store import SessionStore, default_session_db_path
from codedoggy.memory.store import MemoryStore, load_on_disk_store
from codedoggy.memory.tool_injection import inject_memory_provider_tools

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
    "bind_session",
    "build_memory_context_block",
    "build_system_memory_block",
    "commit_session_boundary",
    "default_memory_home",
    "default_session_db_path",
    "discover_memory_providers",
    "get_memory_dir",
    "inject_memory_provider_tools",
    "list_memory_provider_names",
    "load_memory_provider",
    "load_memory_provider_from_env",
    "load_on_disk_store",
    "messages_with_ephemeral_memory",
    "notify_curated_write",
    "on_pre_compress",
    "on_session_close",
    "on_transcript_rewound",
    "on_turn_begin",
    "on_turn_end",
    "prefetch_fenced",
    "redact_secrets",
    "sample_messages_with_memory",
    "sanitize_context",
]
