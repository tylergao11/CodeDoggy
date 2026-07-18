"""In-session context: Grok compaction pipeline + Hermes memory authority."""

from codedoggy.context.budget import (
    ContextBudget,
    budget_status,
    estimate_chars,
    estimate_tokens,
    needs_compaction,
)
from codedoggy.context.live_history import seed_messages, strip_system_messages
from codedoggy.context.tokens import count_text_tokens, tokenizer_backend
from codedoggy.context.compactor import (
    COMPACTION_PREFIX,
    HISTORICAL_IN_PROGRESS_HEADING,
    HISTORICAL_PENDING_ASKS_HEADING,
    HISTORICAL_REMAINING_WORK_HEADING,
    HISTORICAL_TASK_HEADING,
    CompactionResult,
    ContextCompactor,
)
from codedoggy.context.select import plan_fold_regions, snap_to_safe_boundary
from codedoggy.context.memory_flush import (
    FLUSH_SYSTEM_PROMPT,
    FlushResult,
    FlushResultKind,
    MemoryFlushConfig,
    process_flush_response,
    run_memory_flush,
    should_flush,
)
from codedoggy.context.mode import CompactionMode
from codedoggy.context.pruning import (
    collect_p0_footers,
    extract_audit_p0_footer,
    has_audit_p0_footer,
    prune_oversized_tool_results,
    prune_retained_tool_results,
    reinject_missing_p0,
)
from codedoggy.context.segments import compaction_dir, write_segment
from codedoggy.context.suppress import CompactionSuppressor, SuppressLevel

__all__ = [
    "COMPACTION_PREFIX",
    "CompactionMode",
    "CompactionResult",
    "CompactionSuppressor",
    "ContextBudget",
    "ContextCompactor",
    "HISTORICAL_IN_PROGRESS_HEADING",
    "HISTORICAL_PENDING_ASKS_HEADING",
    "HISTORICAL_REMAINING_WORK_HEADING",
    "HISTORICAL_TASK_HEADING",
    "plan_fold_regions",
    "snap_to_safe_boundary",
    "FLUSH_SYSTEM_PROMPT",
    "FlushResult",
    "FlushResultKind",
    "MemoryFlushConfig",
    "SuppressLevel",
    "budget_status",
    "compaction_dir",
    "count_text_tokens",
    "estimate_chars",
    "estimate_tokens",
    "needs_compaction",
    "tokenizer_backend",
    "process_flush_response",
    "collect_p0_footers",
    "extract_audit_p0_footer",
    "has_audit_p0_footer",
    "prune_oversized_tool_results",
    "prune_retained_tool_results",
    "reinject_missing_p0",
    "run_memory_flush",
    "seed_messages",
    "should_flush",
    "strip_system_messages",
    "write_segment",
]
