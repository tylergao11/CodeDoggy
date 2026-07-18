"""Defaults for curated persistent memory."""

from __future__ import annotations

# Character budgets (model-independent). Same ballpark as production agents.
MEMORY_CHAR_LIMIT: int = 2_200
USER_CHAR_LIMIT: int = 1_375

# Entry separator in MEMORY.md / USER.md (entries may be multiline).
ENTRY_DELIMITER: str = "\n§\n"

# Stop spinning the model on failed consolidate-in-this-turn loops.
MAX_CONSOLIDATION_FAILURES_PER_TURN: int = 3
