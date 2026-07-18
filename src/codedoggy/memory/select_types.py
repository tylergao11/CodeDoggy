"""Memory selection request/result — used by Hermes prefetch (not audit)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MemorySelectRequest:
    """Inputs for multi-source memory selection (curated + session FTS)."""

    goal: str | None = None
    # Path / topic hint (prefetch uses a placeholder; tools may pass a real path)
    path: str = ""
    trajectory_summary: str = ""
    session_id: str | None = None
    agent_id: str = "main"
    max_curated_chars: int = 2_000
    max_session_hits: int = 5
    query_hint: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemorySelectResult:
    """Selected memory slices — curated / session / provider kept separate."""

    curated_blocks: list[str] = field(default_factory=list)
    session_hits: list[str] = field(default_factory=list)
    provider_hits: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def combined_text(self, *, max_chars: int = 6_000) -> str:
        parts: list[str] = []
        if self.curated_blocks:
            parts.append("## Curated memory\n" + "\n\n".join(self.curated_blocks))
        if self.session_hits:
            parts.append("## Session recall\n" + "\n\n".join(self.session_hits))
        if self.provider_hits:
            parts.append("## Provider memory\n" + "\n\n".join(self.provider_hits))
        text = "\n\n".join(parts)
        if len(text) > max_chars:
            return text[: max_chars - 20] + "\n… (truncated)"
        return text
