"""Session construction config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SessionConfig:
    """Static options for a new session."""

    cwd: Path
    """Workspace root for path resolution."""

    max_turns: int | None = None
    """Max sampling rounds per prompt (each may include a tool batch); None = unlimited."""

    session_id: str | None = None
    """If omitted, a new id is generated."""

    goal: str | None = None
    """Session-level intent anchor for resident audit (and later Hermes memory select)."""

    enable_memory: bool = False
    """Enable cross-session memory when that subsystem is wired."""

    extra: dict = field(default_factory=dict)
    """Bag for optional fields without changing the dataclass shape."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "cwd", Path(self.cwd).resolve())
