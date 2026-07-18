"""Grok-style auto-compaction suppression after deterministic failures."""

from __future__ import annotations

from enum import IntEnum


class SuppressLevel(IntEnum):
    """Mirrors Grok SUPPRESS_* gates for auto-compact."""

    NONE = 0
    """Normal — auto-compact may run."""

    TURN = 1
    """Skip for current turn; clear at next turn start."""

    STICKY = 2
    """Survives turns; clear only after successful compact or budget change."""

    UNTIL_SUCCESS = 3
    """Account/model failure; clear when a model sample succeeds."""


class CompactionSuppressor:
    """Mutable gate held on ContextCompactor / session."""

    def __init__(self) -> None:
        self.level = SuppressLevel.NONE

    def allow_auto(self) -> bool:
        return self.level is SuppressLevel.NONE

    def mark_turn_failure(self) -> None:
        if self.level < SuppressLevel.TURN:
            self.level = SuppressLevel.TURN

    def mark_sticky_failure(self) -> None:
        self.level = SuppressLevel.STICKY

    def mark_until_success(self) -> None:
        self.level = SuppressLevel.UNTIL_SUCCESS

    def on_turn_start(self) -> None:
        if self.level is SuppressLevel.TURN:
            self.level = SuppressLevel.NONE

    def on_compact_success(self) -> None:
        if self.level in (SuppressLevel.STICKY, SuppressLevel.TURN):
            self.level = SuppressLevel.NONE

    def on_model_success(self) -> None:
        if self.level is SuppressLevel.UNTIL_SUCCESS:
            self.level = SuppressLevel.NONE

    def clear(self) -> None:
        self.level = SuppressLevel.NONE
