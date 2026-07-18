"""Grok CompactionMode — how much pre-compaction history the model can recover."""

from __future__ import annotations

from enum import Enum


class CompactionMode(str, Enum):
    """How compaction exposes pre-compaction history afterwards."""

    SUMMARY = "summary"
    """Summary only — no pointer back to history."""

    TRANSCRIPT = "transcript"
    """Summary + pointer to full raw transcript location."""

    SEGMENTS = "segments"
    """Summary + pointer to segment_*.md store under compaction/."""

    @classmethod
    def parse(cls, value: str | None) -> CompactionMode:
        if not value or not str(value).strip():
            return cls.SUMMARY
        key = str(value).strip().lower()
        for m in cls:
            if m.value == key:
                return m
        return cls.SUMMARY

    def transcript_hint(self, location: str | None) -> str | None:
        if not location:
            return None
        if self is CompactionMode.SUMMARY:
            return None
        if self is CompactionMode.TRANSCRIPT:
            return (
                f"\n\nIf you need specific details from before compaction "
                f"(exact code, errors, or generated content), recover them via "
                f"session_search or the transcript at: {location}"
            )
        # SEGMENTS
        return (
            f"\n\nFull verbatim rollouts of previous segments are available at "
            f"{location}/segment_*.md. See {location}/INDEX.md for a table of contents. "
            f"Use read_file or grep to recover specific details if this summary is "
            f"insufficient. Do NOT recreate these files."
        )
