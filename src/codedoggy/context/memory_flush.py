"""Pre-compaction memory flush — Grok memory_flush spirit → Hermes MemoryStore.

Before folding the live transcript, optionally ask a model to extract durable
facts and write them into curated MEMORY.md so compaction does not lose them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from codedoggy.context.budget import estimate_chars
from codedoggy.turn.types import Message, Role

logger = logging.getLogger(__name__)

FLUSH_SYSTEM_PROMPT = """\
You are a memory assistant. Extract ALL useful information from this conversation \
that would help a coding agent be more effective in future sessions with this user. \
Write a concise markdown summary with ## headers covering:

- **Decisions & rationale** — what was chosen and why
- **Technical context** — architecture, APIs, patterns, tools, file paths discussed
- **Debugging techniques & tools** — CLI commands, query patterns, investigation workflows
- **Problems & solutions** — bugs found, how they were fixed, workarounds

Omit any section where there is nothing substantive to report.
Do NOT include ephemeral progress or one-off temporary paths.
Respond with NO_REPLY if nothing genuinely useful was learned.
"""


class FlushResultKind(str, Enum):
    NOTHING = "nothing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SKIPPED = "skipped"


@dataclass(slots=True)
class FlushResult:
    kind: FlushResultKind
    content: str | None = None
    reason: str | None = None
    entries_written: int = 0
    written_entries: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryFlushConfig:
    enabled: bool = True
    """Run flush when approaching compact threshold."""

    soft_ratio: float = 0.08
    """Flush when usage >= trigger_chars * (1 - soft_ratio) — slightly before compact."""

    max_flush_write_chars: int = 2_000
    """Cap accepted flush body before write."""

    require_markdown_header: bool = True
    """Reject flush body without ## headers (Grok quality gate)."""


def should_flush(
    messages: list[Message],
    *,
    trigger_chars: int,
    config: MemoryFlushConfig,
    last_flush_cycle: int,
    current_cycle: int,
) -> bool:
    """True when usage crosses soft flush threshold for this compaction cycle.

    Once a flush runs for ``current_cycle`` (including cycle 0, before the
    first hard fold), further ensure() calls in that cycle skip flush until
    ``compaction_count`` advances.
    """
    if not config.enabled:
        return False
    # last_flush_cycle starts at -1; after first flush it equals current_cycle
    # (often 0). Must gate cycle 0 too — otherwise every pre-fold sample
    # re-invokes the flush model.
    if last_flush_cycle == current_cycle:
        return False
    usage = estimate_chars(messages)
    flush_at = int(trigger_chars * (1.0 - max(0.0, min(0.5, config.soft_ratio))))
    return usage >= flush_at


def process_flush_response(response: str, config: MemoryFlushConfig) -> FlushResult:
    """Grok process_flush_response quality controls."""
    trimmed = (response or "").strip()
    if not trimmed:
        return FlushResult(kind=FlushResultKind.NOTHING, reason="empty")
    if _is_no_reply(trimmed):
        return FlushResult(kind=FlushResultKind.NOTHING, reason="NO_REPLY")
    if len(trimmed) > config.max_flush_write_chars:
        trimmed = trimmed[: config.max_flush_write_chars]
    if config.require_markdown_header and not re.search(r"^##\s+\S", trimmed, re.M):
        return FlushResult(
            kind=FlushResultKind.REJECTED,
            reason="missing markdown ## headers",
            content=trimmed,
        )
    return FlushResult(kind=FlushResultKind.ACCEPTED, content=trimmed)


def run_memory_flush(
    messages: list[Message],
    *,
    client: Any | None,
    memory_store: Any | None,
    config: MemoryFlushConfig | None = None,
) -> FlushResult:
    """Call model + write durable entries into MemoryStore (target=memory)."""
    config = config or MemoryFlushConfig()
    if not config.enabled or client is None or memory_store is None:
        return FlushResult(kind=FlushResultKind.SKIPPED, reason="disabled or unbound")

    processed = prepare_memory_flush(messages, client=client, config=config)
    return commit_memory_flush(processed, memory_store=memory_store)


def prepare_memory_flush(
    messages: list[Message],
    *,
    client: Any | None,
    config: MemoryFlushConfig | None = None,
) -> FlushResult:
    """Generate and quality-gate a flush without mutating the memory store.

    Prefire runs this phase on a daemon worker.  The owning compactor commits
    the returned result only after joining it, which removes close/write and
    double-flush races.
    """
    config = config or MemoryFlushConfig()
    if not config.enabled or client is None:
        return FlushResult(kind=FlushResultKind.SKIPPED, reason="disabled or unbound")

    sketch = _transcript_for_flush(messages)
    if len(sketch) < 80:
        return FlushResult(kind=FlushResultKind.NOTHING, reason="too little transcript")

    try:
        from codedoggy.model.types import ChatMessage

        result = client.complete(
            [
                ChatMessage(role="system", content=FLUSH_SYSTEM_PROMPT),
                ChatMessage(role="user", content=sketch[:14_000]),
            ],
            temperature=0.1,
            max_tokens=700,
        )
        raw = (result.content or "").strip()
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.I).strip()
    except Exception as e:  # noqa: BLE001
        logger.warning("memory flush model call failed: %s", e)
        return FlushResult(kind=FlushResultKind.SKIPPED, reason=str(e))

    return process_flush_response(raw, config)


def commit_memory_flush(
    processed: FlushResult,
    *,
    memory_store: Any | None,
) -> FlushResult:
    """Commit a prepared result once and retain the exact mirrored entries."""
    if (
        processed.kind is not FlushResultKind.ACCEPTED
        or not processed.content
        or memory_store is None
    ):
        return processed

    written = _write_flush_to_memory(memory_store, processed.content)
    processed.written_entries = written
    processed.entries_written = len(written)
    return processed


def _write_flush_to_memory(store: Any, content: str) -> list[str]:
    """Split ## sections into memory entries (skip empty)."""
    sections = re.split(r"(?=^##\s+)", content, flags=re.M)
    written: list[str] = []
    for sec in sections:
        body = sec.strip()
        if not body or body.upper() == "NO_REPLY":
            continue
        # Prefer full section as one entry
        try:
            resp = store.add("memory", body)
            if isinstance(resp, dict) and resp.get("success"):
                written.append(body)
            elif isinstance(resp, dict) and "already exists" in str(resp.get("message", "")):
                written.append(body)
        except Exception as e:  # noqa: BLE001
            logger.warning("memory flush write failed: %s", e)
    return written


def _is_no_reply(text: str) -> bool:
    t = text.strip().upper()
    return t == "NO_REPLY" or t.startswith("NO_REPLY\n") or t.startswith("NO_REPLY ")


def _transcript_for_flush(messages: list[Message]) -> str:
    lines: list[str] = []
    for m in messages:
        if m.role is Role.SYSTEM:
            continue
        role = m.role.value if isinstance(m.role, Role) else str(m.role)
        body = (m.content or "").strip()
        if m.role is Role.ASSISTANT and m.tool_calls:
            names = ", ".join(tc.name for tc in m.tool_calls)
            lines.append(f"assistant→{names}")
        if body:
            if len(body) > 400:
                body = body[:397] + "…"
            lines.append(f"{role}: {body}")
    return "\n".join(lines)
