"""Hermes memory-context fencing — port of agent/memory_manager.py helpers.

Source of truth: C:\\Ai\\hermes-agent\\agent\\memory_manager.py
  - sanitize_context
  - build_memory_context_block
  - fence tags: <memory-context> … </memory-context>

Do not invent alternate tags. Prefetch is injected into the *current user
message* at sample time only (conversation_loop.py); never into SYSTEM and
never persisted into the live/archive transcript.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Ported from hermes-agent/agent/memory_manager.py
_FENCE_TAG_RE = re.compile(r"</?\s*memory-context\s*>", re.IGNORECASE)
_INTERNAL_CONTEXT_RE = re.compile(
    r"<\s*memory-context\s*>[\s\S]*?</\s*memory-context\s*>",
    re.IGNORECASE,
)
_INTERNAL_NOTE_RE = re.compile(
    r"\[System note:\s*The following is recalled memory context,\s*NOT new user input\.\s*"
    r"Treat as (?:informational background data|authoritative reference data[^\]]*)\.\]\s*",
    re.IGNORECASE,
)

_HERMES_SYSTEM_NOTE = (
    "[System note: The following is recalled memory context, "
    "NOT new user input. Treat as authoritative reference data — "
    "this is the agent's persistent memory and should inform all responses.]"
)


def sanitize_context(text: str) -> str:
    """Strip fence tags / injected blocks / system notes from provider output.

    Hermes: providers must return raw recall text, not pre-wrapped fences.
    """
    if not text:
        return text
    text = _INTERNAL_CONTEXT_RE.sub("", text)
    text = _INTERNAL_NOTE_RE.sub("", text)
    text = _FENCE_TAG_RE.sub("", text)
    return text


def build_memory_context_block(raw_context: str) -> str:
    """Wrap prefetched memory in Hermes ``<memory-context>`` fence.

    Exact shape from hermes-agent MemoryManager.build_memory_context_block.
    """
    if not raw_context or not str(raw_context).strip():
        return ""
    clean = sanitize_context(str(raw_context))
    if clean != str(raw_context):
        logger.warning("memory provider returned pre-wrapped context; stripped")
    if not clean.strip():
        return ""
    return (
        "<memory-context>\n"
        f"{_HERMES_SYSTEM_NOTE}\n\n"
        f"{clean.strip()}\n"
        "</memory-context>"
    )


def strip_memory_context_from_messages(messages: list[Any]) -> list[Any]:
    """Remove any accidental persisted memory-context spans from user contents."""
    from codedoggy.context.live_history import copy_message
    from codedoggy.turn.types import Role

    out: list[Any] = []
    for m in messages:
        if getattr(m, "role", None) is Role.USER and isinstance(m.content, str):
            cleaned = sanitize_context(m.content).strip()
            if cleaned != (m.content or "").strip():
                cm = copy_message(m)
                cm.content = cleaned
                out.append(cm)
                continue
        out.append(m)
    return out


def messages_with_ephemeral_memory(
    messages: list[Any],
    fenced_block: str | None,
) -> list[Any]:
    """Hermes conversation_loop: append fence to *current turn user* for API only.

    Does not mutate the input list. Finds the last USER message and appends
    the fenced block to its content (same as api_msg content injection).
    """
    if not fenced_block or not str(fenced_block).strip():
        return messages
    from codedoggy.context.live_history import copy_message
    from codedoggy.turn.types import Role

    out = [copy_message(m) for m in messages]
    for i in range(len(out) - 1, -1, -1):
        if out[i].role is Role.USER:
            base = out[i].content or ""
            if not isinstance(base, str):
                base = str(base)
            # Avoid double-inject if already present
            if "<memory-context>" in base:
                return out
            out[i].content = base + "\n\n" + str(fenced_block).strip()
            return out
    return out


class StreamingContextScrubber:
    """Stateful scrubber for streaming text that may contain split memory-context spans.

    Ported from hermes-agent ``agent/memory_manager.py`` StreamingContextScrubber.
    One-shot ``sanitize_context`` cannot survive chunk boundaries; this holds
    partial tags and discards in-span content so UI never leaks fence payload.
    """

    _OPEN_TAG = "<memory-context>"
    _CLOSE_TAG = "</memory-context>"

    def __init__(self) -> None:
        self._in_span: bool = False
        self._buf: str = ""
        self._at_block_boundary: bool = True

    def reset(self) -> None:
        self._in_span = False
        self._buf = ""
        self._at_block_boundary = True

    def feed(self, text: str) -> str:
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []

        while buf:
            if self._in_span:
                idx = buf.lower().find(self._CLOSE_TAG)
                if idx == -1:
                    held = self._max_partial_suffix(buf, self._CLOSE_TAG)
                    self._buf = buf[-held:] if held else ""
                    return "".join(out)
                buf = buf[idx + len(self._CLOSE_TAG) :]
                self._in_span = False
            else:
                idx = self._find_boundary_open_tag(buf)
                if idx == -1:
                    held = self._max_pending_open_suffix(buf) or self._max_partial_suffix(
                        buf, self._OPEN_TAG
                    )
                    if held:
                        self._append_visible(out, buf[:-held])
                        self._buf = buf[-held:]
                    else:
                        self._append_visible(out, buf)
                    return "".join(out)
                if idx > 0:
                    self._append_visible(out, buf[:idx])
                buf = buf[idx + len(self._OPEN_TAG) :]
                self._in_span = True

        return "".join(out)

    def flush(self) -> str:
        if self._in_span:
            self._buf = ""
            self._in_span = False
            return ""
        tail = self._buf
        self._buf = ""
        return tail

    @staticmethod
    def _max_partial_suffix(buf: str, tag: str) -> int:
        tag_lower = tag.lower()
        buf_lower = buf.lower()
        max_check = min(len(buf_lower), len(tag_lower) - 1)
        for i in range(max_check, 0, -1):
            if tag_lower.startswith(buf_lower[-i:]):
                return i
        return 0

    def _find_boundary_open_tag(self, buf: str) -> int:
        buf_lower = buf.lower()
        search_start = 0
        while True:
            idx = buf_lower.find(self._OPEN_TAG, search_start)
            if idx == -1:
                return -1
            if self._is_block_boundary(buf, idx) and self._has_block_opener_suffix(buf, idx):
                return idx
            search_start = idx + 1

    def _max_pending_open_suffix(self, buf: str) -> int:
        if not buf.lower().endswith(self._OPEN_TAG):
            return 0
        idx = len(buf) - len(self._OPEN_TAG)
        if not self._is_block_boundary(buf, idx):
            return 0
        return len(self._OPEN_TAG)

    def _has_block_opener_suffix(self, buf: str, idx: int) -> bool:
        after_idx = idx + len(self._OPEN_TAG)
        if after_idx >= len(buf):
            return False
        return buf[after_idx] in "\r\n"

    def _is_block_boundary(self, buf: str, idx: int) -> bool:
        if idx == 0:
            return self._at_block_boundary
        preceding = buf[:idx]
        last_newline = preceding.rfind("\n")
        if last_newline == -1:
            return self._at_block_boundary and preceding.strip() == ""
        return preceding[last_newline + 1 :].strip() == ""

    def _append_visible(self, out: list[str], text: str) -> None:
        if not text:
            return
        out.append(text)
        self._update_block_boundary(text)

    def _update_block_boundary(self, text: str) -> None:
        last_newline = text.rfind("\n")
        if last_newline != -1:
            self._at_block_boundary = text[last_newline + 1 :].strip() == ""
        else:
            self._at_block_boundary = self._at_block_boundary and text.strip() == ""
