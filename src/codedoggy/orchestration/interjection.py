"""Grok interjection framing — port of ``xai-interjection-core``.

Source of truth (do not invent):
  C:\\Ai\\grok-build\\crates\\common\\xai-interjection-core\\src\\format.rs
  C:\\Ai\\grok-build\\crates\\common\\xai-interjection-core\\src\\buffer.rs

Wire contract:
  - One synthetic USER message per pending entry (never merged)
  - Text framed by ``format_interjection`` (same shape as Grok tests)
  - No ``[interjection]`` prefix (CodeDoggy invention — removed)
"""

from __future__ import annotations

from typing import Callable

# format.rs — LARGE_PROMPT_THRESHOLD (Rust String::len = UTF-8 bytes)
LARGE_PROMPT_THRESHOLD = 25_000


def user_query(user_message: str) -> str:
    """Source: ``format.rs::user_query``."""
    return f"<user_query>\n{user_message}\n</user_query>"


def format_interjection(text: str) -> str:
    """Source: ``format.rs::format_interjection``.

    Truncate on UTF-8 byte boundary at LARGE_PROMPT_THRESHOLD, append
    ``... [truncated]``, wrap with mid-turn note + ``user_query``.
    No deferral instruction.
    """
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    raw = text.encode("utf-8")
    if len(raw) > LARGE_PROMPT_THRESHOLD:
        cut = raw[:LARGE_PROMPT_THRESHOLD]
        # Walk back to a valid UTF-8 start boundary (not a continuation byte)
        while cut and (cut[-1] & 0xC0) == 0x80:
            cut = cut[:-1]
        truncated = cut.decode("utf-8", errors="ignore") + "... [truncated]"
    else:
        truncated = text

    return (
        "The user sent a message while you were working:\n"
        + user_query(truncated)
    )


def drain_formatted(
    items: list,
    *,
    sanitize_text: Callable[[str], str] | None = None,
) -> list[str]:
    """Source: ``buffer.rs::drain_formatted`` — one framed string per entry."""
    sanitize = sanitize_text or (lambda t: t)
    out: list[str] = []
    for entry in items:
        raw = getattr(entry, "text", None)
        if raw is None:
            raw = str(entry)
        out.append(format_interjection(sanitize(str(raw))))
    return out
