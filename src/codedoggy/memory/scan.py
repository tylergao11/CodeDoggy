"""Lightweight scan for memory content that should not enter the system prompt."""

from __future__ import annotations

import re
from typing import Optional

# Strict-ish patterns for frozen system-prompt injection. Keep small; expand later.
_THREAT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "prompt_injection",
        re.compile(
            r"(ignore\s+(all\s+)?(previous|prior)\s+instructions"
            r"|disregard\s+(all\s+)?(previous|prior)"
            r"|you\s+are\s+now\s+(?:DAN|jailbreak)"
            r"|system\s*:\s*you\s+must)",
            re.IGNORECASE,
        ),
    ),
    (
        "exfil_marker",
        re.compile(
            r"(exfiltrate|send\s+all\s+(secrets|api\s*keys|credentials)\s+to)",
            re.IGNORECASE,
        ),
    ),
]


def first_threat_message(content: str) -> Optional[str]:
    """Return an error string if content matches a threat pattern, else None."""
    if not content or not content.strip():
        return None
    hits: list[str] = []
    for name, pat in _THREAT_PATTERNS:
        if pat.search(content):
            hits.append(name)
    if not hits:
        return None
    return (
        "Memory entry blocked: content matched threat pattern(s): "
        + ", ".join(hits)
        + ". Rewrite without injection/exfil wording."
    )


def threat_ids(content: str) -> list[str]:
    if not content or not content.strip():
        return []
    return [name for name, pat in _THREAT_PATTERNS if pat.search(content)]
