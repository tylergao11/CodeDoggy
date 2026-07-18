"""Token counting: optional tiktoken, else CJK-aware weighted heuristic.

Grok budgets in real tokens. We use tiktoken when installed
(``pip install tiktoken``) and fall back without making it a hard dependency.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\u3040-\u30ff\uac00-\ud7af\uff00-\uffef]"
)

_enc: Any | None = None
_enc_tried = False
_enc_name: str | None = None


def tokenizer_backend() -> str:
    """``tiktoken:<name>`` or ``heuristic``."""
    _ensure_encoder()
    if _enc is not None and _enc_name:
        return f"tiktoken:{_enc_name}"
    return "heuristic"


def _ensure_encoder() -> None:
    global _enc, _enc_tried, _enc_name
    if _enc_tried:
        return
    _enc_tried = True
    # Default on when package present; disable with CODEDOGGY_TIKTOKEN=0
    if os.environ.get("CODEDOGGY_TIKTOKEN", "1").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }:
        return
    try:
        import tiktoken  # type: ignore[import-not-found]

        name = os.environ.get("CODEDOGGY_TIKTOKEN_ENCODING", "cl100k_base").strip()
        try:
            _enc = tiktoken.get_encoding(name)
            _enc_name = name
        except Exception:
            _enc = tiktoken.get_encoding("cl100k_base")
            _enc_name = "cl100k_base"
        logger.info("token budget using tiktoken encoding=%s", _enc_name)
    except Exception as e:  # noqa: BLE001
        logger.debug("tiktoken unavailable, using heuristic: %s", e)
        _enc = None
        _enc_name = None


def count_text_tokens(text: str | None) -> int:
    if not text:
        return 0
    _ensure_encoder()
    if _enc is not None:
        try:
            return len(_enc.encode(text))
        except Exception:  # noqa: BLE001
            pass
    return max(1, _heuristic_tokens(text))


def _heuristic_tokens(text: str) -> int:
    """~1 token / 4 latin chars; ~1 token per CJK char."""
    if not _CJK_RE.search(text):
        return max(1, (len(text) + 3) // 4)
    n = 0
    for ch in text:
        if _CJK_RE.match(ch):
            n += 1
        else:
            n += 0.25  # type: ignore[assignment]
    return max(1, int(n + 0.999))


def count_messages_tokens(messages: list[Any]) -> int:
    """Approx total tokens for a Message list (OpenAI-ish overhead per msg)."""
    total = 0
    for m in messages:
        total += 4  # role framing
        content = getattr(m, "content", None)
        if content:
            total += count_text_tokens(content)
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                total += count_text_tokens(getattr(tc, "name", "") or "")
                total += count_text_tokens(str(getattr(tc, "arguments", "") or ""))
                total += 4
        name = getattr(m, "name", None)
        if name:
            total += count_text_tokens(name)
    return max(1, total)
