"""Redact secrets before SessionStore / curated MEMORY persistence.

Single source: delegates to ``codedoggy.model.redact.redact_sensitive_text``
(force=True). No second pattern table — truth must not drift.
"""

from __future__ import annotations


def redact_secrets(text: str | None) -> str:
    """Return *text* with known secret patterns replaced. Never raises."""
    if text is None:
        return ""
    if not text:
        return text
    try:
        from codedoggy.model.redact import redact_sensitive_text

        out = redact_sensitive_text(text, force=True)
        return out if isinstance(out, str) else text
    except Exception:  # noqa: BLE001
        return text
