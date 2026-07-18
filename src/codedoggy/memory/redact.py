"""Redact secrets from text before SessionStore persistence.

Write-path only: FTS never indexes unredacted secrets. Pure functions — no I/O.
"""

from __future__ import annotations

import re

# Order matters: more specific patterns first.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # PEM / OpenSSH private key blocks (multiline)
    (
        re.compile(
            r"-----BEGIN[^\n]*PRIVATE KEY-----.*?-----END[^\n]*PRIVATE KEY-----",
            re.DOTALL | re.IGNORECASE,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    # Bearer tokens
    (
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]+=*", re.ASCII),
        "Bearer [REDACTED_TOKEN]",
    ),
    # OpenAI-style / generic sk- secrets (sk-…, sk-proj-…, sk-or-…)
    (
        re.compile(r"\bsk-(?:proj-|or-)?[A-Za-z0-9_\-]{16,}\b"),
        "[REDACTED_API_KEY]",
    ),
    # Common env / assignment forms: API_KEY=…, apiKey: …, "api_key": "…"
    (
        re.compile(
            r"(?i)\b("
            r"api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|"
            r"aws[_-]?secret[_-]?access[_-]?key|aws[_-]?access[_-]?key[_-]?id|"
            r"database[_-]?url|db[_-]?url|mongo(?:db)?[_-]?uri|"
            r"client[_-]?secret|xai[_-]?api[_-]?key|openai[_-]?api[_-]?key|"
            r"anthropic[_-]?api[_-]?key|github[_-]?token|gh[_-]?token|"
            r"password|passwd|pwd"
            r")\b(\s*[=:]\s*)([\"']?)([^\s\"']{8,})(\3)",
        ),
        r"\1\2\3[REDACTED]\3",
    ),
    # URI credentials: postgres://user:pass@host / mysql://...
    (
        re.compile(
            r"(?i)\b((?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|amqp)://)"
            r"([^:\s/@]+):([^@\s/]+)@",
        ),
        r"\1\2:[REDACTED]@",
    ),
    # Authorization: <scheme> <token>
    (
        re.compile(
            r"(?i)(authorization\s*[=:]\s*)([\"']?)(bearer\s+)?([A-Za-z0-9\-._~+/]{12,})(\2)",
        ),
        r"\1\2\3[REDACTED_TOKEN]\2",
    ),
    # Generic long hex/base64 tokens after key= style already covered;
    # xoxb-/xoxp- Slack, ghp_ GitHub PATs, glpat- GitLab
    (
        re.compile(r"\b(?:xox[baprs]-|ghp_|gho_|github_pat_|glpat-)[A-Za-z0-9_\-]{10,}\b"),
        "[REDACTED_TOKEN]",
    ),
]


def redact_secrets(text: str | None) -> str:
    """Return *text* with known secret patterns replaced. Never raises."""
    if text is None:
        return ""
    if not text:
        return text
    out = text
    for pattern, repl in _PATTERNS:
        out = pattern.sub(repl, out)
    return out
