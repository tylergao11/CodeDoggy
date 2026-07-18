"""Format shadow verdicts into model-facing notes (P0 immediate vs deferred).

Product name: **Shadow** (影子) — write-time quality soft-interrupt.
Distinct from normal project/code audits: shadow sits *inside* the agent
loop, never writes the workspace, and only footnotes tool results.
"""

from __future__ import annotations

import re

from codedoggy.audit.types import AuditFinding, AuditVerdict, FindingSeverity

# Product name constants
SHADOW_NAME = "shadow"
SHADOW_NAME_ZH = "影子"

# Stable markers — context pruning/fold must preserve these (see context/pruning.py).
AUDIT_P0_START = "── shadow P0 ──"
AUDIT_P0_END = "── end shadow P0 ──"
AUDIT_DEFERRED_END = "── end shadow summary ──"
# Legacy markers still recognized for prune/reinject of old transcripts
LEGACY_P0_START = "── resident audit P0 ──"
LEGACY_P0_END = "── end P0 ──"

_P0_FOOTER_RE = re.compile(
    rf"(?:{re.escape(AUDIT_P0_START)}|{re.escape(LEGACY_P0_START)})"
    rf"[\s\S]*?"
    rf"(?:{re.escape(AUDIT_P0_END)}|{re.escape(LEGACY_P0_END)})",
    re.MULTILINE,
)


def is_p0(finding: AuditFinding) -> bool:
    """P0 red-card: critical only."""
    sev = finding.severity
    if isinstance(sev, FindingSeverity):
        return sev is FindingSeverity.CRITICAL
    return str(sev).lower() == "critical"


def partition_findings(
    findings: list[AuditFinding],
) -> tuple[list[AuditFinding], list[AuditFinding]]:
    """Split into (p0_immediate, deferred_rest)."""
    p0: list[AuditFinding] = []
    rest: list[AuditFinding] = []
    for f in findings:
        (p0 if is_p0(f) else rest).append(f)
    return p0, rest


def format_p0_red_card(findings: list[AuditFinding]) -> str | None:
    """Immediate tool-observation footnote for P0 only."""
    if not findings:
        return None
    lines = [
        "",
        AUDIT_P0_START,
        "Blocking issues on your last write. Rethink and fix before continuing "
        "in the same direction.",
    ]
    _append_finding_lines(lines, findings)
    lines.append(AUDIT_P0_END)
    return "\n".join(lines)


def has_p0_footer(text: str | None) -> bool:
    """True when content carries a shadow P0 red card."""
    if not text:
        return False
    return AUDIT_P0_START in text or LEGACY_P0_START in text


def extract_p0_footers(text: str | None) -> list[str]:
    """Pull every P0 red-card block from a tool (or other) message body."""
    if not text or (AUDIT_P0_START not in text and LEGACY_P0_START not in text):
        return []
    return [m.group(0).strip() for m in _P0_FOOTER_RE.finditer(text)]


def extract_p0_footers_from_messages(messages: list[object]) -> list[str]:
    """Collect unique P0 footers across a message list (order preserved)."""
    seen: set[str] = set()
    out: list[str] = []
    for m in messages:
        content = getattr(m, "content", None)
        if not isinstance(content, str):
            continue
        for footer in extract_p0_footers(content):
            if footer not in seen:
                seen.add(footer)
                out.append(footer)
    return out


def format_deferred_summary(
    findings: list[AuditFinding],
    *,
    title: str = "shadow (end of turn)",
) -> str | None:
    """End-of-turn batch of non-P0 findings."""
    if not findings:
        return None
    lines = [
        "",
        f"── {title} ──",
        "Non-blocking notes from this turn (review when convenient):",
    ]
    _append_finding_lines(lines, findings)
    lines.append(AUDIT_DEFERRED_END)
    return "\n".join(lines)


def format_audit_observation(verdict: AuditVerdict) -> str | None:
    """Legacy: format all findings as one block (prefer P0/deferred split)."""
    if verdict.ok or not verdict.findings:
        return None
    p0, rest = partition_findings(list(verdict.findings))
    parts: list[str] = []
    card = format_p0_red_card(p0)
    if card:
        parts.append(card.lstrip("\n"))
    # If only non-P0, still show something when caller wants immediate-all.
    if rest and not p0:
        deferred = format_deferred_summary(rest, title="shadow")
        if deferred:
            parts.append(deferred.lstrip("\n"))
    elif rest:
        # Mixed: only P0 in immediate path; rest left for deferred channel.
        pass
    return "\n".join(parts) if parts else None


def _append_finding_lines(lines: list[str], findings: list[AuditFinding]) -> None:
    for i, f in enumerate(findings, start=1):
        sev = f.severity.value if isinstance(f.severity, FindingSeverity) else str(f.severity)
        loc = f" ({f.path})" if f.path else ""
        code = f" [{f.code}]" if f.code else ""
        lines.append(f"{i}. [{sev}]{code}{loc}: {f.message}")
