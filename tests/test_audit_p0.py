"""P0 immediate vs deferred non-P0 audit delivery."""

from __future__ import annotations

from codedoggy.audit.format import (
    format_deferred_summary,
    format_p0_red_card,
    is_p0,
    partition_findings,
)
from codedoggy.audit.types import AuditFinding, FindingSeverity


def test_partition_p0() -> None:
    findings = [
        AuditFinding(message="a", severity=FindingSeverity.CRITICAL),
        AuditFinding(message="b", severity=FindingSeverity.IMPORTANT),
        AuditFinding(message="c", severity=FindingSeverity.SUGGESTION),
    ]
    p0, rest = partition_findings(findings)
    assert len(p0) == 1 and is_p0(p0[0])
    assert [f.message for f in rest] == ["b", "c"]


def test_format_p0_and_deferred() -> None:
    p0 = [AuditFinding(message="stop", severity=FindingSeverity.CRITICAL, path="x")]
    rest = [AuditFinding(message="nit", severity=FindingSeverity.SUGGESTION)]
    card = format_p0_red_card(p0)
    assert card and "P0" in card and "stop" in card
    summary = format_deferred_summary(rest)
    assert summary and "nit" in summary and "Non-blocking" in summary
