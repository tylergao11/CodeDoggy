"""Shadow (影子) — write-time soft quality review inside the agent loop.

Not a normal offline code audit. Package path stays ``codedoggy.audit`` for
import stability; product name is **Shadow**.
"""

from codedoggy.audit.auditor import (
    GoalDriftHeuristicAuditor,
    PassThroughAuditor,
    ResidentAuditor,
    ScriptedAuditor,
)
from codedoggy.audit.format import SHADOW_NAME, SHADOW_NAME_ZH, format_audit_observation
from codedoggy.audit.hooks import ResidentAuditHooks, resolve_audit_hooks
from codedoggy.audit.memory_select import (
    CuratedMemorySelector,
    MemorySelector,
    NoopMemorySelector,
)
from codedoggy.memory.hermes_select import HermesMemorySelector
from codedoggy.audit.model_auditor import ModelAuditor
from codedoggy.audit.services import AuditServices
from codedoggy.audit.trajectory import MutationTrajectory
from codedoggy.audit.types import (
    AuditContext,
    AuditFinding,
    AuditVerdict,
    FindingSeverity,
    MemorySelectRequest,
    MemorySelectResult,
    MutationEvent,
)

# Product-facing aliases (影子)
ShadowAuditor = ModelAuditor
ShadowHooks = ResidentAuditHooks
ShadowServices = AuditServices
ShadowTrajectory = MutationTrajectory

__all__ = [
    "AuditContext",
    "AuditFinding",
    "AuditServices",
    "AuditVerdict",
    "CuratedMemorySelector",
    "FindingSeverity",
    "GoalDriftHeuristicAuditor",
    "HermesMemorySelector",
    "MemorySelectRequest",
    "MemorySelectResult",
    "MemorySelector",
    "ModelAuditor",
    "MutationEvent",
    "MutationTrajectory",
    "NoopMemorySelector",
    "PassThroughAuditor",
    "ResidentAuditHooks",
    "ResidentAuditor",
    "SHADOW_NAME",
    "SHADOW_NAME_ZH",
    "ScriptedAuditor",
    "ShadowAuditor",
    "ShadowHooks",
    "ShadowServices",
    "ShadowTrajectory",
    "format_audit_observation",
    "resolve_audit_hooks",
]
