"""Model-brain **shadow** (影子) — write-time soft review via ChatClient.

Distinct from normal audits: runs inside the agent loop on each mutation,
never writes the workspace, only emits footnotes for the coding model.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from codedoggy.audit.types import (
    AuditContext,
    AuditFinding,
    AuditVerdict,
    FindingSeverity,
)
from codedoggy.model.provider import ChatClient
from codedoggy.model.types import ChatMessage

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are the Shadow (影子) quality reviewer for a coding agent.
You are NOT a normal offline code audit of the whole repo.
You do NOT write files. You only review ONE mutation (file write) and decide
whether it drifts from the session goal or introduces clear quality problems.

Return ONLY a single JSON object (no markdown fences), one of:

{"ok": true}

{"ok": false, "findings": [
  {"severity": "important"|"critical"|"suggestion", "message": "...", "path": "optional"}
]}

Rules:
- Pass silently (ok:true) when the edit is reasonable for the goal.
- Fail only when direction is wrong, edit is clearly harmful, or contradicts goal.
- Prefer short, actionable messages that tell the coding agent how to rethink.
- Do not invent issues. When unsure, ok:true.
"""


class ModelAuditor:
    """Shadow brain backed by a chat model (Ollama / OpenAI-compat).

    Alias: :class:`ShadowAuditor` (same type).
    """

    def __init__(
        self,
        client: ChatClient,
        *,
        temperature: float = 0.1,
        max_tokens: int = 800,
        max_diff_chars: int = 6_000,
        fail_closed: bool = True,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_diff_chars = max_diff_chars
        # When True, model/parse failure is a soft fail finding (not silent pass)
        self.fail_closed = fail_closed

    def review(self, ctx: AuditContext) -> AuditVerdict:
        user = self._build_user_prompt(ctx)
        try:
            result = self.client.complete(
                [
                    ChatMessage(role="system", content=_SYSTEM),
                    ChatMessage(role="user", content=user),
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:  # noqa: BLE001 — never crash the coding loop
            logger.warning("model auditor call failed: %s", e)
            if self.fail_closed:
                return AuditVerdict.fail(
                    [
                        AuditFinding(
                            message=f"Shadow unavailable (call failed): {e}",
                            severity=FindingSeverity.IMPORTANT,
                            path=ctx.mutation.path if ctx.mutation else None,
                        )
                    ]
                )
            return AuditVerdict.pass_silent()

        text = (result.content or "").strip()
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I).strip()
        verdict = _parse_verdict(text)
        if verdict is None:
            logger.warning("model auditor unparseable response: %s", text[:200])
            if self.fail_closed:
                return AuditVerdict.fail(
                    [
                        AuditFinding(
                            message="Shadow returned unparseable output; treat write as needs review",
                            severity=FindingSeverity.IMPORTANT,
                            path=ctx.mutation.path if ctx.mutation else None,
                        )
                    ]
                )
            return AuditVerdict.pass_silent()
        return verdict

    def _build_user_prompt(self, ctx: AuditContext) -> str:
        mut = ctx.mutation
        diff = mut.unified_diff_hint(max_chars=self.max_diff_chars)
        mem = ctx.memory.combined_text(max_chars=3_000)
        parts = [
            f"## Session goal\n{(ctx.goal or '(none)').strip()}",
            f"## Agent\n{ctx.agent_id}  session={ctx.session_id or '-'}",
            f"## Mutation trajectory (summary)\n{ctx.trajectory_summary}",
            f"## This mutation\npath={mut.path}\ntool={mut.tool_name}\n"
            f"create={mut.is_create} delete={getattr(mut, 'is_delete', False)}\n\n{diff}",
        ]
        if mem.strip():
            parts.append(f"## Selected memory\n{mem}")
        # Tool-layer workspace policy (four-pillar fuse) — first-class field,
        # with memory.raw fallback for older call sites.
        policy = getattr(ctx, "policy", None)
        if policy is None and isinstance(getattr(ctx.memory, "raw", None), dict):
            policy = ctx.memory.raw.get("policy")
        if policy:
            import json as _json

            try:
                pol_txt = _json.dumps(policy, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                pol_txt = str(policy)
            parts.append(
                "## Workspace policy (tool layer)\n"
                "Writes already denied by policy never reach you. "
                "For allowed mutations, respect deny lists and session goal.\n"
                f"{pol_txt}"
            )
        # Per-mutation policy stamp from tool layer (if present)
        mut_pol = (ctx.mutation.args or {}).get("_policy")
        if mut_pol:
            parts.append(f"## This write policy decision\n{mut_pol}")
        parts.append("Respond with JSON only.")
        return "\n\n".join(parts)


def _parse_verdict(text: str) -> AuditVerdict | None:
    data = _extract_json_object(text)
    if data is None:
        return None
    ok = data.get("ok")
    if ok is True:
        return AuditVerdict.pass_silent()
    if ok is not True and ok is not False:
        # Allow findings without ok field.
        if "findings" not in data:
            return None
        ok = False
    if ok is True:
        return AuditVerdict.pass_silent()

    raw_findings = data.get("findings") or []
    if not isinstance(raw_findings, list) or not raw_findings:
        # ok:false without findings — soft generic note
        return AuditVerdict.fail(
            [
                AuditFinding(
                    message=str(data.get("message") or "Model auditor rejected this edit."),
                    severity=FindingSeverity.IMPORTANT,
                    code="model",
                )
            ]
        )

    findings: list[AuditFinding] = []
    for item in raw_findings:
        if not isinstance(item, dict):
            continue
        msg = item.get("message") or item.get("text")
        if not msg:
            continue
        sev_raw = str(item.get("severity") or "important").lower()
        try:
            sev = FindingSeverity(sev_raw)
        except ValueError:
            sev = FindingSeverity.IMPORTANT
        findings.append(
            AuditFinding(
                message=str(msg),
                severity=sev,
                path=item.get("path"),
                code=str(item.get("code") or "model"),
            )
        )
    if not findings:
        return AuditVerdict.pass_silent()
    return AuditVerdict.fail(findings)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    # Fenced json
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    # First {...} slice
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None
