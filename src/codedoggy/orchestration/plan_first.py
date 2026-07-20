"""Plan-first enforcement — go-steer core-agent substrate glue.

Source (do not invent):
  https://github.com/go-steer/core-agent/blob/main/docs/plan-first-design.md
  pkg/permissions/gate.go  (RequirePlanArtifact / planRecorded / planExemptTools)
  pkg/tools/record_plan.go (artifact path, seq, revoke, atomic write)

v1 semantics from go-steer Resolved Q1–Q5:
  - any non-empty plan after trim
  - spawn family plan-gated; children inherit parent's planRecorded
  - storage: <agentsDir>/plans/plan-<seq>.md
  - MCP / unknown tools gated by default
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# go-steer recordPlanDir
RECORD_PLAN_DIR = "plans"

# go-steer recordPlanFilenameRegex: plan-<seq>.md | plan-<seq>-revoked.md
_RECORD_PLAN_FILENAME_RE = re.compile(r"^plan-(\d+)(?:-revoked)?\.md$")

# Denial message mirrors gate.go planFirstDenial
PLAN_FIRST_DENIAL = (
    "{tool} denied: plan-first mode requires record_plan to be called "
    "before any mutating tool. Call record_plan(plan:) first, then retry"
)

# go-steer planExemptTools — mapped to CodeDoggy client / short names.
# Research + escape valve + read-only introspection. Everything else gated.
PLAN_EXEMPT_TOOLS: frozenset[str] = frozenset(
    {
        # Read-only filesystem + research (go-steer names + Doggy aliases)
        "read_file",
        "read_many_files",
        "stat",
        "list_dir",
        "glob",
        "grep",
        "json_query",
        "fetch_url",
        "web_fetch",
        "web_search",
        "todo",
        "todo_write",
        "record_plan",
        # Skill namespace / tool
        "skill",
        # Subagent introspection (go-steer list_agents / check_agent)
        "list_agents",
        "check_agent",
        "get_subagent_output",
        "get_task_output",
        "wait_tasks",
        "wait_commands_or_subagents",
        # Doggy research surface (read-only)
        "lsp",
        "code_nav",
        "session_search",
        "memory_search",
        "memory_get",
        "search_tool",
        # Grok plan-mode tools are research/orchestration, not mutations of the
        # workspace under plan-first (writes still blocked by plan edit gate).
        "enter_plan_mode",
        "exit_plan_mode",
        "ask_user_question",
    }
)


@dataclass
class PlanFirstGate:
    """Per-session plan-first flag (go-steer permissions.Gate plan fields)."""

    require_plan_artifact: bool = False
    plan_recorded: bool = False
    agents_dir: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def mark_plan_recorded(self) -> None:
        """go-steer Gate.MarkPlanRecorded."""
        with self._lock:
            self.plan_recorded = True

    def clear_plan_recorded(self) -> None:
        """go-steer Gate.ClearPlanRecorded."""
        with self._lock:
            self.plan_recorded = False

    def is_plan_recorded(self) -> bool:
        """go-steer Gate.IsPlanRecorded."""
        with self._lock:
            return self.plan_recorded

    def plan_required(self) -> bool:
        """go-steer Gate.PlanRequired."""
        return bool(self.require_plan_artifact)

    def resolve_agents_dir(self, cwd: Path | str | None) -> Path | None:
        """agentsDir for artifacts; default cwd/.agents (go-steer .agents/plans)."""
        if self.agents_dir:
            return Path(self.agents_dir)
        if cwd is None:
            return None
        return Path(cwd) / ".agents"


def normalize_tool_name(tool_name: str) -> str:
    """Strip Doggy:/MCP: then map product client name → wire short-id.

    Single source: ``grok_surface.CLIENT_ALIASES``. Prevents exempt-list drift
    when the model uses product names (e.g. get_command_or_subagent_output).
    """
    name = (tool_name or "").strip()
    if ":" in name:
        name = name.split(":", 1)[-1]
    if not name:
        return name
    try:
        from codedoggy.tools.grok_surface import CLIENT_ALIASES

        return CLIENT_ALIASES.get(name, name)
    except Exception:  # noqa: BLE001
        return name


def is_plan_exempt(tool_name: str) -> bool:
    wire = normalize_tool_name(tool_name)
    if wire in PLAN_EXEMPT_TOOLS:
        return True
    # Also accept raw client name if listed (legacy dual entries)
    raw = (tool_name or "").strip()
    if ":" in raw:
        raw = raw.split(":", 1)[-1]
    return raw in PLAN_EXEMPT_TOOLS


def plan_first_denial(gate: PlanFirstGate | None, tool_name: str) -> str | None:
    """Return denial message or None (go-steer planFirstDenial).

    Runs before mode/policy logic. Even yolo/auto-approve respects this when
    RequirePlanArtifact is set.
    """
    if gate is None or not gate.require_plan_artifact:
        return None
    if is_plan_exempt(tool_name):
        return None
    if gate.is_plan_recorded():
        return None
    return PLAN_FIRST_DENIAL.format(tool=normalize_tool_name(tool_name) or tool_name)


def next_plan_seq(plans_dir: Path) -> int:
    """go-steer nextPlanSeq — max(seq)+1 over plan-*.md and *-revoked.md."""
    if not plans_dir.is_dir():
        return 1
    max_seq = 0
    for entry in plans_dir.iterdir():
        if not entry.is_file():
            continue
        m = _RECORD_PLAN_FILENAME_RE.match(entry.name)
        if m is None:
            continue
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        if n > max_seq:
            max_seq = n
    return max_seq + 1


def latest_active_plan(agents_dir: Path | str) -> Path | None:
    """go-steer LatestActivePlan — highest-seq non-revoked plan path."""
    plans_dir = Path(agents_dir) / RECORD_PLAN_DIR
    if not plans_dir.is_dir():
        return None
    best_seq = -1
    best_name: str | None = None
    for entry in plans_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if "-revoked.md" in name:
            continue
        m = _RECORD_PLAN_FILENAME_RE.match(name)
        if m is None:
            continue
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        if n > best_seq:
            best_seq = n
            best_name = name
    if best_name is None:
        return None
    return plans_dir / best_name


def revoke_latest_plan(gate: PlanFirstGate, agents_dir: Path | str) -> Path | None:
    """go-steer RevokeLatestPlan — rename to *-revoked.md + clear flag."""
    try:
        latest = latest_active_plan(agents_dir)
        revoked: Path | None = None
        if latest is not None:
            revoked = latest.with_name(latest.stem + "-revoked.md")
            latest.rename(revoked)
        return revoked
    finally:
        gate.clear_plan_recorded()


def atomic_write_file(path: Path, data: bytes, mode: int = 0o644) -> None:
    """go-steer atomicWriteFile — temp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = None, None
    import tempfile

    try:
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=".plan-", dir=str(path.parent)
        )
        with os.fdopen(tmp_fd, "wb") as f:
            tmp_fd = None
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if os.name != "nt":
            os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def write_plan_artifact(
    agents_dir: Path,
    plan_body: str,
) -> tuple[Path, int]:
    """Persist plan markdown; return (path, sequence). Caller marks gate."""
    body = plan_body.strip()
    if not body:
        raise ValueError("record_plan: plan is required (non-empty markdown)")
    if not body.endswith("\n"):
        body += "\n"
    plans_dir = agents_dir / RECORD_PLAN_DIR
    plans_dir.mkdir(parents=True, exist_ok=True)
    seq = next_plan_seq(plans_dir)
    path = plans_dir / f"plan-{seq}.md"
    atomic_write_file(path, body.encode("utf-8"))
    return path, seq


def resolve_plan_first_gate(extra: dict[str, Any] | None) -> PlanFirstGate | None:
    """Pull PlanFirstGate from tool_extra / kernel (host bag)."""
    bag = extra or {}
    gate = bag.get("plan_first_gate")
    if isinstance(gate, PlanFirstGate):
        return gate
    kernel = bag.get("kernel")
    if kernel is not None:
        gate = getattr(kernel, "plan_first_gate", None)
        if isinstance(gate, PlanFirstGate):
            return gate
    return None


def require_plan_artifact_from_env(
    *,
    default: bool = False,
    environ: dict[str, str] | None = None,
) -> bool:
    """go-steer RequirePlanArtifact from CODEDOGGY_REQUIRE_PLAN_ARTIFACT.

    Substrate default is False (opt-in). Product bootstrap may pass default=True.
    """
    env = environ if environ is not None else os.environ
    raw = str(env.get("CODEDOGGY_REQUIRE_PLAN_ARTIFACT", "")).strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default
