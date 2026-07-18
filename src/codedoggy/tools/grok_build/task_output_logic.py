"""Task output / kill / wait pure logic — source port from Grok.

Ported from:
  grok-build/crates/common/xai-tool-types/src/task.rs
    MAX_MULTI_WAIT_IDS, task_output_waits, resolve_task_ids
    build_task_output_description, build_kill_task_description,
    build_wait_tasks_description, format_resume_footer
  grok-build/crates/codegen/xai-grok-tools/src/implementations/grok_build/task_output/mod.rs
    DEFAULT_WAIT_TIMEOUT, MAX_WAIT_BLOCK, capped_wait_timeout, not_found_result
  grok-build/crates/codegen/xai-grok-tools/src/implementations/task_output/tool.rs
    snapshot_to_result status mapping
  grok-build/crates/codegen/xai-grok-tools/src/types/output.rs
    TaskOutput / KillTask to_prompt_format cards
  grok-build/crates/codegen/xai-grok-tools/src/implementations/grok_build/kill_task/mod.rs
    kill success / already_exited / not-found messages

Function map:
  MAX_MULTI_WAIT_IDS          ↔ MAX_MULTI_WAIT_IDS
  task_output_waits           ↔ task_output_waits
  resolve_task_ids            ↔ resolve_task_ids
  capped_wait_timeout_ms      ↔ capped_wait_timeout
  build_task_output_description / build_kill_task_description / build_wait_tasks_description
  format_task_result_card     ↔ ToolOutput::TaskOutput Result branch
  format_multi_result         ↔ MultiResult branch
  format_kill_result          ↔ KillTask Result branch
  format_resume_footer        ↔ format_resume_footer
  format_subagent_card        ↔ format_subagent_snapshot (subset)
  kill messages               ↔ KillTaskTool messages
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

# ── constants (task.rs / task_output/mod.rs) ─────────────────────────

MAX_MULTI_WAIT_IDS: int = 20
DEFAULT_WAIT_TIMEOUT_MS: int = 30_000
MAX_WAIT_BLOCK_MS: int = 600_000  # 10 minutes

# Product-facing kill success messages (kill_task/mod.rs)
KILL_MSG_TERMINATED = "Task was terminated successfully"
KILL_MSG_ALREADY_EXITED = "Task had already completed"
KILL_MSG_SUBAGENT_CANCEL = "Subagent cancellation initiated"


def task_output_waits(timeout_ms: Optional[int]) -> bool:
    """Positive timeout_ms waits; omit or 0 polls without blocking."""
    return timeout_ms is not None and timeout_ms > 0


def resolve_task_ids(ids: list[str]) -> list[str]:
    """Trimmed, de-duplicated task IDs preserving first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in ids:
        tid = str(raw).strip()
        if tid and tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


def max_wait_block_ms() -> int:
    """Env override GROK_MAX_WAIT_BLOCK_MS (Grok max_wait_block)."""
    raw = os.environ.get("GROK_MAX_WAIT_BLOCK_MS")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return MAX_WAIT_BLOCK_MS


def capped_wait_timeout_ms(timeout_ms: Optional[int]) -> int:
    """Default when omitted (wait mode), then clamp to max_wait_block."""
    base = DEFAULT_WAIT_TIMEOUT_MS if timeout_ms is None else int(timeout_ms)
    if base < 0:
        base = DEFAULT_WAIT_TIMEOUT_MS
    return min(base, max_wait_block_ms())


# ── description builders (task.rs) ───────────────────────────────────


def _lifecycle_target_suffix(*, monitor_present: bool, subagent_present: bool) -> str:
    if monitor_present and subagent_present:
        return ", monitor, or subagent"
    if monitor_present:
        return " or monitor"
    if subagent_present:
        return " or subagent"
    return ""


def _monitor_task_id_note(monitor_tool: Optional[str]) -> str:
    if monitor_tool:
        return f" (a monitor's task_id is returned by {monitor_tool})"
    return ""


def build_task_output_description(
    *,
    monitor_tool: Optional[str] = "monitor",
    read_tool: Optional[str] = "read_file",
    bash_background_param: Optional[str] = "background",
    subagent_background_param: Optional[str] = "background",
) -> str:
    """Grok build_task_output_description (product param names)."""
    monitor_present = monitor_tool is not None
    subagent_present = subagent_background_param is not None
    target_suffix = _lifecycle_target_suffix(
        monitor_present=monitor_present, subagent_present=subagent_present
    )
    sources: list[str] = []
    if bash_background_param:
        sources.append(f"{bash_background_param}=true commands")
    if subagent_background_param:
        # After product rename both params may be "background"; keep distinct labels.
        sources.append(f"{subagent_background_param}=true subagents")
    # de-dupe while preserving order (product renames can collapse param names)
    seen_src: set[str] = set()
    uniq: list[str] = []
    for s in sources:
        if s not in seen_src:
            seen_src.add(s)
            uniq.append(s)
    sources_s = " or ".join(uniq)
    monitor_note = _monitor_task_id_note(monitor_tool)
    read_note = (
        f"\n- If output is large, use {read_tool} on the output_file path"
        if read_tool
        else ""
    )
    return (
        f"Get output and status from a background task{target_suffix}.\n\n"
        f"Usage notes:\n"
        f"- Pass task_ids with one or more ids from {sources_s}{monitor_note}; "
        f"for a single task use a one-element array. Multiple ids with a positive "
        f"timeout_ms wait until all complete\n"
        f"- Omit timeout_ms or pass 0 for a non-blocking status snapshot; set a "
        f"positive timeout_ms to wait up to that many milliseconds, capped at ~10 min\n"
        f"- Returns current output, status, and exit code if completed{read_note}"
    )


def build_kill_task_description(
    *,
    monitor_tool: Optional[str] = "monitor",
    subagent_present: bool = True,
    bash_present: bool = True,
    # CodeDoggy glue: no Job Object — honest taskkill / process-group wording.
    kill_action: Optional[str] = None,
) -> str:
    """Grok build_kill_task_description with portable kill verb (not Job Object)."""
    monitor_present = monitor_tool is not None
    target_suffix = _lifecycle_target_suffix(
        monitor_present=monitor_present, subagent_present=subagent_present
    )
    monitor_note = _monitor_task_id_note(monitor_tool)

    if kill_action is not None:
        action = kill_action
    elif bash_present:
        # Honest Windows: taskkill /F /T; Unix: process group SIGTERM/SIGKILL.
        # Grok Windows says "Terminates the Job Object" — we document that as X.
        parts = [
            "Terminates the process tree of a bash task"
            " (Windows: taskkill /T; Unix: SIGTERM/SIGKILL to process group)"
        ]
        if monitor_present:
            parts[0] = parts[0].replace("a bash task", "a bash task or monitor")
        if subagent_present:
            parts.append("; sends Cancel+Shutdown to a subagent")
        action = "".join(parts)
    elif subagent_present:
        action = "Sends Cancel+Shutdown to a subagent"
    elif monitor_present:
        action = (
            "Terminates the process tree of a monitor "
            "(Windows: taskkill /T; Unix: SIGTERM/SIGKILL to process group)"
        )
    else:
        action = ""

    return (
        f"Terminate a running background task{target_suffix}.\n\n"
        f"Usage notes:\n"
        f"- Pass its task_id{monitor_note}.\n"
        f"- {action}.\n"
        f"- Returns success if the task was killed or had already exited."
    )


def build_wait_tasks_description(
    *,
    background_retrieval_tool: str = "get_command_or_subagent_output",
    bash_background_param: Optional[str] = "background",
    subagent_background_param: Optional[str] = "background",
) -> str:
    """Grok build_wait_tasks_description."""
    sources: list[str] = []
    if bash_background_param:
        sources.append(f"{bash_background_param}=true")
    if subagent_background_param:
        sources.append(f"{subagent_background_param}=true")
    # de-dupe (product renames collapse is_background / run_in_background → background)
    seen_src: set[str] = set()
    uniq: list[str] = []
    for s in sources:
        if s not in seen_src:
            seen_src.add(s)
            uniq.append(s)
    sources_s = " or ".join(uniq)
    return (
        f"Wait for multiple background tasks or subagents to complete.\n\n"
        f"Prefer {background_retrieval_tool} with task_ids and a positive timeout_ms. "
        f"This tool is kept for compatibility.\n\n"
        f"Usage notes:\n"
        f"- task_ids: list of task IDs from {sources_s}\n"
        f"- mode: 'wait_all' or 'wait_any'\n"
        f"- timeout_ms: optional max wait, default 30s, capped at ~10 min"
    )


# ── result cards (output.rs to_prompt_format) ────────────────────────


def _rfc3339(ts: float | None = None) -> str:
    t = time.time() if ts is None else ts
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class TaskOutputResult:
    """Grok TaskOutputResult subset used for model-facing cards."""

    task_id: str
    command: str
    status: str
    exit_code: Optional[int]
    started: str
    ended: Optional[str]
    duration_secs: float
    output: str
    output_file: str
    truncated: bool
    truncation_hint: str = ""
    raw_output_bytes: int = 0

    def is_terminal(self) -> bool:
        return self.status in {"completed", "failed", "cancelled"}


def status_from_snapshot(
    *,
    completed: bool,
    explicitly_killed: bool,
    exit_code: Optional[int],
) -> str:
    """snapshot_to_result status mapping."""
    if not completed:
        return "running"
    if explicitly_killed:
        return "cancelled"
    if exit_code == 0:
        return "completed"
    return "failed"


def snapshot_to_result(
    snap: Any,
    *,
    read_file_name: str = "read_file",
    max_output_chars: int = 40_000,
) -> TaskOutputResult:
    """Convert TaskSnapshot-like object → TaskOutputResult (snapshot_to_result)."""
    raw = snap.output or ""
    raw_bytes = len(raw.encode("utf-8", errors="replace"))
    truncated = bool(getattr(snap, "truncated", False))
    out = raw
    if len(out) > max_output_chars:
        # Grok uses truncate_with_preview; keep a simple head + marker.
        head = max_output_chars // 2
        tail = max_output_chars // 4
        out = (
            out[:head]
            + f"\n\n[Output truncated — showing first {head} and last {tail} chars. "
            f"Use {read_file_name} on {snap.output_file} for full content]\n\n"
            + out[-tail:]
        )
        truncated = True
    truncation_hint = (
        f"[truncated - use {read_file_name} on output_file for full content]"
    )
    cmd = getattr(snap, "display_command", None) or snap.command
    started = _rfc3339(snap.start_time)
    ended = _rfc3339(snap.end_time) if snap.end_time is not None else None
    return TaskOutputResult(
        task_id=snap.task_id,
        command=cmd,
        status=status_from_snapshot(
            completed=bool(snap.completed),
            explicitly_killed=bool(getattr(snap, "explicitly_killed", False)),
            exit_code=snap.exit_code,
        ),
        exit_code=snap.exit_code,
        started=started,
        ended=ended,
        duration_secs=float(snap.duration_secs()),
        output=out,
        output_file=str(snap.output_file or ""),
        truncated=truncated,
        truncation_hint=truncation_hint,
        raw_output_bytes=raw_bytes,
    )


def format_task_result_card(r: TaskOutputResult) -> str:
    """ToolOutput::TaskOutput Result → model text (output.rs)."""
    lines = [
        f"=== Task {r.task_id} ===",
        f"Command: {r.command}",
        f"Status: {r.status}",
        f"Started: {r.started}",
    ]
    if r.ended is not None:
        lines.append(f"Ended: {r.ended}")
    lines.append(f"Duration: {r.duration_secs:.2f}s")
    if r.exit_code is not None:
        lines.append(f"Exit Code: {r.exit_code}")
    lines.append(f"Output File: {r.output_file}")
    lines.append("")
    lines.append("=== Output ===")
    if not r.output:
        lines.append("(no output yet)")
    else:
        lines.append(r.output)
    if r.truncated and r.truncation_hint:
        lines.append(r.truncation_hint)
    return "\n".join(lines)


def not_found_result(task_id: str) -> TaskOutputResult:
    """Grok not_found_result."""
    return TaskOutputResult(
        task_id=task_id,
        command="",
        status="not_found",
        exit_code=None,
        started="",
        ended=None,
        duration_secs=0.0,
        output=f"Task {task_id} not found.",
        output_file="",
        truncated=False,
    )


def format_multi_result(
    results: list[TaskOutputResult],
    *,
    mode: str,
) -> str:
    """ToolOutput::TaskOutput MultiResult → model text (output.rs)."""
    completed_count = sum(1 for r in results if r.is_terminal())
    total = len(results)
    summary = f"{completed_count}/{total} tasks completed ({mode})"
    lines = [f"=== Multi-wait ({mode}) ==="]
    for r in results:
        lines.append(
            f"--- Task {r.task_id} [{r.status}] ---\n"
            f"Command: {r.command}\n"
            f"Duration: {r.duration_secs:.2f}s"
        )
        if r.exit_code is not None:
            lines.append(f"Exit Code: {r.exit_code}")
        if r.output:
            lines.append(r.output)
    lines.append(f"\n{summary}")
    return "\n".join(lines)


def format_kill_result(*, outcome: str, message: str) -> str:
    """KillTask Result → '{outcome}: {message}'."""
    return f"{outcome}: {message}"


def kill_not_found_message(task_id: str, known_bash_ids: list[str]) -> str:
    """Current (non-legacy) kill_task not-found discoverability text."""
    if not known_bash_ids:
        return (
            f"Task or subagent {task_id} not found. No background tasks or "
            "subagents exist in this session."
        )
    return (
        f"Task or subagent {task_id} not found. Known bash task IDs: "
        f"[{', '.join(known_bash_ids)}]"
    )


def task_output_not_found_message(task_id: str, known_ids: list[str]) -> str:
    """Current get_task_output not-found discoverability text."""
    if not known_ids:
        return (
            f"Task {task_id} not found. No background tasks or subagents "
            "exist in this session."
        )
    return f"Task {task_id} not found. Known task IDs: [{', '.join(known_ids)}]"


def format_resume_footer(
    subagent_id: str,
    subagent_type: str,
    persona: Optional[str] = None,
) -> str:
    """Grok format_resume_footer."""
    footer = (
        f"<subagent_result>\n"
        f"subagent_id: {subagent_id}\n"
        f"subagent_type: {subagent_type}\n"
        f'To continue this subagent\'s conversation, use resume_from="{subagent_id}".'
    )
    if persona:
        footer += (
            f'\nThe subagent used persona="{persona}". '
            "Pass the same persona when resuming."
        )
    footer += "\n</subagent_result>"
    return footer


def format_subagent_card(snap: Any) -> TaskOutputResult:
    """format_subagent_snapshot → TaskOutputResult (subset for available fields)."""
    sid = getattr(snap, "subagent_id", "?")
    stype = getattr(snap, "subagent_type", "") or ""
    status = getattr(snap, "status", "unknown") or "unknown"
    desc = getattr(snap, "description", "") or ""
    duration_ms = int(getattr(snap, "duration_ms", 0) or 0)
    duration_secs = duration_ms / 1000.0
    started_at = float(getattr(snap, "started_at", 0) or 0)
    started = _rfc3339(started_at) if started_at else ""
    command = f"[subagent:{stype}] {desc}"
    tool_calls = int(getattr(snap, "tool_calls", 0) or 0)
    turns = int(getattr(snap, "turns", 0) or 0)
    worktree = getattr(snap, "worktree_path", None)
    persona = getattr(snap, "persona", None) or (
        (getattr(snap, "metadata", None) or {}).get("persona")
        if isinstance(getattr(snap, "metadata", None), dict)
        else None
    )
    exit_code: Optional[int] = None
    ended: Optional[str] = None
    output: str

    if status in {"pending", "initializing"}:
        output = (
            "Subagent is initializing (creating worktree, resolving config).\n"
            f"Type: {stype}\n"
            f"Description: {desc}\n"
            f"Elapsed: {duration_secs:.1f}s\n\n"
            "Use timeout_ms to wait for completion."
        )
        status = "initializing" if status == "pending" else status
    elif status == "running":
        output = (
            "Subagent is still running.\n"
            f"Type: {stype}\n"
            f"Description: {desc}\n"
            f"Elapsed: {duration_secs:.1f}s\n"
            f"Progress: {tool_calls} tool calls, {turns} turns\n\n"
            "Use timeout_ms to wait for completion."
        )
    elif status == "completed":
        body = getattr(snap, "output", None) or ""
        output = (
            f"{body}\n\n"
            f"<subagent_meta>id={sid}, type={stype}, tool_calls={tool_calls}, "
            f"turns={turns}, duration_ms={duration_ms}</subagent_meta>"
        )
        if worktree:
            output += f"\n<worktree_path>{worktree}</worktree_path>"
        output += "\n\n" + format_resume_footer(sid, stype, persona)
        exit_code = 0
        if started_at:
            ended = _rfc3339(started_at + duration_ms / 1000.0)
    elif status == "failed":
        output = str(getattr(snap, "error", None) or getattr(snap, "output", None) or "")
        exit_code = 1
        if started_at:
            ended = _rfc3339(started_at + duration_ms / 1000.0)
    elif status == "cancelled":
        output = str(
            getattr(snap, "error", None)
            or getattr(snap, "output", None)
            or "Subagent was cancelled"
        )
        if started_at:
            ended = _rfc3339(started_at + duration_ms / 1000.0)
    else:
        output = str(getattr(snap, "output", None) or getattr(snap, "error", None) or "")

    return TaskOutputResult(
        task_id=str(sid),
        command=command,
        status=status,
        exit_code=exit_code,
        started=started,
        ended=ended,
        duration_secs=duration_secs,
        output=output,
        output_file="",
        truncated=False,
        raw_output_bytes=len(output.encode("utf-8", errors="replace")),
    )
