"""Subagent coordinator — child Session + summary fold-back (Grok subagent).

Subagent = full child session (not a function call). Independent context;
ends with a concise summary returned to the parent.

Isolation:
  - none: child shares parent cwd (edits touch parent tree)
  - worktree: git worktree under ``.codedoggy/worktrees/<id>/``

Resume:
  - completed/failed/cancelled children can be resumed with a new prompt
  - reuses live transcript + worktree when preserved
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from codedoggy.orchestration.agent_def import (
    AgentDefinition,
    build_agent,
    resolve_agent_definition,
)
from codedoggy.orchestration.types import CapabilityMode, IsolationMode
from codedoggy.orchestration.worktree import (
    MergeResult,
    WorktreeError,
    WorktreeHandle,
    branch_for_subagent,
    create_worktree,
    merge_worktree_into_parent,
    reattach_worktree,
    should_cleanup_worktree,
)
from codedoggy.tools.registry import FinalizedToolset

logger = logging.getLogger(__name__)


@dataclass
class SubagentRequest:
    """Grok SubagentRequest (spawn)."""

    subagent_type: str
    prompt: str
    description: str = ""
    parent_session_id: str = ""
    parent_prompt_id: str | None = None
    run_in_background: bool | None = None  # None → definition default
    capability_mode: CapabilityMode | None = None
    isolation: IsolationMode = IsolationMode.NONE
    max_turns: int | None = None
    persona: str | None = None
    id: str = field(default_factory=lambda: f"sub_{uuid.uuid4().hex[:12]}")
    # Resume: continue prior child transcript / worktree
    resume_from: str | None = None
    prior_messages: list[Any] | None = None
    worktree_path: str | None = None
    worktree_branch: str | None = None
    system_prompt: str | None = None


@dataclass
class SubagentSnapshot:
    """Query result for a subagent (Grok coordinator lookup)."""

    subagent_id: str
    subagent_type: str
    status: str  # pending | running | completed | failed | cancelled
    description: str = ""
    output: str | None = None
    error: str | None = None
    tool_calls: int = 0
    turns: int = 0
    duration_ms: int = 0
    worktree_path: str | None = None
    started_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    # Resume payload (serialized message dicts)
    live_messages: list[dict[str, Any]] | None = None
    worktree_branch: str | None = None
    system_prompt: str | None = None
    resume_count: int = 0

    @property
    def is_running(self) -> bool:
        return self.status in {"pending", "running"}

    @property
    def can_resume(self) -> bool:
        # Grok task.rs: source must be *completed* (not running / unknown).
        # Fail-closed: failed/cancelled are not resumable by default.
        return self.status == "completed" and not self.is_running


@dataclass
class _Entry:
    request: SubagentRequest
    snapshot: SubagentSnapshot
    cancel: threading.Event = field(default_factory=threading.Event)
    future: Future | None = None
    # Durable resume state (updated after each run)
    live_messages: list[dict[str, Any]] = field(default_factory=list)
    worktree_path: str | None = None
    worktree_branch: str | None = None
    system_prompt: str | None = None
    resume_count: int = 0


class SubagentCoordinator:
    """Tracks child sessions (Grok SubagentCoordinator)."""

    def __init__(self, *, max_workers: int = 8) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="subagent")

    def lookup(self, subagent_id: str) -> SubagentSnapshot | None:
        with self._lock:
            e = self._entries.get(subagent_id)
            return None if e is None else _copy_snap(e.snapshot)

    def list_for_parent(self, parent_session_id: str) -> list[SubagentSnapshot]:
        with self._lock:
            return [
                _copy_snap(e.snapshot)
                for e in self._entries.values()
                if e.request.parent_session_id == parent_session_id
            ]

    def cancel(self, subagent_id: str) -> bool:
        with self._lock:
            e = self._entries.get(subagent_id)
            if e is None:
                return False
            e.cancel.set()
            e.snapshot.status = "cancelled"
            return True

    def spawn(
        self,
        request: SubagentRequest,
        *,
        run_fn: Callable[[SubagentRequest, threading.Event], SubagentSnapshot],
    ) -> SubagentSnapshot:
        """Register + schedule child. Returns initial snapshot (may still be running)."""
        # Resume path: reuse id and prior state
        if request.resume_from:
            return self.resume(
                request.resume_from,
                request.prompt,
                run_fn=run_fn,
                description=request.description,
                run_in_background=request.run_in_background,
                max_turns=request.max_turns,
            )

        definition = resolve_agent_definition(request.subagent_type)
        if definition is None:
            snap = SubagentSnapshot(
                subagent_id=request.id,
                subagent_type=request.subagent_type,
                status="failed",
                description=request.description,
                error=f"Unknown subagent type: {request.subagent_type}",
            )
            with self._lock:
                self._entries[request.id] = _Entry(request=request, snapshot=snap)
            return _copy_snap(snap)

        bg = (
            request.run_in_background
            if request.run_in_background is not None
            else definition.background
        )
        start = time.time()
        snap = SubagentSnapshot(
            subagent_id=request.id,
            subagent_type=request.subagent_type,
            status="pending",
            description=request.description or definition.description,
            started_at=start,
        )
        entry = _Entry(request=request, snapshot=snap)
        with self._lock:
            self._entries[request.id] = entry

        return self._schedule(entry, request, run_fn=run_fn, background=bg, start=start)

    def resume(
        self,
        subagent_id: str,
        prompt: str,
        *,
        run_fn: Callable[[SubagentRequest, threading.Event], SubagentSnapshot],
        description: str = "",
        run_in_background: bool | None = None,
        max_turns: int | None = None,
        parent_session_id: str | None = None,
        subagent_type: str | None = None,
    ) -> SubagentSnapshot:
        """Continue a *completed* subagent (Grok ``resume_from``).

        Source: ``xai-tool-types/task.rs`` + ``subagent/handle_request.rs``
          - source must be completed (not running)
          - same parent session
          - same subagent_type
          - inherits raw transcript; new prompt appended as next user turn
          - worktree/cwd inherited when present
        """
        prompt = (prompt or "").strip()
        if not prompt:
            return SubagentSnapshot(
                subagent_id=subagent_id,
                subagent_type="?",
                status="failed",
                error="resume requires a non-empty prompt",
            )
        with self._lock:
            e = self._entries.get(subagent_id)
            if e is None:
                return SubagentSnapshot(
                    subagent_id=subagent_id,
                    subagent_type="?",
                    status="failed",
                    error=(
                        f"Cannot resume from subagent '{subagent_id}': not found. "
                        "The subagent may have been evicted or the ID is invalid."
                    ),
                )
            if e.snapshot.is_running:
                return SubagentSnapshot(
                    subagent_id=subagent_id,
                    subagent_type=e.snapshot.subagent_type,
                    status="failed",
                    error=(
                        f"Cannot resume from subagent '{subagent_id}': it is still running. "
                        "Wait for it to complete before resuming."
                    ),
                    metadata=dict(e.snapshot.metadata),
                )
            if e.snapshot.status != "completed":
                return SubagentSnapshot(
                    subagent_id=subagent_id,
                    subagent_type=e.snapshot.subagent_type,
                    status="failed",
                    error=(
                        f"Cannot resume from subagent '{subagent_id}': status is "
                        f"{e.snapshot.status!r} (must be completed)."
                    ),
                    metadata=dict(e.snapshot.metadata),
                )
            # Same parent session (Grok identity check)
            if (
                parent_session_id is not None
                and e.request.parent_session_id
                and parent_session_id != e.request.parent_session_id
            ):
                return SubagentSnapshot(
                    subagent_id=subagent_id,
                    subagent_type=e.snapshot.subagent_type,
                    status="failed",
                    error="Cannot resume: subagent belongs to a different parent session.",
                )
            # Same subagent_type when caller specifies it
            if (
                subagent_type is not None
                and subagent_type.strip()
                and subagent_type.strip().lower() != e.request.subagent_type.lower()
            ):
                return SubagentSnapshot(
                    subagent_id=subagent_id,
                    subagent_type=e.snapshot.subagent_type,
                    status="failed",
                    error=(
                        f"Cannot resume: subagent_type mismatch "
                        f"(source={e.request.subagent_type!r}, requested={subagent_type!r})."
                    ),
                )
            # Build resume request carrying prior state
            prior = list(e.live_messages or e.snapshot.live_messages or [])
            req = SubagentRequest(
                subagent_type=e.request.subagent_type,
                prompt=prompt,
                description=description or e.request.description,
                parent_session_id=e.request.parent_session_id,
                parent_prompt_id=e.request.parent_prompt_id,
                run_in_background=run_in_background
                if run_in_background is not None
                else e.request.run_in_background,
                capability_mode=e.request.capability_mode,
                isolation=e.request.isolation,
                max_turns=max_turns if max_turns is not None else e.request.max_turns,
                persona=e.request.persona,
                id=subagent_id,
                resume_from=subagent_id,
                prior_messages=prior,
                worktree_path=e.worktree_path or e.snapshot.worktree_path,
                worktree_branch=e.worktree_branch or e.snapshot.worktree_branch,
                system_prompt=e.system_prompt or e.snapshot.system_prompt,
            )
            e.request = req
            e.resume_count += 1
            e.cancel = threading.Event()
            start = time.time()
            e.snapshot = SubagentSnapshot(
                subagent_id=subagent_id,
                subagent_type=req.subagent_type,
                status="pending",
                description=req.description,
                started_at=start,
                worktree_path=req.worktree_path,
                worktree_branch=req.worktree_branch,
                system_prompt=req.system_prompt,
                live_messages=prior,
                resume_count=e.resume_count,
                metadata={"resumed": True, "resume_count": e.resume_count},
            )
            entry = e
            bg = (
                req.run_in_background
                if req.run_in_background is not None
                else True
            )
            definition = resolve_agent_definition(req.subagent_type)
            if definition is not None and req.run_in_background is None:
                bg = definition.background

        return self._schedule(entry, req, run_fn=run_fn, background=bool(bg), start=start)

    def merge_worktree(
        self,
        subagent_id: str,
        parent_cwd: Path | str,
        *,
        strategy: str = "merge",
        commit_message: str | None = None,
        cleanup_worktree: bool = False,
        delete_branch: bool = False,
    ) -> MergeResult:
        """Merge a completed subagent's worktree branch into the parent repo."""
        with self._lock:
            e = self._entries.get(subagent_id)
            if e is None:
                return MergeResult(
                    ok=False,
                    message=f"Unknown subagent id: {subagent_id}",
                )
            branch = e.worktree_branch or e.snapshot.worktree_branch or branch_for_subagent(
                subagent_id
            )
            wt_path = e.worktree_path or e.snapshot.worktree_path
        result = merge_worktree_into_parent(
            Path(parent_cwd),
            branch=branch,
            worktree_path=wt_path,
            subagent_id=subagent_id,
            strategy=strategy,
            commit_message=commit_message,
            cleanup_worktree=cleanup_worktree,
            delete_branch=delete_branch,
        )
        if result.ok and cleanup_worktree:
            with self._lock:
                e = self._entries.get(subagent_id)
                if e is not None:
                    e.worktree_path = None
                    e.snapshot.worktree_path = None
                    e.snapshot.metadata["worktree_merged"] = True
                    e.snapshot.metadata["merge_commit"] = result.commit
        return result

    def wait(
        self,
        subagent_id: str,
        *,
        timeout_ms: int | None = 30_000,
    ) -> SubagentSnapshot | None:
        with self._lock:
            e = self._entries.get(subagent_id)
            if e is None:
                return None
            fut = e.future
        if fut is None:
            return self.lookup(subagent_id)
        timeout = None if timeout_ms is None else max(0.0, timeout_ms / 1000.0)
        try:
            fut.result(timeout=timeout)
        except Exception:  # noqa: BLE001
            pass
        return self.lookup(subagent_id)

    def spawn_many(
        self,
        requests: list[SubagentRequest],
        *,
        run_fn: Callable[[SubagentRequest, threading.Event], SubagentSnapshot],
    ) -> list[SubagentSnapshot]:
        """Register and schedule many children for true parallel fan-out.

        Each request is forced to ``run_in_background=True`` so the pool runs
        them concurrently; call :meth:`wait_all` (or the tool) to join.
        """
        snaps: list[SubagentSnapshot] = []
        for req in requests:
            # Force background so spawn returns immediately and children overlap.
            req.run_in_background = True
            snaps.append(self.spawn(req, run_fn=run_fn))
        return snaps

    def wait_all(
        self,
        subagent_ids: list[str],
        *,
        timeout_ms: int | None = 120_000,
    ) -> list[SubagentSnapshot]:
        """Wait until every listed subagent leaves pending/running (or budget expires)."""
        if not subagent_ids:
            return []
        deadline = None if timeout_ms is None else time.time() + max(0.0, timeout_ms / 1000.0)
        out: list[SubagentSnapshot] = []
        for sid in subagent_ids:
            remaining: int | None
            if deadline is None:
                remaining = None
            else:
                remaining = max(0, int((deadline - time.time()) * 1000))
            snap = self.wait(sid, timeout_ms=remaining)
            if snap is None:
                out.append(
                    SubagentSnapshot(
                        subagent_id=sid,
                        subagent_type="?",
                        status="failed",
                        error=f"Unknown subagent id: {sid}",
                    )
                )
            else:
                out.append(snap)
        return out

    def shutdown(self, wait: bool = False) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=True)

    def _schedule(
        self,
        entry: _Entry,
        request: SubagentRequest,
        *,
        run_fn: Callable[[SubagentRequest, threading.Event], SubagentSnapshot],
        background: bool,
        start: float,
    ) -> SubagentSnapshot:
        def _job() -> SubagentSnapshot:
            with self._lock:
                entry.snapshot.status = "running"
            try:
                result = run_fn(request, entry.cancel)
            except Exception as e:  # noqa: BLE001
                logger.exception("subagent failed id=%s", request.id)
                result = SubagentSnapshot(
                    subagent_id=request.id,
                    subagent_type=request.subagent_type,
                    status="failed",
                    description=request.description,
                    error=f"{type(e).__name__}: {e}",
                    duration_ms=int((time.time() - start) * 1000),
                    started_at=start,
                )
            with self._lock:
                # Persist resume state from result
                entry.snapshot = result
                if result.live_messages is not None:
                    entry.live_messages = list(result.live_messages)
                if result.worktree_path:
                    entry.worktree_path = result.worktree_path
                if result.worktree_branch:
                    entry.worktree_branch = result.worktree_branch
                if result.system_prompt:
                    entry.system_prompt = result.system_prompt
                entry.resume_count = max(entry.resume_count, result.resume_count)
            return result

        fut = self._pool.submit(_job)
        entry.future = fut

        if not background:
            try:
                return fut.result()
            except Exception as e:  # noqa: BLE001
                return SubagentSnapshot(
                    subagent_id=request.id,
                    subagent_type=request.subagent_type,
                    status="failed",
                    error=str(e),
                    started_at=start,
                    duration_ms=int((time.time() - start) * 1000),
                )
        return _copy_snap(entry.snapshot)


def _copy_snap(s: SubagentSnapshot) -> SubagentSnapshot:
    return SubagentSnapshot(
        subagent_id=s.subagent_id,
        subagent_type=s.subagent_type,
        status=s.status,
        description=s.description,
        output=s.output,
        error=s.error,
        tool_calls=s.tool_calls,
        turns=s.turns,
        duration_ms=s.duration_ms,
        worktree_path=s.worktree_path,
        started_at=s.started_at,
        metadata=dict(s.metadata),
        live_messages=list(s.live_messages) if s.live_messages else None,
        worktree_branch=s.worktree_branch,
        system_prompt=s.system_prompt,
        resume_count=s.resume_count,
    )


def make_child_runner(
    *,
    parent_cwd: Path,
    parent_tools: FinalizedToolset,
    parent_sampler: Any,
    parent_system_prompt: str | None,
    parent_session: Any = None,
    context_compactor_factory: Callable[[], Any] | None = None,
) -> Callable[[SubagentRequest, threading.Event], SubagentSnapshot]:
    """Build the function that actually runs a child agentic loop."""

    def run(request: SubagentRequest, cancel: threading.Event) -> SubagentSnapshot:
        from codedoggy.turn.loop import run_agent_loop
        from codedoggy.turn.types import Message, Role

        start = time.time()
        definition = resolve_agent_definition(request.subagent_type)
        if definition is None:
            return SubagentSnapshot(
                subagent_id=request.id,
                subagent_type=request.subagent_type,
                status="failed",
                error=f"Unknown subagent type: {request.subagent_type}",
            )

        isolation = request.isolation or definition.isolation
        if isolation is IsolationMode.NONE and definition.isolation is not IsolationMode.NONE:
            isolation = definition.isolation
        definition = AgentDefinition(
            name=definition.name,
            description=definition.description,
            tools=list(definition.tools),
            capability_mode=request.capability_mode or definition.capability_mode,
            isolation=isolation,
            prompt_mode=definition.prompt_mode,
            system_prompt_body=definition.system_prompt_body,
            session_mode=definition.session_mode,
            max_turns=request.max_turns if request.max_turns is not None else definition.max_turns,
            background=definition.background,
        )

        # Resume may pin a prior system prompt (frozen at first spawn)
        base_prompt = request.system_prompt or parent_system_prompt
        agent = build_agent(
            definition,
            parent_tools=parent_tools,
            base_system_prompt=base_prompt if request.system_prompt is None else None,
        )
        system_prompt = request.system_prompt or agent.system_prompt
        child_tools = _strip_nested_spawn(agent.tools)

        compactor = None
        if context_compactor_factory is not None:
            try:
                compactor = context_compactor_factory()
            except Exception:  # noqa: BLE001
                logger.debug("child compactor factory failed", exc_info=True)

        max_turns = definition.max_turns
        if max_turns is None and parent_session is not None:
            max_turns = getattr(parent_session, "max_turns", None)

        def _cancelled() -> bool:
            return cancel.is_set()

        mode_state = None
        if definition.session_mode.value == "plan":
            from codedoggy.orchestration.session_mode import SessionModeState

            mode_state = SessionModeState()
            mode_state.enter_plan()

        child_cwd = Path(parent_cwd).resolve()
        wt: WorktreeHandle | None = None
        isolation_mode = definition.isolation
        is_resume = bool(request.resume_from or request.prior_messages)

        if isolation_mode is IsolationMode.WORKTREE:
            try:
                if is_resume and (request.worktree_path or request.worktree_branch):
                    wt = reattach_worktree(
                        child_cwd,
                        subagent_id=request.id,
                        branch=request.worktree_branch,
                        existing_path=request.worktree_path,
                    )
                else:
                    wt = create_worktree(child_cwd, subagent_id=request.id)
                child_cwd = wt.path
            except WorktreeError as e:
                return SubagentSnapshot(
                    subagent_id=request.id,
                    subagent_type=request.subagent_type,
                    status="failed",
                    description=request.description or definition.description,
                    error=f"worktree isolation failed: {e}",
                    duration_ms=int((time.time() - start) * 1000),
                    started_at=start,
                    metadata={"isolation": "worktree", "resumed": is_resume},
                    resume_count=1 if is_resume else 0,
                )

        # Product path: no Shadow/audit hooks on children.
        child_hooks = None

        tool_extra: dict[str, Any] = {
            "is_subagent": True,
            "subagent_id": request.id,
            "subagent_type": request.subagent_type,
            "session_mode_state": mode_state,
            "isolation": isolation_mode.value,
            "resumed": is_resume,
        }
        if wt is not None:
            tool_extra["worktree_path"] = str(wt.path)
            tool_extra["worktree_branch"] = wt.branch

        if parent_session is not None and wt is None:
            ext = getattr(parent_session, "extensions", None)
            pol = getattr(ext, "policy", None) if ext else None
            if pol is not None:
                tool_extra["policy"] = pol
        if wt is not None:
            try:
                from codedoggy.tools.policy import WorkspacePolicy

                tool_extra["policy"] = WorkspacePolicy(cwd=child_cwd)
            except Exception:  # noqa: BLE001
                logger.debug("child worktree policy bind failed", exc_info=True)

        prior = _hydrate_prior_messages(request.prior_messages)

        try:
            loop = run_agent_loop(
                user_text=request.prompt,
                sampler=parent_sampler,
                tools=child_tools,
                cwd=child_cwd,
                max_turns=max_turns,
                system_prompt=system_prompt,
                is_cancelled=_cancelled,
                hooks=child_hooks,
                # Keep session=None so child does not mutate parent phase/live.
                session=None,
                session_id=f"{request.parent_session_id}:{request.id}",
                tool_extra=tool_extra,
                context_compactor=compactor,
                prior_messages=prior,
            )
        finally:
            if wt is not None and should_cleanup_worktree() and not is_resume:
                # On resume we usually preserve; cleanup env still honored
                try:
                    wt.cleanup(force=True)
                except Exception:  # noqa: BLE001
                    logger.warning("worktree cleanup failed path=%s", wt.path, exc_info=True)

        summary = _summarize_child(loop.final_text, loop.messages, request)
        status = "completed" if loop.completed else (
            "cancelled" if loop.cancelled else ("failed" if loop.error else "completed")
        )
        if loop.max_turns_reached and not loop.completed:
            status = "completed"

        live = _serialize_messages(loop.messages)
        return SubagentSnapshot(
            subagent_id=request.id,
            subagent_type=request.subagent_type,
            status=status,
            description=request.description or definition.description,
            output=summary,
            error=loop.error,
            tool_calls=len(loop.tools_called),
            turns=loop.rounds,
            duration_ms=int((time.time() - start) * 1000),
            worktree_path=str(wt.path) if wt is not None else None,
            worktree_branch=wt.branch if wt is not None else request.worktree_branch,
            system_prompt=system_prompt,
            live_messages=live,
            resume_count=1 if is_resume else 0,
            started_at=start,
            metadata={
                "max_turns_reached": loop.max_turns_reached,
                "tools": list(loop.tools_called),
                "isolation": isolation_mode.value,
                "worktree_preserved": bool(
                    wt is not None and not should_cleanup_worktree()
                ),
                "resumed": is_resume,
                "message_count": len(live),
            },
        )

    return run


def format_parallel_dispatched(snaps: list[SubagentSnapshot]) -> str:
    """Notice after MAIN chose wait=false: children running; join is MAIN's next choice."""
    lines: list[str] = [
        f"## Parallel fan-out started ({len(snaps)} tasks running in background)",
        "",
        "You chose wait=false. Children are running. Continue whatever serial / other work "
        "you still own, then join when you need their results "
        "(wait_commands_or_subagents / get_command_or_subagent_output) and aggregate yourself.",
        "",
        "### Dispatched",
    ]
    ids: list[str] = []
    for i, s in enumerate(snaps, start=1):
        title = (s.description or s.subagent_type or "task").strip() or "task"
        ids.append(s.subagent_id)
        lines.append(
            f"{i}. `{s.subagent_id}` — {title} "
            f"(type={s.subagent_type}, status={s.status})"
        )
    lines.append("")
    lines.append(f"task_ids: {ids!r}")
    return "\n".join(lines).rstrip() + "\n"


def format_parallel_aggregate(snaps: list[SubagentSnapshot]) -> str:
    """Format multi-child results after MAIN chose to wait (or after join)."""
    lines: list[str] = [
        f"## Parallel fan-out complete ({len(snaps)} tasks)",
        "",
        "Child reports below. As MAIN, synthesise the final user-facing answer "
        "(include any work you did yourself). Retry a child only if it failed and you need it.",
        "",
    ]
    completed = failed = running = 0
    for i, s in enumerate(snaps, start=1):
        status = (s.status or "?").lower()
        if status == "completed":
            completed += 1
        elif status in {"failed", "cancelled"}:
            failed += 1
        elif status in {"pending", "running"}:
            running += 1
        title = (s.description or s.subagent_type or "task").strip() or "task"
        lines.append(f"### [{i}] {title} (`{s.subagent_id}`) — {status}")
        lines.append(f"- type: {s.subagent_type}")
        if s.duration_ms:
            lines.append(f"- duration_ms: {s.duration_ms}")
        if s.error:
            lines.append(f"- error: {s.error}")
        body = (s.output or "").strip()
        if body:
            lines.append("")
            lines.append(body)
        else:
            lines.append("")
            lines.append("(no output)")
        lines.append("")
    lines.append(
        f"## Counts: completed={completed} failed={failed} "
        f"still_running={running} total={len(snaps)}"
    )
    return "\n".join(lines).rstrip() + "\n"


def _strip_nested_spawn(tools: FinalizedToolset) -> FinalizedToolset:
    """Children must not spawn further subagents (Grok default)."""
    from codedoggy.tools.kinds import ToolKind

    deny = {
        "task",
        "spawn_subagent",
        "spawn_agent",
        "parallel_tasks",
    }
    by = {}
    for name, ft in tools.by_client_name.items():
        short = getattr(ft, "short_id", None) or ""
        if name in deny or short in deny:
            continue
        if getattr(ft, "kind", None) is ToolKind.Task:
            continue
        by[name] = ft
    if len(by) == len(tools.by_client_name):
        return tools
    return FinalizedToolset(by_client_name=by)


def _serialize_messages(messages: list) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages or []:
        role = getattr(m, "role", None)
        role_s = getattr(role, "value", None) or str(role or "user")
        if role_s == "system":
            continue  # system rebuilt each run
        d: dict[str, Any] = {
            "role": role_s,
            "content": getattr(m, "content", None),
        }
        name = getattr(m, "name", None)
        if name:
            d["name"] = name
        tc_id = getattr(m, "tool_call_id", None)
        if tc_id:
            d["tool_call_id"] = tc_id
        tcs = getattr(m, "tool_calls", None)
        if tcs:
            d["tool_calls"] = [
                {
                    "id": getattr(tc, "id", ""),
                    "name": getattr(tc, "name", ""),
                    "arguments": getattr(tc, "arguments", {}) or {},
                }
                for tc in tcs
            ]
        out.append(d)
    return out


def _hydrate_prior_messages(prior: list[Any] | None) -> list | None:
    if not prior:
        return None
    from codedoggy.turn.types import Message, Role, ToolCall

    out: list[Message] = []
    for item in prior:
        if isinstance(item, Message):
            if item.role is Role.SYSTEM:
                continue
            out.append(item)
            continue
        if not isinstance(item, dict):
            continue
        role_s = str(item.get("role") or "user")
        if role_s == "system":
            continue
        try:
            role = Role(role_s)
        except ValueError:
            role = Role.USER
        tcs = None
        raw_tcs = item.get("tool_calls")
        if isinstance(raw_tcs, list) and raw_tcs:
            tcs = [
                ToolCall(
                    id=str(tc.get("id") or ""),
                    name=str(tc.get("name") or ""),
                    arguments=tc.get("arguments") or {},
                )
                for tc in raw_tcs
                if isinstance(tc, dict)
            ]
        out.append(
            Message(
                role=role,
                content=item.get("content"),
                name=item.get("name"),
                tool_call_id=item.get("tool_call_id"),
                tool_calls=tcs,
            )
        )
    return out or None


def _summarize_child(final_text: str | None, messages: list, request: SubagentRequest) -> str:
    """Fold child result for parent (Grok: independent context, summary back)."""
    body = (final_text or "").strip()
    if not body:
        for msg in reversed(messages or []):
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None) or ""
            if str(getattr(role, "value", role)) == "assistant" and content.strip():
                body = content.strip()
                break
    if not body:
        body = "(subagent finished with no text)"
    resume_tag = " resume" if request.resume_from or request.prior_messages else ""
    header = (
        f"[subagent:{request.subagent_type} id={request.id}{resume_tag}"
        + (f" — {request.description}" if request.description else "")
        + "]\n"
    )
    max_chars = 8_000
    if len(body) > max_chars:
        body = body[: max_chars - 20] + "\n…[truncated]"
    return header + body
