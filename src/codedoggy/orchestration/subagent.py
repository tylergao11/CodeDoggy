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

import json
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, wait as wait_futures
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
    # Grok TaskTool model pin — child sampler uses this model when set.
    model: str | None = None
    # Explicit child working directory (mutually exclusive with worktree isolation).
    cwd: str | None = None
    # Nesting depth of the *parent* that is spawning (MAIN = 0).
    spawn_depth: int = 0


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


@dataclass
class _ChildToolResources:
    """Child-local mutable state plus parent-owned Grok runtime handles."""

    task_manager: Any
    scheduler: Any
    todo_state: Any
    session_mode_state: Any = None
    goal_log: list[dict[str, Any]] = field(default_factory=list)
    goal_blocked_streak: int = 0
    goal_active: bool = False
    goal_completed: bool = False
    goal_blocked: bool = False
    goal_blocked_reason: str | None = None
    goal_completion_message: str | None = None

    def enter_plan_mode(self, plan_file: str | None = None) -> None:
        if self.session_mode_state is None:
            from codedoggy.orchestration.session_mode import SessionModeState

            self.session_mode_state = SessionModeState()
        self.session_mode_state.enter_plan(plan_file)

    def exit_plan_mode(self, *, approved: bool = True) -> None:
        if self.session_mode_state is not None:
            self.session_mode_state.exit_plan(approved=approved)

    def enter_goal_mode(self) -> None:
        if self.session_mode_state is None:
            from codedoggy.orchestration.session_mode import SessionModeState

            self.session_mode_state = SessionModeState()
        self.session_mode_state.enter_goal()
        self.goal_active = True

    def exit_goal_mode(self) -> None:
        if self.session_mode_state is not None:
            exit_goal = getattr(self.session_mode_state, "exit_goal", None)
            if callable(exit_goal):
                exit_goal(reason="exit")
        self.goal_active = False


class SubagentCoordinator:
    """Tracks child sessions (Grok SubagentCoordinator)."""

    def __init__(self, *, max_workers: int = 8) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="subagent")
        self._closed = False
        # Mid-run live subscribers: cb(snapshot, message|None) — may run on worker threads.
        self._listeners: list[Callable[[SubagentSnapshot, Any], None]] = []

    def add_listener(
        self, callback: Callable[[SubagentSnapshot, Any], None]
    ) -> None:
        """Subscribe to mid-run live message publishes (and final snapshot writes)."""
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def remove_listener(
        self, callback: Callable[[SubagentSnapshot, Any], None]
    ) -> None:
        with self._lock:
            # Bound-method objects are recreated on attribute access.  Equality
            # matches the same instance+function pair; identity leaks TUI
            # listeners forever across repeated attach/detach cycles.
            self._listeners = [c for c in self._listeners if c != callback]

    def publish_live_message(self, subagent_id: str, message: Any) -> None:
        """Append one live transcript message while a child is still running.

        Called from the child ``on_archive_message`` path so the parent TUI can
        update without waiting for the child future to finish.
        """
        with self._lock:
            entry = self._entries.get(subagent_id)
            if entry is None:
                return
            serialized = _serialize_messages([message])
            if serialized:
                entry.live_messages.append(serialized[0])
                entry.snapshot.live_messages = list(entry.live_messages)
            if entry.snapshot.status in {"pending", "running", ""}:
                entry.snapshot.status = "running"
            snap = _copy_snap(entry.snapshot)
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(snap, message)
            except Exception:  # noqa: BLE001
                logger.debug("subagent live listener failed", exc_info=True)

    def lookup(self, subagent_id: str) -> SubagentSnapshot | None:
        with self._lock:
            e = self._entries.get(subagent_id)
            return None if e is None else _copy_snap(e.snapshot)

    def todo_state_for(self, subagent_id: str) -> Any | None:
        """Child-local TodoState (never MAIN's). For TUI / resume inspect."""
        with self._lock:
            e = self._entries.get(subagent_id)
            if e is None:
                return None
            rs = getattr(e.request, "_child_runtime_state", None) or {}
            res = rs.get("resources")
            return getattr(res, "todo_state", None) if res is not None else None

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
            child_runtime_state = getattr(e.request, "_child_runtime_state", None)
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
            if child_runtime_state:
                setattr(req, "_child_runtime_state", child_runtime_state)
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

    def shutdown(
        self,
        wait: bool = False,
        *,
        cancel_running: bool = True,
        timeout_s: float | None = None,
    ) -> bool:
        """Close dispatch, signal running children, and optionally await them.

        Returns whether every known child future reached a terminal state inside
        the requested wait budget. The signature stays compatible with the
        existing ``shutdown(wait=False)`` host call.
        """
        with self._lock:
            self._closed = True
            entries = list(self._entries.values())
            if cancel_running:
                for entry in entries:
                    if entry.snapshot.is_running:
                        entry.cancel.set()
            futures = [entry.future for entry in entries if entry.future is not None]
        self._pool.shutdown(wait=False, cancel_futures=True)
        if not wait:
            return all(future.done() for future in futures)
        if not futures:
            return True
        _done, pending = wait_futures(
            futures,
            timeout=None if timeout_s is None else max(0.0, float(timeout_s)),
        )
        if not pending:
            self._pool.shutdown(wait=True, cancel_futures=True)
            return True
        return False

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
                snap = _copy_snap(entry.snapshot)
                listeners = list(self._listeners)
            # Terminal notify — same channel as live messages so TUI does not
            # wait on the 0.35s idle poll after MAIN exits.
            for cb in listeners:
                try:
                    cb(snap, None)
                except Exception:  # noqa: BLE001
                    logger.debug("subagent terminal listener failed", exc_info=True)
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


def resolve_subagent_model(
    subagent_type: str,
    *,
    explicit: str | None = None,
) -> str | None:
    """Resolve child model: explicit Task.model > per-type env > None (inherit).

    Env (Grok ``[subagents.models]`` spirit):
      CODEDOGGY_SUBAGENT_MODELS=explore=grok-3,general-purpose=grok-4.5
      CODEDOGGY_SUBAGENT_MODEL_EXPLORE=grok-3
    """
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    import os

    key = (subagent_type or "").strip().lower().replace("-", "_")
    if key:
        per = os.environ.get(f"CODEDOGGY_SUBAGENT_MODEL_{key.upper()}")
        if per and per.strip():
            return per.strip()
    raw = (os.environ.get("CODEDOGGY_SUBAGENT_MODELS") or "").strip()
    if not raw:
        return None
    # JSON object or comma map
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                for k, v in data.items():
                    if str(k).strip().lower().replace("-", "_") == key and str(v).strip():
                        return str(v).strip()
        except Exception:  # noqa: BLE001
            pass
    for part in raw.split(","):
        if "=" not in part:
            continue
        k, _, v = part.partition("=")
        if k.strip().lower().replace("-", "_") == key and v.strip():
            return v.strip()
    return None


def pin_sampler_model(sampler: Any, model: str) -> Any:
    """Return a ChatSampler whose client uses ``model`` (same provider/auth)."""
    if not model or not str(model).strip():
        return sampler
    client = getattr(sampler, "client", None)
    if client is None:
        return sampler
    cfg = getattr(client, "config", None) or getattr(client, "_config", None)
    if cfg is None:
        return sampler
    try:
        from dataclasses import replace

        from codedoggy.model.chat_sampler import ChatSampler

        new_cfg = replace(cfg, model=str(model).strip())
        profile = getattr(client, "profile", None) or getattr(client, "_profile", None)
        try:
            if profile is not None:
                new_client = type(client)(new_cfg, profile=profile)
            else:
                new_client = type(client)(new_cfg)
        except TypeError:
            new_client = type(client)(new_cfg)
        return ChatSampler(new_client)
    except Exception:  # noqa: BLE001
        logger.debug("pin_sampler_model failed model=%s", model, exc_info=True)
        return sampler


def make_child_runner(
    *,
    parent_cwd: Path,
    parent_tools: FinalizedToolset,
    parent_sampler: Any,
    parent_system_prompt: str | None,
    parent_session: Any = None,
    context_compactor_factory: Callable[[], Any] | None = None,
    parent_sampler_factory: Callable[[], Any] | None = None,
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

        # System prompt: Grok subagent_prompt base + role-instructions (product).
        # Do not inherit full MAIN prompt — matches Grok child prompt isolation.
        if request.system_prompt:
            system_prompt = request.system_prompt
            agent = build_agent(
                definition,
                parent_tools=parent_tools,
                base_system_prompt=None,
            )
        else:
            agent = build_agent(
                definition,
                parent_tools=parent_tools,
                base_system_prompt=None,
            )
            system_prompt = None  # filled after worktree cwd known
        parent_memory_manager = _parent_resource(parent_session, "memory_manager")
        child_depth = max(0, int(getattr(request, "spawn_depth", 0) or 0) + 1)
        from codedoggy.orchestration.subagent_policy import effective_max_subagent_depth

        allow_nested_spawn = child_depth < effective_max_subagent_depth()
        child_tools = _strip_child_private_tools(
            agent.tools,
            memory_manager=parent_memory_manager,
            allow_nested_spawn=allow_nested_spawn,
        )

        compactor = None
        if context_compactor_factory is not None:
            try:
                compactor = context_compactor_factory()
            except Exception:  # noqa: BLE001
                logger.debug("child compactor factory failed", exc_info=True)

        # One sampler per child: ChatSampler contains mutable tool-call id and
        # streaming state.  The factory also resolves the latest connection
        # generation, so children spawned after a TUI provider switch do not
        # stay pinned to the bootstrap client.
        runtime_sampler = parent_sampler
        if parent_sampler_factory is not None:
            try:
                runtime_sampler = parent_sampler_factory()
            except Exception:  # noqa: BLE001
                logger.debug("child sampler factory failed", exc_info=True)
        # Grok Task.model: pin child model (explicit or per-type env map).
        pinned_model = resolve_subagent_model(
            request.subagent_type, explicit=request.model
        )
        if pinned_model:
            runtime_sampler = pin_sampler_model(runtime_sampler, pinned_model)
        if compactor is not None:
            try:
                from codedoggy.turn.runner import _bind_compactor_model_window

                _bind_compactor_model_window(compactor, runtime_sampler)
            except Exception:  # noqa: BLE001
                logger.debug("child model-window bind failed", exc_info=True)

        max_turns = definition.max_turns
        if max_turns is None and parent_session is not None:
            max_turns = getattr(parent_session, "max_turns", None)

        def _cancelled() -> bool:
            return cancel.is_set()

        child_cwd = Path(parent_cwd).resolve()
        # Explicit cwd (schema-validated by Task tool) when not using worktree.
        if request.cwd and isolation is IsolationMode.NONE:
            try:
                override = Path(str(request.cwd)).expanduser().resolve()
                if override.is_dir():
                    child_cwd = override
            except Exception:  # noqa: BLE001
                logger.debug("child cwd override failed", exc_info=True)
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

        child_hooks = None
        visible_tools = _toolset_names(child_tools)
        runtime_state = dict(getattr(request, "_child_runtime_state", None) or {})
        prior_resources = runtime_state.get("resources")
        if not isinstance(prior_resources, _ChildToolResources):
            prior_resources = None

        # Grok child sessions reuse the parent's terminal backend + scheduler
        # handles. They must not create orphan actors that the parent cannot
        # poll, query, notify, or close. Fallbacks exist only for standalone
        # child-runner use (tests/embedders) and are closed below.
        parent_task_manager = _parent_resource(parent_session, "task_manager")
        owned_task_manager = None
        if parent_task_manager is None:
            from codedoggy.tools.task_manager import BackgroundTaskManager

            parent_task_manager = BackgroundTaskManager()
            owned_task_manager = parent_task_manager

        parent_scheduler = _parent_resource(parent_session, "scheduler")
        if parent_scheduler is None:
            prior_scheduler = (
                prior_resources.scheduler if prior_resources is not None else None
            )
            if prior_scheduler is not None:
                parent_scheduler = prior_scheduler
            else:
                from codedoggy.tools.scheduler import Scheduler

                parent_scheduler = Scheduler()

        from codedoggy.tools.grok_build.todo_logic import (
            TodoState,
            load_todo_state,
            save_todo_state,
        )

        # Grok: each subagent has its own TodoState. Never share MAIN's list —
        # only resume *this* child's prior resources (same subagent_id).
        todo_state = (
            prior_resources.todo_state if prior_resources is not None else None
        )
        if not isinstance(todo_state, TodoState):
            todo_state = TodoState()
        parent_todo = _parent_resource(parent_session, "todo_state")
        if parent_todo is not None and todo_state is parent_todo:
            todo_state = TodoState()
        # Disk resume under *parent* cwd (survive worktree cleanup).
        # Child session_id is ``parent:subagent_id`` (same as loop).
        child_todo_sid = f"{request.parent_session_id}:{request.id}"
        todo_disk_cwd = Path(parent_cwd).resolve()
        if todo_state.is_empty():
            try:
                disk_todo = load_todo_state(
                    cwd=todo_disk_cwd, session_id=child_todo_sid
                )
                if disk_todo is not None and not disk_todo.is_empty():
                    todo_state = disk_todo
            except Exception:  # noqa: BLE001
                logger.debug("load child todo_state failed", exc_info=True)

        mode_state = (
            prior_resources.session_mode_state
            if prior_resources is not None
            else None
        )
        # Fresh child: Inactive plan tracker (do not inherit parent plan gate).
        parent_mode = _parent_resource(parent_session, "session_mode_state")
        if mode_state is not None and parent_mode is not None and mode_state is parent_mode:
            mode_state = None
        if definition.session_mode.value == "plan" and mode_state is None:
            from codedoggy.orchestration.session_mode import SessionModeState

            mode_state = SessionModeState()
            mode_state.enter_plan()

        child_resources = _ChildToolResources(
            task_manager=parent_task_manager,
            scheduler=parent_scheduler,
            todo_state=todo_state,
            session_mode_state=mode_state,
            goal_log=list(prior_resources.goal_log) if prior_resources else [],
            goal_blocked_streak=(
                prior_resources.goal_blocked_streak if prior_resources else 0
            ),
            goal_active=prior_resources.goal_active if prior_resources else False,
            goal_completed=(
                prior_resources.goal_completed if prior_resources else False
            ),
            goal_blocked=prior_resources.goal_blocked if prior_resources else False,
            goal_blocked_reason=(
                prior_resources.goal_blocked_reason if prior_resources else None
            ),
            goal_completion_message=(
                prior_resources.goal_completion_message if prior_resources else None
            ),
        )

        parent_coord = _parent_resource(parent_session, "subagent_coordinator")
        parent_run_fn = _parent_resource(parent_session, "subagent_run_fn")
        tool_extra: dict[str, Any] = {
            "kernel": child_resources,
            "is_subagent": True,
            "subagent_id": request.id,
            "subagent_type": request.subagent_type,
            "parent_session_id": request.parent_session_id,
            "session_id": child_todo_sid,
            "platform": "subagent",
            "session_mode_state": mode_state,
            "task_manager": parent_task_manager,
            "scheduler": parent_scheduler,
            "todo_state": todo_state,
            "mutations": [],
            "isolation": isolation_mode.value,
            "resumed": is_resume,
            # Nesting: child Task.spawn sees this as its depth (MAIN = 0).
            "subagent_depth": child_depth,
            "subagent_coordinator": parent_coord if allow_nested_spawn else None,
            "subagent_run_fn": parent_run_fn if allow_nested_spawn else None,
        }
        # Media extras follow the parent's ActiveConnection (same login as MAIN).
        parent_connection = _parent_resource(parent_session, "connection")
        if parent_connection is not None:
            tool_extra["connection"] = parent_connection
        if wt is not None:
            tool_extra["worktree_path"] = str(wt.path)
            tool_extra["worktree_branch"] = wt.branch

        policy = _parent_resource(parent_session, "policy") if wt is None else None
        if policy is not None:
            tool_extra["policy"] = policy
        if wt is not None:
            try:
                from codedoggy.tools.policy import WorkspacePolicy

                policy = WorkspacePolicy(cwd=child_cwd)
                tool_extra["policy"] = policy
            except Exception:  # noqa: BLE001
                logger.debug("child worktree policy bind failed", exc_info=True)

        session_store = _parent_resource(parent_session, "session_store")
        if session_store is not None:
            # Hermes skip_memory still passes the parent session DB so the
            # read-only session_search tool remains real and usable.
            tool_extra["session_store"] = session_store

        owned_graph = None
        if "code_nav" in visible_tools:
            parent_graph = _parent_resource(parent_session, "graph")
            if wt is None and parent_graph is not None:
                tool_extra["graph"] = parent_graph
            else:
                try:
                    from codedoggy.graph.handle import CodebaseGraph

                    owned_graph = CodebaseGraph(child_cwd, policy=policy)
                    tool_extra["graph"] = owned_graph
                except Exception:  # noqa: BLE001
                    logger.debug("child graph bind failed", exc_info=True)

        if "run_terminal_cmd" in visible_tools:
            from codedoggy.tools.util.shell_state import ShellState, ensure_shell_state

            shell_state = runtime_state.get("shell_state")
            if isinstance(shell_state, ShellState):
                tool_extra["shell_state"] = shell_state
            else:
                shell_state = ensure_shell_state(tool_extra, child_cwd)
        else:
            shell_state = None

        prior = _hydrate_prior_messages(request.prior_messages)

        if system_prompt is None:
            from codedoggy.prompt.grok_system import build_subagent_system_prompt

            system_prompt = build_subagent_system_prompt(
                _child_role_instructions(definition.system_prompt_body),
                cwd=child_cwd,
                memory_enabled=False,
                persona_instructions=request.persona,
            )

        def _on_archive(msg: Any) -> None:
            if parent_coord is None:
                return
            try:
                parent_coord.publish_live_message(request.id, msg)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "publish_live_message failed id=%s",
                    request.id,
                    exc_info=True,
                )

        try:
            loop = run_agent_loop(
                user_text=request.prompt,
                sampler=runtime_sampler,
                tools=child_tools,
                cwd=child_cwd,
                max_turns=max_turns,
                system_prompt=system_prompt,
                is_cancelled=_cancelled,
                cancel_event=cancel,
                hooks=child_hooks,
                # Keep session=None so child does not mutate parent phase/live.
                session=None,
                session_id=f"{request.parent_session_id}:{request.id}",
                tool_extra=tool_extra,
                context_compactor=compactor,
                prior_messages=prior,
                on_archive_message=_on_archive,
            )
        finally:
            # Keep passive child state for Grok-style in-process resume. Active
            # handles remain parent-owned; standalone fallback handles close.
            setattr(
                request,
                "_child_runtime_state",
                {
                    "resources": child_resources,
                    "shell_state": shell_state,
                },
            )
            if owned_graph is not None:
                if not (wt is not None and should_cleanup_worktree() and not is_resume):
                    try:
                        owned_graph.persist_if_dirty()
                    except Exception:  # noqa: BLE001
                        logger.debug("child graph persist failed", exc_info=True)
                try:
                    owned_graph.close()
                except Exception:  # noqa: BLE001
                    logger.debug("child graph close failed", exc_info=True)
            if owned_task_manager is not None:
                try:
                    owned_task_manager.close()
                except Exception:  # noqa: BLE001
                    logger.debug("child task manager close failed", exc_info=True)
            if wt is not None and should_cleanup_worktree() and not is_resume:
                # On resume we usually preserve; cleanup env still honored
                try:
                    wt.cleanup(force=True)
                except Exception:  # noqa: BLE001
                    logger.warning("worktree cleanup failed path=%s", wt.path, exc_info=True)

        summary = _summarize_child(loop.final_text, loop.messages, request)
        if loop.cancelled:
            status = "cancelled"
            child_error = loop.error
        elif loop.completed:
            status = "completed"
            child_error = loop.error
        else:
            status = "failed"
            if loop.max_turns_reached:
                child_error = loop.error or "Maximum turn limit reached before completion."
            elif loop.aborted:
                child_error = loop.error or "Child turn aborted before completion."
            else:
                child_error = loop.error or "Child agent stopped without completion."

        # Hermes: parent memory provider observes delegation (child has no provider)
        parent_session_still_owned = (
            parent_session is not None
            and str(getattr(parent_session, "id", "")) == request.parent_session_id
        )
        if parent_session_still_owned and status == "completed":
            try:
                from codedoggy.memory.hermes_seam import on_delegation

                on_delegation(
                    parent_memory_manager,
                    task=request.prompt or "",
                    result=summary or "",
                    child_session_id=request.id,
                )
            except Exception:  # noqa: BLE001
                logger.debug("parent on_delegation failed", exc_info=True)

        live = _serialize_messages(loop.messages)
        todo_meta: dict[str, Any] = {}
        try:
            from codedoggy.tools.grok_build.todo_logic import count_todos

            if isinstance(todo_state, TodoState):
                # Flush under parent cwd so worktree cleanup cannot drop it.
                try:
                    save_todo_state(
                        todo_state, cwd=todo_disk_cwd, session_id=child_todo_sid
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("save child todo_state failed", exc_info=True)
                if not todo_state.is_empty():
                    badge = count_todos(todo_state).badge_text()
                    todo_meta["todo_badge"] = badge
                    todo_meta["todos"] = [
                        {
                            "id": tid,
                            "content": item.content,
                            "status": item.status,
                        }
                        for tid, item in todo_state.todo_items_with_ids()
                    ]
        except Exception:  # noqa: BLE001
            logger.debug("child todo metadata failed", exc_info=True)

        return SubagentSnapshot(
            subagent_id=request.id,
            subagent_type=request.subagent_type,
            status=status,
            description=request.description or definition.description,
            output=summary,
            error=child_error,
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
                "exit_reason": loop.exit_reason,
                "tools": list(loop.tools_called),
                "isolation": isolation_mode.value,
                "worktree_preserved": bool(
                    wt is not None and not should_cleanup_worktree()
                ),
                "resumed": is_resume,
                "message_count": len(live),
                "cwd": str(child_cwd),
                **(
                    {"model": pinned_model}
                    if pinned_model
                    else {}
                ),
                **todo_meta,
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


def _parent_resource(parent_session: Any, name: str) -> Any:
    """Resolve one parent-owned resource without passing the parent kernel."""
    if parent_session is None:
        return None
    ext = getattr(parent_session, "extensions", None) or getattr(
        parent_session, "_ext", None
    )
    kernel = getattr(ext, "kernel", None) if ext is not None else None
    for owner in (kernel, ext, parent_session):
        if owner is None:
            continue
        value = getattr(owner, name, None)
        if value is not None:
            return value
    return None


def _toolset_names(tools: FinalizedToolset) -> set[str]:
    names = set(tools.by_client_name)
    for tool in tools.by_client_name.values():
        short = getattr(tool, "short_id", None)
        if short:
            names.add(str(short))
    return names


def _child_role_instructions(body: str | None) -> str | None:
    """Remove role text that advertises child-disabled curated memory tools."""
    if not body:
        return body
    return body.replace("session_search / memory_search", "session_search")


def _strip_child_private_tools(
    tools: FinalizedToolset,
    *,
    memory_manager: Any = None,
    allow_nested_spawn: bool = False,
) -> FinalizedToolset:
    """Apply Grok nesting and Hermes child-memory visibility boundaries.

    When ``allow_nested_spawn`` is True (``CODEDOGGY_MAX_SUBAGENT_DEPTH`` > 1
    and this child has remaining depth budget), Task / parallel_tasks stay
    available so deeper fan-out is possible. Memory tools remain denied.
    """
    from codedoggy.tools.kinds import ToolKind

    deny = {
        # Hermes children run with skip_memory=True. Session search remains
        # available through the parent's read-only SessionStore handle.
        "memory",
        "memory_search",
        "memory_get",
    }
    if not allow_nested_spawn:
        deny.update(
            {
                "task",
                "spawn_subagent",
                "spawn_agent",
                "parallel_tasks",
            }
        )
    get_memory_names = getattr(memory_manager, "get_all_tool_names", None)
    if callable(get_memory_names):
        try:
            deny.update(str(name) for name in get_memory_names() or set())
        except Exception:  # noqa: BLE001
            logger.debug("memory provider tool-name lookup failed", exc_info=True)
    by = {}
    for name, ft in tools.by_client_name.items():
        short = getattr(ft, "short_id", None) or ""
        if name in deny or short in deny:
            continue
        if not allow_nested_spawn and getattr(ft, "kind", None) is ToolKind.Task:
            # Block Task-kind tools (incl. merge_subagent_worktree is Task —
            # merge is MAIN-only; children should not land into parent).
            continue
        if allow_nested_spawn and short in {
            "merge_subagent_worktree",
            "merge_worktree",
        }:
            # Merge lands into parent — MAIN only.
            continue
        by[name] = ft
    if len(by) == len(tools.by_client_name):
        return tools
    return FinalizedToolset(by_client_name=by)


def _strip_nested_spawn(tools: FinalizedToolset) -> FinalizedToolset:
    """Backward-compatible helper for callers that only need nesting denial."""
    return _strip_child_private_tools(tools)


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
                    **(
                        {"provider_data": dict(tc.provider_data)}
                        if isinstance(getattr(tc, "provider_data", None), dict)
                        else {}
                    ),
                }
                for tc in tcs
            ]
        reasoning = getattr(m, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning:
            d["reasoning_content"] = reasoning
        provider_data = getattr(m, "provider_data", None)
        if isinstance(provider_data, dict) and provider_data:
            d["provider_data"] = dict(provider_data)
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
                    provider_data=(
                        dict(tc["provider_data"])
                        if isinstance(tc.get("provider_data"), dict)
                        else None
                    ),
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
                reasoning_content=item.get("reasoning_content"),
                provider_data=(
                    dict(item["provider_data"])
                    if isinstance(item.get("provider_data"), dict)
                    else None
                ),
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
