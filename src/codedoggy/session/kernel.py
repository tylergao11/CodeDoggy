"""RuntimeKernel — single owner of session execution state.

Owns (or is the sole writer of):
  - goal + base system prompt
  - live transcript (via runner)
  - lifecycle phase coordination on close
  - hydrate from SessionStore

Session remains the public handle; Kernel is the spine extensions hang from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codedoggy.turn.types import Message, Role, ToolCall

logger = logging.getLogger(__name__)


def _default_task_model_validator(slug: str) -> str | None:
    """Permissive Task.model check: non-empty only (host may replace with catalog)."""
    if not str(slug or "").strip():
        return "model is empty"
    return None


@dataclass
class RuntimeKernel:
    """Unified runtime state for one Session."""

    cwd: Path
    session_id: str
    goal: str | None = None
    base_system_prompt: str = ""
    # Bound handles (same objects as SessionExtensions — single identity)
    turn_runner: Any = None
    tools: Any = None
    context: Any = None
    memory: Any = None
    memory_manager: Any = None
    session_store: Any = None
    policy: Any = None
    graph: Any = None
    connection: Any = None  # ConnectionService — unified model truth
    mcp_runtime: Any = None
    # Grok orchestration handles
    session_mode_state: Any = None  # SessionModeState
    interjection_buffer: Any = None  # InterjectionBuffer
    prompt_queue: Any = None  # PromptQueue
    subagent_coordinator: Any = None  # SubagentCoordinator
    subagent_run_fn: Any = None  # Callable for child runs
    task_manager: Any = None  # BackgroundTaskManager
    scheduler: Any = None  # Scheduler
    agent: Any = None  # optional Agent config package
    # Goal / todo state (model tools)
    todo_state: Any = None
    goal_log: list = field(default_factory=list)
    goal_completed: bool = False
    goal_blocked: bool = False
    goal_blocked_reason: str | None = None
    goal_completion_message: str | None = None
    # Lifecycle
    closed: bool = False
    # Extra bag for ToolExecutor / loop
    tool_extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cwd = Path(self.cwd).resolve()
        if self.task_manager is None:
            from codedoggy.tools.task_manager import BackgroundTaskManager

            self.task_manager = BackgroundTaskManager(
                work_dir=self.cwd / ".codedoggy" / "tasks" / self.session_id
            )
        if self.scheduler is None:
            from codedoggy.tools.scheduler import Scheduler

            self.scheduler = Scheduler()
        self.refresh_tool_extra()

    # Kernel-owned keys rewritten every refresh. Host keys (mcp_*, shell_state,
    # stream hooks, prefetch_user_block, memory_backend, ask_user_fn, …) are
    # preserved across rebuilds — they are NOT in this set.
    _MANAGED_TOOL_EXTRA_KEYS: frozenset[str] = frozenset(
        {
            "kernel",
            "connection",
            "memory_store",
            "session_store",
            "policy",
            "memory_manager",
            "graph",
            "mcp_runtime",
            "session_mode_state",
            "interjection_buffer",
            "subagent_coordinator",
            "subagent_run_fn",
            "task_manager",
            "scheduler",
            "todo_state",
        }
    )

    # Host adapter keys that survive refresh (documentation + wire_default_host_extras).
    HOST_TOOL_EXTRA_KEYS: frozenset[str] = frozenset(
        {
            "memory_backend",
            "ask_user_fn",
            "lsp_backend",
            "mcp_inner_dispatch",
            "mcp_dispatch",
            "mcp_tools",
            "mcp_servers",
            "mcp_status",
            "mcp_tool_index",
            "mcp_initialized",
            "mcp_runtime",
            "scheduler_tick",
            "shell_state",
            "plan_file_path",
            "plan_mode_consent_fn",
            "plan_mode_exit_fn",
            "todo_changed_fn",
            "goal_ack_fn",
        }
    )

    def refresh_tool_extra(self) -> None:
        """Rebuild dict injected into every tool call mid-turn.

        Always includes memory_manager / memory_store / session_store / policy
        when bound so provider tools and session_search work without a second
        protocol. Host-injected MCP hooks (mcp_dispatch, mcp_tools, …) and
        other non-managed keys survive the rebuild.
        """
        prev = dict(self.tool_extra or {})
        extra: dict[str, Any] = {"kernel": self}
        # Media / search extras follow ActiveConnection (same login as MAIN chat).
        if self.connection is not None:
            extra["connection"] = self.connection
        if self.memory is not None:
            extra["memory_store"] = self.memory
        if self.session_store is not None:
            extra["session_store"] = self.session_store
        if self.policy is not None:
            extra["policy"] = self.policy
        if self.memory_manager is not None:
            extra["memory_manager"] = self.memory_manager
        if self.graph is not None:
            extra["graph"] = self.graph
        if self.mcp_runtime is not None:
            populate = getattr(self.mcp_runtime, "populate_tool_extra", None)
            if callable(populate):
                populate(extra)
        if self.session_mode_state is not None:
            extra["session_mode_state"] = self.session_mode_state
            # Always publish authoritative session plan path (Grok PlanFilePath).
            try:
                extra["plan_file_path"] = self._resolve_plan_file_path(
                    getattr(self.session_mode_state, "plan_file", None) or None
                )
            except Exception:  # noqa: BLE001
                pass
        if self.interjection_buffer is not None:
            extra["interjection_buffer"] = self.interjection_buffer
        if self.subagent_coordinator is not None:
            extra["subagent_coordinator"] = self.subagent_coordinator
        if self.subagent_run_fn is not None:
            extra["subagent_run_fn"] = self.subagent_run_fn
        # Task.model validation: permissive non-empty check so spawn can pin
        # a model without a full catalog (Grok host installs a richer one).
        if "task_model_validator" not in prev:
            extra["task_model_validator"] = _default_task_model_validator
        if self.task_manager is not None:
            extra["task_manager"] = self.task_manager
        if self.scheduler is not None:
            extra["scheduler"] = self.scheduler
        if self.todo_state is not None:
            extra["todo_state"] = self.todo_state
        # Preserve host extras: mcp_*, memory_backend, ask_user_fn, shell_state,
        # stream_sample / on_sample_delta / scheduler_tick / …
        for key, value in prev.items():
            if key not in self._MANAGED_TOOL_EXTRA_KEYS and key not in extra:
                extra[key] = value
        self.tool_extra = extra

    def wire_host_adapters(self, **opts: Any) -> dict[str, Any]:
        """Attach optional product host adapters into tool_extra.

        See ``codedoggy.host.wire_default_host_extras``. MCP is owned directly
        by this kernel when ``mcp_runtime`` is bound; external hosts may still
        inject compatible hooks when it is absent.
        """
        from codedoggy.host import wire_default_host_extras

        # Ensure managed bag is current, then layer host keys on top.
        self.refresh_tool_extra()
        return wire_default_host_extras(self, **opts)

    def _resolve_plan_file_path(self, plan_file: str | None = None) -> str:
        """Grok PlanModeTracker: ``session_dir/plan.md`` for this session.

        Fallback without session id: tool-layer ``cwd/.grok/plan.md``.
        """
        if not plan_file:
            from codedoggy.orchestration.session_mode import plan_file_for_session

            sid = str(getattr(self, "session_id", "") or "").strip()
            if sid:
                return str(plan_file_for_session(self.cwd, sid).resolve())
            from codedoggy.tools.grok_build.plan_mode import PLAN_FILE_RELATIVE_PATH

            return str((Path(self.cwd) / PLAN_FILE_RELATIVE_PATH).resolve())
        p = Path(plan_file)
        if p.is_absolute():
            return str(p)
        return str((Path(self.cwd) / p).resolve())

    def persist_plan_mode_state(self) -> None:
        """Write plan_mode.json after every lifecycle transition (Grok)."""
        state = self.session_mode_state
        if state is None:
            return
        try:
            from codedoggy.orchestration.session_mode import save_plan_mode_state

            save_plan_mode_state(
                state, cwd=self.cwd, session_id=str(self.session_id)
            )
        except Exception:  # noqa: BLE001
            logger.debug("persist_plan_mode_state failed", exc_info=True)

    def load_plan_mode_state(self) -> bool:
        """Restore plan_mode.json into session_mode_state. Returns True if loaded."""
        try:
            from codedoggy.orchestration.session_mode import (
                load_plan_mode_state,
            )

            plan_file = self._resolve_plan_file_path(None)
            restored = load_plan_mode_state(
                cwd=self.cwd,
                session_id=str(self.session_id),
                plan_file=plan_file,
            )
            if restored is None:
                return False
            self.session_mode_state = restored
            self.refresh_tool_extra()
            logger.info(
                "kernel restored plan_mode session_id=%s phase=%s awaiting=%s",
                self.session_id,
                restored.plan_phase,
                restored.awaiting_plan_approval,
            )
            return True
        except Exception:  # noqa: BLE001
            logger.debug("load_plan_mode_state failed", exc_info=True)
            return False

    def persist_todo_state(self) -> None:
        """Write todo_state.json (Grok write_plan_state / tool TodoState)."""
        st = self.todo_state
        if st is None:
            bag = self.tool_extra if isinstance(self.tool_extra, dict) else {}
            st = bag.get("todo_state")
        if st is None:
            return
        try:
            from codedoggy.tools.grok_build.todo_logic import save_todo_state

            save_todo_state(st, cwd=self.cwd, session_id=str(self.session_id))
        except Exception:  # noqa: BLE001
            logger.debug("persist_todo_state failed", exc_info=True)

    def load_todo_state(self) -> bool:
        """Restore todo_state.json. Returns True if loaded."""
        try:
            from codedoggy.tools.grok_build.todo_logic import load_todo_state

            restored = load_todo_state(
                cwd=self.cwd, session_id=str(self.session_id)
            )
            if restored is None:
                return False
            self.todo_state = restored
            self.refresh_tool_extra()
            logger.info(
                "kernel restored todo_state session_id=%s items=%s",
                self.session_id,
                sum(1 for _ in restored.todo_items()),
            )
            return True
        except Exception:  # noqa: BLE001
            logger.debug("load_todo_state failed", exc_info=True)
            return False

    def enter_plan_mode(self, plan_file: str | None = None) -> None:
        """Grok enter plan mode — hard activate (tool path → Active)."""
        state = self.session_mode_state
        if state is None:
            from codedoggy.orchestration.session_mode import SessionModeState

            state = SessionModeState()
            self.session_mode_state = state
        state.enter_plan(self._resolve_plan_file_path(plan_file))
        self.refresh_tool_extra()
        self.persist_plan_mode_state()

    def enter_plan_mode_pending(self, plan_file: str | None = None) -> bool:
        """User Tab: Pending until next prompt (Grok enter_pending)."""
        state = self.session_mode_state
        if state is None:
            from codedoggy.orchestration.session_mode import SessionModeState

            state = SessionModeState()
            self.session_mode_state = state
        changed = state.enter_plan_pending(self._resolve_plan_file_path(plan_file))
        self.refresh_tool_extra()
        self.persist_plan_mode_state()
        return changed

    def user_exit_plan_mode(self, *, turn_in_flight: bool) -> None:
        """User Tab off: ExitPending if turn running, else Inactive."""
        state = self.session_mode_state
        if state is None:
            return
        if hasattr(state, "user_exit"):
            state.user_exit(turn_in_flight=turn_in_flight)
        else:
            state.exit_plan(approved=False)
        self.refresh_tool_extra()
        self.persist_plan_mode_state()

    def exit_plan_mode(self, *, approved: bool = True, reason: str | None = None) -> None:
        state = self.session_mode_state
        if state is not None:
            try:
                state.exit_plan(approved=approved, reason=reason)
            except TypeError:
                state.exit_plan(approved=approved)
        self.refresh_tool_extra()
        self.persist_plan_mode_state()

    # Grok PLAN_APPROVED_IMPLEMENT_MESSAGE (tool_calls.rs)
    PLAN_APPROVED_IMPLEMENT_MESSAGE = (
        "The user approved the plan. Implement the plan in plan.md."
    )

    def wait_or_resolve_parked_plan_approval(self) -> str | None:
        """Grok resume re-park: if awaiting_plan_approval, run host approval.

        Returns optional text to prepend to the next user turn (implement nudge
        or revise guidance). Returns None when nothing parked / no inject.
        """
        state = self.session_mode_state
        if state is None or not getattr(state, "awaiting_plan_approval", False):
            return None

        plan_path = str(getattr(state, "plan_file", "") or "")
        plan_content: str | None = None
        try:
            p = Path(plan_path) if plan_path else None
            if p is not None and not p.is_absolute():
                p = (Path(self.cwd) / p).resolve()
            if p is not None and p.is_file():
                text = p.read_text(encoding="utf-8", errors="replace")
                plan_content = text if text.strip() else None
        except OSError:
            plan_content = None

        bag = self.tool_extra if isinstance(self.tool_extra, dict) else {}
        exit_fn = bag.get("plan_mode_exit_fn")
        outcome = "cancelled"
        feedback: str | None = None
        if callable(exit_fn):
            try:
                result = exit_fn(
                    {
                        "plan_content": plan_content,
                        "plan_file_path": plan_path,
                        "resume": True,
                        "tool_call_id": "resume-plan-approval",
                    }
                )
            except Exception:  # noqa: BLE001
                logger.exception("parked plan approval host failed")
                result = {"outcome": "cancelled", "feedback": "approval host failed"}
            if isinstance(result, dict):
                outcome = str(result.get("outcome") or "cancelled")
                fb = result.get("feedback")
                feedback = str(fb) if fb is not None else None
            elif isinstance(result, str):
                outcome = result
        else:
            # Headless: leave parked; remind model not to implement yet.
            return (
                "Plan mode approval is still outstanding from a previous session. "
                "Do not implement until the user approves the plan "
                "(exit_plan_mode / host approval)."
            )

        if outcome == "approved":
            self.exit_plan_mode(approved=True, reason="approved")
            return self.PLAN_APPROVED_IMPLEMENT_MESSAGE
        if outcome == "abandoned":
            self.exit_plan_mode(approved=False, reason="abandoned")
            return (
                "The user abandoned the plan. Plan mode is off. "
                "Do not call exit_plan_mode again unless asked to re-enter plan mode."
            )
        # cancelled / revise — stay in plan mode; clear park so we do not re-block.
        state.awaiting_plan_approval = False
        if state.plan_phase != "active":
            state.enter_plan(plan_path or None)
        self.persist_plan_mode_state()
        self.refresh_tool_extra()
        if plan_content is None:
            return (
                "The user does not want to exit plan mode. "
                "Continue planning and ask the user what they would like to do."
            )
        fb = (feedback or "").strip()
        if not fb:
            return (
                "The user wants to revise the plan. "
                "Ask the user what changes they would like to make."
            )
        return f"The user wants to revise the plan. The user said:\n{fb}"

    def enter_goal_mode(self) -> None:
        """Enter goal mode — hard tool gate when blocked (orchestration).

        Clears plan lifecycle (modes exclusive, Grok session mode spirit).
        """
        state = self.session_mode_state
        if state is None:
            from codedoggy.orchestration.session_mode import SessionModeState

            state = SessionModeState()
            self.session_mode_state = state
        # Leaving plan while entering goal — cancel outstanding approval park.
        if getattr(state, "awaiting_plan_approval", False):
            state.awaiting_plan_approval = False
        state.enter_goal()
        self.refresh_tool_extra()
        self.persist_plan_mode_state()

    def exit_goal_mode(self) -> None:
        state = self.session_mode_state
        if state is not None:
            try:
                state.exit_goal(reason="exit")
            except TypeError:
                state.exit_goal()
        self.refresh_tool_extra()
        self.persist_plan_mode_state()

    def interject(self, text: str, *, prompt_id: str | None = None) -> None:
        """Push a mid-turn user message (Grok pending_interjections).

        Drained at next safe point in the turn loop (not mid-stream invent).
        """
        buf = self.interjection_buffer
        if buf is None:
            from codedoggy.orchestration.prompt_queue import InterjectionBuffer

            buf = InterjectionBuffer()
            self.interjection_buffer = buf
            self.refresh_tool_extra()
        buf.push(text, prompt_id=prompt_id)

    def enqueue_prompt(
        self,
        text: str,
        *,
        prompt_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Park a full prompt for after the current turn (not an interjection).

        Creates ``prompt_queue`` if missing. Does not start a turn.
        Returns the queue length after push.
        """
        q = self.prompt_queue
        if q is None:
            from codedoggy.orchestration.prompt_queue import PromptQueue

            q = PromptQueue()
            self.prompt_queue = q
        from codedoggy.orchestration.prompt_queue import PromptQueueItem

        q.push(
            PromptQueueItem(
                text=text,
                prompt_id=prompt_id,
                metadata=dict(metadata or {}),
            )
        )
        return len(q)

    def new_session(
        self,
        *,
        title: str | None = None,
        clear_live: bool = True,
        reason: str = "new_session",
    ) -> str:
        """Hermes ``new_session`` spirit — rotate id, notify memory providers.

        Does not tear down curated MEMORY/USER; only rebinds session-scoped
        providers via ``commit_session_boundary_async``.
        """
        from codedoggy.session.types import SessionId

        old_id = self.session_id
        new_id = str(SessionId.new())
        # Snapshot live messages for end-of-session extraction
        snap: list[Any] = []
        if self.turn_runner is not None:
            live = getattr(self.turn_runner, "live_messages", None) or []
            snap = list(live)
            if clear_live:
                clear = getattr(self.turn_runner, "clear_live_history", None)
                if callable(clear):
                    clear()
        # A new session is a context lifetime boundary, not only a transcript
        # id change.  Otherwise iterative summary/checkpoint state can inject
        # the previous session into the first fold or rewind of the new one.
        seen_contexts: set[int] = set()
        for context in (
            self.context,
            getattr(self.turn_runner, "context_compactor", None),
        ):
            if context is None or id(context) in seen_contexts:
                continue
            seen_contexts.add(id(context))
            reset = getattr(context, "on_session_end", None)
            if callable(reset):
                try:
                    reset()
                except Exception:  # noqa: BLE001
                    logger.exception("context reset at session boundary failed")
        # Archive boundary on store if present
        if self.session_store is not None:
            try:
                self.session_store.ensure_session(
                    new_id, cwd=str(self.cwd), goal=self.goal, title=title
                )
            except Exception:  # noqa: BLE001
                logger.debug("ensure_session for new id failed", exc_info=True)

        # Hermes seam: end → switch on single worker (or sync fallback)
        from codedoggy.memory.hermes_seam import commit_session_boundary

        commit_session_boundary(
            self.memory_manager,
            snap,
            new_session_id=new_id,
            parent_session_id=old_id,
            reason=reason,
        )

        self.session_id = new_id
        self.refresh_tool_extra()
        logger.info("kernel new_session old=%s new=%s reason=%s", old_id, new_id, reason)
        return new_id

    # ----- goal / system (single writer) -----

    def set_goal(self, goal: str | None) -> None:
        if self.closed:
            raise RuntimeError("kernel is closed")
        g = goal.strip() if isinstance(goal, str) and goal.strip() else goal
        self.goal = g
        # Rebuild base system so main model sees new goal next turn
        self.base_system_prompt = rebuild_system_prompt(self.base_system_prompt, g)
        if self.turn_runner is not None:
            self.turn_runner.system_prompt = self.base_system_prompt
        if self.session_store is not None:
            try:
                self.session_store.ensure_session(
                    self.session_id, cwd=str(self.cwd), goal=g
                )
            except Exception:  # noqa: BLE001
                logger.debug("session_store goal update failed", exc_info=True)

    # ----- transcript -----

    @property
    def live_messages(self) -> list[Message]:
        r = self.turn_runner
        if r is None:
            return []
        return list(getattr(r, "live_messages", None) or [])

    def set_live_messages(self, messages: list[Message]) -> None:
        if self.turn_runner is not None:
            self.turn_runner.live_messages = list(messages)

    def hydrate_from_store(self, *, limit: int = 200) -> int:
        """Load the *newest* archived messages (tail), not the oldest.

        Grok resume is recent-context first. ORDER BY id DESC LIMIT then reverse.
        Also restores plan_mode.json lifecycle when present.
        """
        # Plan mode + todos independent of message rows.
        self.load_plan_mode_state()
        self.load_todo_state()

        if self.session_store is None or self.turn_runner is None:
            return 0
        try:
            get_snapshot = getattr(self.session_store, "get_context_snapshot", None)
            rows = get_snapshot(self.session_id) if callable(get_snapshot) else []
            if rows:
                pass
            elif callable(getattr(self.session_store, "get_messages_tail", None)):
                get_tail = self.session_store.get_messages_tail
                rows = get_tail(self.session_id, limit=limit)
            else:
                # Fallback: load all then slice tail (small stores only)
                rows = self.session_store.get_messages(self.session_id, limit=None)
                if limit is not None and len(rows) > limit:
                    rows = rows[-int(limit) :]
        except Exception:  # noqa: BLE001
            logger.warning("hydrate get_messages failed", exc_info=True)
            return 0
        if not rows:
            return 0
        messages = [_row_to_message(r) for r in rows]
        messages = [m for m in messages if m.role is not Role.SYSTEM]
        from codedoggy.context.select import sanitize_tool_pairs

        messages = sanitize_tool_pairs(messages)
        self.turn_runner.live_messages = messages
        logger.info(
            "kernel hydrated session_id=%s messages=%s (canonical-or-tail)",
            self.session_id,
            len(messages),
        )
        return len(messages)

    # ----- close -----

    def close(self) -> None:
        """Tear down runtime: context, live history, memory, subagents, store."""
        if self.closed:
            return
        # Flush plan/todo before marking closed (Grok persist on session end).
        try:
            self.persist_plan_mode_state()
        except Exception:  # noqa: BLE001
            logger.debug("close: persist_plan_mode_state failed", exc_info=True)
        try:
            self.persist_todo_state()
        except Exception:  # noqa: BLE001
            logger.debug("close: persist_todo_state failed", exc_info=True)
        self.closed = True
        # Stop host scheduler tick thread if running
        try:
            handle = (self.tool_extra or {}).get("scheduler_runtime")
            stop = getattr(handle, "stop", None)
            if callable(stop):
                stop()
        except Exception:  # noqa: BLE001
            logger.debug("scheduler_runtime stop failed", exc_info=True)
        # Children borrow parent-owned memory/graph/task handles. Cancel and
        # join them before tearing any of those resources down.
        coord = self.subagent_coordinator
        if coord is not None:
            shutdown = getattr(coord, "shutdown", None)
            if callable(shutdown):
                try:
                    import inspect

                    candidates = {
                        "wait": True,
                        "cancel_running": True,
                        "timeout_s": 5.0,
                    }
                    try:
                        params = inspect.signature(shutdown).parameters
                        has_varkw = any(
                            p.kind is inspect.Parameter.VAR_KEYWORD
                            for p in params.values()
                        )
                        kwargs = (
                            candidates
                            if has_varkw
                            else {k: v for k, v in candidates.items() if k in params}
                        )
                    except (TypeError, ValueError):
                        kwargs = {"wait": True}
                    shutdown(**kwargs)
                except Exception:  # noqa: BLE001
                    logger.exception("subagent coordinator shutdown failed")
        # Grok: the Session owns MCP clients/dispatcher/restart tasks. Children
        # are joined first because they may still be borrowing the live tool
        # index or dispatch hook.
        mcp = self.mcp_runtime
        if mcp is not None:
            close_mcp = getattr(mcp, "close", None)
            if callable(close_mcp):
                try:
                    close_mcp()
                except Exception:  # noqa: BLE001
                    logger.exception("MCP runtime shutdown failed")
        # Stop background shell/process producers before persisting Graph or
        # tearing down the memory/context consumers they may still reference.
        tm = self.task_manager
        if tm is not None:
            close_tm = getattr(tm, "close", None)
            if callable(close_tm):
                try:
                    close_tm()
                except Exception:  # noqa: BLE001
                    logger.exception("task_manager close failed")
        # Drain prefire / context.  The runner normally references the same
        # compactor as ``self.context``; call each identity once.
        seen_contexts: set[int] = set()
        for obj in (self.context, getattr(self.turn_runner, "context_compactor", None)):
            if obj is None:
                continue
            oid = id(obj)
            if oid in seen_contexts:
                continue
            seen_contexts.add(oid)
            on_end = getattr(obj, "on_session_end", None)
            if callable(on_end):
                try:
                    on_end()
                except Exception:  # noqa: BLE001
                    logger.exception("context on_session_end failed")
        # Snapshot live before clear so Hermes on_session_end can extract
        snap: list[Any] = []
        if self.turn_runner is not None:
            snap = list(getattr(self.turn_runner, "live_messages", None) or [])
        clear = getattr(self.turn_runner, "clear_live_history", None)
        if callable(clear):
            try:
                clear()
            except Exception:  # noqa: BLE001
                logger.exception("clear_live_history failed")
        # Hermes seam: on_session_end → flush → shutdown_all
        from codedoggy.memory.hermes_seam import on_session_close

        on_session_close(self.memory_manager, messages=snap, timeout_s=5.0)
        # Curated memory store (if it exposes close)
        mem = self.memory
        if mem is not None:
            close_mem = getattr(mem, "close", None)
            if callable(close_mem):
                try:
                    close_mem()
                except Exception:  # noqa: BLE001
                    logger.debug("memory store close failed", exc_info=True)
        # Persist graph if dirty
        graph = self.graph
        if graph is not None:
            close_graph = getattr(graph, "close", None)
            if callable(close_graph):
                try:
                    close_graph()
                except Exception:  # noqa: BLE001
                    logger.debug("graph close failed", exc_info=True)
            else:
                save = getattr(graph, "persist_if_dirty", None)
                if callable(save):
                    try:
                        save()
                    except Exception:  # noqa: BLE001
                        logger.debug("graph persist on close failed", exc_info=True)
                stop = getattr(graph, "stop_watch", None)
                if callable(stop):
                    try:
                        stop()
                    except Exception:  # noqa: BLE001
                        pass
        # Session archive SQLite
        ss = self.session_store
        if ss is not None:
            close = getattr(ss, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    logger.debug("session_store close failed", exc_info=True)


def rebuild_system_prompt(base: str, goal: str | None) -> str:
    """Replace or append Session goal line in base system prompt."""
    lines = (base or "").splitlines()
    out: list[str] = []
    for line in lines:
        if line.startswith("Session goal:"):
            continue
        out.append(line)
    text = "\n".join(out).rstrip()
    if goal and str(goal).strip():
        text = f"{text}\nSession goal: {str(goal).strip()}" if text else f"Session goal: {str(goal).strip()}"
    return text


def _row_to_message(row: dict[str, Any]) -> Message:
    role_s = str(row.get("role") or "user").lower()
    try:
        role = Role(role_s)
    except ValueError:
        role = Role.USER
    tool_calls = None
    raw_tc = row.get("tool_calls")
    if isinstance(raw_tc, list) and raw_tc:
        tool_calls = []
        for tc in raw_tc:
            if not isinstance(tc, dict):
                continue
            args = tc.get("arguments")
            if isinstance(args, str):
                import json

                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            if not isinstance(args, dict):
                args = {}
            tool_calls.append(
                ToolCall(
                    id=str(tc.get("id") or ""),
                    name=str(tc.get("name") or ""),
                    arguments=args,
                    provider_data=(
                        dict(tc["provider_data"])
                        if isinstance(tc.get("provider_data"), dict)
                        else None
                    ),
                )
            )
    return Message(
        role=role,
        content=row.get("content"),
        name=row.get("tool_name") or row.get("name"),
        tool_call_id=row.get("tool_call_id"),
        tool_calls=tool_calls,
        reasoning_content=row.get("reasoning_content"),
        provider_data=(
            dict(row["provider_data"])
            if isinstance(row.get("provider_data"), dict)
            else None
        ),
    )
