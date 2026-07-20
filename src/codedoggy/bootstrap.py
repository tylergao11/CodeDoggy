"""One-shot wiring: main brain + context + memory manager + tools policy + graph."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from codedoggy.memory.manager import MemoryManager
from codedoggy.memory.session_store import SessionStore, default_session_db_path
from codedoggy.memory.store import MemoryStore
from codedoggy.model.chat_sampler import ChatSampler
from codedoggy.model.connection import ConnectionService
from codedoggy.model.profiles import ModelProfiles, model_profiles_from_env
from codedoggy.model.provider import ChatClient
from codedoggy.session.extensions import SessionExtensions
from codedoggy.session.session import Session
from codedoggy.tools.policy import WorkspacePolicy
from codedoggy.tools.registry import FinalizedToolset, ToolRegistryBuilder
from codedoggy.turn.runner import AgentTurnRunner

logger = logging.getLogger(__name__)


def build_session(
    cwd: str | Path,
    *,
    goal: str | None = None,
    max_turns: int | None = None,
    system_prompt: str | None = None,
    enable_memory: bool = True,
    enable_session_store: bool = True,
    enable_policy: bool = True,
    enable_graph: bool = True,
    enable_mcp: bool | None = None,
    mcp_servers: Mapping[str, Mapping[str, Any]] | Sequence[Any] | None = None,
    mcp_watch: bool = True,
    mcp_auto_restart: bool = True,
    memory_dir: str | Path | None = None,
    session_db: str | Path | None = None,
    profiles: ModelProfiles | None = None,
    main_client: ChatClient | None = None,
    tools: FinalizedToolset | None = None,
    session_id: str | None = None,
) -> Session:
    """Create a Session with the product pillars fused.

    - **Context**: ``ContextCompactor`` (Grok pipeline + prefire + rewind)
    - **Memory**: ``MemoryManager`` (curated + FTS + external slot)
    - **Tools**: ``FinalizedToolset`` + ``WorkspacePolicy`` + ``code_nav``
    - **MAIN parallel bias**: system prompt + tools; MAIN decides when to fan out
      (harness does **not** auto-split or auto-parallelize work for MAIN)
    - **Graph**: ``CodebaseGraph`` (xai-codebase-graph API spirit)
    - **MCP**: Grok ``McpState`` + dispatcher + stdio/HTTP transports
    """
    prof = profiles or model_profiles_from_env()
    # Soft auth at bootstrap so TUI can open the login wizard when unauthenticated.
    # Hard gate happens when starting a turn / reloading after login.
    if main_client is not None:
        main = main_client
    else:
        from codedoggy.model.registry import create_client

        try:
            main = create_client(prof.main, require_auth=False)
        except Exception as exc:
            from codedoggy.model.auth.base import LoginRequired

            if isinstance(exc, LoginRequired):
                raise LoginRequired(
                    exc.provider,
                    f"{exc} — open the TUI auth gate (Street HUD / Ctrl+L).",
                ) from exc
            raise
    cwd_path = Path(cwd).resolve()

    # Memory pillar FIRST so external provider tools can be injected into the
    # model toolset (Hermes inject_memory_provider_tools). Finalize before load
    # left notes_append etc. half-wired — schemas on manager, never on toolset.
    memory: MemoryStore | None = None
    if enable_memory:
        # Honor CODEDOGGY_MEMORY_CHAR_LIMIT / CODEDOGGY_USER_CHAR_LIMIT (same as CLI).
        from codedoggy.memory.store import load_on_disk_store

        memory = load_on_disk_store(memory_dir=memory_dir)

    session_store: SessionStore | None = None
    if enable_session_store:
        db = Path(session_db) if session_db else default_session_db_path()
        session_store = SessionStore(db)
        if session_id:
            ownership = session_store.claim_session(
                session_id,
                cwd_path,
                allow_unbound=False,
                goal=goal,
            )
            if not ownership.allowed:
                session_store.close()
                raise ValueError(
                    "refusing to hydrate session "
                    f"{session_id!r} in workspace {ownership.requested_cwd!r}: "
                    f"{ownership.reason}; stored workspace="
                    f"{ownership.stored_cwd!r}"
                )

    # Memory pillar orchestration (Hermes MemoryManager + optional external plugin)
    memory_manager: MemoryManager | None = None
    if enable_memory or enable_session_store:
        memory_manager = MemoryManager.create_default(
            curated=memory,
            session_store=session_store,
        )
        # Hermes memory.provider selection → CODEDOGGY_MEMORY_PROVIDER
        try:
            from codedoggy.memory.plugins import load_memory_provider_from_env

            ext = load_memory_provider_from_env()
            if ext is not None and memory_manager is not None:
                if memory_manager.add_provider(ext):
                    pass  # registered
        except Exception:  # noqa: BLE001
            pass

    # Tools after memory so provider schemas can join the model-visible set.
    # Always inject when mm is present — even if the caller passed a custom
    # pre-finalized toolset (otherwise notes_append etc. stay half-wired:
    # schemas on the manager, never on the model tool surface).
    toolset = tools or ToolRegistryBuilder.new().finalize()
    if memory_manager is not None:
        try:
            from codedoggy.memory.tool_injection import inject_memory_provider_tools

            inject_memory_provider_tools(toolset, memory_manager)
        except Exception:  # noqa: BLE001
            pass

    policy: WorkspacePolicy | None = None
    if enable_policy:
        policy = WorkspacePolicy.from_env(cwd_path)

    graph = None
    if enable_graph:
        from codedoggy.graph.handle import CodebaseGraph

        # Glue: attach workspace policy so reindex/cache writes honor allow_writes
        graph = CodebaseGraph(cwd_path, policy=policy if enable_policy else None)
        # Index construction remains lazy until code_nav first needs it.

    default_system = system_prompt
    if default_system is None:
        default_system = _default_system_prompt(goal)

    from codedoggy.context.compactor import ContextCompactor

    # Optional fold summarizer: use aux model when distinct from main
    summary_client = None
    try:
        if prof.aux.model != prof.main.model or prof.aux.base_url != prof.main.base_url:
            summary_client = prof.aux_client()
    except Exception:  # noqa: BLE001
        summary_client = None

    # Budget follows the active main model (provider+model), not a blind 32k.
    main_cfg = prof.main if prof is not None else None
    compactor = ContextCompactor.from_env(
        summary_client=summary_client,
        memory_store=memory,
        session_store=session_store,
        memory_manager=memory_manager,
        provider=getattr(main_cfg, "provider", None) if main_cfg else None,
        model=getattr(main_cfg, "model", None) if main_cfg else None,
        base_url=getattr(main_cfg, "base_url", None) if main_cfg else None,
        context_window=getattr(main_cfg, "context_window", None) if main_cfg else None,
    )

    cw = getattr(main_cfg, "context_window", None) if main_cfg else None
    mt = getattr(main_cfg, "max_tokens", None) if main_cfg else None
    if hasattr(compactor, "bind_model_window"):
        try:
            compactor.bind_model_window(
                context_window=int(cw) if cw else None,
                max_completion_tokens=int(mt) if mt else None,
            )
        except Exception:  # noqa: BLE001
            pass

    runner = AgentTurnRunner(
        sampler=ChatSampler(main),
        tools=toolset,
        system_prompt=default_system,
        context_compactor=compactor,
    )

    # Unified model/provider truth — env is import-only after this point.
    connection = ConnectionService.bootstrap(
        prof.main,
        aux=prof.aux,
        client=main,
        runner=runner,
    )

    from codedoggy.session.kernel import RuntimeKernel

    # Temporary session id if not provided (Session.create assigns one)
    provisional_id = session_id

    session = Session.create(
        cwd_path,
        max_turns=max_turns,
        session_id=provisional_id,
        goal=goal,
        extensions=SessionExtensions(
            turn_runner=runner,
            tools=toolset,
            context=compactor,
            memory=memory,
            memory_manager=memory_manager,
            session_store=session_store,
            policy=policy,
            graph=graph,
            connection=connection,
        ),
    )

    # Grok orchestration spine: mode + interjection + subagent coordinator
    from codedoggy.orchestration.prompt_queue import InterjectionBuffer, PromptQueue
    from codedoggy.orchestration.session_mode import SessionModeState
    from codedoggy.orchestration.subagent import SubagentCoordinator, make_child_runner

    mode_state = SessionModeState()
    # Session-level goal → enter goal mode so hard gates + update_goal apply
    if goal and str(goal).strip():
        mode_state.enter_goal()
    interjections = InterjectionBuffer()
    prompt_queue = PromptQueue()
    # High pool: main agent defaults to parallel fan-out of many children.
    subagent_coord = SubagentCoordinator(max_workers=8)
    subagent_run = make_child_runner(
        parent_cwd=cwd_path,
        parent_tools=toolset,
        parent_sampler=ChatSampler(main),
        parent_system_prompt=default_system,
        parent_session=None,  # bound after session exists
        context_compactor_factory=lambda: ContextCompactor.from_env(
            summary_client=None,
            memory_store=None,  # child: independent memory window
            session_store=None,
            memory_manager=None,
        ),
    )

    kernel = RuntimeKernel(
        cwd=cwd_path,
        session_id=str(session.id),
        goal=goal,
        base_system_prompt=default_system,
        turn_runner=runner,
        tools=toolset,
        context=compactor,
        memory=memory,
        memory_manager=memory_manager,
        session_store=session_store,
        policy=policy,
        graph=graph,
        connection=connection,
        session_mode_state=mode_state,
        interjection_buffer=interjections,
        prompt_queue=prompt_queue,
        subagent_coordinator=subagent_coord,
        subagent_run_fn=subagent_run,
    )
    connection.bind_runtime(runner=runner, kernel=kernel, context=compactor)
    # Re-bind parent_session into runner after session object exists
    kernel.subagent_run_fn = make_child_runner(
        parent_cwd=cwd_path,
        parent_tools=toolset,
        parent_sampler=ChatSampler(main),
        parent_system_prompt=default_system,
        parent_session=session,
        parent_sampler_factory=connection.new_sampler,
        context_compactor_factory=lambda: ContextCompactor.from_env(
            summary_client=None,
            memory_store=None,
            session_store=None,
            memory_manager=None,
        ),
    )
    session.bind_extensions(session.extensions.with_kernel(kernel))
    if graph is not None:
        # Acquire this Session's watcher lease only after the runtime spine is
        # bound. Watching does not trigger a full repository build.
        graph.start_watch()

    # Hermes seam: bind providers to this session (initialize_all) so external
    # tools (notes_append, …) see session_id on first turn. Must run after
    # Session.create assigns the real id.
    if memory_manager is not None:
        from codedoggy.memory.hermes_seam import bind_session
        from codedoggy.memory.paths import default_memory_home

        bind_session(
            memory_manager,
            session_id=str(session.id),
            cwd=str(session.cwd),
            platform="cli",
            agent_context="primary",
            hermes_home=str(default_memory_home()),
        )
        # Refresh after bind so tool_extra identity matches live session handles
        kernel.refresh_tool_extra()

    # Host adapters: memory_backend always when store present.
    # ask_user_cli: auto on TTY unless CODEDOGGY_ASK_USER_CLI=0 (tests stay quiet).
    # scheduler: auto-start light tick thread unless CODEDOGGY_SCHEDULER_TICK=0.
    try:
        import os as _os
        import sys as _sys

        def _env_flag(name: str) -> str | None:
            raw = _os.environ.get(name, "").strip().lower()
            if raw in {"1", "true", "yes", "on"}:
                return "on"
            if raw in {"0", "false", "no", "off"}:
                return "off"
            return None

        _ask_flag = _env_flag("CODEDOGGY_ASK_USER_CLI")
        if _ask_flag == "on":
            _ask = True
        elif _ask_flag == "off":
            _ask = False
        else:
            try:
                _ask = bool(_sys.stdin.isatty() and _sys.stdout.isatty())
            except Exception:  # noqa: BLE001
                _ask = False

        _tick_flag = _env_flag("CODEDOGGY_SCHEDULER_TICK")
        # Default ON for product sessions; tests can set CODEDOGGY_SCHEDULER_TICK=0
        _tick = True if _tick_flag is None else (_tick_flag == "on")
        # Avoid daemon threads under pytest unless explicitly requested
        if _os.environ.get("PYTEST_CURRENT_TEST") and _tick_flag is None:
            _tick = False

        # Memory = Hermes (no default Grok memory_backend / memory_search).
        # Opt-in: CODEDOGGY_GROK_MEMORY_BACKEND=1 + register_optional_grok_memory_tools.
        _mem_be = _env_flag("CODEDOGGY_GROK_MEMORY_BACKEND") == "on"
        kernel.wire_host_adapters(
            enable_memory_backend=_mem_be,
            enable_ask_user_cli=_ask,
            enable_scheduler_tick=_tick,
            start_scheduler_thread=_tick,
            submit_prompt=session.submit_prompt,
        )
    except Exception:  # noqa: BLE001
        pass

    if session_store is not None:
        ownership = session_store.claim_session(
            str(session.id),
            session.cwd,
            goal=goal,
            title=(goal or "")[:80] or None,
        )
        if not ownership.allowed:
            try:
                session.close()
            finally:
                raise ValueError(
                    "refusing to hydrate session "
                    f"{str(session.id)!r} in workspace "
                    f"{ownership.requested_cwd!r}: {ownership.reason}; "
                    f"stored workspace={ownership.stored_cwd!r}"
                )
        # True restore: hydrate live transcript + plan_mode.json when resuming
        if provisional_id:
            n = kernel.hydrate_from_store()
            if n:
                pass  # live_messages filled
        else:
            # Fresh session id may still reattach if files exist for this id
            try:
                kernel.load_plan_mode_state()
            except Exception:  # noqa: BLE001
                pass
            try:
                kernel.load_todo_state()
            except Exception:  # noqa: BLE001
                pass

    # Grok MCP runtime: Session-owned clients, progressive initialization,
    # dispatcher, config diff/reload, and bounded recovery. Start it last so a
    # later bootstrap failure cannot orphan transport threads or child servers.
    # Under pytest the implicit product default is disabled so a developer's
    # global ~/.grok servers cannot leak into isolated unit tests.
    try:
        import os as _os

        if enable_mcp is None:
            raw_mcp = _os.environ.get("CODEDOGGY_MCP", "").strip().lower()
            if mcp_servers is not None:
                _enable_mcp = True
            elif raw_mcp in {"1", "true", "yes", "on"}:
                _enable_mcp = True
            elif raw_mcp in {"0", "false", "no", "off"}:
                _enable_mcp = False
            else:
                _enable_mcp = not bool(_os.environ.get("PYTEST_CURRENT_TEST"))
        else:
            _enable_mcp = bool(enable_mcp)

        if _enable_mcp:
            from codedoggy.mcp.runtime import McpRuntime

            mcp_runtime = McpRuntime(
                cwd_path,
                session_id=str(session.id),
                configs=mcp_servers,
                watch=mcp_watch,
                auto_restart=mcp_auto_restart,
            )
            mcp_runtime.start()
            mcp_runtime.attach_kernel(kernel)
    except Exception:  # noqa: BLE001
        logger.exception("failed to start Grok-aligned MCP runtime")
        if enable_mcp is True or mcp_servers is not None:
            try:
                session.close()
            finally:
                raise
    return session


def _default_system_prompt(goal: str | None) -> str:
    """Grok ``prompt.md`` structure (source-level) + CodeDoggy product appendix."""
    from codedoggy.prompt.grok_system import build_main_system_prompt

    return build_main_system_prompt(goal)
