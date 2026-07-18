"""One-shot wiring: dual brains + context + memory manager + tools policy + audit."""

from __future__ import annotations

from pathlib import Path

from codedoggy.audit.model_auditor import ModelAuditor
from codedoggy.audit.services import AuditServices
from codedoggy.memory.manager import MemoryManager
from codedoggy.memory.session_store import SessionStore, default_session_db_path
from codedoggy.memory.store import MemoryStore
from codedoggy.model.chat_sampler import ChatSampler
from codedoggy.model.profiles import ModelProfiles, model_profiles_from_env
from codedoggy.model.provider import ChatClient
from codedoggy.session.extensions import SessionExtensions
from codedoggy.session.session import Session
from codedoggy.tools.policy import WorkspacePolicy
from codedoggy.tools.registry import FinalizedToolset, ToolRegistryBuilder
from codedoggy.turn.runner import AgentTurnRunner


def build_session(
    cwd: str | Path,
    *,
    goal: str | None = None,
    max_turns: int | None = 32,
    system_prompt: str | None = None,
    enable_memory: bool = True,
    enable_session_store: bool = True,
    enable_audit: bool = True,
    enable_policy: bool = True,
    enable_graph: bool = True,
    memory_dir: str | Path | None = None,
    session_db: str | Path | None = None,
    profiles: ModelProfiles | None = None,
    main_client: ChatClient | None = None,
    audit_client: ChatClient | None = None,
    tools: FinalizedToolset | None = None,
    session_id: str | None = None,
) -> Session:
    """Create a Session with all four pillars fused.

    - **Context**: ``ContextCompactor`` (Grok pipeline + prefire + rewind)
    - **Memory**: ``MemoryManager`` (curated + FTS + external slot)
    - **Tools**: ``FinalizedToolset`` + ``WorkspacePolicy`` + ``code_nav``
    - **Shadow**: ``ModelAuditor`` + selector from MemoryManager
    - **Graph**: ``CodebaseGraph`` (xai-codebase-graph API spirit)
    """
    prof = profiles or model_profiles_from_env()
    main = main_client or prof.main_client()
    audit_cli = audit_client or prof.audit_client()
    cwd_path = Path(cwd).resolve()

    toolset = tools or ToolRegistryBuilder.new().finalize()

    memory: MemoryStore | None = None
    if enable_memory:
        memory = MemoryStore(memory_dir=memory_dir) if memory_dir else MemoryStore()
        memory.load_from_disk()

    session_store: SessionStore | None = None
    if enable_session_store:
        db = Path(session_db) if session_db else default_session_db_path()
        session_store = SessionStore(db)

    # Memory pillar orchestration
    memory_manager: MemoryManager | None = None
    if enable_memory or enable_session_store:
        memory_manager = MemoryManager.create_default(
            curated=memory,
            session_store=session_store,
        )

    selector = memory_manager.as_audit_selector() if memory_manager else None

    audit_svc: AuditServices | None = None
    if enable_audit:
        audit_svc = AuditServices.create(
            auditor=ModelAuditor(audit_cli),
            memory_selector=selector,
            memory_store=memory,
            agent_id="main",
        )

    policy: WorkspacePolicy | None = None
    if enable_policy:
        policy = WorkspacePolicy.from_env(cwd_path)

    graph = None
    if enable_graph:
        from codedoggy.graph.handle import CodebaseGraph

        graph = CodebaseGraph(cwd_path)
        # Lazy index on first code_nav use (ensure_indexed)

    default_system = system_prompt
    if default_system is None:
        default_system = _default_system_prompt(goal)

    from codedoggy.context.compactor import ContextCompactor

    compactor = ContextCompactor.from_env(
        summary_client=audit_cli if enable_audit else None,
        memory_store=memory,
        session_store=session_store,
        memory_manager=memory_manager,
    )

    runner = AgentTurnRunner(
        sampler=ChatSampler(main),
        tools=toolset,
        system_prompt=default_system,
        context_compactor=compactor,
    )

    session = Session.create(
        cwd_path,
        max_turns=max_turns,
        session_id=session_id,
        goal=goal,
        extensions=SessionExtensions(
            turn_runner=runner,
            tools=toolset,
            context=compactor,
            memory=memory,
            memory_manager=memory_manager,
            session_store=session_store,
            audit=audit_svc,
            policy=policy,
            graph=graph,
        ),
    )
    if session_store is not None:
        session_store.ensure_session(
            str(session.id),
            cwd=str(session.cwd),
            goal=goal,
            title=(goal or "")[:80] or None,
        )
    return session


def _default_system_prompt(goal: str | None) -> str:
    lines = [
        "You are CodeDoggy, a coding agent with full tool access in the workspace.",
        "Prefer dedicated tools (read_file, search_replace, grep, list_dir, code_nav) over shell when possible.",
        "Use code_nav for go-to-definition / find-references (code graph); grep for free text.",
        "Use session_search for past conversations; curated MEMORY.md is injected at session start.",
        "After edits, if shadow P0 feedback (影子) appears in tool results, address it before continuing.",
        "Shadow is write-time soft review — not a separate offline audit of the repo.",
        "Workspace policy may deny writes to protected paths (.git, .env, …).",
    ]
    if goal and goal.strip():
        lines.append(f"Session goal: {goal.strip()}")
    return "\n".join(lines)
