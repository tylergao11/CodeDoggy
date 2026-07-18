"""Bundle of resident-audit services bound on a Session."""

from __future__ import annotations

from dataclasses import dataclass

from codedoggy.audit.auditor import PassThroughAuditor, ResidentAuditor
from codedoggy.audit.memory_select import (
    CuratedMemorySelector,
    MemorySelector,
    NoopMemorySelector,
)
from codedoggy.audit.trajectory import MutationTrajectory


@dataclass(slots=True)
class AuditServices:
    """Everything the loop needs for write-time resident audit.

    Attach via ``SessionExtensions.audit``. Interfaces are stable so Hermes
    memory selection and model-brain auditors can swap in without retouching
    the turn loop.
    """

    trajectory: MutationTrajectory
    auditor: ResidentAuditor
    memory_selector: MemorySelector
    agent_id: str = "main"

    @classmethod
    def create(
        cls,
        *,
        auditor: ResidentAuditor | None = None,
        memory_selector: MemorySelector | None = None,
        trajectory: MutationTrajectory | None = None,
        agent_id: str = "main",
        memory_store: object | None = None,
        session_store: object | None = None,
        # Convenience: build ModelAuditor from env/Ollama when auditor is None
        # and use_model_auditor is True.
        use_model_auditor: bool = False,
        model_client: object | None = None,
    ) -> AuditServices:
        sel = memory_selector
        if sel is None:
            if memory_store is not None or session_store is not None:
                from codedoggy.memory.hermes_select import HermesMemorySelector

                sel = HermesMemorySelector(
                    curated_store=memory_store,
                    session_store=session_store,  # type: ignore[arg-type]
                )
            else:
                sel = NoopMemorySelector()
        elif (
            isinstance(sel, CuratedMemorySelector)
            and sel.store is None
            and memory_store is not None
        ):
            sel.bind_store(memory_store)

        resolved_auditor = auditor
        if resolved_auditor is None and use_model_auditor:
            from codedoggy.audit.model_auditor import ModelAuditor
            from codedoggy.model.registry import create_client, model_config_from_env

            client = model_client
            if client is None:
                client = create_client(model_config_from_env())
            resolved_auditor = ModelAuditor(client)  # type: ignore[arg-type]
        if resolved_auditor is None:
            resolved_auditor = PassThroughAuditor()

        return cls(
            trajectory=trajectory if trajectory is not None else MutationTrajectory(),
            auditor=resolved_auditor,
            memory_selector=sel,
            agent_id=agent_id,
        )
