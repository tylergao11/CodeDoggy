"""Thread-safe presentation snapshots for the boss-view TUI."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock


@dataclass(slots=True)
class AgentView:
    """One Agent participating in one boss task."""

    id: str
    label: str
    status: str = "waiting"
    output: str = ""
    description: str = ""


@dataclass(slots=True)
class TaskView:
    """A boss-level task and the reports that belong to it."""

    id: str
    title: str
    status: str = "running"
    phase: str = "dispatching"
    reporter: str = "MAIN"
    report: str = ""
    agents: list[AgentView] = field(default_factory=list)
    # GrokBuild plan lifecycle hung on the task card (not go-steer plan-first).
    # none | consent | planning | awaiting_approval | approved | abandoned
    plan_state: str = "none"
    plan_file: str = ""
    # Wall-clock duration for the homepage card (set on create / terminal).
    started_at: float = 0.0
    ended_at: float | None = None


class TaskLedger:
    """Single synchronized source for the TUI's task hierarchy."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._tasks: list[TaskView] = []
        self._next_task = 1

    def create(self, prompt: str) -> TaskView:
        # Keep full title; the task list wraps instead of hard-cropping storage.
        title = " ".join(prompt.strip().split()) or "未命名任务"
        with self._lock:
            task_id = f"task_{self._next_task:03d}"
            self._next_task += 1
            task = TaskView(
                id=task_id,
                title=title,
                agents=[AgentView(id=f"{task_id}:main", label="MAIN", status="running")],
                started_at=time.time(),
            )
            self._tasks.append(task)
            return deepcopy(task)

    def snapshots(self) -> list[TaskView]:
        with self._lock:
            return deepcopy(self._tasks)

    def get_agent(self, task_id: str, agent_id: str) -> AgentView | None:
        with self._lock:
            task = self._find_task(task_id)
            if task is None:
                return None
            agent = next((item for item in task.agents if item.id == agent_id), None)
            return deepcopy(agent) if agent is not None else None

    def set_report(self, task_id: str, reporter: str, report: str) -> None:
        with self._lock:
            task = self._find_task(task_id)
            if task is None:
                return
            task.reporter = reporter.strip().upper() or "MAIN"
            task.report = report.strip()

    def set_task_status(
        self,
        task_id: str,
        status: str,
    ) -> None:
        with self._lock:
            task = self._find_task(task_id)
            if task is None:
                return
            task.status = status
            if status in {"completed", "failed", "cancelled", "max_turns"}:
                if task.ended_at is None:
                    task.ended_at = time.time()
                if task.started_at <= 0:
                    task.started_at = task.ended_at

    def finish_task(self, task_id: str, status: str) -> None:
        """Commit the one terminal state used by the task list."""
        with self._lock:
            task = self._find_task(task_id)
            if task is None:
                return
            task.status = status
            task.phase = {
                "completed": "done",
                "cancelled": "cancelled",
            }.get(status, "failed")
            if task.ended_at is None:
                task.ended_at = time.time()
            if task.started_at <= 0:
                task.started_at = task.ended_at
            # Finished cards must not keep draft/review chrome ("计划起草中").
            if task.plan_state in {
                "planning",
                "consent",
                "awaiting_approval",
            }:
                task.plan_state = "none"

    def set_task_phase(self, task_id: str, phase: str) -> None:
        with self._lock:
            task = self._find_task(task_id)
            if task is not None:
                task.phase = phase

    def set_plan_state(
        self,
        task_id: str,
        plan_state: str,
        *,
        plan_file: str | None = None,
    ) -> None:
        with self._lock:
            task = self._find_task(task_id)
            if task is None:
                return
            task.plan_state = plan_state
            if plan_file is not None:
                task.plan_file = plan_file
            if plan_state == "planning":
                task.phase = "planning"
            elif plan_state == "awaiting_approval":
                task.phase = "plan_review"
            elif plan_state in {"approved", "abandoned", "none"}:
                # Leave execution phases alone once work is underway.
                if task.phase in {"planning", "plan_review", "dispatching"}:
                    task.phase = "dispatching"

    def update_agent(
        self,
        task_id: str,
        agent_id: str,
        *,
        label: str,
        status: str,
        output: str | None = None,
        description: str | None = None,
    ) -> None:
        with self._lock:
            task = self._find_task(task_id)
            if task is None:
                return
            agent = next((item for item in task.agents if item.id == agent_id), None)
            if agent is None:
                agent = AgentView(id=agent_id, label=label.strip().upper() or "AGENT")
                task.agents.append(agent)
            agent.label = label.strip().upper() or agent.label
            agent.status = status
            if output is not None:
                agent.output = output.strip()
            if description is not None:
                agent.description = description.strip()

    def apply_agent_status(
        self,
        task_id: str,
        agent_id: str,
        *,
        label: str,
        status: str,
        output: str | None = None,
        description: str | None = None,
    ) -> bool:
        """Route live vs terminal agent writes through the correct fence.

        Non-terminal statuses use ``update_live_agent`` (refuse revive after
        task/agent done). Terminal statuses use ``update_agent`` only when the
        task is still open or we are reconciling a terminal snap.
        """
        st = (status or "").strip().lower()
        if st in {"pending", "running", "waiting"}:
            return self.update_live_agent(
                task_id,
                agent_id,
                label=label,
                status=st,
                output=output,
                description=description,
            )
        with self._lock:
            task = self._find_task(task_id)
            if task is None:
                return False
            # Allow terminal reconcile even after task finished; skip only if
            # task cancelled and agent already cancelled (no-op noise).
        self.update_agent(
            task_id,
            agent_id,
            label=label,
            status=st or status,
            output=output,
            description=description,
        )
        return True

    def update_live_agent(
        self,
        task_id: str,
        agent_id: str,
        *,
        label: str,
        status: str,
        output: str | None = None,
        description: str | None = None,
    ) -> bool:
        """Apply a non-terminal callback only while its task/agent is live.

        Model stream owners are intentionally abandonable on cancellation.  A
        callback already in flight may therefore arrive after the turn commits
        its terminal state.  This atomic ledger fence prevents that old write
        from reviving a completed/failed/cancelled card as ``running``.
        """
        with self._lock:
            task = self._find_task(task_id)
            if task is None or task.status in {"completed", "failed", "cancelled"}:
                return False
            agent = next((item for item in task.agents if item.id == agent_id), None)
            if agent is not None and agent.status in {
                "completed",
                "failed",
                "cancelled",
            }:
                return False
            if agent is None:
                agent = AgentView(id=agent_id, label=label.strip().upper() or "AGENT")
                task.agents.append(agent)
            agent.label = label.strip().upper() or agent.label
            agent.status = status
            if output is not None:
                agent.output = output.strip()
            if description is not None:
                agent.description = description.strip()
            return True

    def _find_task(self, task_id: str) -> TaskView | None:
        return next((item for item in self._tasks if item.id == task_id), None)
