"""Thread-safe presentation snapshots for the boss-view TUI."""

from __future__ import annotations

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


class TaskLedger:
    """Single synchronized source for the TUI's task hierarchy."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._tasks: list[TaskView] = []
        self._next_task = 1

    def create(self, prompt: str) -> TaskView:
        clean = " ".join(prompt.strip().split())
        title = clean if len(clean) <= 48 else clean[:47].rstrip() + "…"
        with self._lock:
            task_id = f"task_{self._next_task:03d}"
            self._next_task += 1
            task = TaskView(
                id=task_id,
                title=title or "未命名任务",
                agents=[AgentView(id=f"{task_id}:main", label="MAIN", status="running")],
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

    def set_task_phase(self, task_id: str, phase: str) -> None:
        with self._lock:
            task = self._find_task(task_id)
            if task is not None:
                task.phase = phase

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

    def _find_task(self, task_id: str) -> TaskView | None:
        return next((item for item in self._tasks if item.id == task_id), None)
