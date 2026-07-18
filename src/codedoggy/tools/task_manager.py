"""Background task manager — Grok LocalTerminalBackend spirit (Python).

Owns shell (and similar) background processes for one session:
  - spawn → immediate task_id + output_file
  - snapshot / wait / kill / list
  - output drained to a session-scoped file (model reads via get_task_output)

Not a full actor loop; thread-safe enough for tool dispatch + path locks.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


# Grok BACKGROUND_MAX_RUNTIME = 10h
DEFAULT_BACKGROUND_MAX_RUNTIME_S = 36_000.0
# Grok COMPLETED_TASK_TTL
DEFAULT_COMPLETED_TTL_S = 300.0
MAX_COMPLETED_SNAPSHOTS = 100
# Max retained output in memory/file for model reads
DEFAULT_MAX_OUTPUT_CHARS = 64 * 1024 * 1024


def _rfc3339(ts: float | None = None) -> str:
    t = time.time() if ts is None else ts
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class TaskSnapshot:
    """Grok TaskSnapshot subset (model + kill/output tools)."""

    task_id: str
    command: str
    cwd: str
    start_time: float
    end_time: float | None = None
    output: str = ""
    output_file: str = ""
    truncated: bool = False
    exit_code: int | None = None
    signal: str | None = None
    completed: bool = False
    kind: str = "bash"  # bash | monitor
    explicitly_killed: bool = False
    owner_session_id: str | None = None
    description: str = ""
    pid: int | None = None
    display_command: str | None = None

    def duration_secs(self) -> float:
        end = self.end_time if self.end_time is not None else time.time()
        return max(0.0, end - self.start_time)

    def status_label(self) -> str:
        if not self.completed:
            return "running"
        if self.explicitly_killed:
            return "cancelled"
        if self.exit_code == 0:
            return "completed"
        return "failed"

    def copy(self) -> TaskSnapshot:
        return TaskSnapshot(
            task_id=self.task_id,
            command=self.command,
            cwd=self.cwd,
            start_time=self.start_time,
            end_time=self.end_time,
            output=self.output,
            output_file=self.output_file,
            truncated=self.truncated,
            exit_code=self.exit_code,
            signal=self.signal,
            completed=self.completed,
            kind=self.kind,
            explicitly_killed=self.explicitly_killed,
            owner_session_id=self.owner_session_id,
            description=self.description,
            pid=self.pid,
            display_command=self.display_command,
        )


@dataclass
class BackgroundHandle:
    task_id: str
    output_file: str
    pid: int | None = None


@dataclass
class _LiveTask:
    snap: TaskSnapshot
    proc: subprocess.Popen[bytes] | None
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    kill_requested: bool = False


class BackgroundTaskManager:
    """Session-scoped registry of background shell tasks."""

    def __init__(
        self,
        *,
        work_dir: Path | None = None,
        max_runtime_s: float = DEFAULT_BACKGROUND_MAX_RUNTIME_S,
        completed_ttl_s: float = DEFAULT_COMPLETED_TTL_S,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        self._lock = threading.RLock()
        self._tasks: dict[str, _LiveTask] = {}
        self._max_runtime_s = max_runtime_s
        self._completed_ttl_s = completed_ttl_s
        self._max_output_chars = max_output_chars
        base = work_dir or Path(tempfile.gettempdir()) / "codedoggy-tasks"
        self._work_dir = Path(base)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._closed = False

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    def spawn(
        self,
        argv: list[str],
        *,
        command: str,
        cwd: str | Path,
        env: dict[str, str] | None = None,
        description: str = "",
        owner_session_id: str | None = None,
        kind: str = "bash",
        max_runtime_s: float | None = None,
        display_command: str | None = None,
        popen_kwargs: dict[str, Any] | None = None,
    ) -> BackgroundHandle:
        """Start a process in the background; return handle immediately."""
        if self._closed:
            raise RuntimeError("task manager is closed")
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        out_path = self._work_dir / f"{task_id}.log"
        out_path.write_bytes(b"")
        start = time.time()
        snap = TaskSnapshot(
            task_id=task_id,
            command=command,
            display_command=display_command,
            cwd=str(Path(cwd).resolve()),
            start_time=start,
            output_file=str(out_path),
            kind=kind,
            owner_session_id=owner_session_id,
            description=description,
        )
        live = _LiveTask(snap=snap, proc=None)
        with self._lock:
            self._tasks[task_id] = live
            self._evict_completed_locked()

        try:
            kwargs: dict[str, Any] = {
                "cwd": str(Path(cwd).resolve()),
                "env": env,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "stdin": subprocess.DEVNULL,
            }
            if popen_kwargs:
                kwargs.update(popen_kwargs)
            proc = subprocess.Popen(argv, **kwargs)
        except OSError as e:
            with live.lock:
                live.snap.completed = True
                live.snap.end_time = time.time()
                live.snap.exit_code = -1
                live.snap.signal = "spawn_failed"
                live.snap.output = f"Failed to spawn: {e}"
                try:
                    out_path.write_text(live.snap.output, encoding="utf-8", errors="replace")
                except OSError:
                    pass
            live.done.set()
            return BackgroundHandle(task_id=task_id, output_file=str(out_path), pid=None)

        if sys.platform == "win32":
            try:
                from codedoggy.tools.util.job_object import create_and_assign_job

                create_and_assign_job(proc.pid)
            except Exception:  # noqa: BLE001
                pass

        with live.lock:
            live.proc = proc
            live.snap.pid = proc.pid

        runtime = self._max_runtime_s if max_runtime_s is None else max_runtime_s
        t = threading.Thread(
            target=self._reap,
            args=(task_id, out_path, runtime),
            name=f"bg-task-{task_id}",
            daemon=True,
        )
        t.start()
        return BackgroundHandle(task_id=task_id, output_file=str(out_path), pid=proc.pid)

    def adopt(
        self,
        proc: subprocess.Popen[bytes],
        *,
        command: str,
        cwd: str | Path,
        description: str = "",
        owner_session_id: str | None = None,
        kind: str = "bash",
        partial_output: bytes = b"",
        max_runtime_s: float | None = None,
        signal: str = "auto_backgrounded",
        display_command: str | None = None,
    ) -> BackgroundHandle:
        """Adopt an already-running process (e.g. FG timed out → background)."""
        if self._closed:
            raise RuntimeError("task manager is closed")
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        out_path = self._work_dir / f"{task_id}.log"
        if partial_output:
            try:
                out_path.write_bytes(partial_output)
            except OSError:
                out_path.write_bytes(b"")
        else:
            out_path.write_bytes(b"")
        start = time.time()
        text0 = partial_output.decode("utf-8", errors="replace") if partial_output else ""
        snap = TaskSnapshot(
            task_id=task_id,
            command=command,
            display_command=display_command,
            cwd=str(Path(cwd).resolve()),
            start_time=start,
            output=text0,
            output_file=str(out_path),
            kind=kind,
            owner_session_id=owner_session_id,
            description=description,
            pid=proc.pid,
            signal=signal,
        )
        # FG→BG adopt: ensure Job Object if FG path did not assign (or re-use existing).
        if sys.platform == "win32":
            try:
                from codedoggy.tools.util.job_object import create_and_assign_job

                create_and_assign_job(proc.pid)
            except Exception:  # noqa: BLE001
                pass

        live = _LiveTask(snap=snap, proc=proc)
        with self._lock:
            self._tasks[task_id] = live
            self._evict_completed_locked()
        runtime = self._max_runtime_s if max_runtime_s is None else max_runtime_s
        t = threading.Thread(
            target=self._reap,
            args=(task_id, out_path, runtime),
            name=f"bg-task-{task_id}",
            daemon=True,
        )
        t.start()
        return BackgroundHandle(task_id=task_id, output_file=str(out_path), pid=proc.pid)

    def get(self, task_id: str) -> TaskSnapshot | None:
        with self._lock:
            live = self._tasks.get(task_id)
            if live is None:
                return None
            with live.lock:
                self._refresh_output_locked(live)
                return live.snap.copy()

    def list_tasks(self) -> list[TaskSnapshot]:
        with self._lock:
            out: list[TaskSnapshot] = []
            for live in self._tasks.values():
                with live.lock:
                    self._refresh_output_locked(live)
                    out.append(live.snap.copy())
            return out

    def wait(self, task_id: str, *, timeout_ms: float | None = None) -> TaskSnapshot | None:
        """Wait for completion. timeout_ms=None waits forever (capped by caller)."""
        with self._lock:
            live = self._tasks.get(task_id)
        if live is None:
            return None
        if timeout_ms is None:
            live.done.wait()
        else:
            live.done.wait(timeout=max(0.0, timeout_ms / 1000.0))
        return self.get(task_id)

    def kill(self, task_id: str) -> tuple[str, str]:
        """Kill a task. Returns (outcome, message) like Grok KillOutcome.

        Outcomes match Grok: ``killed`` | ``already_exited`` | ``not_found``.
        Product-facing message strings live in kill_task tool (KillTaskResult).
        """
        with self._lock:
            live = self._tasks.get(task_id)
        if live is None:
            return "not_found", f"Task {task_id} not found"
        with live.lock:
            if live.snap.completed:
                return (
                    "already_exited",
                    "Task had already completed",
                )
            live.kill_requested = True
            proc = live.proc
        if proc is not None:
            _kill_process_tree(proc)
        # Wait briefly for reaper
        live.done.wait(timeout=3.0)
        with live.lock:
            live.snap.explicitly_killed = True
            if not live.snap.completed:
                live.snap.completed = True
                live.snap.end_time = time.time()
                live.snap.signal = live.snap.signal or "killed"
                live.snap.exit_code = live.snap.exit_code if live.snap.exit_code is not None else -1
                live.done.set()
        return "killed", "Task was terminated successfully"

    def known_ids(self) -> list[str]:
        with self._lock:
            return list(self._tasks.keys())

    def close(self) -> None:
        """Kill all running tasks (session teardown)."""
        self._closed = True
        with self._lock:
            ids = list(self._tasks.keys())
        for tid in ids:
            try:
                self.kill(tid)
            except Exception:  # noqa: BLE001
                pass

    # ----- internals -----

    def _reap(self, task_id: str, out_path: Path, max_runtime_s: float) -> None:
        with self._lock:
            live = self._tasks.get(task_id)
        if live is None:
            return
        with live.lock:
            proc = live.proc
        if proc is None:
            live.done.set()
            return

        chunks: list[bytes] = []
        total = 0
        deadline = time.time() + max_runtime_s if max_runtime_s > 0 else None

        def _reader() -> None:
            nonlocal total
            assert proc.stdout is not None
            try:
                while True:
                    block = proc.stdout.read(8192)
                    if not block:
                        break
                    chunks.append(block)
                    total += len(block)
                    # Bound retained bytes
                    if total > self._max_output_chars:
                        # keep head + mark truncated on disk later
                        pass
                    try:
                        with open(out_path, "ab") as f:
                            f.write(block)
                    except OSError:
                        pass
            except Exception:  # noqa: BLE001
                pass

        reader = threading.Thread(target=_reader, name=f"bg-read-{task_id}", daemon=True)
        reader.start()

        exit_code: int | None = None
        signal_name: str | None = None
        try:
            while True:
                if live.kill_requested:
                    _kill_process_tree(proc)
                    try:
                        exit_code = proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        exit_code = -1
                    signal_name = "killed"
                    break
                if deadline is not None and time.time() >= deadline:
                    _kill_process_tree(proc)
                    try:
                        exit_code = proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        exit_code = -1
                    signal_name = "max_runtime"
                    break
                rc = proc.poll()
                if rc is not None:
                    exit_code = rc
                    break
                time.sleep(0.05)
        finally:
            reader.join(timeout=2.0)
            try:
                if proc.stdout:
                    proc.stdout.close()
            except OSError:
                pass

        raw = b"".join(chunks)
        text = raw.decode("utf-8", errors="replace")
        truncated = len(text) > self._max_output_chars
        if truncated:
            text = text[: self._max_output_chars]

        with live.lock:
            live.snap.output = text
            live.snap.truncated = truncated or live.snap.truncated
            live.snap.exit_code = exit_code
            live.snap.signal = signal_name or live.snap.signal
            live.snap.completed = True
            live.snap.end_time = time.time()
            if live.kill_requested:
                live.snap.explicitly_killed = True
        live.done.set()

    def _refresh_output_locked(self, live: _LiveTask) -> None:
        """If still running, refresh output from file for snapshots."""
        if live.snap.completed:
            return
        path = live.snap.output_file
        if not path:
            return
        try:
            data = Path(path).read_bytes()
        except OSError:
            return
        text = data.decode("utf-8", errors="replace")
        if len(text) > self._max_output_chars:
            live.snap.output = text[: self._max_output_chars]
            live.snap.truncated = True
        else:
            live.snap.output = text

    def _evict_completed_locked(self) -> None:
        now = time.time()
        completed = [
            (tid, live)
            for tid, live in self._tasks.items()
            if live.snap.completed and live.snap.end_time is not None
        ]
        # Drop by TTL
        for tid, live in completed:
            end = live.snap.end_time or 0.0
            if now - end > self._completed_ttl_s:
                del self._tasks[tid]
        # Cap count
        completed = [
            (tid, live)
            for tid, live in self._tasks.items()
            if live.snap.completed
        ]
        if len(completed) > MAX_COMPLETED_SNAPSHOTS:
            completed.sort(key=lambda x: x[1].snap.end_time or 0.0)
            for tid, _ in completed[: len(completed) - MAX_COMPLETED_SNAPSHOTS]:
                self._tasks.pop(tid, None)


def _kill_process_tree(proc: subprocess.Popen[bytes]) -> None:
    """Shared with run_terminal_cmd — Job Object on Windows, process group on POSIX."""
    from codedoggy.tools.util.job_object import kill_process_tree as _kill

    _kill(proc)


def format_background_started(
    handle: BackgroundHandle,
    *,
    command: str,
    description: str = "",
    retrieval_hint: str | None = None,
) -> str:
    """Grok BackgroundTaskStarted model-facing XML envelope."""
    summary = description.strip() or f"Background: {command[:120]}"
    hint = retrieval_hint or (
        "Use get_task_output with this task_id to check status/output. "
        "Use kill_task to terminate if needed."
    )
    return (
        f"<task-id>{handle.task_id}</task-id>\n"
        f"<task-type>bash</task-type>\n"
        f"<output-file>{handle.output_file}</output-file>\n"
        f"<status>running</status>\n"
        f"<summary>{summary}</summary>\n"
        f"{hint}"
    )


def format_task_snapshot(
    snap: TaskSnapshot,
    *,
    max_output_chars: int = 40_000,
    read_file_name: str = "read_file",
) -> str:
    """Grok TaskOutput Result card (delegates to task_output_logic)."""
    from codedoggy.tools.grok_build.task_output_logic import (
        format_task_result_card,
        snapshot_to_result,
    )

    return format_task_result_card(
        snapshot_to_result(
            snap,
            read_file_name=read_file_name,
            max_output_chars=max_output_chars,
        )
    )


def ensure_task_manager(extra: dict[str, Any] | None) -> BackgroundTaskManager:
    """Get or create a task manager from tool extra bag."""
    bag = extra if extra is not None else {}
    tm = bag.get("task_manager")
    if isinstance(tm, BackgroundTaskManager):
        return tm
    # Lazy per-call manager (tests / tools without kernel)
    tm = BackgroundTaskManager()
    bag["task_manager"] = tm
    return tm
