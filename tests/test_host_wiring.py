"""Product host wiring: memory_backend, scheduler fire inject, ask_user flags."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from codedoggy.host.memory_backend import build_memory_backend
from codedoggy.host.scheduler_runtime import start_scheduler_runtime
from codedoggy.memory.store import MemoryStore
from codedoggy.orchestration.prompt_queue import InterjectionBuffer
from codedoggy.tools.runtime import ToolCallContext
from codedoggy.tools.scheduler import Scheduler


def test_memory_backend_search_shapes(tmp_path: Path) -> None:
    store = MemoryStore(memory_dir=tmp_path / "m")
    store.load_from_disk()
    store.add("memory", "use ripgrep for code search in this repo")
    be = build_memory_backend(store)
    assert be is not None
    hits = be.search("ripgrep code")
    assert hits
    assert hits[0]["snippet"]
    assert "score" in hits[0]


def test_memory_search_tool_with_backend(tmp_path: Path) -> None:
    from codedoggy.tools import ToolRegistryBuilder

    store = MemoryStore(memory_dir=tmp_path / "m")
    store.load_from_disk()
    store.add("memory", "prefer small diffs")
    be = build_memory_backend(store)
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={"memory_store": store, "memory_backend": be},
    )
    out = tools.call("memory_search", {"query": "small diffs"}, ctx)
    assert "Found " in out
    assert "small diffs" in out.lower()


def test_scheduler_runtime_injects_interjection(tmp_path: Path) -> None:
    class FakeKernel:
        def __init__(self) -> None:
            self.scheduler = Scheduler()
            self.interjection_buffer = InterjectionBuffer()
            self.prompt_queue = None
            self.tool_extra: dict = {}

    k = FakeKernel()
    tid = k.scheduler.create(
        interval="60s",
        prompt="scheduled hello from tick",
        recurring=False,
        fire_immediately=True,
    )
    assert tid
    handle = start_scheduler_runtime(k, start_thread=False)
    assert handle is not None
    results = handle.poll_once()
    assert results
    assert results[0].prompt == "scheduled hello from tick"
    # on_fire should have pushed into interjection buffer
    drained = k.interjection_buffer.drain()
    assert drained
    assert "scheduled hello from tick" in drained[0].text
    handle.stop()


def test_job_object_module_imports() -> None:
    from codedoggy.tools.util import job_object

    assert hasattr(job_object, "create_and_assign_job")
    assert hasattr(job_object, "terminate_job_for_pid")
    assert hasattr(job_object, "kill_process_tree")
    # On non-Windows, create returns False
    import sys

    if sys.platform != "win32":
        assert job_object.create_and_assign_job(1) is False


def test_kill_process_tree_shared_import() -> None:
    """bash and task_manager must share the same killer (no drift)."""
    from codedoggy.tools.builtins.run_terminal_cmd import kill_process_tree as k1
    from codedoggy.tools.task_manager import _kill_process_tree as k2
    from codedoggy.tools.util.job_object import kill_process_tree as k0

    # Wrappers exist and call through (smoke: no-op on dead-like mock)
    class _Dead:
        pid = 0

        def poll(self):
            return 0

    k0(_Dead())
    k1(_Dead())  # type: ignore[arg-type]
    k2(_Dead())  # type: ignore[arg-type]
