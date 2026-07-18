"""parallel_tasks — main-agent aggressive parallel fan-out + aggregate."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from codedoggy.bootstrap import build_session, _default_system_prompt
from codedoggy.orchestration.agent_def import resolve_agent_definition
from codedoggy.orchestration.subagent import (
    SubagentCoordinator,
    SubagentRequest,
    SubagentSnapshot,
    format_parallel_aggregate,
)
from codedoggy.tools.builtins.parallel_tasks import (
    MAX_PARALLEL_TASKS,
    ParallelTasksTool,
)
from codedoggy.tools.registry import ToolRegistryBuilder
from codedoggy.tools.runtime import ToolCallContext, ToolError


def _ctx(
    tmp_path: Path,
    *,
    coord: SubagentCoordinator | None = None,
    run_fn=None,
    extra: dict | None = None,
    session_id: str = "parent-session",
) -> ToolCallContext:
    bag = dict(extra or {})
    if coord is not None:
        bag["subagent_coordinator"] = coord
    if run_fn is not None:
        bag["subagent_run_fn"] = run_fn
    return ToolCallContext(cwd=tmp_path, session_id=session_id, extra=bag)


def _slow_parallel_run(*, delay: float = 0.15, marker: str = "ok"):
    """Children that overlap when truly parallel."""

    active = 0
    peak = 0
    lock = threading.Lock()

    def run_fn(req: SubagentRequest, cancel) -> SubagentSnapshot:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(delay)
            return SubagentSnapshot(
                subagent_id=req.id,
                subagent_type=req.subagent_type,
                status="completed",
                description=req.description,
                output=f"{marker}:{req.description}",
                tool_calls=1,
                turns=1,
                duration_ms=int(delay * 1000),
            )
        finally:
            with lock:
                active -= 1

    run_fn.peak = lambda: peak  # type: ignore[attr-defined]
    return run_fn


def test_general_purpose_agent_resolves() -> None:
    d = resolve_agent_definition("general-purpose")
    assert d is not None
    assert d.name == "general-purpose"
    assert d.background is True


def test_format_parallel_aggregate() -> None:
    snaps = [
        SubagentSnapshot(
            subagent_id="sub_a",
            subagent_type="explore",
            status="completed",
            description="scan auth",
            output="found AuthService",
        ),
        SubagentSnapshot(
            subagent_id="sub_b",
            subagent_type="explore",
            status="failed",
            description="scan payments",
            error="timeout",
        ),
    ]
    text = format_parallel_aggregate(snaps)
    assert "Parallel fan-out complete (2 tasks)" in text
    assert "scan auth" in text
    assert "found AuthService" in text
    assert "failed" in text
    assert "MAIN" in text or "synthesise" in text.lower() or "synthesize" in text.lower()
    assert "completed=1" in text
    assert "failed=1" in text


def test_spawn_many_and_wait_all_overlap() -> None:
    coord = SubagentCoordinator(max_workers=4)
    run_fn = _slow_parallel_run(delay=0.2)
    reqs = [
        SubagentRequest(
            subagent_type="general-purpose",
            prompt=f"work {i}",
            description=f"slice-{i}",
            parent_session_id="p1",
        )
        for i in range(3)
    ]
    t0 = time.perf_counter()
    snaps = coord.spawn_many(reqs, run_fn=run_fn)
    ids = [s.subagent_id for s in snaps]
    assert all(s.status in {"pending", "running", "completed"} for s in snaps)
    final = coord.wait_all(ids, timeout_ms=10_000)
    elapsed = time.perf_counter() - t0
    assert all(s.status == "completed" for s in final)
    # True parallel: wall clock ~ one delay, not 3×
    assert elapsed < 0.55, f"expected parallel wall time, got {elapsed:.2f}s"
    assert run_fn.peak() >= 2  # type: ignore[attr-defined]
    coord.shutdown(wait=False)


def test_parallel_tasks_tool_aggregate(tmp_path: Path) -> None:
    coord = SubagentCoordinator(max_workers=4)
    run_fn = _slow_parallel_run(delay=0.05, marker="child")
    tool = ParallelTasksTool()
    ctx = _ctx(tmp_path, coord=coord, run_fn=run_fn)
    out = tool.run(
        ctx,
        {
            "tasks": [
                {
                    "prompt": "find auth bugs",
                    "description": "auth slice",
                    "subagent_type": "explore",
                },
                {
                    "prompt": "find cache bugs",
                    "description": "cache slice",
                    "subagent_type": "explore",
                },
            ],
            "timeout_ms": 10_000,
        },
    )
    assert "Parallel fan-out complete (2 tasks)" in out
    assert "auth slice" in out
    assert "cache slice" in out
    assert "child:auth slice" in out
    assert "child:cache slice" in out
    assert "MAIN" in out
    coord.shutdown(wait=False)


def test_parallel_tasks_empty_and_cap(tmp_path: Path) -> None:
    coord = SubagentCoordinator()
    tool = ParallelTasksTool()
    ctx = _ctx(tmp_path, coord=coord, run_fn=_slow_parallel_run(delay=0.01))
    with pytest.raises(ToolError):
        tool.run(ctx, {"tasks": []})
    too_many = [
        {"prompt": f"p{i}", "description": f"d{i}", "subagent_type": "explore"}
        for i in range(MAX_PARALLEL_TASKS + 1)
    ]
    with pytest.raises(ToolError):
        tool.run(ctx, {"tasks": too_many})
    coord.shutdown(wait=False)


def test_parallel_tasks_missing_backend(tmp_path: Path) -> None:
    tool = ParallelTasksTool()
    ctx = ToolCallContext(cwd=tmp_path, session_id="s", extra={})
    with pytest.raises(ToolError) as ei:
        tool.run(
            ctx,
            {"tasks": [{"prompt": "x", "description": "y", "subagent_type": "explore"}]},
        )
    assert "missing" in str(ei.value).lower() or "coordinator" in str(ei.value).lower()


def test_tool_registry_includes_parallel_tasks() -> None:
    names = set(ToolRegistryBuilder.new().finalize().client_names())
    assert "parallel_tasks" in names


def test_system_prompt_main_parallel_tendency_not_auto() -> None:
    text = _default_system_prompt(None)
    low = text.lower()
    assert "parallel_tasks" in low
    assert "main" in low
    assert "shadow" not in low
    # Agency: MAIN decides; harness does not auto-fan-out
    assert "does not" in low or "not" in low
    assert "auto" in low or "auto-fan" in low or "auto-fans" in low
    assert "bias" in low or "tendency" in low or "prefer" in low
    assert "wait=false" in low or "wait=false" in text
    assert "serial" in low


def test_parallel_tasks_wait_false_returns_immediately(tmp_path: Path) -> None:
    coord = SubagentCoordinator(max_workers=4)
    run_fn = _slow_parallel_run(delay=0.4)
    tool = ParallelTasksTool()
    ctx = _ctx(tmp_path, coord=coord, run_fn=run_fn)
    t0 = time.perf_counter()
    out = tool.run(
        ctx,
        {
            "wait": False,
            "tasks": [
                {
                    "prompt": "slice a",
                    "description": "lane-a",
                    "subagent_type": "explore",
                },
                {
                    "prompt": "slice b",
                    "description": "lane-b",
                    "subagent_type": "explore",
                },
            ],
        },
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.25, f"wait=false must not block on children, got {elapsed:.2f}s"
    assert "Parallel fan-out started" in out
    assert "wait=false" in out.lower() or "you chose" in out.lower()
    assert "task_ids:" in out
    assert "lane-a" in out and "lane-b" in out
    # Children still finish in background
    import re

    ids = re.findall(r"`(sub_[a-f0-9]+)`", out)
    if not ids:
        # fallback: task_ids list form
        m = re.search(r"task_ids:\s*(\[[^\]]+\])", out)
        assert m, out
        ids = eval(m.group(1))  # noqa: S307 — test-only id list
    final = coord.wait_all(ids, timeout_ms=10_000)
    assert all(s.status == "completed" for s in final)
    coord.shutdown(wait=False)


def test_build_session_wires_parallel_coordinator(tmp_path: Path) -> None:
    from codedoggy.model import ModelConfig, CompletionResult
    from codedoggy.model.types import ChatMessage

    class Fake:
        def __init__(self) -> None:
            self.config = ModelConfig(
                provider="fake", model="m", base_url="http://x", api_key="x"
            )
            self.n = 0

        def complete(self, messages, **kwargs):
            self.n += 1
            return CompletionResult(content="done", model="m")

    s = build_session(tmp_path, main_client=Fake(), enable_memory=False, enable_graph=False)
    try:
        k = s.extensions.kernel
        assert k is not None
        assert k.subagent_coordinator is not None
        assert "parallel_tasks" in (k.tool_extra or {}) or "subagent_coordinator" in (
            k.tool_extra or {}
        )
        assert k.tool_extra.get("subagent_coordinator") is k.subagent_coordinator
        # Children strip nested parallel_tasks
        from codedoggy.orchestration.subagent import _strip_nested_spawn

        stripped = _strip_nested_spawn(s.extensions.tools)
        assert "parallel_tasks" not in set(stripped.client_names())
    finally:
        s.close()
