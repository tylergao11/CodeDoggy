"""Grok-aligned batch tool dispatch: path-lock parallel phase 2."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from codedoggy.orchestration.tool_pipeline import (
    execute_approved_batch,
    execute_tool_calls_two_phase,
    prepare_tool_batch,
)
from codedoggy.orchestration.types import PrecheckVerdict
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.registry import FinalizedToolset, ToolRegistryBuilder
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolId,
)
from codedoggy.turn.loop import run_agent_loop
from codedoggy.turn.types import Message, Role, SampleResult, ToolCall


class _SlowReadTool(Tool):
    """Sleepy read that tracks peak concurrency."""

    active = 0
    peak = 0
    lock = threading.Lock()

    def id(self) -> ToolId:
        return ToolId("slow_read")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Read

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="slow_read", description="slow test read")

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"target_file": {"type": "string"}},
            "required": ["target_file"],
        }

    def run(self, ctx: ToolCallContext, args: dict) -> str:
        with _SlowReadTool.lock:
            _SlowReadTool.active += 1
            _SlowReadTool.peak = max(_SlowReadTool.peak, _SlowReadTool.active)
        try:
            time.sleep(0.2)
            path = Path(ctx.cwd) / str(args.get("target_file") or "x")
            return path.read_text(encoding="utf-8") if path.is_file() else "missing"
        finally:
            with _SlowReadTool.lock:
                _SlowReadTool.active -= 1


class _SlowWriteTool(Tool):
    order: list[str] = []
    lock = threading.Lock()

    def id(self) -> ToolId:
        return ToolId("slow_write")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Write

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="slow_write", description="slow test write")

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "contents": {"type": "string"},
            },
            "required": ["file_path", "contents"],
        }

    def run(self, ctx: ToolCallContext, args: dict) -> str:
        tag = str(args.get("contents") or "")
        path = Path(ctx.cwd) / str(args.get("file_path") or "w.txt")
        # Hold work under path lock from pipeline — sleep while "writing"
        time.sleep(0.15)
        path.write_text(tag, encoding="utf-8")
        with _SlowWriteTool.lock:
            _SlowWriteTool.order.append(tag)
        return f"wrote {tag}"


def _with_slow_tools() -> FinalizedToolset:
    b = ToolRegistryBuilder.empty()
    b.register(_SlowReadTool())
    b.register(_SlowWriteTool())
    # product surface allow-list would drop these; finalize without product list
    return b.finalize(product_surface=False)


class ScriptedSampler:
    def __init__(self, script: list[SampleResult]) -> None:
        self.script = list(script)
        self.calls = 0

    def sample(self, messages: list[Message], tools) -> SampleResult:
        if self.calls >= len(self.script):
            return SampleResult(content="(done)")
        out = self.script[self.calls]
        self.calls += 1
        return out


def test_execute_approved_batch_reads_overlap(tmp_path: Path) -> None:
    tools = _with_slow_tools()
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    _SlowReadTool.active = 0
    _SlowReadTool.peak = 0
    calls = [
        ToolCall(id="1", name="slow_read", arguments={"target_file": "a.txt"}),
        ToolCall(id="2", name="slow_read", arguments={"target_file": "b.txt"}),
    ]
    phase1 = prepare_tool_batch(tools, calls, cwd=tmp_path)
    assert len(phase1.approved) == 2
    t0 = time.perf_counter()
    out = execute_approved_batch(tools, phase1.approved, cwd=tmp_path, parallel=True)
    elapsed = time.perf_counter() - t0
    assert out[0].ok and out[1].ok
    assert elapsed < 0.35, f"expected parallel wall clock, got {elapsed:.2f}s"
    assert _SlowReadTool.peak >= 2


def test_same_path_writes_serialized(tmp_path: Path) -> None:
    tools = _with_slow_tools()
    _SlowWriteTool.order = []
    calls = [
        ToolCall(
            id="1",
            name="slow_write",
            arguments={"file_path": "same.txt", "contents": "first"},
        ),
        ToolCall(
            id="2",
            name="slow_write",
            arguments={"file_path": "same.txt", "contents": "second"},
        ),
    ]
    phase1 = prepare_tool_batch(tools, calls, cwd=tmp_path)
    t0 = time.perf_counter()
    execute_approved_batch(tools, phase1.approved, cwd=tmp_path, parallel=True)
    elapsed = time.perf_counter() - t0
    # Same path lock → roughly serial wall time
    assert elapsed >= 0.25, f"same-path should serialize, got {elapsed:.2f}s"
    assert (tmp_path / "same.txt").read_text(encoding="utf-8") in {"first", "second"}
    assert len(_SlowWriteTool.order) == 2


def test_two_phase_parallel_flag_in_metadata(tmp_path: Path) -> None:
    tools = _with_slow_tools()
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    batch = execute_tool_calls_two_phase(
        tools,
        [
            ToolCall(id="1", name="slow_read", arguments={"target_file": "a.txt"}),
            ToolCall(id="2", name="slow_read", arguments={"target_file": "b.txt"}),
        ],
        cwd=tmp_path,
        parallel=True,
    )
    assert batch.metadata.get("parallel") is True
    assert all(r.ok for r in batch.records)


def test_loop_uses_path_lock_parallel_dispatch(tmp_path: Path) -> None:
    """Main agent loop phase-2 must overlap independent tools (Grok dispatch)."""
    tools = _with_slow_tools()
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    _SlowReadTool.active = 0
    _SlowReadTool.peak = 0
    sampler = ScriptedSampler(
        [
            SampleResult(
                content="reading",
                tool_calls=[
                    ToolCall(
                        id="1", name="slow_read", arguments={"target_file": "a.txt"}
                    ),
                    ToolCall(
                        id="2", name="slow_read", arguments={"target_file": "b.txt"}
                    ),
                ],
            ),
            SampleResult(content="done"),
        ]
    )
    t0 = time.perf_counter()
    result = run_agent_loop(
        user_text="go",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        max_turns=5,
    )
    elapsed = time.perf_counter() - t0
    assert result.completed
    assert result.final_text == "done"
    assert "slow_read" in result.tools_called
    assert elapsed < 0.45, f"loop batch should parallelize reads, got {elapsed:.2f}s"
    assert _SlowReadTool.peak >= 2
    # Observations still in model emission order
    tool_msgs = [m for m in result.messages if m.role is Role.TOOL]
    assert len(tool_msgs) >= 2
