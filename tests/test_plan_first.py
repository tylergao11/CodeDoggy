"""go-steer plan-first substrate tests (RequirePlanArtifact + record_plan).

Mirrors:
  docs/plan-first-design.md
  pkg/permissions gate planFirstDenial
  pkg/tools/record_plan.go
"""

from __future__ import annotations

from pathlib import Path

from codedoggy.orchestration.plan_first import (
    PLAN_EXEMPT_TOOLS,
    PlanFirstGate,
    latest_active_plan,
    next_plan_seq,
    plan_first_denial,
    require_plan_artifact_from_env,
    revoke_latest_plan,
    write_plan_artifact,
)
from codedoggy.orchestration.tool_pipeline import prepare_tool_batch, prepare_tool_call
from codedoggy.orchestration.types import PrecheckVerdict, ToolLoopOutcome
from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.builtins.record_plan import RecordPlanTool
from codedoggy.tools.runtime import ToolCallContext
from codedoggy.turn.types import ToolCall


def _tools():
    return ToolRegistryBuilder.new().finalize()


def test_env_opt_in_default_false() -> None:
    assert require_plan_artifact_from_env(default=False, environ={}) is False
    assert require_plan_artifact_from_env(default=True, environ={}) is True
    assert (
        require_plan_artifact_from_env(
            default=False, environ={"CODEDOGGY_REQUIRE_PLAN_ARTIFACT": "1"}
        )
        is True
    )
    assert (
        require_plan_artifact_from_env(
            default=True, environ={"CODEDOGGY_REQUIRE_PLAN_ARTIFACT": "0"}
        )
        is False
    )


def test_exempt_includes_record_plan_and_reads() -> None:
    from codedoggy.orchestration.plan_first import is_plan_exempt

    assert "record_plan" in PLAN_EXEMPT_TOOLS
    assert "read_file" in PLAN_EXEMPT_TOOLS
    assert "grep" in PLAN_EXEMPT_TOOLS
    assert "spawn_subagent" not in PLAN_EXEMPT_TOOLS
    assert "write" not in PLAN_EXEMPT_TOOLS
    assert "run_terminal_cmd" not in PLAN_EXEMPT_TOOLS
    # Product client name must resolve via CLIENT_ALIASES (no hardcode drift)
    assert is_plan_exempt("get_command_or_subagent_output")
    assert is_plan_exempt("wait_commands_or_subagents")
    assert not is_plan_exempt("spawn_subagent")


def test_denial_before_record(tmp_path: Path) -> None:
    gate = PlanFirstGate(require_plan_artifact=True, agents_dir=str(tmp_path / ".agents"))
    msg = plan_first_denial(gate, "write")
    assert msg is not None
    assert "record_plan" in msg
    assert plan_first_denial(gate, "read_file") is None
    assert plan_first_denial(gate, "record_plan") is None


def test_denial_clears_after_mark(tmp_path: Path) -> None:
    gate = PlanFirstGate(require_plan_artifact=True, agents_dir=str(tmp_path / ".agents"))
    gate.mark_plan_recorded()
    assert plan_first_denial(gate, "write") is None
    assert plan_first_denial(gate, "spawn_subagent") is None


def test_denial_off_when_not_required(tmp_path: Path) -> None:
    gate = PlanFirstGate(require_plan_artifact=False)
    assert plan_first_denial(gate, "write") is None


def test_empty_plan_rejected(tmp_path: Path) -> None:
    agents = tmp_path / ".agents"
    try:
        write_plan_artifact(agents, "   ")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "non-empty" in str(e)


def test_record_plan_seq_and_atomic(tmp_path: Path) -> None:
    agents = tmp_path / ".agents"
    p1, s1 = write_plan_artifact(agents, "# Goal\nDo X")
    assert s1 == 1
    assert p1.name == "plan-1.md"
    assert p1.read_text(encoding="utf-8").endswith("\n")
    p2, s2 = write_plan_artifact(agents, "# Goal\nDo Y")
    assert s2 == 2
    assert p2.name == "plan-2.md"
    assert next_plan_seq(agents / "plans") == 3


def test_revoke_latest(tmp_path: Path) -> None:
    agents = tmp_path / ".agents"
    gate = PlanFirstGate(require_plan_artifact=True, agents_dir=str(agents))
    write_plan_artifact(agents, "plan A")
    gate.mark_plan_recorded()
    assert gate.is_plan_recorded()
    latest = latest_active_plan(agents)
    assert latest is not None
    revoked = revoke_latest_plan(gate, agents)
    assert revoked is not None
    assert revoked.name.endswith("-revoked.md")
    assert not gate.is_plan_recorded()
    assert latest_active_plan(agents) is None


def test_record_plan_tool_flips_gate(tmp_path: Path) -> None:
    gate = PlanFirstGate(
        require_plan_artifact=True, agents_dir=str(tmp_path / ".agents")
    )
    tool = RecordPlanTool()
    ctx = ToolCallContext(cwd=tmp_path, extra={"plan_first_gate": gate})
    out = tool.run(ctx, {"plan": "## Goal\nShip plan-first\n"})
    assert "Plan recorded" in out
    assert gate.is_plan_recorded()
    path = tmp_path / ".agents" / "plans" / "plan-1.md"
    assert path.is_file()


def test_prepare_blocks_write_until_record(tmp_path: Path) -> None:
    tools = _tools()
    gate = PlanFirstGate(
        require_plan_artifact=True, agents_dir=str(tmp_path / ".agents")
    )
    call = ToolCall(
        id="c1",
        name="write",
        arguments={"file_path": str(tmp_path / "a.py"), "content": "x"},
    )
    pre = prepare_tool_call(
        tools,
        call,
        cwd=tmp_path,
        extra={"plan_first_gate": gate},
    )
    assert pre.verdict is PrecheckVerdict.HOOK_DENY
    assert "record_plan" in (pre.reason or "")

    # research still ok
    read_call = ToolCall(
        id="c2",
        name="read_file",
        arguments={"target_file": str(tmp_path / "missing.py")},
    )
    pre_r = prepare_tool_call(
        tools,
        read_call,
        cwd=tmp_path,
        extra={"plan_first_gate": gate},
    )
    # may fail later on missing file at execute; prepare should APPROVE
    assert pre_r.verdict is PrecheckVerdict.APPROVE

    RecordPlanTool().run(
        ToolCallContext(cwd=tmp_path, extra={"plan_first_gate": gate}),
        {"plan": "Implement a.py safely"},
    )
    pre2 = prepare_tool_call(
        tools,
        call,
        cwd=tmp_path,
        extra={"plan_first_gate": gate},
    )
    assert pre2.verdict is PrecheckVerdict.APPROVE


def test_plan_first_soft_deny_keeps_record_plan_in_batch(tmp_path: Path) -> None:
    """write-before-record must not hard-abort the batch (escape valve)."""
    tools = _tools()
    gate = PlanFirstGate(
        require_plan_artifact=True, agents_dir=str(tmp_path / ".agents")
    )
    batch = prepare_tool_batch(
        tools,
        [
            ToolCall(
                id="1",
                name="write",
                arguments={"file_path": str(tmp_path / "a.py"), "content": "x"},
            ),
            ToolCall(
                id="2",
                name="record_plan",
                arguments={"plan": "## Goal\nDo the work"},
            ),
        ],
        cwd=tmp_path,
        extra={"plan_first_gate": gate},
    )
    assert batch.outcome is ToolLoopOutcome.CONTINUE
    assert batch.prechecks[0].verdict is PrecheckVerdict.HOOK_DENY
    assert batch.prechecks[1].verdict is PrecheckVerdict.APPROVE
    assert len(batch.approved) == 1
    assert batch.approved[0][1].tool_name == "record_plan"


def test_prepare_gates_spawn_family(tmp_path: Path) -> None:
    tools = _tools()
    gate = PlanFirstGate(require_plan_artifact=True, agents_dir=str(tmp_path / ".agents"))
    cases = {
        "spawn_subagent": {
            "prompt": "explore",
            "description": "explore auth",
            "subagent_type": "explore",
        },
        "parallel_tasks": {
            "tasks": [
                {
                    "prompt": "explore auth",
                    "description": "explore auth",
                    "subagent_type": "explore",
                },
            ]
        },
        "run_terminal_cmd": {"command": "echo hi", "description": "noop"},
    }
    for name, args in cases.items():
        if name not in tools.by_client_name:
            continue
        pre = prepare_tool_call(
            tools,
            ToolCall(id="x", name=name, arguments=args),
            cwd=tmp_path,
            extra={"plan_first_gate": gate},
        )
        assert pre.verdict is PrecheckVerdict.HOOK_DENY, (
            f"{name}: {pre.verdict} {pre.reason}"
        )
