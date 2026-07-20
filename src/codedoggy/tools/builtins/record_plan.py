"""record_plan — go-steer plan-first escape valve.

Source:
  github.com/go-steer/core-agent/pkg/tools/record_plan.go
  docs/plan-first-design.md

Writes ``<agentsDir>/plans/plan-<seq>.md`` and flips PlanFirstGate.plan_recorded.
ALWAYS allowed by the plan-first pre-check (exempt). No invented schema beyond
non-empty markdown (go-steer Q2).
"""

from __future__ import annotations

from typing import Any

from codedoggy.orchestration.plan_first import (
    PlanFirstGate,
    resolve_plan_first_gate,
    write_plan_artifact,
)
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolError,
    ToolId,
)

# go-steer RecordPlan Description (tool names adapted to CodeDoggy surface)
_DESC = """\
Record the agent's implementation plan as a markdown artifact and unblock \
mutating tools when plan-first gating is enabled. Call this BEFORE any \
write / search_replace / apply_patch / run_terminal_cmd / spawn_subagent / \
parallel_tasks call when require_plan_artifact is on; otherwise those calls \
are denied with a 'plan required' error. Plan is free-form markdown — typical \
shape: goal, files to change, approach, risks, test plan, out of scope. The \
plan is persisted to .agents/plans/plan-<seq>.md. To revise an existing plan, \
just call record_plan again — each call writes a new plan file with the next \
sequence number.
"""


class RecordPlanTool(Tool):
    def id(self) -> ToolId:
        return ToolId("record_plan")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        # Escape valve; treat as non-mutating for capability filtering.
        return ToolKind.EnterPlan

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="record_plan", description=_DESC.strip())

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": (
                        "the plan as markdown — required. free-form structure; "
                        "typical shape: goal, files to change, approach, risks, "
                        "test plan, out of scope"
                    ),
                },
            },
            "required": ["plan"],
            "additionalProperties": False,
        }

    def capabilities(self) -> dict[str, Any]:
        return {"is_read_only": True, "tool_scope": "Read"}

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        plan = args.get("plan")
        if not isinstance(plan, str):
            raise ToolError.invalid_arguments("plan must be a string")
        body = plan.strip()
        if not body:
            raise ToolError.invalid_arguments(
                "record_plan: plan is required (non-empty markdown)"
            )

        bag = ctx.extra if ctx.extra is not None else {}
        gate = resolve_plan_first_gate(bag)
        if gate is None:
            # Host forgot to wire — still persist under cwd/.agents so the
            # artifact exists; flip a local gate into the bag for this call.
            gate = PlanFirstGate(require_plan_artifact=False)
            bag["plan_first_gate"] = gate

        agents_dir = gate.resolve_agents_dir(ctx.cwd)
        if agents_dir is None:
            raise ToolError(
                "record_plan: agentsDir is required "
                "(set plan_first_gate.agents_dir or run with a workspace cwd)",
                code="missing_resource",
            )

        try:
            path, seq = write_plan_artifact(agents_dir, body)
        except ValueError as e:
            raise ToolError.invalid_arguments(str(e)) from e
        except OSError as e:
            raise ToolError(
                f"record_plan: write failed: {e}",
                code="io_error",
            ) from e

        gate.mark_plan_recorded()
        # Keep bag / kernel in sync for mid-turn subsequent prepares
        bag["plan_first_gate"] = gate
        kernel = bag.get("kernel")
        if kernel is not None and getattr(kernel, "plan_first_gate", None) is None:
            try:
                kernel.plan_first_gate = gate
            except Exception:  # noqa: BLE001
                pass

        return (
            f"Plan recorded at {path}. Mutating tools are now unblocked for "
            f"this session (sequence={seq}). The operator can revoke via "
            f"replan, which clears the gate flag and forces a redraft."
        )
