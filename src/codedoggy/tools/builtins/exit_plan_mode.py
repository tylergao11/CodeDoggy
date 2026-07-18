"""exit_plan_mode — leave plan mode and present the plan (Grok ExitPlanMode).

Ported from grok-build:
  crates/codegen/xai-grok-tools/src/implementations/grok_build/exit_plan_mode/mod.rs
  crates/codegen/xai-grok-tools/src/implementations/grok_build/exit_plan_mode/types.rs
    (ACP ExitPlanModeExtRequest/Response — host contract only; not model schema)
  crates/codegen/xai-grok-tools/src/types/output.rs (ExitPlanModeOutput, to_prompt_format)
  crates/codegen/xai-grok-tools/src/types/resources.rs (require_plan_file_path)

Host vs Grok Resources (honest X/C):
  - Plan file read from disk (NOT from tool args) — same as Grok
  - Empty model input ``{}`` — approval UI is client/host side
  - PlanModeExited notification → host inject kernel.exit_plan_mode /
    session_mode_state.exit_plan when wired; optional plan_mode_exit_fn for
    ACP-style outcome (approved/cancelled/abandoned)
  - Does **not** invent a full plan kernel or accept plan content as input
"""

from __future__ import annotations

from typing import Any

from codedoggy.tools.grok_build.plan_mode import (
    EMPTY_PLAN_MESSAGE,
    PLAN_READY_MESSAGE,
    format_exit_plan_empty,
    format_exit_plan_ready,
    read_plan_content,
    require_plan_file_path,
    resolve_configured_plan_path,
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

# Grok description_template — exact
_DESC = """\
Exit plan mode and present your plan to the user.

Use this after you have finished writing your plan to the plan file in plan mode.
"""


class ExitPlanModeTool(Tool):
    def id(self) -> ToolId:
        return ToolId("exit_plan_mode")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.ExitPlan

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="exit_plan_mode", description=_DESC.strip())

    def parameters_schema(self) -> dict[str, Any]:
        # Empty object — plan is read from disk, NOT passed as a parameter
        # (Grok ExitPlanModeInput {})
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    def capabilities(self) -> dict[str, Any]:
        return {"is_read_only": True, "tool_scope": "Read"}

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        _ = args  # empty input — do not accept plan content or approved flag
        bag = ctx.extra if ctx.extra is not None else {}

        configured = resolve_configured_plan_path(bag)
        # Prefer session mode plan file only when already in plan mode
        # (absolute path set by enter_plan_mode host inject).
        if configured is None:
            mode_state = bag.get("session_mode_state")
            if mode_state is None:
                kernel = bag.get("kernel")
                if kernel is not None:
                    mode_state = getattr(kernel, "session_mode_state", None)
            if mode_state is not None and getattr(mode_state, "is_plan", lambda: False)():
                pf = getattr(mode_state, "plan_file", None)
                if pf:
                    configured = str(pf)

        try:
            plan_path, plan_file_path = require_plan_file_path(
                cwd=ctx.cwd,
                plan_file_path=configured,
            )
        except ValueError as e:
            raise ToolError(str(e), code="missing_resource") from e

        plan_content = read_plan_content(plan_path)

        # Optional host ACP-style exit callback (types.rs ExtResponse stand-in)
        exit_fn = bag.get("plan_mode_exit_fn")
        host_outcome: str | None = None
        host_feedback: str | None = None
        if callable(exit_fn):
            try:
                result = exit_fn(
                    {
                        "plan_content": plan_content,
                        "plan_file_path": plan_file_path,
                        "tool_call_id": bag.get("tool_call_id"),
                    }
                )
            except Exception as e:  # noqa: BLE001
                raise ToolError(
                    f"plan mode exit host failed: {e}",
                    code="plan_exit_failed",
                ) from e
            if isinstance(result, dict):
                host_outcome = str(result.get("outcome") or "")
                fb = result.get("feedback")
                host_feedback = str(fb) if fb is not None else None
            elif isinstance(result, str):
                host_outcome = result

        # Session mode switch (NotificationHandle stand-in) — best-effort
        approved = host_outcome not in {"cancelled", "abandoned"}
        kernel = bag.get("kernel")
        mode_state = bag.get("session_mode_state")
        if mode_state is None and kernel is not None:
            mode_state = getattr(kernel, "session_mode_state", None)

        if host_outcome in {"cancelled", "abandoned"}:
            # Host rejected — exit plan mode without "approved" model message
            if kernel is not None and hasattr(kernel, "exit_plan_mode"):
                try:
                    kernel.exit_plan_mode(approved=False)
                except Exception:  # noqa: BLE001
                    pass
            elif mode_state is not None and hasattr(mode_state, "exit_plan"):
                mode_state.exit_plan(approved=False)
            if host_outcome == "cancelled":
                if host_feedback:
                    return (
                        "Plan mode exit cancelled. User feedback: "
                        f"{host_feedback}"
                    )
                return "Plan mode exit cancelled."
            return "Plan mode exit abandoned."

        if kernel is not None and hasattr(kernel, "exit_plan_mode"):
            try:
                kernel.exit_plan_mode(approved=True)
            except Exception:  # noqa: BLE001
                pass
        elif mode_state is not None and hasattr(mode_state, "exit_plan"):
            mode_state.exit_plan(approved=True)

        bag["last_plan_content"] = plan_content
        bag["plan_file_path"] = plan_file_path

        if plan_content is not None:
            return format_exit_plan_ready(
                message=PLAN_READY_MESSAGE,
                plan_content=plan_content,
                plan_file_path=plan_file_path,
            )
        return format_exit_plan_empty(message=EMPTY_PLAN_MESSAGE)
