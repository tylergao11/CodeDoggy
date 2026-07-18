"""enter_plan_mode — agent-initiated plan mode (Grok EnterPlanMode).

Ported from grok-build:
  crates/codegen/xai-grok-tools/src/implementations/grok_build/enter_plan_mode/mod.rs
  crates/codegen/xai-grok-tools/src/types/output.rs (EnterPlanModeOutput, seed status)
  crates/codegen/xai-grok-tools/src/types/resources.rs (plan path resolve)

Host vs Grok Resources (honest X/C):
  - PlanFilePath → ctx.extra["plan_file_path"] (optional absolute path)
  - Cwd → ctx.cwd → default ``.grok/plan.md``
  - FileSystem → pathlib (seed empty file; never truncate)
  - NotificationHandle / PlanModeEntered → host session inject via
    kernel.enter_plan_mode / session_mode_state.enter_plan when wired
  - TemplateRenderer tool hints → extra["plan_tool_hints"] or defaults
  - User consent UI → extra["plan_mode_consent_fn"] → ``User declined...``
  Does **not** invent a full plan kernel; mode flip is host/orchestration only.
"""

from __future__ import annotations

from typing import Any

from codedoggy.tools.grok_build.plan_mode import (
    ENTERED_MESSAGE,
    USER_DECLINED_ENTER,
    PlanFileSeedFailure,
    PlanFileSeedStatus,
    format_enter_plan_prompt,
    probe_or_create_empty_plan_file,
    resolve_configured_plan_path,
    resolve_plan_file_path,
    tool_hints_from_extra,
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

# Grok description_template (ToolMetadata) — exact
_DESC = """\
Use this tool when a task has ambiguity about the right approach or when the user asks you to write a plan. This tool enables a read-only plan mode where you explore the codebase and create an implementation plan for the user.
"""


class EnterPlanModeTool(Tool):
    def id(self) -> ToolId:
        return ToolId("enter_plan_mode")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.EnterPlan

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="enter_plan_mode", description=_DESC.strip())

    def parameters_schema(self) -> dict[str, Any]:
        # Empty object — no parameters (Grok EnterPlanModeInput {})
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    def capabilities(self) -> dict[str, Any]:
        # Read-only for permission UX; only FS write is seeding the session plan file.
        return {"is_read_only": True, "tool_scope": "Read"}

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        _ = args  # empty input
        bag = ctx.extra if ctx.extra is not None else {}

        # Optional host consent (Grok UI requires approval before execute)
        consent = bag.get("plan_mode_consent_fn")
        if callable(consent):
            try:
                ok = bool(consent())
            except Exception as e:  # noqa: BLE001
                raise ToolError(
                    f"plan mode consent failed: {e}",
                    code="consent_failed",
                ) from e
            if not ok:
                return USER_DECLINED_ENTER

        configured = resolve_configured_plan_path(bag)
        seed_target, plan_file_path = resolve_plan_file_path(
            cwd=ctx.cwd,
            plan_file_path=configured,
        )
        tool_hints = tool_hints_from_extra(bag)

        # Host session switch (NotificationHandle stand-in) — best-effort
        kernel = bag.get("kernel")
        mode_state = bag.get("session_mode_state")
        if mode_state is None and kernel is not None:
            mode_state = getattr(kernel, "session_mode_state", None)

        # Prefer absolute display for mode gate; pass path used for plan file
        plan_for_mode = plan_file_path
        if seed_target is not None:
            plan_for_mode = str(seed_target)

        if kernel is not None and hasattr(kernel, "enter_plan_mode"):
            try:
                kernel.enter_plan_mode(plan_for_mode)
            except Exception:  # noqa: BLE001
                pass
            mode_state = getattr(kernel, "session_mode_state", mode_state)
        elif mode_state is not None and hasattr(mode_state, "enter_plan"):
            mode_state.enter_plan(plan_for_mode)
        # else: Grok works without NotificationHandle — still return Entered

        # Seed only with absolute target; never write a relative path or truncate
        if seed_target is not None:
            plan_file_seed = probe_or_create_empty_plan_file(seed_target)
        else:
            plan_file_seed = PlanFileSeedStatus.missing(
                PlanFileSeedFailure.UNAVAILABLE
            )

        # Stash path for exit / hosts that poll tool_extra
        bag["plan_file_path"] = plan_file_path
        bag["plan_file_seed"] = plan_file_seed.kind.value
        if mode_state is not None:
            bag["session_mode_state"] = mode_state

        return format_enter_plan_prompt(
            message=ENTERED_MESSAGE,
            plan_file_path=plan_file_path,
            tool_hints=tool_hints,
            plan_file_seed=plan_file_seed,
        )
