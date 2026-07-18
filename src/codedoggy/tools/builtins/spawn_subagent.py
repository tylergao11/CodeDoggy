"""task tool — Grok TaskTool (product client name: spawn_subagent).

Ported from grok-build:
  crates/codegen/xai-grok-tools/src/implementations/grok_build/task/mod.rs
    TaskTool id/kind, depth check, resume/model/cwd sanitization,
    type validation error strings, background vs blocking output
  crates/common/xai-tool-types/src/task.rs
    TaskToolInput schema fields + descriptions (exact schemars text)
  crates/codegen/xai-grok-tools/.../task/types.rs + backend.rs
    Channel contract is host-side; Doggy uses ctx.extra hooks only

Wire id is ``task`` (GrokBuild). Product surface renames to ``spawn_subagent``
and ``run_in_background`` → ``background`` via grok_surface.

Host dispatch (no invented multi-agent kernel):
  ctx.extra["subagent_coordinator"]  — spawn / resume
  ctx.extra["subagent_run_fn"]       — child run callable
  ctx.extra["subagent_depth"]        — optional nesting depth (default 0)
  ctx.extra["task_model_validator"]  — optional callable(slug)→error|None
  ctx.extra["subagent_available_types"] — optional list[str] for Unknown suffix
  ctx.extra["task_output_tool_name"] — optional poll tool name in notices

Fidelity notes:
  - Schema / descriptions / format strings: **S**
  - isolation=worktree: host SubagentCoordinator path exists (**C**);
    not full Grok fast-worktree kernel
  - cwd override: schema+validation **S**; not applied on SubagentRequest (**X**)
  - model: constraints messaging + optional validator **S**; not pinned on child (**X**)
  - general-purpose advertised in schema (Grok); host agent catalog may only
    resolve explore/plan until agent_def grows (**X** for that type)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from codedoggy.orchestration.subagent import SubagentRequest, SubagentSnapshot
from codedoggy.orchestration.types import CapabilityMode, IsolationMode
from codedoggy.tools.grok_build.task_format import (
    DEFAULT_SUBAGENT_TYPE,
    MAX_SUBAGENT_DEPTH,
    MODEL_PARAM_DESCRIPTION,
    cwd_does_not_exist_message,
    cwd_not_directory_message,
    cwd_worktree_mutex_message,
    default_task_description,
    depth_limit_error_message,
    format_auto_backgrounded_notice,
    format_subagent_completed,
    format_subagent_started_background,
    is_valid_resume_id,
    missing_backend_message,
    parse_lenient_bool,
    sanitize_cwd_value,
    sanitize_optional_arg,
    unknown_subagent_type_message,
    validation_unavailable_type_message,
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

# Grok TaskToolInput field descriptions (schemars, exact).
_PROMPT_DESC = "The full task prompt for the subagent to execute."
_DESCRIPTION_DESC = "Short description of the task (3-5 words)."
_SUBAGENT_TYPE_DESC = (
    'Name of the subagent type to launch. Built-in types: "general-purpose", '
    '"explore", "plan". Additional user-defined types may also be available.'
)
_RUN_IN_BG_DESC = (
    "Returns immediately with a subagent_id. Use the task output tool to "
    "retrieve results. This is set to true by default."
)
_CAPABILITY_DESC = (
    'Capability mode: "read-only", "read-write", "execute", or "all". '
    "Controls which tool classes the child can use. Default is determined by the role."
)
_ISOLATION_DESC = (
    'Isolation mode: "none" (default, shared workspace) or "worktree" '
    "(isolated git worktree). Worktree mode prevents the child's edits from "
    "affecting the parent workspace until explicitly merged."
)
_RESUME_DESC = (
    "Resume from a previously completed subagent's conversation. "
    "Pass the subagent_id returned by a prior task call. The new subagent "
    "continues the previous one's raw transcript with the new task prompt "
    "appended. The source must be completed (not running), belong to the "
    "current session, and use the same subagent_type."
)
_CWD_DESC = (
    "Explicit working directory for the subagent. The path must exist and "
    'be a directory. Mutually exclusive with isolation="worktree". '
    "Ignored when resume_from is set (the resumed child inherits "
    "its source's cwd/worktree)."
)


class TaskTool(Tool):
    """Grok TaskTool — wire id ``task``."""

    def id(self) -> ToolId:
        return ToolId("task")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Task

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        # Grok injects build_task_description via ToolConfig; Doggy builds
        # the same body with product naming (spawn_subagent / background).
        model_slugs = None
        if _ctx is not None:
            # ListToolsContext is empty today; room for host model catalog later.
            pass
        text = default_task_description(model_slugs=model_slugs)
        return ToolDescription(name="task", description=text)

    def parameters_schema(self) -> dict[str, Any]:
        # Exact Grok TaskToolInput schemars field set (task_id is #[schemars(skip)]).
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": _PROMPT_DESC,
                },
                "description": {
                    "type": "string",
                    "description": _DESCRIPTION_DESC,
                },
                "subagent_type": {
                    "type": "string",
                    "description": _SUBAGENT_TYPE_DESC,
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": _RUN_IN_BG_DESC,
                },
                "capability_mode": {
                    "type": "string",
                    "description": _CAPABILITY_DESC,
                },
                "isolation": {
                    "type": "string",
                    "description": _ISOLATION_DESC,
                },
                "resume_from": {
                    "type": "string",
                    "description": _RESUME_DESC,
                },
                "cwd": {
                    "type": "string",
                    "description": _CWD_DESC,
                },
                "model": {
                    "type": "string",
                    "description": MODEL_PARAM_DESCRIPTION,
                },
            },
            "required": ["prompt", "description"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        bag = ctx.extra or {}
        coord = bag.get("subagent_coordinator")
        run_fn = bag.get("subagent_run_fn")
        if coord is None or run_fn is None:
            # Grok: missing SubagentBackendResource
            raise ToolError(missing_backend_message(), code="missing_resource")

        # 1. Depth check (Grok MAX_SUBAGENT_DEPTH)
        depth = _read_depth(bag)
        if depth >= MAX_SUBAGENT_DEPTH:
            raise ToolError.invalid_arguments(
                depth_limit_error_message(depth, MAX_SUBAGENT_DEPTH)
            )

        prompt = str(args.get("prompt") or "").strip()
        if not prompt:
            raise ToolError.invalid_arguments("prompt is required")

        description = str(args.get("description") or "").strip()
        # Schema requires description; empty is still accepted at runtime
        # (models sometimes emit blank short labels).

        # resume_from: blank/null/none/undefined → absent
        resume_raw = args.get("resume_from")
        resume_from: str | None = None
        if isinstance(resume_raw, str) and is_valid_resume_id(resume_raw):
            resume_from = resume_raw.strip()

        # subagent_type default general-purpose (Grok default_subagent_type)
        st = str(args.get("subagent_type") or "").strip() or DEFAULT_SUBAGENT_TYPE

        # model: soft-ignored on resume
        model = sanitize_optional_arg(
            str(args["model"]) if args.get("model") is not None else None
        )
        if resume_from is not None:
            model = None

        # cwd sanitize + mutual exclusion with isolation=worktree
        isolation_mode, isolation_explicit = _parse_isolation(args.get("isolation"))
        cwd = sanitize_cwd_value(
            str(args["cwd"]) if args.get("cwd") is not None else None
        )
        if cwd is not None and isolation_mode is IsolationMode.WORKTREE:
            p = Path(cwd)
            if p.is_dir():
                raise ToolError.invalid_arguments(cwd_worktree_mutex_message())
            # Non-existent path alongside worktree — clear so worktree wins.
            cwd = None

        # Validate cwd exists (skip when resuming)
        if cwd is not None and resume_from is None:
            p = Path(cwd)
            if p.exists() and not p.is_dir():
                raise ToolError.invalid_arguments(cwd_not_directory_message(cwd))
            if not p.is_dir():
                raise ToolError.invalid_arguments(cwd_does_not_exist_message(cwd))
        # Host SubagentRequest has no cwd field yet (**X**): validated only.

        # 2. Eager type validation (Grok backend.validate_type)
        _validate_subagent_type(bag, st, parent_session_id=ctx.session_id or "")

        # Model catalog validation when explicit model is requested
        if model is not None:
            _validate_model(bag, model)

        # run_in_background default true; product remap may already have renamed
        bg_raw = args.get("run_in_background")
        if bg_raw is None and "background" in args:
            bg_raw = args.get("background")
        run_in_background = parse_lenient_bool(bg_raw, default=True)

        cap = None
        raw_cap = args.get("capability_mode")
        if isinstance(raw_cap, str) and raw_cap.strip():
            cap = CapabilityMode.parse(raw_cap)

        parent_sid = ctx.session_id or ""
        task_output_name = str(
            bag.get("task_output_tool_name") or "get_command_or_subagent_output"
        )

        if resume_from:
            snap = coord.resume(
                resume_from,
                prompt,
                run_fn=run_fn,
                description=description,
                run_in_background=run_in_background,
                parent_session_id=parent_sid or None,
                subagent_type=st or None,
            )
            return _format_result(
                snap,
                run_in_background=run_in_background,
                task_output_name=task_output_name,
                description=description,
            )

        req = SubagentRequest(
            subagent_type=st,
            prompt=prompt,
            description=description,
            parent_session_id=parent_sid,
            run_in_background=run_in_background,
            capability_mode=cap,
            isolation=isolation_mode if isolation_explicit else IsolationMode.NONE,
        )
        snap: SubagentSnapshot = coord.spawn(req, run_fn=run_fn)
        return _format_result(
            snap,
            run_in_background=run_in_background,
            task_output_name=task_output_name,
            description=description,
        )


# Back-compat import name
SpawnSubagentTool = TaskTool


class GetSubagentOutputTool(Tool):
    """Legacy Doggy helper. Prefer get_command_or_subagent_output.

    Not in grok_build product toolset by default. (Doggy enhancement.)
    """

    def id(self) -> ToolId:
        return ToolId("get_subagent_output")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.BackgroundTaskAction

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(
            name="get_subagent_output",
            description=(
                "Legacy alias: query a subagent. Prefer "
                "get_command_or_subagent_output with task_ids."
            ),
        )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "subagent_id": {
                    "type": "string",
                    "description": "Id returned by spawn_subagent",
                },
                "block": {
                    "type": "boolean",
                    "description": "Wait until complete (default false)",
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Wait timeout when block=true (default 30000)",
                },
            },
            "required": ["subagent_id"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        from codedoggy.tools.builtins.get_task_output import GetTaskOutputTool

        sid = str(args.get("subagent_id") or "").strip()
        if not sid:
            raise ToolError.invalid_arguments("subagent_id is required")
        block = bool(args.get("block"))
        timeout = args.get("timeout_ms")
        if block:
            if timeout is None:
                timeout = 30_000
            return GetTaskOutputTool().run(
                ctx, {"task_ids": [sid], "timeout_ms": int(timeout)}
            )
        return GetTaskOutputTool().run(ctx, {"task_ids": [sid], "timeout_ms": 0})


# ── helpers ──────────────────────────────────────────────────────────


def _read_depth(bag: dict[str, Any]) -> int:
    raw = bag.get("subagent_depth")
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _parse_isolation(raw: Any) -> tuple[IsolationMode, bool]:
    """Return (mode, was_explicitly_set)."""
    if raw is None:
        return IsolationMode.NONE, False
    if not isinstance(raw, str) or not raw.strip():
        return IsolationMode.NONE, False
    return IsolationMode.parse(raw), True


def _available_types(bag: dict[str, Any]) -> list[str]:
    raw = bag.get("subagent_available_types")
    if isinstance(raw, (list, tuple)) and raw:
        return [str(x) for x in raw]
    # Honest host catalog: what resolve_agent_definition knows today.
    try:
        from codedoggy.orchestration.agent_def import BUILTIN_AGENTS

        return sorted(BUILTIN_AGENTS.keys())
    except Exception:  # noqa: BLE001
        return ["explore", "plan"]


def _validate_subagent_type(
    bag: dict[str, Any],
    subagent_type: str,
    *,
    parent_session_id: str,
) -> None:
    """Grok eager validate_type — host hooks or local agent catalog."""
    validate_fn: Callable[..., Any] | None = bag.get("subagent_validate_fn")
    if callable(validate_fn):
        outcome = validate_fn(subagent_type, parent_session_id)
        if outcome is None or outcome is True or outcome == "ok":
            return
        if outcome == "disabled":
            from codedoggy.tools.grok_build.task_format import (
                disabled_subagent_type_message,
            )

            raise ToolError.invalid_arguments(
                disabled_subagent_type_message(subagent_type)
            )
        if outcome == "unavailable":
            raise ToolError(
                validation_unavailable_type_message(subagent_type),
                code="validation_unavailable",
            )
        if isinstance(outcome, dict):
            kind = str(outcome.get("kind") or outcome.get("status") or "")
            if kind == "unknown":
                avail = outcome.get("available") or _available_types(bag)
                raise ToolError.invalid_arguments(
                    unknown_subagent_type_message(subagent_type, list(avail))
                )
            if kind == "not_allowed":
                from codedoggy.tools.grok_build.task_format import (
                    not_allowed_subagent_type_message,
                )

                allowed = outcome.get("allowed") or []
                raise ToolError.invalid_arguments(
                    not_allowed_subagent_type_message(subagent_type, list(allowed))
                )
            if kind == "disabled":
                from codedoggy.tools.grok_build.task_format import (
                    disabled_subagent_type_message,
                )

                raise ToolError.invalid_arguments(
                    disabled_subagent_type_message(subagent_type)
                )
        # Unknown outcome shape — fail open only if explicitly allowed
        if outcome is False:
            raise ToolError.invalid_arguments(
                unknown_subagent_type_message(subagent_type, _available_types(bag))
            )
        return

    # Local catalog check (no full validate channel).
    # Grok general-purpose is advertised; host may not resolve it yet — still
    # allow the spawn so the coordinator returns its own Unknown error when
    # appropriate, OR reject early when clearly not in available list and
    # strict mode is on.
    try:
        from codedoggy.orchestration.agent_def import resolve_agent_definition
    except Exception:  # noqa: BLE001
        return

    # Resume path still validates type name when provided; for spawn we only
    # reject types that are neither resolvable nor in the advertised Grok set
    # when host supplies an explicit available list.
    explicit = bag.get("subagent_available_types")
    if isinstance(explicit, (list, tuple)) and explicit:
        names = {str(x).strip().lower() for x in explicit}
        if subagent_type.strip().lower() not in names:
            raise ToolError.invalid_arguments(
                unknown_subagent_type_message(subagent_type, sorted(names))
            )
        return

    # Default: allow any type through to coordinator (Grok ValidationUnavailable
    # is transport-only). Coordinator returns "Unknown subagent type" on fail.
    # Soft pre-check only when we can resolve and know it's unknown AND it's not
    # a Grok built-in name still awaiting host registration.
    defn = resolve_agent_definition(subagent_type)
    if defn is not None:
        return
    # Grok built-ins advertised even if host catalog is incomplete — defer.
    if subagent_type.strip().lower() in {
        "general-purpose",
        "explore",
        "plan",
    }:
        return
    raise ToolError.invalid_arguments(
        unknown_subagent_type_message(subagent_type, _available_types(bag))
    )


def _validate_model(bag: dict[str, Any], model: str) -> None:
    """Grok TaskModelValidator — optional host hook."""
    validator = bag.get("task_model_validator")
    if validator is None:
        raise ToolError(
            "Cannot validate Task.model: model catalog validator is unavailable.",
            code="validation_unavailable",
        )
    if not callable(validator):
        raise ToolError(
            "Cannot validate Task.model: model catalog validator is unavailable.",
            code="validation_unavailable",
        )
    err = validator(model)
    if err:
        raise ToolError.invalid_arguments(str(err))


def _format_result(
    snap: SubagentSnapshot,
    *,
    run_in_background: bool,
    task_output_name: str,
    description: str,
) -> str:
    """Map host snapshot to Grok model-facing text."""
    # Auto-backgrounded (host may set metadata flag)
    if getattr(snap, "metadata", None) and snap.metadata.get("backgrounded"):
        return format_auto_backgrounded_notice(
            snap.subagent_id,
            snap.subagent_type,
            description or snap.description or "",
            task_output_name,
        )

    if run_in_background and snap.is_running:
        return format_subagent_started_background(
            snap.subagent_id,
            snap.subagent_type,
            description or snap.description or "",
            task_output_name,
        )

    if snap.status == "failed" or snap.error:
        raise ToolError.invalid_arguments(
            snap.error or "Unknown subagent error"
        )

    if snap.status == "cancelled":
        raise ToolError.invalid_arguments(
            snap.error or "Unknown subagent error"
        )

    if snap.is_running:
        # Blocking mode still running (no backgrounded flag) — treat as
        # auto-background notice so the model can poll.
        return format_auto_backgrounded_notice(
            snap.subagent_id,
            snap.subagent_type,
            description or snap.description or "",
            task_output_name,
        )

    return format_subagent_completed(
        snap.output or "",
        snap.subagent_id,
        snap.subagent_type,
        int(snap.tool_calls or 0),
        int(snap.turns or 0),
        int(snap.duration_ms or 0),
        persona=None,
    )
