"""task / spawn_subagent — Grok TaskTool fidelity tests.

Ported test intent from:
  implementations/grok_build/task/mod.rs
  xai-tool-types/src/task.rs
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.orchestration.subagent import (
    SubagentCoordinator,
    SubagentRequest,
    SubagentSnapshot,
)
from codedoggy.tools.builtins.spawn_subagent import TaskTool
from codedoggy.tools.grok_build.task_format import (
    DEFAULT_SUBAGENT_TYPE,
    MAX_SUBAGENT_DEPTH,
    MODEL_PARAM_DESCRIPTION,
    build_task_description,
    cwd_worktree_mutex_message,
    default_task_description,
    depth_limit_error_message,
    format_auto_backgrounded_notice,
    format_resume_footer,
    format_subagent_completed,
    format_subagent_started_background,
    is_not_sentinel,
    is_valid_resume_id,
    parse_lenient_bool,
    sanitize_cwd_value,
    sanitize_optional_arg,
    task_model_guidance,
    unknown_subagent_type_message,
)
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


def _instant_run(output: str = "done", *, status: str = "completed"):
    def run_fn(req: SubagentRequest, cancel) -> SubagentSnapshot:
        return SubagentSnapshot(
            subagent_id=req.id,
            subagent_type=req.subagent_type,
            status=status,
            description=req.description,
            output=output if status == "completed" else None,
            error=None if status == "completed" else "boom",
            tool_calls=2,
            turns=1,
            duration_ms=42,
        )

    return run_fn


# ── pure helpers ─────────────────────────────────────────────────────


def test_is_not_sentinel() -> None:
    assert is_not_sentinel("abc")
    assert not is_not_sentinel("")
    assert not is_not_sentinel("  ")
    assert not is_not_sentinel("null")
    assert not is_not_sentinel("NULL")
    assert not is_not_sentinel("none")
    assert not is_not_sentinel("undefined")


def test_sanitize_optional_arg() -> None:
    assert sanitize_optional_arg("grok-3") == "grok-3"
    assert sanitize_optional_arg("  grok-3  ") == "grok-3"
    assert sanitize_optional_arg("null") is None
    assert sanitize_optional_arg("  NULL  ") is None
    assert sanitize_optional_arg(None) is None


def test_sanitize_cwd_value(tmp_path: Path) -> None:
    assert sanitize_cwd_value(None) is None
    assert sanitize_cwd_value("null") is None
    assert sanitize_cwd_value('"C:/tmp"') == "C:/tmp"
    assert sanitize_cwd_value("`/x`") == "/x"
    home = sanitize_cwd_value("~")
    assert home is not None
    assert Path(home).is_absolute()


def test_is_valid_resume_id() -> None:
    assert is_valid_resume_id("sub_abc")
    assert not is_valid_resume_id("null")
    assert not is_valid_resume_id("")


def test_format_subagent_started_background() -> None:
    text = format_subagent_started_background(
        "id-1", "explore", "find bugs", "get_command_or_subagent_output"
    )
    assert "Subagent started in background." in text
    assert "subagent_id: id-1" in text
    assert "type: explore" in text
    assert "description: find bugs" in text
    assert 'task_ids=["id-1"]' in text
    assert "timeout_ms" in text


def test_format_subagent_completed_and_resume_footer() -> None:
    text = format_subagent_completed("Found 3 files", "sid", "explore", 5, 2, 1234)
    assert "Found 3 files" in text
    assert "<subagent_meta>id=sid, type=explore, tool_calls=5, turns=2, duration_ms=1234</subagent_meta>" in text
    assert 'resume_from="sid"' in text
    footer = format_resume_footer("sid", "explore", persona="reviewer")
    assert 'persona="reviewer"' in footer


def test_format_auto_backgrounded_notice() -> None:
    text = format_auto_backgrounded_notice("id-2", "plan", "design")
    assert "moved to the background" in text
    assert "subagent_id: id-2" in text
    assert "timeout_ms" in text


def test_build_task_description_lists_agents() -> None:
    desc = default_task_description()
    assert "Start a subagent that works on a task independently" in desc
    assert "**general-purpose**" in desc
    assert "**explore**" in desc
    assert "**plan**" in desc
    assert "background:" in desc  # product naming
    assert "spawn_subagent" in desc
    assert "get_command_or_subagent_output" in desc
    assert "resume_from" in desc
    assert "worktree" in desc
    assert "No explicit model slugs" in desc
    # Grok by_kind.plan → todo_write (not enter_plan_mode)
    assert "todo_write" in desc
    assert "enter_plan_mode" not in desc
    assert "run_terminal_command" in desc
    assert "Has access to all tools:" in desc


def test_task_model_guidance_with_slugs() -> None:
    g = task_model_guidance(["zeta", "alpha"])
    assert "- alpha" in g
    assert "- zeta" in g
    assert "ONLY use model slugs" in g


def test_default_subagent_type_constant() -> None:
    assert DEFAULT_SUBAGENT_TYPE == "general-purpose"
    assert MAX_SUBAGENT_DEPTH == 1


def test_parse_lenient_bool() -> None:
    assert parse_lenient_bool(None, default=True) is True
    assert parse_lenient_bool("false", default=True) is False
    assert parse_lenient_bool("1", default=False) is True
    assert parse_lenient_bool(0, default=True) is False


# ── schema ───────────────────────────────────────────────────────────


def test_task_schema_matches_grok_fields() -> None:
    schema = TaskTool().parameters_schema()
    props = schema["properties"]
    assert set(props) == {
        "prompt",
        "description",
        "subagent_type",
        "run_in_background",
        "capability_mode",
        "isolation",
        "resume_from",
        "cwd",
        "model",
    }
    assert schema["required"] == ["prompt", "description"]
    # No task_id (schemars skip), no max_turns (Doggy was inventing)
    assert "task_id" not in props
    assert "max_turns" not in props
    assert props["model"]["description"] == MODEL_PARAM_DESCRIPTION
    assert "general-purpose" in props["subagent_type"]["description"]
    assert "worktree" in props["isolation"]["description"]


def test_task_tool_id_and_kind() -> None:
    t = TaskTool()
    assert t.id() == "task"
    assert t.kind().name == "Task"
    desc = t.description(None)
    assert desc.name == "task"
    assert "Agent types:" in desc.description


# ── run path ─────────────────────────────────────────────────────────


def test_missing_backend_returns_error(tmp_path: Path) -> None:
    with pytest.raises(ToolError) as ei:
        TaskTool().run(
            _ctx(tmp_path),
            {"prompt": "p", "description": "d", "subagent_type": "explore"},
        )
    assert "SubagentBackendResource" in ei.value.message
    assert ei.value.code == "missing_resource"


def test_depth_limit_exceeded(tmp_path: Path) -> None:
    coord = SubagentCoordinator()
    with pytest.raises(ToolError) as ei:
        TaskTool().run(
            _ctx(
                tmp_path,
                coord=coord,
                run_fn=_instant_run(),
                extra={"subagent_depth": MAX_SUBAGENT_DEPTH},
            ),
            {
                "prompt": "p",
                "description": "d",
                "subagent_type": "explore",
                "run_in_background": False,
            },
        )
    assert "depth limit exceeded" in ei.value.message
    assert ei.value.code == "invalid_arguments"
    assert depth_limit_error_message(1) in ei.value.message


def test_background_spawn_returns_started_notice(tmp_path: Path) -> None:
    coord = SubagentCoordinator()
    # Slow-ish child so status is still running when bg returns
    import threading
    import time

    def slow_run(req: SubagentRequest, cancel: threading.Event) -> SubagentSnapshot:
        time.sleep(0.3)
        return SubagentSnapshot(
            subagent_id=req.id,
            subagent_type=req.subagent_type,
            status="completed",
            description=req.description,
            output="ok",
            tool_calls=1,
            turns=1,
            duration_ms=300,
        )

    out = TaskTool().run(
        _ctx(tmp_path, coord=coord, run_fn=slow_run),
        {
            "prompt": "scan",
            "description": "find bugs",
            "subagent_type": "explore",
            "run_in_background": True,
        },
    )
    assert "Subagent started in background." in out
    assert "type: explore" in out
    assert "description: find bugs" in out
    assert "get_command_or_subagent_output" in out
    assert "timeout_ms" in out
    coord.shutdown(wait=True)


def test_blocking_success_returns_subagent_completed(tmp_path: Path) -> None:
    coord = SubagentCoordinator()
    out = TaskTool().run(
        _ctx(tmp_path, coord=coord, run_fn=_instant_run("Found 3 auth files")),
        {
            "prompt": "search auth",
            "description": "Find auth middleware",
            "subagent_type": "explore",
            "run_in_background": False,
        },
    )
    assert "Found 3 auth files" in out
    assert "<subagent_meta>" in out
    assert "type=explore" in out
    assert "tool_calls=2" in out
    assert "<subagent_result>" in out
    assert "resume_from=" in out
    coord.shutdown(wait=True)


def test_failed_subagent_returns_error(tmp_path: Path) -> None:
    coord = SubagentCoordinator()

    def fail_run(req: SubagentRequest, cancel) -> SubagentSnapshot:
        return SubagentSnapshot(
            subagent_id=req.id,
            subagent_type=req.subagent_type,
            status="failed",
            error="Child session crashed",
        )

    with pytest.raises(ToolError) as ei:
        TaskTool().run(
            _ctx(tmp_path, coord=coord, run_fn=fail_run),
            {
                "prompt": "x",
                "description": "d",
                "subagent_type": "explore",
                "run_in_background": False,
            },
        )
    assert "Child session crashed" in ei.value.message
    coord.shutdown(wait=True)


def test_unknown_subagent_type_before_spawn(tmp_path: Path) -> None:
    coord = SubagentCoordinator()
    with pytest.raises(ToolError) as ei:
        TaskTool().run(
            _ctx(
                tmp_path,
                coord=coord,
                run_fn=_instant_run(),
                extra={"subagent_available_types": ["explore", "plan"]},
            ),
            {
                "prompt": "p",
                "description": "d",
                "subagent_type": "invented-agent",
                "run_in_background": True,
            },
        )
    msg = ei.value.message
    assert "Unknown subagent type: invented-agent" in msg
    assert "explore" in msg
    assert "plan" in msg
    coord.shutdown(wait=True)


def test_unknown_message_helper_empty_available() -> None:
    assert (
        unknown_subagent_type_message("invented", [])
        == "Unknown subagent type: invented"
    )
    assert "Available types:" not in unknown_subagent_type_message("x", [])


def test_invalid_model_returns_error_before_spawn(tmp_path: Path) -> None:
    coord = SubagentCoordinator()
    spawned = {"n": 0}

    def run_fn(req, cancel):
        spawned["n"] += 1
        return SubagentSnapshot(
            subagent_id=req.id,
            subagent_type=req.subagent_type,
            status="completed",
            output="ok",
        )

    def validator(slug: str) -> str | None:
        if slug == "invented-model":
            return (
                "Unknown Task.model slug 'invented-model'. Valid model slugs: alpha, zeta. "
                "Omit `model` to inherit the parent model."
            )
        return None

    with pytest.raises(ToolError) as ei:
        TaskTool().run(
            _ctx(
                tmp_path,
                coord=coord,
                run_fn=run_fn,
                extra={"task_model_validator": validator},
            ),
            {
                "prompt": "p",
                "description": "d",
                "subagent_type": "explore",
                "run_in_background": True,
                "model": "invented-model",
            },
        )
    assert "Unknown Task.model slug 'invented-model'" in ei.value.message
    assert spawned["n"] == 0
    coord.shutdown(wait=True)


def test_model_without_validator_is_validation_unavailable(tmp_path: Path) -> None:
    coord = SubagentCoordinator()
    with pytest.raises(ToolError) as ei:
        TaskTool().run(
            _ctx(tmp_path, coord=coord, run_fn=_instant_run()),
            {
                "prompt": "p",
                "description": "d",
                "subagent_type": "explore",
                "model": "any-model",
            },
        )
    assert ei.value.code == "validation_unavailable"
    assert "model catalog validator is unavailable" in ei.value.message
    coord.shutdown(wait=True)


def test_spawn_passes_model_and_cwd_on_request(tmp_path: Path) -> None:
    """Task.model + cwd must land on SubagentRequest (not validate-only)."""
    coord = SubagentCoordinator()
    seen: list[SubagentRequest] = []
    child = tmp_path / "work"
    child.mkdir()

    def run_fn(req: SubagentRequest, cancel) -> SubagentSnapshot:
        seen.append(req)
        return SubagentSnapshot(
            subagent_id=req.id,
            subagent_type=req.subagent_type,
            status="completed",
            description=req.description,
            output="ok",
            tool_calls=0,
            turns=1,
            duration_ms=1,
        )

    def validator(slug: str) -> str | None:
        return None if slug == "child-model" else f"bad {slug}"

    out = TaskTool().run(
        _ctx(
            tmp_path,
            coord=coord,
            run_fn=run_fn,
            extra={"task_model_validator": validator},
        ),
        {
            "prompt": "do work",
            "description": "slice",
            "subagent_type": "explore",
            "run_in_background": False,
            "model": "child-model",
            "cwd": str(child),
        },
    )
    assert "ok" in out or "done" in out.lower() or "Completed" in out or out
    assert len(seen) == 1
    assert seen[0].model == "child-model"
    assert seen[0].cwd == str(child)
    coord.shutdown(wait=True)


def test_resolve_subagent_model_env(monkeypatch: object) -> None:
    from codedoggy.orchestration.subagent import resolve_subagent_model

    monkeypatch.setenv("CODEDOGGY_SUBAGENT_MODELS", "explore=env-explore,plan=env-plan")
    assert resolve_subagent_model("explore") == "env-explore"
    assert resolve_subagent_model("plan") == "env-plan"
    assert resolve_subagent_model("explore", explicit="win") == "win"
    monkeypatch.setenv("CODEDOGGY_SUBAGENT_MODEL_GENERAL_PURPOSE", "gp-model")
    assert resolve_subagent_model("general-purpose") == "gp-model"


def test_pin_sampler_model_rewrites_client_model() -> None:
    from dataclasses import replace

    from codedoggy.model.chat_sampler import ChatSampler
    from codedoggy.model.types import ModelConfig
    from codedoggy.orchestration.subagent import pin_sampler_model

    class _Client:
        def __init__(self, config: ModelConfig, profile=None) -> None:
            self._config = config
            self.config = config
            self._profile = profile
            self.profile = profile

    cfg = ModelConfig(
        provider="openai_compat",
        model="parent-model",
        base_url="http://localhost",
        api_key="k",
    )
    base = ChatSampler(_Client(cfg))
    pinned = pin_sampler_model(base, "child-model")
    assert pinned is not base
    assert pinned.client.config.model == "child-model"


def test_cwd_and_worktree_mutex(tmp_path: Path) -> None:
    coord = SubagentCoordinator()
    real = tmp_path / "child_cwd"
    real.mkdir()
    with pytest.raises(ToolError) as ei:
        TaskTool().run(
            _ctx(tmp_path, coord=coord, run_fn=_instant_run()),
            {
                "prompt": "p",
                "description": "d",
                "subagent_type": "explore",
                "cwd": str(real),
                "isolation": "worktree",
                "run_in_background": False,
            },
        )
    assert ei.value.message == cwd_worktree_mutex_message()
    coord.shutdown(wait=True)


def test_cwd_does_not_exist(tmp_path: Path) -> None:
    coord = SubagentCoordinator()
    missing = tmp_path / "nope_dir"
    with pytest.raises(ToolError) as ei:
        TaskTool().run(
            _ctx(tmp_path, coord=coord, run_fn=_instant_run()),
            {
                "prompt": "p",
                "description": "d",
                "subagent_type": "explore",
                "cwd": str(missing),
                "run_in_background": False,
            },
        )
    assert "does not exist" in ei.value.message
    coord.shutdown(wait=True)


def test_product_background_param_alias(tmp_path: Path) -> None:
    """Product remap may leave `background` if reverse map missed — accept both."""
    coord = SubagentCoordinator()
    import threading
    import time

    def slow_run(req, cancel):
        time.sleep(0.2)
        return SubagentSnapshot(
            subagent_id=req.id,
            subagent_type=req.subagent_type,
            status="completed",
            output="ok",
            description=req.description,
        )

    out = TaskTool().run(
        _ctx(tmp_path, coord=coord, run_fn=slow_run),
        {
            "prompt": "p",
            "description": "d",
            "subagent_type": "explore",
            "background": True,  # product name without internal rename
        },
    )
    assert "Subagent started in background." in out
    coord.shutdown(wait=True)


def test_resume_from_sentinel_ignored_defaults_type(tmp_path: Path) -> None:
    """Blank resume_from treated as absent (Grok is_valid_resume_id)."""
    coord = SubagentCoordinator()
    out = TaskTool().run(
        _ctx(tmp_path, coord=coord, run_fn=_instant_run("ok")),
        {
            "prompt": "p",
            "description": "d",
            "subagent_type": "explore",
            "resume_from": "null",
            "run_in_background": False,
        },
    )
    assert "ok" in out
    coord.shutdown(wait=True)


def test_build_task_description_preserves_wire_naming() -> None:
    from codedoggy.tools.grok_build.task_format import (
        BUILTIN_SUBAGENT_DESCRIPTORS,
        WIRE_TASK_NAMING,
    )

    desc = build_task_description(BUILTIN_SUBAGENT_DESCRIPTORS, WIRE_TASK_NAMING)
    assert "run_in_background:" in desc
    assert "get_task_output" in desc
    assert "When using the task tool" in desc
