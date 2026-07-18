"""Focused tests for Grok enter_plan_mode / exit_plan_mode port."""

from __future__ import annotations

from pathlib import Path

from codedoggy.orchestration.session_mode import SessionModeState
from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.builtins.enter_plan_mode import EnterPlanModeTool
from codedoggy.tools.builtins.exit_plan_mode import ExitPlanModeTool
from codedoggy.tools.grok_build.plan_mode import (
    EMPTY_PLAN_MESSAGE,
    ENTERED_MESSAGE,
    PLAN_FILE_RELATIVE_PATH,
    PLAN_READY_MESSAGE,
    USER_DECLINED_ENTER,
    EnterPlanModeToolHints,
    PlanFileSeedFailure,
    PlanFileSeedStatus,
    PlanFileSeedStatusKind,
    format_enter_plan_prompt,
    format_exit_plan_ready,
    probe_or_create_empty_plan_file,
    resolve_plan_file_path,
)
from codedoggy.tools.runtime import ToolCallContext


def _tools():
    return ToolRegistryBuilder.new().finalize()


def _ctx(tmp_path: Path, **extra) -> ToolCallContext:
    return ToolCallContext(cwd=tmp_path, extra=dict(extra))


# ── pure logic ───────────────────────────────────────────────────────────


def test_resolve_defaults_to_grok_plan_md(tmp_path: Path) -> None:
    abs_t, display = resolve_plan_file_path(cwd=tmp_path, plan_file_path=None)
    assert abs_t is not None
    assert abs_t == (tmp_path / PLAN_FILE_RELATIVE_PATH).resolve() or abs_t == (
        tmp_path / PLAN_FILE_RELATIVE_PATH
    )
    assert display.endswith(".grok/plan.md") or display.endswith(".grok\\plan.md")


def test_resolve_without_cwd_is_relative_unavailable() -> None:
    abs_t, display = resolve_plan_file_path(cwd=None, plan_file_path=None)
    assert abs_t is None
    # Path.display is OS-native (backslash on Windows)
    assert display.replace("\\", "/") == PLAN_FILE_RELATIVE_PATH


def test_probe_creates_empty_with_parents(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "plan.md"
    status = probe_or_create_empty_plan_file(path)
    assert status.kind is PlanFileSeedStatusKind.EMPTY
    assert path.is_file()
    assert path.read_bytes() == b""


def test_probe_does_not_truncate_nonempty(tmp_path: Path) -> None:
    path = tmp_path / "plan.md"
    path.write_bytes(b"# prior plan\n")
    status = probe_or_create_empty_plan_file(path)
    assert status.kind is PlanFileSeedStatusKind.NON_EMPTY
    assert path.read_bytes() == b"# prior plan\n"


def test_probe_existing_empty_without_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "plan.md"
    path.write_bytes(b"")
    status = probe_or_create_empty_plan_file(path)
    assert status.kind is PlanFileSeedStatusKind.EMPTY
    assert path.read_bytes() == b""


def test_probe_directory_is_not_a_file(tmp_path: Path) -> None:
    path = tmp_path / "plan.md"
    path.mkdir()
    status = probe_or_create_empty_plan_file(path)
    assert status == PlanFileSeedStatus.missing(PlanFileSeedFailure.NOT_A_FILE)


def test_enter_prompt_format_empty_seed() -> None:
    prompt = format_enter_plan_prompt(
        plan_file_path="/sess/plan.md",
        plan_file_seed=PlanFileSeedStatus.empty(),
        tool_hints=EnterPlanModeToolHints(),
    )
    assert ENTERED_MESSAGE in prompt
    assert "The file exists and is empty." in prompt
    assert "5. Write your plan to the plan file above" in prompt
    assert "6. When ready, use exit_plan_mode to present your plan to the user" in prompt
    assert "ask_user_question" in prompt
    assert "create it at that path first if needed" not in prompt


def test_enter_prompt_unavailable_seed() -> None:
    prompt = format_enter_plan_prompt(
        plan_file_path=".grok/plan.md",
        plan_file_seed=PlanFileSeedStatus.missing(PlanFileSeedFailure.UNAVAILABLE),
    )
    assert "The plan file location is unavailable." in prompt
    assert "5. Write your plan to the plan file above" in prompt


def test_enter_prompt_task_hint() -> None:
    prompt = format_enter_plan_prompt(
        plan_file_path="/p.md",
        plan_file_seed=PlanFileSeedStatus.empty(),
        tool_hints=EnterPlanModeToolHints(task="delegate"),
    )
    assert 'subagent_type="explore"' in prompt
    assert "delegate" in prompt


def test_exit_prompt_includes_plan() -> None:
    text = format_exit_plan_ready(
        plan_content="Step 1\nStep 2",
        plan_file_path="/ws/.grok/plan.md",
    )
    assert PLAN_READY_MESSAGE in text
    assert "saved at:" in text
    assert "Step 1" in text
    assert "## Plan:" in text


# ── tool wire ────────────────────────────────────────────────────────────


def test_enter_plan_mode_returns_confirmation(tmp_path: Path) -> None:
    tools = _tools()
    mode = SessionModeState()
    out = tools.call("enter_plan_mode", {}, _ctx(tmp_path, session_mode_state=mode))
    assert "entered plan mode" in out.lower()
    assert "exploring the codebase" in out
    assert "implementation plan" in out
    assert mode.is_plan()
    plan = tmp_path / ".grok" / "plan.md"
    assert plan.is_file()
    assert plan.read_bytes() == b""
    assert "The file exists and is empty." in out
    assert "exit_plan_mode" in out


def test_enter_empty_schema() -> None:
    schema = EnterPlanModeTool().parameters_schema()
    assert schema.get("properties") == {}
    assert "approved" not in schema.get("properties", {})


def test_exit_empty_schema() -> None:
    schema = ExitPlanModeTool().parameters_schema()
    assert schema.get("properties") == {}
    assert "approved" not in schema.get("properties", {})


def test_enter_user_declined(tmp_path: Path) -> None:
    tools = _tools()
    mode = SessionModeState()
    out = tools.call(
        "enter_plan_mode",
        {},
        _ctx(
            tmp_path,
            session_mode_state=mode,
            plan_mode_consent_fn=lambda: False,
        ),
    )
    assert out == USER_DECLINED_ENTER
    assert not mode.is_plan()


def test_enter_does_not_truncate_existing(tmp_path: Path) -> None:
    tools = _tools()
    plan = tmp_path / ".grok" / "plan.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# prior plan\n", encoding="utf-8")
    mode = SessionModeState()
    out = tools.call("enter_plan_mode", {}, _ctx(tmp_path, session_mode_state=mode))
    assert "The file exists but is not empty." in out
    assert plan.read_text(encoding="utf-8") == "# prior plan\n"


def test_enter_custom_plan_file_path(tmp_path: Path) -> None:
    tools = _tools()
    custom = tmp_path / "session" / "plan.md"
    mode = SessionModeState()
    out = tools.call(
        "enter_plan_mode",
        {},
        _ctx(
            tmp_path,
            session_mode_state=mode,
            plan_file_path=str(custom),
        ),
    )
    assert custom.is_file()
    assert str(custom) in out or custom.as_posix() in out.replace("\\", "/")
    assert mode.is_plan()


def test_enter_tool_hints_from_extra(tmp_path: Path) -> None:
    tools = _tools()
    out = tools.call(
        "enter_plan_mode",
        {},
        _ctx(
            tmp_path,
            session_mode_state=SessionModeState(),
            plan_tool_hints={
                "ask_user": "AskUser",
                "exit_plan": "FinishPlan",
                "task": "delegate",
            },
        ),
    )
    assert "AskUser" in out
    assert "FinishPlan" in out
    assert "delegate" in out


def test_enter_works_without_mode_state(tmp_path: Path) -> None:
    """Grok works without NotificationHandle — seed + prompt still succeed."""
    tools = _tools()
    out = tools.call("enter_plan_mode", {}, _ctx(tmp_path))
    assert "entered plan mode" in out.lower()
    assert (tmp_path / ".grok" / "plan.md").is_file()


def test_exit_with_plan_content(tmp_path: Path) -> None:
    tools = _tools()
    mode = SessionModeState()
    ctx = _ctx(tmp_path, session_mode_state=mode)
    tools.call("enter_plan_mode", {}, ctx)
    plan = tmp_path / ".grok" / "plan.md"
    plan.write_text("# My Plan\n\n1. Do thing A\n2. Do thing B\n", encoding="utf-8")
    out = tools.call("exit_plan_mode", {}, ctx)
    assert "plan has been approved" in out.lower()
    assert "start coding" in out
    assert "Do thing A" in out
    assert "Do thing B" in out
    assert "## Plan:" in out
    assert not mode.is_plan()


def test_exit_with_empty_plan_file(tmp_path: Path) -> None:
    tools = _tools()
    mode = SessionModeState()
    ctx = _ctx(tmp_path, session_mode_state=mode)
    tools.call("enter_plan_mode", {}, ctx)
    out = tools.call("exit_plan_mode", {}, ctx)
    assert out == EMPTY_PLAN_MESSAGE
    assert not mode.is_plan()


def test_exit_with_missing_plan_file(tmp_path: Path) -> None:
    tools = _tools()
    mode = SessionModeState()
    # Exit without enter — still resolves cwd/.grok/plan.md
    out = tools.call(
        "exit_plan_mode",
        {},
        _ctx(tmp_path, session_mode_state=mode),
    )
    assert "No plan content was found" in out
    assert "you can proceed" in out


def test_exit_host_cancelled(tmp_path: Path) -> None:
    tools = _tools()
    mode = SessionModeState()
    ctx = _ctx(
        tmp_path,
        session_mode_state=mode,
        plan_mode_exit_fn=lambda _payload: {
            "outcome": "cancelled",
            "feedback": "Please add error handling",
        },
    )
    tools.call("enter_plan_mode", {}, ctx)
    (tmp_path / ".grok" / "plan.md").write_text("plan", encoding="utf-8")
    out = tools.call("exit_plan_mode", {}, ctx)
    assert "cancelled" in out.lower()
    assert "Please add error handling" in out
    assert not mode.is_plan()


def test_descriptions_match_grok() -> None:
    enter = EnterPlanModeTool().description().description
    assert "ambiguity about the right approach" in enter
    assert "read-only plan mode" in enter
    exit_d = ExitPlanModeTool().description().description
    assert "Exit plan mode" in exit_d
    assert "plan file" in exit_d
