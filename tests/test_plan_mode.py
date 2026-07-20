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
    """Tool-layer fallback without PlanFilePath: cwd/.grok/plan.md (Grok resources)."""
    abs_t, display = resolve_plan_file_path(cwd=tmp_path, plan_file_path=None)
    assert abs_t is not None
    assert abs_t == (tmp_path / PLAN_FILE_RELATIVE_PATH).resolve() or abs_t == (
        tmp_path / PLAN_FILE_RELATIVE_PATH
    )
    assert display.endswith(".grok/plan.md") or display.endswith(".grok\\plan.md")


def test_session_plan_file_is_under_session_dir(tmp_path: Path) -> None:
    """Grok PlanModeTracker: session_dir/plan.md (product authority)."""
    from codedoggy.orchestration.session_mode import (
        plan_file_for_session,
        plan_mode_session_dir,
    )

    plan = plan_file_for_session(tmp_path, "sess-abc")
    assert plan == plan_mode_session_dir(tmp_path, "sess-abc") / "plan.md"
    assert plan.name == "plan.md"
    assert "sessions" in plan.parts


def test_edit_gate_off_during_exit_pending(tmp_path: Path) -> None:
    """Grok is_active() only — ExitPending must not block non-plan edits."""
    from codedoggy.orchestration.session_mode import PlanEditGate, plan_mode_edit_gate
    from codedoggy.tools.kinds import ToolKind

    state = SessionModeState()
    state.enter_plan(str(tmp_path / ".grok" / "sessions" / "s1" / "plan.md"))
    state.user_exit(turn_in_flight=True)
    assert state.plan_phase == "exit_pending"
    assert not state.is_plan()
    gate = plan_mode_edit_gate(
        state,
        cwd=tmp_path,
        kind=ToolKind.Edit,
        tool_name="search_replace",
        args={"file_path": "main.py", "old_string": "a", "new_string": "b"},
    )
    assert gate == PlanEditGate.ALLOW


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


def test_exit_host_cancelled_keeps_plan_mode(tmp_path: Path) -> None:
    """Grok Cancelled: stay in plan mode; revise message with feedback."""
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
    assert "The user wants to revise the plan" in out
    assert "Please add error handling" in out
    assert mode.is_plan(), "cancelled must not deactivate plan mode (Grok)"


def test_exit_host_cancelled_empty_plan_stays_in_mode(tmp_path: Path) -> None:
    tools = _tools()
    mode = SessionModeState()
    ctx = _ctx(
        tmp_path,
        session_mode_state=mode,
        plan_mode_exit_fn=lambda _payload: {"outcome": "cancelled"},
    )
    tools.call("enter_plan_mode", {}, ctx)
    out = tools.call("exit_plan_mode", {}, ctx)
    assert "does not want to exit plan mode" in out
    assert mode.is_plan()


def test_exit_host_abandoned_leaves_plan_mode(tmp_path: Path) -> None:
    tools = _tools()
    mode = SessionModeState()
    ctx = _ctx(
        tmp_path,
        session_mode_state=mode,
        plan_mode_exit_fn=lambda _payload: {"outcome": "abandoned"},
    )
    tools.call("enter_plan_mode", {}, ctx)
    (tmp_path / ".grok" / "plan.md").write_text("plan", encoding="utf-8")
    out = tools.call("exit_plan_mode", {}, ctx)
    assert "abandoned" in out.lower()
    assert not mode.is_plan()


def test_descriptions_match_grok() -> None:
    enter = EnterPlanModeTool().description().description
    assert "ambiguity about the right approach" in enter
    assert "read-only plan mode" in enter
    exit_d = ExitPlanModeTool().description().description
    assert "Exit plan mode" in exit_d
    assert "plan file" in exit_d


def test_plan_lifecycle_pending_activates_on_begin_turn() -> None:
    state = SessionModeState()
    assert state.enter_plan_pending(".grok/plan.md")
    assert state.plan_phase == "pending"
    assert not state.is_plan()  # gate off until Active
    assert state.is_plan_ui()
    rem = state.begin_turn()
    assert rem is not None
    assert "Plan mode is active" in rem or "Returning to Plan Mode" in rem
    assert state.plan_phase == "active"
    assert state.is_plan()


def test_plan_lifecycle_user_exit_deferred() -> None:
    state = SessionModeState()
    state.enter_plan(".grok/plan.md")
    state.user_exit(turn_in_flight=True)
    assert state.plan_phase == "exit_pending"
    # Grok is_active() only — edit gate off during ExitPending; UI chrome stays.
    assert not state.is_plan()
    assert state.is_plan_ui()
    state.end_turn()
    assert state.plan_phase == "inactive"
    assert not state.is_plan()
    assert state.pending_exit_reminder
    rem = state.begin_turn()
    assert rem is not None
    assert "exited plan mode" in rem


def test_plan_lifecycle_pending_gate_allows_edits(tmp_path: Path) -> None:
    from codedoggy.orchestration.session_mode import plan_mode_edit_gate, PlanEditGate
    from codedoggy.tools.kinds import ToolKind

    state = SessionModeState()
    state.enter_plan_pending(".grok/plan.md")
    gate = plan_mode_edit_gate(
        state,
        cwd=tmp_path,
        kind=ToolKind.Edit,
        tool_name="search_replace",
        args={"file_path": "main.py", "old_string": "a", "new_string": "b"},
    )
    assert gate == PlanEditGate.ALLOW


def test_plan_lifecycle_reentry_reminder() -> None:
    state = SessionModeState()
    state.enter_plan(".grok/plan.md")
    state.exit_plan(approved=True)
    state.enter_plan_pending(".grok/plan.md")
    rem = state.begin_turn()
    assert rem is not None
    assert "Returning to Plan Mode" in rem


def test_abandoned_exit_reason_metadata() -> None:
    state = SessionModeState()
    state.enter_plan(".grok/plan.md")
    state.exit_plan(approved=False, reason="abandoned")
    assert state.metadata.get("last_plan_exit") == "abandoned"
    assert not state.is_plan()


def test_reset_after_compaction_resets_reminder_count() -> None:
    state = SessionModeState()
    state.enter_plan(".grok/plan.md")
    state.begin_turn()  # advances reminder_count
    assert state.reminder_count >= 1
    state.reset_after_compaction()
    assert state.reminder_count == 0


def test_wait_or_resolve_parked_plan_approval(tmp_path: Path) -> None:
    from codedoggy.session.kernel import RuntimeKernel

    plan = tmp_path / ".grok" / "plan.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# implement me\n", encoding="utf-8")

    k = RuntimeKernel(cwd=tmp_path, session_id="park1")
    k.enter_plan_mode(str(plan.resolve()))
    assert k.session_mode_state is not None
    k.session_mode_state.awaiting_plan_approval = True
    k.persist_plan_mode_state()

    # approved
    k.tool_extra = {
        "plan_mode_exit_fn": lambda _p: {"outcome": "approved"},
    }
    msg = k.wait_or_resolve_parked_plan_approval()
    assert msg is not None and "approved" in msg.lower()
    assert not k.session_mode_state.is_plan()

    # revise keeps plan
    k2 = RuntimeKernel(cwd=tmp_path, session_id="park2")
    k2.enter_plan_mode(str(plan.resolve()))
    assert k2.session_mode_state is not None
    k2.session_mode_state.awaiting_plan_approval = True
    k2.tool_extra = {
        "plan_mode_exit_fn": lambda _p: {
            "outcome": "cancelled",
            "feedback": "add tests",
        },
    }
    msg2 = k2.wait_or_resolve_parked_plan_approval()
    assert msg2 is not None and "revise" in msg2.lower()
    assert k2.session_mode_state.is_plan()
    assert not k2.session_mode_state.awaiting_plan_approval


def test_plan_mode_json_round_trip_active(tmp_path: Path) -> None:
    from codedoggy.orchestration.session_mode import (
        load_plan_mode_state,
        plan_mode_json_path,
        save_plan_mode_state,
    )

    state = SessionModeState()
    state.enter_plan(str(tmp_path / ".grok" / "plan.md"))
    state.awaiting_plan_approval = True
    state.reminder_count = 3
    path = save_plan_mode_state(state, cwd=tmp_path, session_id="sess-1")
    assert path is not None
    assert path == plan_mode_json_path(tmp_path, "sess-1")
    assert path.is_file()

    restored = load_plan_mode_state(cwd=tmp_path, session_id="sess-1")
    assert restored is not None
    assert restored.plan_phase == "active"
    assert restored.is_plan()
    assert restored.awaiting_plan_approval is True
    assert restored.reminder_count == 3
    assert restored.was_previously_active is True


def test_plan_mode_json_collapses_pending_and_exit_pending(tmp_path: Path) -> None:
    from codedoggy.orchestration.session_mode import (
        load_plan_mode_state,
        save_plan_mode_state,
    )

    pending = SessionModeState()
    pending.enter_plan_pending(".grok/plan.md")
    save_plan_mode_state(pending, cwd=tmp_path, session_id="p1")
    r1 = load_plan_mode_state(cwd=tmp_path, session_id="p1")
    assert r1 is not None
    assert r1.plan_phase == "inactive"
    assert not r1.is_plan()

    active = SessionModeState()
    active.enter_plan(".grok/plan.md")
    active.user_exit(turn_in_flight=True)
    assert active.plan_phase == "exit_pending"
    save_plan_mode_state(active, cwd=tmp_path, session_id="e1")
    r2 = load_plan_mode_state(cwd=tmp_path, session_id="e1")
    assert r2 is not None
    assert r2.plan_phase == "inactive"
    assert r2.pending_exit_reminder is True


def test_kernel_persist_and_load_plan_mode(tmp_path: Path) -> None:
    from codedoggy.session.kernel import RuntimeKernel

    k = RuntimeKernel(cwd=tmp_path, session_id="k1")
    k.enter_plan_mode()
    assert k.session_mode_state is not None
    assert k.session_mode_state.is_plan()
    path = tmp_path / ".grok" / "sessions" / "k1" / "plan_mode.json"
    assert path.is_file()

    k2 = RuntimeKernel(cwd=tmp_path, session_id="k1")
    assert k2.load_plan_mode_state() is True
    assert k2.session_mode_state is not None
    assert k2.session_mode_state.is_plan()
