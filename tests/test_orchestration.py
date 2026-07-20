"""Grok-aligned orchestration: two-phase tools, plan gate, subagent, interjection."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.orchestration.agent_def import (
    build_agent,
    builtin_explore,
    filter_toolset,
    resolve_agent_definition,
)
from codedoggy.orchestration.session_mode import (
    PlanEditGate,
    SessionModeState,
    plan_mode_edit_gate,
)
from codedoggy.orchestration.subagent import (
    SubagentCoordinator,
    SubagentRequest,
    make_child_runner,
)
from codedoggy.orchestration.tool_pipeline import (
    execute_tool_calls_two_phase,
    prepare_tool_call,
)
from codedoggy.orchestration.types import (
    CapabilityMode,
    PrecheckVerdict,
    ToolLoopOutcome,
)
from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.kinds import ToolKind
from codedoggy.tools.policy import WorkspacePolicy
from codedoggy.turn.hooks import HookContext, NoopHooks
from codedoggy.turn.loop import run_agent_loop
from codedoggy.turn.types import HookDecision, Message, Role, SampleResult, ToolCall


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


def test_prepare_unknown_tool_is_soft(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    pre = prepare_tool_call(
        tools,
        ToolCall(id="1", name="no_such_tool", arguments={}),
        cwd=tmp_path,
    )
    assert pre.verdict is PrecheckVerdict.NON_EXISTING
    assert pre.observation and "not_found" in pre.observation


def test_plan_mode_rejects_non_plan_edit(tmp_path: Path) -> None:
    state = SessionModeState()
    state.enter_plan(".grok/plan.md")
    tools = ToolRegistryBuilder.new().finalize()
    pre = prepare_tool_call(
        tools,
        ToolCall(
            id="1",
            name="search_replace",
            arguments={
                "file_path": "main.py",
                "old_string": "a",
                "new_string": "b",
            },
        ),
        cwd=tmp_path,
        mode_state=state,
    )
    assert pre.verdict is PrecheckVerdict.PLAN_REJECT

    # Plan file itself is allowed through the gate (policy may still apply)
    gate = plan_mode_edit_gate(
        state,
        cwd=tmp_path,
        kind=ToolKind.Edit,
        tool_name="search_replace",
        args={"file_path": ".grok/plan.md"},
    )
    assert gate == PlanEditGate.ALLOW


def test_plan_mode_allows_shell_and_rejects_apply_patch(tmp_path: Path) -> None:
    """Grok: non-edit tools pass the plan gate; apply_patch always rejected."""
    from codedoggy.tools.kinds import ToolKind as TK

    state = SessionModeState()
    state.enter_plan(".grok/plan.md")
    shell = plan_mode_edit_gate(
        state,
        cwd=tmp_path,
        kind=TK.Execute,
        tool_name="run_terminal_command",
        args={"command": "echo hi"},
    )
    assert shell == PlanEditGate.ALLOW
    spawn = plan_mode_edit_gate(
        state,
        cwd=tmp_path,
        kind=TK.Task,
        tool_name="spawn_subagent",
        args={"prompt": "explore", "description": "x"},
    )
    assert spawn == PlanEditGate.ALLOW
    patch = plan_mode_edit_gate(
        state,
        cwd=tmp_path,
        kind=TK.Edit,
        tool_name="apply_patch",
        args={"patch": "noop"},
    )
    assert patch == PlanEditGate.REJECT_NON_PLAN_FILE


def test_update_goal_completed_exits_goal_mode(tmp_path: Path) -> None:
    """update_goal(completed) exits goal session flag (tool surface + glue)."""
    from codedoggy.tools.builtins.update_goal import UpdateGoalTool
    from codedoggy.tools.runtime import ToolCallContext

    state = SessionModeState()
    state.enter_goal()
    tool = UpdateGoalTool()
    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={"session_mode_state": state},
    )
    out = tool.run(ctx, {"completed": True, "message": "done"})
    # Grok CompletedWithoutClassifier model-facing string
    assert out == "Goal marked complete."
    assert not state.is_goal()


def test_builtin_explore_is_read_only_filtered() -> None:
    parent = ToolRegistryBuilder.new().finalize()
    agent = build_agent(builtin_explore(), parent_tools=parent)
    names = set(agent.tools.by_client_name)
    assert "read_file" in names or "grep" in names
    # Must not expose write tools even if parent has them
    assert "search_replace" not in names
    assert "run_terminal_command" not in names or agent.definition.capability_mode.value == "read-only"
    assert "explore" in (agent.system_prompt or "").lower() or "read-only" in (
        agent.system_prompt or ""
    ).lower()


def test_permission_reject_stops_batch(tmp_path: Path) -> None:
    """Grok: PermissionReject cancels remaining tools in the batch."""
    tools = ToolRegistryBuilder.new().finalize()
    policy = WorkspacePolicy(
        cwd=tmp_path,
        deny_write_globs=["secret.txt"],
    )
    # Write a deny by path — use protected .env style via custom policy check
    class DenySecret:
        def check_write(self, path: str):
            from codedoggy.tools.policy import PolicyDecision

            if "secret" in path.replace("\\", "/"):
                return PolicyDecision(allowed=False, code="policy_denied", reason="no secret")
            return PolicyDecision(allowed=True, code="ok", reason="")

        def check_shell(self, cmd: str):
            return None

    (tmp_path / "ok.txt").write_text("x\n", encoding="utf-8")
    batch = execute_tool_calls_two_phase(
        tools,
        [
            ToolCall(
                id="1",
                name="search_replace",
                arguments={
                    "file_path": "secret.txt",
                    "old_string": "a",
                    "new_string": "b",
                },
            ),
            ToolCall(
                id="2",
                name="read_file",
                arguments={"target_file": "ok.txt"},
            ),
        ],
        cwd=tmp_path,
        extra={"policy": DenySecret()},
        parallel=False,
    )
    assert batch.outcome is ToolLoopOutcome.PERMISSION_REJECT
    assert len(batch.records) == 2
    assert batch.records[0].ok is False
    assert "batch_cancelled" in (batch.records[1].error_code or "") or "cancelled" in (
        batch.records[1].content or ""
    ).lower()


def test_hook_deny_is_non_terminal(tmp_path: Path) -> None:
    """Grok HookDenied: soft deny observation, other tools still run."""
    tools = ToolRegistryBuilder.new().finalize()
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")

    def pre_hook(call: ToolCall, ctx):
        if call.name == "grep":
            return HookDecision(
                abort=True,
                abort_reason="grep not allowed by test hook",
                append_observation="Error (hook_denied): grep blocked",
            )
        return None

    batch = execute_tool_calls_two_phase(
        tools,
        [
            ToolCall(id="1", name="grep", arguments={"pattern": "x", "path": "."}),
            ToolCall(id="2", name="read_file", arguments={"target_file": "a.txt"}),
        ],
        cwd=tmp_path,
        pre_tool_hook=pre_hook,
        parallel=False,
    )
    # Soft deny must NOT set PermissionReject
    assert batch.outcome is ToolLoopOutcome.CONTINUE
    assert any("hook_denied" in (r.content or "") for r in batch.records)
    assert any(r.ok and r.call.name == "read_file" for r in batch.records)


def test_capability_filters_explore_tools() -> None:
    parent = ToolRegistryBuilder.new().finalize()
    agent = build_agent(builtin_explore(), parent_tools=parent)
    names = set(agent.tools.client_names())
    assert "read_file" in names
    assert "grep" in names
    assert "search_replace" not in names
    assert "run_terminal_cmd" not in names
    assert "run_terminal_command" not in names
    assert "spawn_subagent" not in names


def test_filter_toolset_read_only() -> None:
    parent = ToolRegistryBuilder.new().finalize()
    child = filter_toolset(parent, capability=CapabilityMode.READ_ONLY)
    names = set(child.client_names())
    assert "read_file" in names
    assert "search_replace" not in names


def test_loop_uses_two_phase_and_exit_reason(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("payload\n", encoding="utf-8")
    tools = ToolRegistryBuilder.new().finalize()
    sampler = ScriptedSampler(
        [
            SampleResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="read_file",
                        arguments={"target_file": "a.txt"},
                    )
                ],
            ),
            SampleResult(content="got it"),
        ]
    )
    result = run_agent_loop(
        user_text="read",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        max_turns=5,
    )
    assert result.completed
    assert result.exit_reason == "completed"
    assert "read_file" in result.tools_called


def test_interjection_drains_into_messages(tmp_path: Path) -> None:
    from codedoggy.orchestration.prompt_queue import InterjectionBuffer

    tools = ToolRegistryBuilder.new().finalize()
    buf = InterjectionBuffer()
    buf.push("change of plan: stop")

    class CapSampler:
        def __init__(self) -> None:
            self.seen: list[str] = []
            self.n = 0

        def sample(self, messages, tools):
            self.n += 1
            blob = "\n".join((m.content or "") for m in messages if m.role is Role.USER)
            self.seen.append(blob)
            if self.n == 1:
                return SampleResult(content="ok after interrupt")
            return SampleResult(content="ok")

    sampler = CapSampler()
    result = run_agent_loop(
        user_text="start",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        max_turns=3,
        tool_extra={"interjection_buffer": buf},
    )
    assert result.completed
    # Grok format_interjection (xai-interjection-core) — not [interjection]
    assert any(
        "The user sent a message while you were working:" in s
        and "<user_query>" in s
        and "change of plan" in s
        for s in sampler.seen
    )


def test_session_handle_prompt_interjects_while_busy(tmp_path: Path) -> None:
    """Session API: handle_prompt during TURN_RUNNING → interject soft result."""
    import threading

    from codedoggy.orchestration.prompt_queue import InterjectionBuffer
    from codedoggy.session import Session, SessionExtensions, TurnStatus
    from codedoggy.session.kernel import RuntimeKernel
    from codedoggy.session.types import TurnRequest, TurnResult

    gate = threading.Event()
    go = threading.Event()
    buf = InterjectionBuffer()

    class SlowRunner:
        def run(self, request: TurnRequest, *, session: Session) -> TurnResult:
            gate.set()
            go.wait(timeout=5.0)
            # Mid-turn interjection should already be buffered
            assert not buf.is_empty()
            return TurnResult(status=TurnStatus.COMPLETED, final_text="main-done")

    runner = SlowRunner()
    kernel = RuntimeKernel(
        cwd=tmp_path,
        session_id="orch-inj",
        turn_runner=runner,
        interjection_buffer=buf,
    )
    s = Session.create(
        tmp_path,
        extensions=SessionExtensions(turn_runner=runner, kernel=kernel),
    )
    out: list[TurnResult] = []

    def _run() -> None:
        out.append(s.handle_prompt("long task"))

    th = threading.Thread(target=_run)
    th.start()
    assert gate.wait(timeout=5.0)
    soft = s.handle_prompt("interject now")
    assert soft.status is TurnStatus.QUEUED  # not COMPLETED (false success)
    assert soft.metadata.get("interjected") is True
    go.set()
    th.join(timeout=5.0)
    assert out and out[0].final_text == "main-done"
    # Loop path would drain; buffer may still hold if runner didn't drain
    s.close()


def test_subagent_explore_summary_foldback(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("secret-findings-42\n", encoding="utf-8")
    parent_tools = ToolRegistryBuilder.new().finalize()

    child_script = [
        SampleResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="read_file",
                    arguments={"target_file": "note.txt"},
                )
            ],
        ),
        SampleResult(content="Found secret-findings-42 in note.txt"),
    ]
    child_sampler = ScriptedSampler(child_script)
    run_fn = make_child_runner(
        parent_cwd=tmp_path,
        parent_tools=parent_tools,
        parent_sampler=child_sampler,
        parent_system_prompt="parent base",
    )
    coord = SubagentCoordinator()
    snap = coord.spawn(
        SubagentRequest(
            subagent_type="explore",
            prompt="find secrets",
            description="scan notes",
            parent_session_id="parent1",
            run_in_background=False,
        ),
        run_fn=run_fn,
    )
    assert snap.status == "completed"
    assert snap.output is not None
    assert "secret-findings-42" in snap.output
    assert "subagent:explore" in snap.output
    # Child must not expose write tools — if it tried search_replace it would fail
    assert snap.tool_calls >= 1


def test_resolve_builtin_agents() -> None:
    assert resolve_agent_definition("explore") is not None
    assert resolve_agent_definition("plan") is not None
    assert resolve_agent_definition("nope") is None


def test_worktree_create_and_isolate(tmp_path: Path) -> None:
    """Git worktree isolation: child path under .codedoggy/worktrees, parent clean."""
    import subprocess

    from codedoggy.orchestration.types import IsolationMode
    from codedoggy.orchestration.worktree import (
        create_worktree,
        find_git_root,
        remove_worktree,
        should_cleanup_worktree,
    )

    # Init a tiny repo
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "t"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "root.txt").write_text("parent\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    assert find_git_root(tmp_path) == tmp_path.resolve()
    handle = create_worktree(tmp_path, subagent_id="sub_testwt01")
    try:
        assert handle.path.exists()
        assert ".codedoggy" in handle.path.parts
        assert "worktrees" in handle.path.parts
        # Write only in worktree
        (handle.path / "child_only.txt").write_text("isolated\n", encoding="utf-8")
        assert not (tmp_path / "child_only.txt").exists()
        assert (handle.path / "child_only.txt").exists()
    finally:
        remove_worktree(tmp_path, handle.path, force=True)

    # Subagent with isolation=worktree uses worktree cwd
    parent_tools = ToolRegistryBuilder.new().finalize()
    seen_cwd: list[str] = []

    class CwdSampler:
        def sample(self, messages, tools):
            # tools may touch cwd via extra — we observe through a write tool later
            return SampleResult(content=f"cwd-ok")

    run_fn = make_child_runner(
        parent_cwd=tmp_path,
        parent_tools=parent_tools,
        parent_sampler=CwdSampler(),
        parent_system_prompt="base",
    )
    coord = SubagentCoordinator()
    snap = coord.spawn(
        SubagentRequest(
            subagent_type="explore",
            prompt="ping",
            parent_session_id="p",
            run_in_background=False,
            isolation=IsolationMode.WORKTREE,
        ),
        run_fn=run_fn,
    )
    assert snap.status == "completed"
    assert snap.worktree_path is not None
    assert Path(snap.worktree_path).exists() or snap.metadata.get("isolation") == "worktree"
    assert snap.metadata.get("isolation") == "worktree"
    # Preserve by default
    assert should_cleanup_worktree() is False
    if snap.worktree_path and Path(snap.worktree_path).exists():
        remove_worktree(tmp_path, Path(snap.worktree_path), force=True)


def test_host_stream_delta_callback(tmp_path: Path) -> None:
    """Glue: on_sample_delta receives chunks when host opts into stream."""
    from codedoggy.turn.loop import run_agent_loop

    deltas: list[str] = []

    class StreamSampler:
        stream = False
        on_delta = None

        def sample(self, messages, tools):
            if self.stream and callable(self.on_delta):
                self.on_delta("Hel")
                self.on_delta("lo")
            return SampleResult(content="Hello", raw={"usage": {"prompt_tokens": 10}})

    tools = ToolRegistryBuilder.new().finalize()
    result = run_agent_loop(
        user_text="hi",
        sampler=StreamSampler(),
        tools=tools,
        cwd=tmp_path,
        max_turns=2,
        tool_extra={"on_sample_delta": deltas.append, "stream_sample": True},
    )
    assert result.final_text == "Hello"
    assert deltas == ["Hel", "lo"]


def test_format_interjection_matches_grok_source() -> None:
    """Lock wire shape to xai-interjection-core tests."""
    from codedoggy.orchestration.interjection import (
        LARGE_PROMPT_THRESHOLD,
        format_interjection,
        user_query,
    )

    wrapped = format_interjection("please also add tests")
    assert wrapped.startswith("The user sent a message while you were working:\n")
    assert "<user_query>\nplease also add tests\n</user_query>" in wrapped
    assert wrapped.strip().endswith("</user_query>")
    assert "After completing your current task" not in wrapped
    assert user_query("hi") == "<user_query>\nhi\n</user_query>"
    # Truncation
    huge = "x" * (LARGE_PROMPT_THRESHOLD + 100)
    t = format_interjection(huge)
    assert "... [truncated]" in t


def test_worktree_merge_into_parent(tmp_path: Path) -> None:
    import subprocess

    from codedoggy.orchestration.worktree import (
        commit_worktree_changes,
        create_worktree,
        merge_worktree_into_parent,
        remove_worktree,
    )

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "t"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True
    )

    handle = create_worktree(tmp_path, subagent_id="merge_me")
    try:
        (handle.path / "from_child.txt").write_text("child-change\n", encoding="utf-8")
        sha = commit_worktree_changes(handle.path, message="child commit")
        assert sha
        # Parent must not have the file yet
        assert not (tmp_path / "from_child.txt").exists()
        result = merge_worktree_into_parent(
            tmp_path,
            branch=handle.branch,
            worktree_path=handle.path,
            subagent_id="merge_me",
            strategy="merge",
            cleanup_worktree=True,
        )
        assert result.ok, result.message
        assert (tmp_path / "from_child.txt").read_text(encoding="utf-8") == "child-change\n"
        assert result.commit
    finally:
        if handle.path.exists():
            remove_worktree(tmp_path, handle.path, force=True)


def test_subagent_resume_continues_transcript(tmp_path: Path) -> None:
    """Resume reuses prior messages so the child sees earlier work."""
    parent_tools = ToolRegistryBuilder.new().finalize()
    seen_user: list[str] = []

    class ResumeSampler:
        def __init__(self) -> None:
            self.n = 0

        def sample(self, messages, tools):
            self.n += 1
            users = [
                m.content or ""
                for m in messages
                if getattr(m.role, "value", m.role) == "user"
            ]
            seen_user.append("|".join(users))
            if self.n == 1:
                return SampleResult(content="first-pass-result")
            # Second run (resume) must still see first user prompt in prior
            assert any("first task" in u for u in users)
            assert any("continue work" in u for u in users)
            return SampleResult(content="second-pass-result")

    run_fn = make_child_runner(
        parent_cwd=tmp_path,
        parent_tools=parent_tools,
        parent_sampler=ResumeSampler(),
        parent_system_prompt="base",
    )
    coord = SubagentCoordinator()
    first = coord.spawn(
        SubagentRequest(
            subagent_type="explore",
            prompt="first task",
            parent_session_id="p1",
            run_in_background=False,
        ),
        run_fn=run_fn,
    )
    assert first.status == "completed"
    assert first.live_messages
    assert first.can_resume

    second = coord.resume(
        first.subagent_id,
        "continue work",
        run_fn=run_fn,
        run_in_background=False,
    )
    assert second.status == "completed"
    assert "second-pass-result" in (second.output or "")
    assert second.metadata.get("resumed") is True or "resume" in (second.output or "")


def test_path_lock_serializes_same_file(tmp_path: Path) -> None:
    from codedoggy.orchestration.path_lock import lock_path_for_args

    assert lock_path_for_args({"file_path": "a.py"}) == "a.py"
    assert lock_path_for_args({"target_file": "b.py"}) == "b.py"
    assert lock_path_for_args({"target_directory": "src"}) is None
