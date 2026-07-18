"""Background shell tasks + orchestration tools (GrokBuild surface)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from codedoggy.orchestration.session_mode import SessionModeState
from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.grok_build.monitor_event import (
    LineProcessor,
    batch_lines,
    wrap_monitor_event,
)
from codedoggy.tools.grok_build.monitor_rate_limiter import (
    MonitorRateLimiter,
    RateLimitKind,
    TokenBucket,
)
from codedoggy.tools.grok_build.monitor_types import (
    DEFAULT_TIMEOUT_MS,
    resolved_timeout_ms,
    validate_monitor_input,
    MonitorError,
)
from codedoggy.tools.grok_build.task_output_logic import (
    MAX_MULTI_WAIT_IDS,
    build_kill_task_description,
    build_task_output_description,
    build_wait_tasks_description,
    capped_wait_timeout_ms,
    DEFAULT_WAIT_TIMEOUT_MS,
    MAX_WAIT_BLOCK_MS,
    task_output_waits,
)
from codedoggy.tools.runtime import ToolCallContext, ToolError
from codedoggy.tools.task_manager import BackgroundTaskManager


def _tools():
    return ToolRegistryBuilder.new().finalize()


def test_run_background_returns_task_id(tmp_path: Path) -> None:
    tools = _tools()
    tm = BackgroundTaskManager(work_dir=tmp_path / "tasks")
    ctx = ToolCallContext(cwd=tmp_path, extra={"task_manager": tm})
    out = tools.call(
        "run_terminal_cmd",
        {
            "command": 'python -c "import time; time.sleep(2); print(42)"',
            "description": "sleep then print",
            "is_background": True,
        },
        ctx,
    )
    assert "<task-id>" in out
    assert "<status>running</status>" in out
    # extract id
    tid = out.split("<task-id>")[1].split("</task-id>")[0]
    snap = tools.call(
        "get_task_output",
        {"task_ids": [tid], "timeout_ms": 10_000},
        ctx,
    )
    assert "=== Task" in snap
    assert "42" in snap or "Status: completed" in snap or "Exit Code" in snap
    # cleanup
    tools.call("kill_task", {"task_id": tid}, ctx)


def test_kill_running_background(tmp_path: Path) -> None:
    tools = _tools()
    tm = BackgroundTaskManager(work_dir=tmp_path / "tasks")
    ctx = ToolCallContext(cwd=tmp_path, extra={"task_manager": tm})
    out = tools.call(
        "run_terminal_cmd",
        {
            "command": 'python -c "import time; time.sleep(30)"',
            "description": "long sleep",
            "is_background": True,
        },
        ctx,
    )
    tid = out.split("<task-id>")[1].split("</task-id>")[0]
    killed = tools.call("kill_task", {"task_id": tid}, ctx)
    # Grok: "killed: Task was terminated successfully"
    assert killed.startswith("killed:") or killed.startswith("already_exited:")
    assert "Task was terminated successfully" in killed or "already" in killed.lower()
    snap = tools.call("get_task_output", {"task_ids": [tid]}, ctx)
    assert "cancelled" in snap or "Exit Code" in snap or "Status:" in snap


def test_get_task_output_not_found(tmp_path: Path) -> None:
    tools = _tools()
    tm = BackgroundTaskManager(work_dir=tmp_path / "tasks")
    ctx = ToolCallContext(cwd=tmp_path, extra={"task_manager": tm})
    out = tools.call("get_task_output", {"task_ids": ["missing_xyz"]}, ctx)
    assert "not found" in out.lower()
    assert "No background tasks or subagents exist" in out


def test_get_task_output_empty_ids_error(tmp_path: Path) -> None:
    tools = _tools()
    tm = BackgroundTaskManager(work_dir=tmp_path / "tasks")
    ctx = ToolCallContext(cwd=tmp_path, extra={"task_manager": tm})
    with pytest.raises(ToolError, match="non-empty|task_ids"):
        tools.call("get_task_output", {"task_ids": []}, ctx)


def test_get_task_output_max_multi_ids(tmp_path: Path) -> None:
    tools = _tools()
    tm = BackgroundTaskManager(work_dir=tmp_path / "tasks")
    ctx = ToolCallContext(cwd=tmp_path, extra={"task_manager": tm})
    ids = [f"t{i}" for i in range(MAX_MULTI_WAIT_IDS + 1)]
    with pytest.raises(ToolError, match="maximum|exceeds"):
        tools.call("get_task_output", {"task_ids": ids}, ctx)


def test_get_task_output_multi_poll_format(tmp_path: Path) -> None:
    tools = _tools()
    tm = BackgroundTaskManager(work_dir=tmp_path / "tasks")
    ctx = ToolCallContext(cwd=tmp_path, extra={"task_manager": tm})
    out = tools.call(
        "get_task_output",
        {"task_ids": ["a", "b"], "timeout_ms": 0},
        ctx,
    )
    assert "=== Multi-wait (poll) ===" in out
    assert "not_found" in out
    assert "0/2 tasks completed (poll)" in out


def test_wait_tasks_wait_all(tmp_path: Path) -> None:
    tools = _tools()
    tm = BackgroundTaskManager(work_dir=tmp_path / "tasks")
    ctx = ToolCallContext(cwd=tmp_path, extra={"task_manager": tm})
    out = tools.call(
        "run_terminal_cmd",
        {
            "command": 'python -c "print(99)"',
            "description": "print",
            "is_background": True,
        },
        ctx,
    )
    tid = out.split("<task-id>")[1].split("</task-id>")[0]
    waited = tools.call(
        "wait_tasks",
        {"task_ids": [tid], "mode": "wait_all", "timeout_ms": 10_000},
        ctx,
    )
    # multi path always used for wait_tasks
    assert "Multi-wait" in waited or "=== Task" in waited or "99" in waited
    assert "wait_all" in waited or "completed" in waited.lower()


def test_wait_tasks_rejects_empty(tmp_path: Path) -> None:
    tools = _tools()
    tm = BackgroundTaskManager(work_dir=tmp_path / "tasks")
    ctx = ToolCallContext(cwd=tmp_path, extra={"task_manager": tm})
    with pytest.raises(ToolError, match="empty"):
        tools.call("wait_tasks", {"task_ids": [], "mode": "wait_all"}, ctx)


def test_kill_not_found_message(tmp_path: Path) -> None:
    tools = _tools()
    tm = BackgroundTaskManager(work_dir=tmp_path / "tasks")
    ctx = ToolCallContext(cwd=tmp_path, extra={"task_manager": tm})
    out = tools.call("kill_task", {"task_id": "nope"}, ctx)
    assert "not found" in out.lower()
    assert "No background tasks or subagents exist" in out


def test_monitor_started_message(tmp_path: Path) -> None:
    tools = _tools()
    tm = BackgroundTaskManager(work_dir=tmp_path / "tasks")
    ctx = ToolCallContext(cwd=tmp_path, extra={"task_manager": tm})
    out = tools.call(
        "monitor",
        {
            "command": 'python -c "import time; print(1); time.sleep(5)"',
            "description": "tick once",
            "timeout_ms": 600_000,
        },
        ctx,
    )
    assert out.startswith("Monitor started (task ")
    assert "timeout 600000ms" in out
    assert "do not poll or sleep" in out
    assert "<task-id>" not in out
    # extract task id and kill
    # "Monitor started (task task_xxx, timeout ..."
    mid = out.split("task ", 1)[1].split(",", 1)[0].strip()
    tools.call("kill_task", {"task_id": mid}, ctx)


def test_monitor_persistent_message(tmp_path: Path) -> None:
    tools = _tools()
    tm = BackgroundTaskManager(work_dir=tmp_path / "tasks")
    ctx = ToolCallContext(cwd=tmp_path, extra={"task_manager": tm})
    out = tools.call(
        "monitor",
        {
            "command": 'python -c "import time; time.sleep(30)"',
            "description": "persist",
            "persistent": True,
        },
        ctx,
    )
    assert "persistent" in out
    assert "kill_command_or_subagent" in out or "session end" in out
    mid = out.split("task ", 1)[1].split(",", 1)[0].strip()
    tools.call("kill_task", {"task_id": mid}, ctx)


def test_monitor_timeout_exceeds_max(tmp_path: Path) -> None:
    tools = _tools()
    tm = BackgroundTaskManager(work_dir=tmp_path / "tasks")
    ctx = ToolCallContext(cwd=tmp_path, extra={"task_manager": tm})
    with pytest.raises(ToolError, match="persistent must be true"):
        tools.call(
            "monitor",
            {
                "command": "echo x",
                "description": "bad",
                "timeout_ms": DEFAULT_TIMEOUT_MS + 1,
                "persistent": False,
            },
            ctx,
        )


def test_capped_wait_timeout_clamps() -> None:
    assert capped_wait_timeout_ms(None) == DEFAULT_WAIT_TIMEOUT_MS
    assert capped_wait_timeout_ms(5_000) == 5_000
    assert capped_wait_timeout_ms(36_000_000) == MAX_WAIT_BLOCK_MS
    assert capped_wait_timeout_ms(600_000) == MAX_WAIT_BLOCK_MS
    assert not task_output_waits(None)
    assert not task_output_waits(0)
    assert task_output_waits(1)


def test_description_builders_product_names() -> None:
    """Lock Grok xai-tool-types task.rs cli_default product strings."""
    import sys

    # get_task_output — task_output_matches_cli_default
    td = build_task_output_description()
    assert td == (
        "Get output and status from a background task, monitor, or subagent.\n\n"
        "Usage notes:\n"
        "- Pass task_ids with one or more ids from background=true commands or "
        "background=true subagents (a monitor's task_id is returned by monitor); "
        "for a single task use a one-element array. Multiple ids with a positive "
        "timeout_ms wait until all complete\n"
        "- Omit timeout_ms or pass 0 for a non-blocking status snapshot; set a "
        "positive timeout_ms to wait up to that many milliseconds, capped at ~10 min\n"
        "- Returns current output, status, and exit code if completed\n"
        "- If output is large, use read_file on the output_file path"
    )

    # kill_task — OS verb from is_windows (Grok kill_task_matches_cli_default_*)
    kd_win = build_kill_task_description(is_windows=True)
    assert kd_win == (
        "Terminate a running background task, monitor, or subagent.\n\n"
        "Usage notes:\n"
        "- Pass its task_id (a monitor's task_id is returned by monitor).\n"
        "- Terminates the Job Object of a bash task or monitor; "
        "sends Cancel+Shutdown to a subagent.\n"
        "- Returns success if the task was killed or had already exited."
    )
    kd_posix = build_kill_task_description(is_windows=False)
    assert kd_posix == (
        "Terminate a running background task, monitor, or subagent.\n\n"
        "Usage notes:\n"
        "- Pass its task_id (a monitor's task_id is returned by monitor).\n"
        "- Sends SIGTERM/SIGKILL to a bash task or monitor; "
        "sends Cancel+Shutdown to a subagent.\n"
        "- Returns success if the task was killed or had already exited."
    )
    # Default follows host OS
    kd = build_kill_task_description()
    if sys.platform == "win32":
        assert "Job Object" in kd
    else:
        assert "SIGTERM/SIGKILL" in kd

    # wait_tasks — wait_tasks_matches_cli_default (no source de-dupe)
    wd = build_wait_tasks_description()
    assert wd == (
        "Wait for multiple background tasks or subagents to complete.\n\n"
        "Prefer get_command_or_subagent_output with task_ids and a positive "
        "timeout_ms. This tool is kept for compatibility.\n\n"
        "Usage notes:\n"
        "- task_ids: list of task IDs from background=true or background=true\n"
        "- mode: 'wait_all' or 'wait_any'\n"
        "- timeout_ms: optional max wait, default 30s, capped at ~10 min"
    )


def test_kill_task_subagent_only_toolbox() -> None:
    """Grok kill_task_subagent_only_toolbox."""
    desc = build_kill_task_description(
        monitor_tool=None,
        subagent_present=True,
        bash_present=False,
        is_windows=False,
    )
    assert desc == (
        "Terminate a running background task or subagent.\n\n"
        "Usage notes:\n"
        "- Pass its task_id.\n"
        "- Sends Cancel+Shutdown to a subagent.\n"
        "- Returns success if the task was killed or had already exited."
    )


def test_monitor_line_processor_and_wrap() -> None:
    proc = LineProcessor()
    assert proc.push(b"hello world\n") == ["hello world"]
    assert batch_lines(["a", "b"]) == "a\nb"
    wrapped = wrap_monitor_event('watch "prod"\nlogs', "line", "t-1")
    assert wrapped.startswith(
        '<monitor-event description="watch \'prod\' logs" task_id="t-1">'
    )


def test_monitor_rate_limiter_bucket() -> None:
    bucket = TokenBucket(3, 2000)
    assert bucket.try_consume()
    assert bucket.try_consume()
    assert bucket.try_consume()
    assert not bucket.try_consume()
    rl = MonitorRateLimiter.new(2, 2000)
    assert rl.process_event("t").kind == RateLimitKind.Allowed
    assert rl.process_event("t").kind == RateLimitKind.Allowed
    assert rl.process_event("t").kind == RateLimitKind.Suppressed


def test_monitor_resolved_timeout() -> None:
    assert resolved_timeout_ms(timeout_ms=None, persistent=False) == DEFAULT_TIMEOUT_MS
    assert resolved_timeout_ms(timeout_ms=None, persistent=True) == 0
    assert resolved_timeout_ms(timeout_ms=600_000, persistent=False) == 600_000
    with pytest.raises(MonitorError):
        validate_monitor_input(timeout_ms=DEFAULT_TIMEOUT_MS + 1, persistent=False)
    validate_monitor_input(timeout_ms=DEFAULT_TIMEOUT_MS + 1, persistent=True)


def test_trailing_ampersand_rejected_when_disallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grok default allow_background_operator=true; reject only when allow=false."""
    import codedoggy.tools.builtins.run_terminal_cmd as rtc
    import codedoggy.tools.defaults as defaults

    monkeypatch.setattr(defaults, "BASH_ALLOW_BACKGROUND_OPERATOR", False)
    monkeypatch.setattr(rtc, "BASH_ALLOW_BACKGROUND_OPERATOR", False)
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path)
    with pytest.raises(ToolError, match="background|&"):
        tools.call(
            "run_terminal_cmd",
            {
                "command": 'python -c "print(1)" &',
                "description": "reject bare ampersand",
            },
            ctx,
        )


def test_todo_write_merge(tmp_path: Path) -> None:
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    out = tools.call(
        "todo_write",
        {
            "merge": False,
            "todos": [
                {"id": "1", "content": "first", "status": "pending"},
                {"id": "2", "content": "second", "status": "in_progress"},
            ],
        },
        ctx,
    )
    assert "first" in out and "second" in out
    out2 = tools.call(
        "todo_write",
        {
            "merge": True,
            "todos": [{"id": "2", "status": "completed"}],
        },
        ctx,
    )
    assert "completed" in out2.lower() or "[x]" in out2


def test_enter_exit_plan_mode(tmp_path: Path) -> None:
    tools = _tools()
    mode = SessionModeState()
    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={"session_mode_state": mode},
    )
    out = tools.call("enter_plan_mode", {}, ctx)
    assert "entered plan mode" in out.lower()
    assert mode.is_plan()
    plan = tmp_path / ".grok" / "plan.md"
    assert plan.exists()
    plan.write_text("# Plan\n1. ship it\n", encoding="utf-8")
    out2 = tools.call("exit_plan_mode", {}, ctx)
    assert "plan has been approved" in out2.lower()
    assert "ship it" in out2
    assert not mode.is_plan()


def test_update_goal_progress(tmp_path: Path) -> None:
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    out = tools.call("update_goal", {"message": "halfway"}, ctx)
    assert out == "halfway."
    out2 = tools.call(
        "update_goal",
        {"completed": True, "message": "done"},
        ctx,
    )
    # Grok CompletedWithoutClassifier
    assert out2 == "Goal marked complete."


def test_ask_user_with_callback(tmp_path: Path) -> None:
    tools = _tools()

    def answer(questions):
        return {"answers": ["opt-a"]}

    ctx = ToolCallContext(cwd=tmp_path, extra={"ask_user_fn": answer})
    out = tools.call(
        "ask_user_question",
        {
            "questions": [
                {
                    "question": "Pick one?",
                    "options": [
                        {"label": "A (Recommended)", "description": "first"},
                        {"label": "B", "description": "second"},
                    ],
                }
            ]
        },
        ctx,
    )
    assert "opt-a" in out


def test_scheduler_create_list_delete(tmp_path: Path) -> None:
    import re

    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    out = tools.call(
        "scheduler_create",
        {"interval": "5m", "prompt": "check status", "recurring": True},
        ctx,
    )
    # Grok ToolOutput::SchedulerCreate model text
    assert "Scheduled task created (ID: " in out
    assert "every 5 minutes" in out
    m = re.search(r"ID: ([0-9a-f]{12}),", out)
    assert m, out
    tid = m.group(1)
    listed = tools.call("scheduler_list", {}, ctx)
    assert "check status" in listed
    deleted = tools.call("scheduler_delete", {"id": tid}, ctx)
    assert deleted == f"Scheduled task {tid} cancelled."


def test_ssrf_blocks_private(tmp_path: Path) -> None:
    from codedoggy.tools.util.ssrf import check_ssrf_url, is_blocked_ip

    assert is_blocked_ip("10.0.0.1")
    assert is_blocked_ip("192.168.1.1")
    assert is_blocked_ip("169.254.169.254")
    assert not is_blocked_ip("127.0.0.1")
    with pytest.raises(ValueError, match="SSRF"):
        check_ssrf_url("http://192.168.0.5/secret")


def test_grep_files_with_matches(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("hello world\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("nothing\n", encoding="utf-8")
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path)
    out = tools.call(
        "grep",
        {"pattern": "hello", "path": str(tmp_path), "output_mode": "files_with_matches"},
        ctx,
    )
    assert "a.py" in out
    assert "Found " in out
