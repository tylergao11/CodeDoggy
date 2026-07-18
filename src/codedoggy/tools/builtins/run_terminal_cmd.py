"""run_terminal_cmd — foreground shell command (no background subsystem)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Any

from codedoggy.tools.defaults import (
    BASH_ALLOW_BACKGROUND_OPERATOR,
    BASH_DEFAULT_MAX_TIMEOUT_MS,
    BASH_DEFAULT_TIMEOUT_MS,
    DEFAULT_TOOL_OUTPUT_CHARS,
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
from codedoggy.tools.util.shell import (
    chain_separator,
    has_unix_utilities,
    shell_command_argv,
)


def _description() -> str:
    shell_label = "shell command" if sys.platform == "win32" else "bash command"
    if sys.platform == "win32":
        kill_note = (
            "  - On timeout the process tree is terminated (taskkill /T on Windows)."
        )
    else:
        kill_note = (
            "  - On timeout the child process group is killed (SIGTERM, then SIGKILL "
            "after a short grace period). Descendants that did not leave the group "
            "are also killed."
        )
    lines = [
        f"Run a {shell_label} and return its output.",
        "",
        "Usage notes:",
        f"  - You can specify an optional timeout in milliseconds (up to {BASH_DEFAULT_MAX_TIMEOUT_MS}ms). "
        f"If not specified, commands will timeout after {BASH_DEFAULT_TIMEOUT_MS}ms.",
        kill_note,
        f"  - If the output exceeds {DEFAULT_TOOL_OUTPUT_CHARS} characters, output will be truncated "
        "before being returned to you.",
        "  - Working directory is the session cwd unless the command changes it.",
        "  - Output is returned as: first line `exit: <code>`, then the combined stdout/stderr.",
        "  - Do not end commands with a bare background `&`; run them in the foreground.",
    ]
    if sys.platform == "win32" and chain_separator() == ";":
        lines.append("  - '&&' is not supported in this shell; chain sequential commands with ';'.")
    if not has_unix_utilities():
        lines.append(
            "  - The Unix utilities `grep`, `head`, `tail`, `sed`, `awk`, and `find` are NOT "
            "available in this shell. Use the dedicated tools instead."
        )
    return "\n".join(lines)


def resolve_fg_timeout_ms(timeout: Any) -> int:
    """Foreground timeout: None/0 → default 120s; positive clamped to max 300s."""
    if timeout is None:
        return BASH_DEFAULT_TIMEOUT_MS
    try:
        timeout_ms = int(timeout)
    except (TypeError, ValueError) as e:
        raise ToolError.invalid_arguments(f"invalid timeout: {timeout}") from e
    if timeout_ms < 0:
        raise ToolError.invalid_arguments("timeout must be non-negative")
    if timeout_ms == 0:
        return BASH_DEFAULT_TIMEOUT_MS
    return min(timeout_ms, BASH_DEFAULT_MAX_TIMEOUT_MS)


def contains_trailing_background_op(command: str) -> bool:
    s = command.rstrip()
    if not s.endswith("&") or s.endswith("&&"):
        return False
    before = s[:-1].rstrip()
    if before.endswith("&"):
        return False
    return True


def format_shell_observation(exit_code: int | str, body: str) -> str:
    """Model-facing card: exit header then output."""
    header = f"exit: {exit_code}"
    if not body:
        return header
    return f"{header}\n{body}"


def _popen_kwargs() -> dict[str, Any]:
    """Spawn options so the child is killable as a group/tree."""
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        # New process group so we can signal the tree via taskkill /T.
        create_new_process_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        kwargs["creationflags"] = create_new_process_group
    else:
        # Own session → process group leader; killpg covers descendants.
        kwargs["start_new_session"] = True
    return kwargs


def kill_process_tree(proc: subprocess.Popen[bytes]) -> None:
    """Best-effort kill of the spawned process and its descendants."""
    if proc.poll() is not None:
        return
    pid = proc.pid
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass
        return

    # POSIX: SIGTERM the whole group, then escalate.
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except OSError:
            pass
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def run_command_with_timeout(
    argv: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout_ms: int,
) -> tuple[int | str, bytes, bytes]:
    """
    Run argv; return (exit_code_or_timeout_label, stdout, stderr).
    On timeout: kill process tree and return ("killed (timeout)", partial_out, partial_err).
    """
    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_popen_kwargs(),
        )
    except OSError as e:
        raise ToolError(f"Failed to spawn command: {e}", code="spawn_failed") from e

    try:
        stdout, stderr = proc.communicate(timeout=timeout_ms / 1000.0)
        return proc.returncode if proc.returncode is not None else -1, stdout, stderr
    except subprocess.TimeoutExpired as e:
        kill_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            stdout = e.stdout or b""
            stderr = e.stderr or b""
            try:
                proc.kill()
            except OSError:
                pass
        return "killed (timeout)", stdout or b"", stderr or b""


class RunTerminalCmdTool(Tool):
    def id(self) -> ToolId:
        return ToolId("run_terminal_cmd")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Execute

    def description(self, ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="run_terminal_cmd", description=_description())

    def parameters_schema(self) -> dict[str, Any]:
        cmd_desc = "The command to run." if sys.platform == "win32" else "The bash command to run."
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": cmd_desc,
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        f"Optional timeout in milliseconds (max {BASH_DEFAULT_MAX_TIMEOUT_MS}). "
                        f"Default: {BASH_DEFAULT_TIMEOUT_MS}."
                    ),
                    "maximum": BASH_DEFAULT_MAX_TIMEOUT_MS,
                },
                "description": {
                    "type": "string",
                    "description": (
                        "One sentence explanation as to why this command needs to be run "
                        "and how it contributes to the goal."
                    ),
                },
            },
            "required": ["command", "description"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolError.invalid_arguments("command is required")

        desc = args.get("description")
        if not isinstance(desc, str) or not desc.strip():
            raise ToolError.invalid_arguments("description is required")

        if not BASH_ALLOW_BACKGROUND_OPERATOR and contains_trailing_background_op(command):
            raise ToolError(
                "Command must not end with '&'. Remove the '&' and run in the foreground.",
                code="invalid_arguments",
            )

        policy = (ctx.extra or {}).get("policy")
        if policy is not None:
            check = getattr(policy, "check_shell", None)
            if callable(check):
                decision = check(command)
                if decision is not None and not getattr(decision, "allowed", True):
                    raise ToolError(
                        getattr(decision, "reason", None) or "shell denied by policy",
                        code=getattr(decision, "code", None) or "policy_denied",
                    )
            # Close write hole: shell redirects / python open must pass check_write
            try:
                from codedoggy.tools.util.write_detect import detect_shell_write_paths

                check_w = getattr(policy, "check_write", None)
                if callable(check_w):
                    for wp in detect_shell_write_paths(command):
                        wd = check_w(wp)
                        if wd is not None and not getattr(wd, "allowed", True):
                            raise ToolError(
                                getattr(wd, "reason", None)
                                or f"shell write denied for {wp}",
                                code=getattr(wd, "code", None) or "policy_denied",
                            )
            except ToolError:
                raise
            except Exception:
                pass

        timeout_ms = resolve_fg_timeout_ms(args.get("timeout"))
        inv = shell_command_argv(command)
        env = {**os.environ, **inv.env}

        code, out_b, err_b = run_command_with_timeout(
            [inv.program, *inv.args],
            cwd=str(ctx.cwd),
            env=env,
            timeout_ms=timeout_ms,
        )

        out = out_b.decode("utf-8", errors="replace")
        err = err_b.decode("utf-8", errors="replace")
        combined = out
        if err:
            combined = f"{out}\n{err}" if out else err
        combined = _truncate_chars(combined, DEFAULT_TOOL_OUTPUT_CHARS)

        if code == "killed (timeout)":
            body = combined if combined else "(no output before timeout)"
            return format_shell_observation("killed (timeout)", body)

        # Resident audit: shell redirects / PS write cmdlets → mutation event
        # so quality review is not limited to search_replace.
        if isinstance(code, int) and code == 0:
            try:
                from codedoggy.tools.util.write_detect import record_shell_mutations

                record_shell_mutations(
                    ctx,
                    command,
                    exit_ok=True,
                    tool_name="run_terminal_cmd",
                )
            except Exception:  # noqa: BLE001 — never break shell observation
                pass

        if not combined:
            combined = "(no output)"
        return format_shell_observation(code, combined)


def _truncate_chars(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n… (output truncated)"
