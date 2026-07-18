"""run_terminal_cmd — Grok BashTool wire + portable runtime.

Ported from:
  grok-build/.../implementations/grok_build/bash/mod.rs
  computer/local/shell_state.rs (shell_env_overrides subset)

Description matches Grok product Job Object wording on Windows.
Kill path: TerminateJobObject + child kill (Grok terminal.rs; no taskkill).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Any

from codedoggy.tools.defaults import (
    BASH_ALLOW_BACKGROUND_OPERATOR,
    BASH_AUTO_BACKGROUND_ON_TIMEOUT,
    BASH_BACKGROUND_MAX_RUNTIME_S,
    BASH_DEFAULT_MAX_TIMEOUT_MS,
    BASH_DEFAULT_TIMEOUT_MS,
    BASH_ENABLED_BACKGROUND,
    BASH_PERSISTENT_SHELL_STATE,
    DEFAULT_TOOL_OUTPUT_CHARS,
)
from codedoggy.tools.grok_build.bash_bg_op import (
    ampersand_semantics_for_host,
    rejection_message,
    should_reject_background_op,
)
from codedoggy.tools.grok_build.bash_params import (
    effective_auto_bg_wait_ms,
    resolve_bg_max_runtime_s as _resolve_bg_s,
    resolve_fg_timeout_ms as _resolve_fg_ms,
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
from codedoggy.tools.task_manager import (
    format_background_started,
    ensure_task_manager,
)
from codedoggy.tools.util.shell import (
    chain_separator,
    has_unix_utilities,
    shell_command_argv,
)
from codedoggy.tools.util.shell_state import (
    ensure_shell_state,
    make_state_file,
    read_pwd_probe,
    shell_env_overrides,
    wrap_command_with_pwd_probe,
)


def _description() -> str:
    """Grok default_description_template_enabled/disabled (product kill name)."""
    shell_label = "shell command" if sys.platform == "win32" else "bash command"
    # Wire name; finalize renames is_background → background on product surface.
    bg_param = "is_background"
    kill_name = "kill_command_or_subagent"
    if BASH_AUTO_BACKGROUND_ON_TIMEOUT and BASH_ENABLED_BACKGROUND:
        timeout_line = (
            f"  - You can specify an optional timeout in milliseconds "
            f"(up to {BASH_DEFAULT_MAX_TIMEOUT_MS}ms). "
            "If not specified, commands exceeding the default timeout will be automatically "
            "backgrounded instead of killed. You will receive a task_id to check output later."
        )
    else:
        timeout_line = (
            f"  - You can specify an optional timeout in milliseconds "
            f"(up to {BASH_DEFAULT_MAX_TIMEOUT_MS}ms). "
            f"If not specified, commands will timeout after {BASH_DEFAULT_TIMEOUT_MS}ms."
        )
    # Grok template: Windows Job Object wording (matches runtime TerminateJobObject).
    if sys.platform == "win32":
        timeout_enforcement = (
            "  - Timeout enforcement: when the timeout fires, the wrapper terminates the child's "
            "Job Object, killing every descendant process immediately "
            "(no graceful-termination grace period). "
            f"`timeout: 0` in `{bg_param}: true` mode disables the wrapper timeout entirely; "
            f"the child's lifetime is owned by the model via {kill_name}."
        )
    else:
        timeout_enforcement = (
            "  - Timeout enforcement: when the timeout fires, the wrapper kills the child process "
            "group (SIGTERM, escalated to SIGKILL after a ~1s grace period). Descendants that did "
            f"not detach via `setsid` / `nohup` will also be killed. `timeout: 0` in `{bg_param}: true` "
            f"mode disables the wrapper timeout entirely; the child's lifetime is owned by the model "
            f"via {kill_name}."
        )
    lines = [
        f"Run a {shell_label} and return its output.",
        "",
        "Usage notes:",
        timeout_line,
        timeout_enforcement,
        f"  - If the output exceeds {DEFAULT_TOOL_OUTPUT_CHARS} characters, output will be truncated "
        "before being returned to you.",
    ]
    if BASH_ENABLED_BACKGROUND:
        unix_amp = (
            " You do not need to use '&' at the end of the command when using this parameter."
            if has_unix_utilities()
            else ""
        )
        lines.append(
            f"  - You can use the {bg_param} parameter to run the command in the background "
            f"(e.g., dev servers, long builds): it returns a task_id immediately and keeps "
            f"running in the background. You are notified on completion, so do not poll or "
            f"sleep-wait for it.{unix_amp}"
        )
    if sys.platform == "win32" and chain_separator() == ";":
        lines.append(
            "  - '&&' is not supported in this shell; chain sequential commands with ';'."
        )
    if not has_unix_utilities():
        lines.append(
            "  - The Unix utilities `grep`, `head`, `tail`, `sed`, `awk`, and `find` are NOT "
            "available in this shell. Use the dedicated tools instead."
        )
    return "\n".join(lines)


def resolve_fg_timeout_ms(timeout: Any) -> int:
    """Foreground timeout: None/0 → default 120s; positive clamped to max 300s."""
    if timeout is None:
        return _resolve_fg_ms(None)
    try:
        timeout_ms = int(timeout)
    except (TypeError, ValueError) as e:
        raise ToolError.invalid_arguments(f"invalid timeout: {timeout}") from e
    try:
        return _resolve_fg_ms(timeout_ms)
    except ValueError as e:
        raise ToolError.invalid_arguments(str(e)) from e


def resolve_bg_max_runtime_s(timeout: Any) -> float:
    """Background max runtime. timeout 0/None → session max (model owns via kill)."""
    if timeout is None:
        return _resolve_bg_s(None, max_s=BASH_BACKGROUND_MAX_RUNTIME_S)
    try:
        timeout_ms = int(timeout)
    except (TypeError, ValueError) as e:
        raise ToolError.invalid_arguments(f"invalid timeout: {timeout}") from e
    try:
        return _resolve_bg_s(timeout_ms, max_s=BASH_BACKGROUND_MAX_RUNTIME_S)
    except ValueError as e:
        raise ToolError.invalid_arguments(str(e)) from e


def contains_trailing_background_op(command: str) -> bool:
    """Compatibility alias — prefer bash_bg_op.has_trailing_background_operator."""
    from codedoggy.tools.grok_build.bash_bg_op import has_trailing_background_operator

    return has_trailing_background_operator(command)


# Source-ported formatters (bash/mod.rs) — re-export for callers/tests
from codedoggy.tools.grok_build.bash_format import (  # noqa: E402
    format_default_prompt as format_shell_observation_v2,
    strip_ansi,
)


def format_shell_observation(
    exit_code: int | str,
    body: str,
    *,
    truncated: bool = False,
    shown_bytes: int | None = None,
    total_bytes: int | None = None,
    output_file: str | None = None,
    signal: str | None = None,
) -> str:
    """Delegate to source-ported format_default_prompt."""
    if isinstance(exit_code, str) and exit_code.startswith("killed"):
        # map "killed (timeout)" → signal timeout
        sig = signal
        if exit_code == "killed (timeout)":
            sig = "timeout"
        elif exit_code.startswith("killed ("):
            sig = exit_code[len("killed (") : -1]
        return format_shell_observation_v2(
            exit_code=None,
            output=body,
            signal=sig,
            truncated=truncated,
            total_bytes=total_bytes or shown_bytes or 0,
            output_file=output_file,
        )
    return format_shell_observation_v2(
        exit_code=int(exit_code) if not isinstance(exit_code, int) else exit_code,
        output=body,
        signal=signal,
        truncated=truncated,
        total_bytes=total_bytes or shown_bytes or 0,
        output_file=output_file,
    )


def format_moved_to_background(
    *,
    partial: str,
    output_file: str,
    cwd: str,
    shown_bytes: int,
    total_bytes: int,
    task_xml: str = "",
) -> str:
    body = format_shell_observation_v2(
        output=partial,
        signal="auto_backgrounded",
        total_bytes=total_bytes or shown_bytes,
        output_file=output_file,
        current_dir=cwd,
    )
    if task_xml:
        return f"{body}\n\n{task_xml}"
    return body


def _popen_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        create_new_process_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        kwargs["creationflags"] = create_new_process_group
    else:
        kwargs["start_new_session"] = True
    return kwargs


_SCRUB_ENV_KEYS = frozenset(
    {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CODEDOGGY_API_KEY",
        "CODEDOGGY_AUDIT_API_KEY",
        "CODEDOGGY_AUX_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "NPM_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    }
)


def scrub_shell_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base if base is not None else os.environ)
    for k in list(env.keys()):
        ku = k.upper()
        if ku in _SCRUB_ENV_KEYS or ku.endswith("_API_KEY") or ku.endswith("_SECRET"):
            env.pop(k, None)
    # Grok shell_env_overrides last so they win over ambient env.
    env.update(shell_env_overrides())
    return env


def kill_process_tree(proc: subprocess.Popen[bytes]) -> None:
    """Delegate to shared Job Object / process-group killer (single implementation)."""
    from codedoggy.tools.util.job_object import kill_process_tree as _kill

    _kill(proc)


def run_command_with_timeout(
    argv: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout_ms: int,
    auto_background: bool = False,
) -> tuple[str, int | str | None, bytes, bytes, subprocess.Popen[bytes] | None]:
    """
    Run argv.

    Returns (status, exit_code_or_label, stdout, stderr, proc_or_none)
      status: "complete" | "timeout_killed" | "auto_backgrounded"
    """
    try:
        # Merge stdout for simpler adopt when auto-bg
        merge = auto_background
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge else subprocess.PIPE,
            **_popen_kwargs(),
        )
    except OSError as e:
        raise ToolError(f"Failed to spawn command: {e}", code="spawn_failed") from e

    # Windows: assign to Job Object ASAP so timeout kill reaps the tree.
    if sys.platform == "win32":
        try:
            from codedoggy.tools.util.job_object import create_and_assign_job

            create_and_assign_job(proc.pid)
        except Exception:  # noqa: BLE001
            pass

    try:
        stdout, stderr = proc.communicate(timeout=timeout_ms / 1000.0)
        if merge:
            stderr = b""
        code = proc.returncode if proc.returncode is not None else -1
        if sys.platform == "win32":
            try:
                from codedoggy.tools.util.job_object import release_job_for_pid

                release_job_for_pid(proc.pid)
            except Exception:  # noqa: BLE001
                pass
        return "complete", code, stdout or b"", stderr or b"", None
    except subprocess.TimeoutExpired as e:
        partial_out = e.stdout or b""
        partial_err = b"" if merge else (e.stderr or b"")
        if auto_background and BASH_ENABLED_BACKGROUND:
            # Leave process running; caller adopts into task manager.
            return "auto_backgrounded", None, partial_out, partial_err, proc
        kill_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=2.0)
            if merge:
                stderr = b""
        except subprocess.TimeoutExpired:
            stdout = partial_out
            stderr = partial_err
            try:
                proc.kill()
            except OSError:
                pass
        return "timeout_killed", "killed (timeout)", stdout or b"", stderr or b"", None


def _policy_check_shell(ctx: ToolCallContext, command: str) -> None:
    policy = (ctx.extra or {}).get("policy")
    if policy is None:
        return
    check = getattr(policy, "check_shell", None)
    if callable(check):
        decision = check(command)
        if decision is not None and not getattr(decision, "allowed", True):
            raise ToolError(
                getattr(decision, "reason", None) or "shell denied by policy",
                code=getattr(decision, "code", None) or "policy_denied",
            )
    try:
        from codedoggy.tools.util.write_detect import detect_shell_write_paths

        check_w = getattr(policy, "check_write", None)
        if callable(check_w):
            for wp in detect_shell_write_paths(command):
                wd = check_w(wp)
                if wd is not None and not getattr(wd, "allowed", True):
                    raise ToolError(
                        getattr(wd, "reason", None) or f"shell write denied for {wp}",
                        code=getattr(wd, "code", None) or "policy_denied",
                    )
    except ToolError:
        raise
    except Exception:
        pass


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
        props: dict[str, Any] = {
            "command": {
                "type": "string",
                "description": cmd_desc,
            },
            "timeout": {
                "type": "integer",
                "description": (
                    f"Optional timeout in milliseconds (max {BASH_DEFAULT_MAX_TIMEOUT_MS}). "
                    f"Default: {BASH_DEFAULT_TIMEOUT_MS}. "
                    "In is_background=true mode, timeout: 0 disables the wrapper timeout."
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
        }
        if BASH_ENABLED_BACKGROUND:
            props["is_background"] = {
                "type": "boolean",
                "description": (
                    "Set to true for long-running commands that should run in the background "
                    "(e.g., dev servers, long builds). Returns a task_id immediately while the "
                    "command keeps running; use get_task_output / kill_task to manage it."
                ),
            }
        return {
            "type": "object",
            "properties": props,
            "required": ["command", "description"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolError.invalid_arguments("command is required")

        desc = args.get("description")
        if not isinstance(desc, str) or not desc.strip():
            raise ToolError.invalid_arguments("description is required")

        # Product surface: `background`; wire: `is_background`
        bg_raw = args.get("is_background")
        if bg_raw is None:
            bg_raw = args.get("background")
        is_background = bool(bg_raw) if BASH_ENABLED_BACKGROUND else False

        # Grok should_reject_background_op (product param name = background)
        violation = should_reject_background_op(
            is_background=is_background,
            allow_background_operator=BASH_ALLOW_BACKGROUND_OPERATOR,
            background_enabled=BASH_ENABLED_BACKGROUND,
            semantics=ampersand_semantics_for_host(),
            command=command,
            is_legacy=False,
        )
        if violation is not None:
            raise ToolError(
                rejection_message(
                    violation,
                    background_enabled=BASH_ENABLED_BACKGROUND,
                    param_name="background",
                    is_legacy=False,
                ),
                code="invalid_arguments",
            )

        _policy_check_shell(ctx, command)

        # Persistent shell state (tool-layer)
        shell_st = None
        run_command = command
        state_file = None
        if BASH_PERSISTENT_SHELL_STATE:
            shell_st = ensure_shell_state(ctx.extra, ctx.cwd)
            cwd = str(shell_st.cwd)
            state_file = make_state_file()
            run_command = wrap_command_with_pwd_probe(command, state_file)
        else:
            cwd = str(ctx.cwd)

        inv = shell_command_argv(run_command)
        env = scrub_shell_env({**os.environ, **inv.env})
        if shell_st is not None:
            env = shell_st.apply_env(env)
        argv = [inv.program, *inv.args]

        if is_background:
            # Background uses original command (no pwd probe wrapper required;
            # cwd already set from shell state)
            inv_bg = shell_command_argv(command)
            env_bg = scrub_shell_env({**os.environ, **inv_bg.env})
            if shell_st is not None:
                env_bg = shell_st.apply_env(env_bg)
            result = self._run_background(
                ctx,
                argv=[inv_bg.program, *inv_bg.args],
                command=command,
                cwd=cwd,
                env=env_bg,
                description=desc.strip(),
                timeout=args.get("timeout"),
            )
            if state_file is not None:
                try:
                    state_file.unlink(missing_ok=True)  # type: ignore[arg-type]
                except TypeError:
                    try:
                        if state_file.exists():
                            state_file.unlink()
                    except OSError:
                        pass
            return result

        timeout_ms = resolve_fg_timeout_ms(args.get("timeout"))
        auto_bg = (
            BASH_AUTO_BACKGROUND_ON_TIMEOUT
            and BASH_ENABLED_BACKGROUND
            and not is_background
        )
        # Grok: FG wait when auto-bg is on is min(resolved_timeout, FG budget).
        timeout_ms = effective_auto_bg_wait_ms(
            timeout_ms,
            auto_background_on_timeout=auto_bg,
        )

        status, code, out_b, err_b, live_proc = run_command_with_timeout(
            argv,
            cwd=cwd,
            env=env,
            timeout_ms=timeout_ms,
            auto_background=auto_bg,
        )

        # Update persistent cwd from probe (foreground complete path)
        if shell_st is not None and state_file is not None and status == "complete":
            new_cwd = read_pwd_probe(state_file)
            if new_cwd is not None:
                shell_st.cwd = new_cwd
            try:
                state_file.unlink(missing_ok=True)  # type: ignore[arg-type]
            except TypeError:
                try:
                    if state_file.exists():
                        state_file.unlink()
                except OSError:
                    pass
        elif state_file is not None:
            try:
                state_file.unlink(missing_ok=True)  # type: ignore[arg-type]
            except TypeError:
                try:
                    if state_file.exists():
                        state_file.unlink()
                except OSError:
                    pass

        if status == "auto_backgrounded" and live_proc is not None:
            tm = ensure_task_manager(ctx.extra)
            partial_b = out_b + (err_b or b"")
            handle = tm.adopt(
                live_proc,
                command=command,
                cwd=cwd,
                description=desc.strip(),
                owner_session_id=ctx.session_id,
                kind="bash",
                partial_output=partial_b,
                max_runtime_s=BASH_BACKGROUND_MAX_RUNTIME_S,
                signal="auto_backgrounded",
            )
            preview = strip_ansi(partial_b.decode("utf-8", errors="replace"))
            preview = _truncate_chars(preview, min(2000, DEFAULT_TOOL_OUTPUT_CHARS))
            xml = format_background_started(
                handle,
                command=command,
                description=desc.strip(),
                retrieval_hint=(
                    "Use get_command_or_subagent_output with this task_id to retrieve the output. "
                    "Use kill_command_or_subagent to terminate if needed."
                ),
            )
            return format_moved_to_background(
                partial=preview or "(no output yet)",
                output_file=handle.output_file,
                cwd=cwd,
                shown_bytes=len(partial_b),
                total_bytes=len(partial_b),
                task_xml=xml,
            )

        out = strip_ansi(out_b.decode("utf-8", errors="replace"))
        err = strip_ansi(err_b.decode("utf-8", errors="replace"))
        combined = out
        if err:
            combined = f"{out}\n{err}" if out else err
        raw_len = len(combined.encode("utf-8", errors="replace"))
        was_trunc = len(combined) > DEFAULT_TOOL_OUTPUT_CHARS
        combined = _truncate_chars(combined, DEFAULT_TOOL_OUTPUT_CHARS)
        shown_len = len(combined.encode("utf-8", errors="replace"))

        try:
            from codedoggy.tools.util.write_detect import record_shell_mutations

            record_shell_mutations(
                ctx,
                command,
                exit_ok=(isinstance(code, int) and code == 0),
                tool_name="run_terminal_cmd",
                call_id="",
            )
        except Exception:
            pass

        if status == "timeout_killed" or code == "killed (timeout)":
            body = combined if combined else "(no output before timeout)"
            return format_shell_observation(
                "killed (timeout)",
                body,
                truncated=was_trunc,
                shown_bytes=shown_len,
                total_bytes=raw_len,
            )

        if not combined:
            combined = "(no output)"
        return format_shell_observation(
            code if code is not None else -1,
            combined,
            truncated=was_trunc,
            shown_bytes=shown_len,
            total_bytes=raw_len,
        )

    def _run_background(
        self,
        ctx: ToolCallContext,
        *,
        argv: list[str],
        command: str,
        cwd: str,
        env: dict[str, str],
        description: str,
        timeout: Any,
    ) -> str:
        tm = ensure_task_manager(ctx.extra)
        max_runtime = resolve_bg_max_runtime_s(timeout)
        handle = tm.spawn(
            argv,
            command=command,
            cwd=cwd,
            env=env,
            description=description,
            owner_session_id=ctx.session_id,
            kind="bash",
            max_runtime_s=max_runtime,
            popen_kwargs=_popen_kwargs(),
        )
        # Explicit is_background → Grok BackgroundTaskStarted XML envelope
        return format_background_started(
            handle,
            command=command,
            description=description,
            retrieval_hint=(
                "Use get_command_or_subagent_output with this task_id to check status/output. "
                "Use kill_command_or_subagent to terminate if needed. You are notified on "
                "completion, so do not poll or sleep-wait for it."
            ),
        )


def _truncate_chars(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n… (output truncated)"
