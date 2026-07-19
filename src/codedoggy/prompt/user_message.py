"""Grok MAIN user-message framing for the model-facing sample view.

Source ports:
  - xai-grok-shell/src/session/user_message.rs
  - xai-grok-workspace/src/file_system/git_status.rs

The caller owns transcript persistence.  These helpers only build the
ephemeral prefix/query representation sent to the sampler.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

_GIT_STATUS_TIMEOUT_S = 2.0
_MAX_GIT_STATUS_CHARS = 1_000


def user_query(user_message: str) -> str:
    """Grok ``user_message::user_query`` envelope."""
    return f"<user_query>\n{user_message}\n</user_query>"


def construct_user_message_minimal(working_directory: Path | str) -> str:
    """Grok minimal ``<user_info>`` block using local host information."""
    cwd = Path(working_directory).resolve()
    if sys.platform == "win32":
        os_name = "windows"
    elif sys.platform == "darwin":
        os_name = "macos"
    elif sys.platform.startswith("linux"):
        os_name = "linux"
    else:
        os_name = sys.platform
    shell = _resolve_shell_display()
    today = date.today().isoformat()
    return (
        "<user_info>\n"
        f"OS Version: {os_name}\n"
        f"Shell: {shell}\n"
        f"Workspace Path: {cwd}\n"
        f"Today's date: {today}\n"
        "Note: Prefer using relative paths over absolute paths as tool call "
        "args when possible.\n"
        "</user_info>"
    )


def format_git_status_block(status: str) -> str:
    """Grok ``format_vcs_status_block`` for a Git workspace."""
    return (
        "\n\n<git_status>\n"
        "This is the git status at the start of the conversation. Note that this "
        "status is a snapshot in time, and will not update during the conversation.\n"
        f"{status}\n"
        "</git_status>\n"
    )


def construct_user_message(working_directory: Path | str) -> str:
    """Build Grok MAIN prefix: ``<user_info>`` plus optional Git snapshot."""
    cwd = Path(working_directory).resolve()
    prefix = construct_user_message_minimal(cwd)
    status = _compute_git_status(cwd)
    if status:
        prefix += format_git_status_block(status)
    return prefix


def _resolve_shell_display() -> str:
    if sys.platform != "win32":
        return os.environ.get("SHELL") or "/bin/sh"
    try:
        from codedoggy.tools.util.shell import detect_windows_shell

        return detect_windows_shell().value
    except Exception:  # noqa: BLE001 - prompt construction must retain user_info
        return "powershell"


def _compute_git_status(cwd: Path) -> str | None:
    """Port Grok's compact, read-only Git snapshot with one 2s budget."""
    deadline = time.monotonic() + _GIT_STATUS_TIMEOUT_S
    branch = _run_git(cwd, ["rev-parse", "--abbrev-ref", "HEAD"], deadline)
    if not branch:
        return None

    lines: list[str] = []
    if branch == "HEAD":
        commit = _run_git(cwd, ["rev-parse", "--short", "HEAD"], deadline)
        if commit:
            lines.append(f"HEAD detached at {commit}")
    else:
        lines.append(f"On branch {branch}")

    upstream = _run_git(
        cwd,
        ["rev-parse", "--abbrev-ref", "@{upstream}"],
        deadline,
    )
    if upstream:
        counts = _run_git(
            cwd,
            ["rev-list", "--count", "--left-right", "@{upstream}...HEAD"],
            deadline,
        )
        if counts:
            parts = counts.split()
            if len(parts) >= 2:
                try:
                    behind, ahead = int(parts[0]), int(parts[1])
                except ValueError:
                    behind = ahead = 0
                if ahead == 0 and behind == 0:
                    lines.append(f"Your branch is up to date with '{upstream}'.")
                elif behind == 0:
                    suffix = "" if ahead == 1 else "s"
                    lines.append(
                        f"Your branch is ahead of '{upstream}' by {ahead} commit{suffix}."
                    )
                elif ahead == 0:
                    suffix = "" if behind == 1 else "s"
                    lines.append(
                        f"Your branch is behind '{upstream}' by {behind} commit{suffix}."
                    )
                else:
                    lines.append(
                        f"Your branch and '{upstream}' have diverged "
                        f"({ahead} ahead, {behind} behind)."
                    )

    staged_raw = _run_git(
        cwd,
        ["diff", "--cached", "--name-status", "HEAD"],
        deadline,
    )
    staged: list[str] = []
    for raw_line in (staged_raw or "").splitlines():
        parts = raw_line.split("\t", 1)
        if len(parts) != 2 or not parts[1]:
            continue
        status, path = parts
        action = {
            "A": "new file",
            "M": "modified",
            "D": "deleted",
            "R": "renamed",
        }.get(status[:1], status)
        staged.append(f"\t{action}: {path}")

    if not staged:
        lines.extend(["", "nothing to commit, working tree clean"])
        return "\n".join(lines)[:_MAX_GIT_STATUS_CHARS].rstrip()

    lines.extend(["", "Changes to be committed:"])
    reserve = 50
    for index, item in enumerate(staged):
        candidate = "\n".join([*lines, item])
        if len(candidate) > _MAX_GIT_STATUS_CHARS - reserve:
            remaining = len(staged) - index
            if remaining:
                lines.append(f"\t... and {remaining} more staged")
            break
        lines.append(item)
    return "\n".join(lines)[:_MAX_GIT_STATUS_CHARS].rstrip()


def _run_git(cwd: Path, args: list[str], deadline: float) -> str | None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=remaining,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    text = completed.stdout.decode("utf-8", errors="replace").strip()
    return text or None
