"""Persistent shell cwd/env for run_terminal_cmd (tool-layer only).

Lives in ToolCallContext.extra['shell_state']. No host/kernel wiring required.
After each command, a shell-specific probe writes the final PWD to a state file
so the next call starts in the same directory.

Ported from (subset):
  grok-build/.../computer/local/shell_state.rs

Grok full ShellState dumps env/functions/aliases via extra FDs (Unix).
This is the **portable subset**: cwd + env overlays + pwd probe.
Fidelity: A (not full dump-via-fd actor).
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codedoggy.tools.util.shell import WindowsShell, detect_windows_shell


@dataclass
class ShellState:
    """Session-scoped shell working directory + env overlays."""

    cwd: Path
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cwd = Path(self.cwd).resolve()

    def apply_env(self, base: dict[str, str]) -> dict[str, str]:
        out = dict(base)
        out.update(self.env)
        return out


def ensure_shell_state(extra: dict[str, Any] | None, default_cwd: Path) -> ShellState:
    bag = extra if extra is not None else {}
    st = bag.get("shell_state")
    if isinstance(st, ShellState):
        return st
    st = ShellState(cwd=Path(default_cwd).resolve())
    bag["shell_state"] = st
    return st


def wrap_command_with_pwd_probe(command: str, state_file: Path) -> str:
    """Append a PWD write that preserves the command's exit code."""
    sf = str(state_file)
    if sys.platform == "win32":
        kind = detect_windows_shell()
        if kind == WindowsShell.GIT_BASH:
            return (
                command
                + '\n__cd_ec=$?\npwd > "'
                + sf
                + '"\nexit $__cd_ec'
            )
        if kind == WindowsShell.CMD:
            return command + ' & echo %CD%> "' + sf + '"'
        # PowerShell (default on Windows for this project)
        return (
            command
            + "; $__ec = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } "
            + "elseif ($?) { 0 } else { 1 }; Set-Content -LiteralPath '"
            + sf
            + "' -Value (Get-Location).Path -Encoding utf8; exit $__ec"
        )
    return command + "\n__cd_ec=$?\npwd > '" + sf + "'\nexit $__cd_ec"


def read_pwd_probe(state_file: Path) -> Path | None:
    if not state_file.is_file():
        return None
    try:
        text = state_file.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not text:
        return None
    # last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    line = lines[-1]
    try:
        p = Path(line)
        if p.is_dir():
            return p.resolve()
    except OSError:
        return None
    return None


def make_state_file() -> Path:
    fd, name = tempfile.mkstemp(prefix="codedoggy-pwd-", suffix=".txt")
    os.close(fd)
    return Path(name)


def shell_env_overrides() -> dict[str, str]:
    """Grok computer/local/shell_state.rs::shell_env_overrides.

    Applied to every agent shell spawn (prevents color/pager noise).
    """
    return {
        "TERM": "dumb",
        "NO_COLOR": "1",
        "FORCE_COLOR": "0",
        "GROK_AGENT": "1",  # Grok GROK_AGENT_ENV marker (name simplified)
    }
