"""Detect host shell and build argv for a command string."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache


class WindowsShell(str, Enum):
    PWSH = "pwsh"
    POWERSHELL = "powershell"
    GIT_BASH = "bash"
    CMD = "cmd"


@dataclass(frozen=True, slots=True)
class ShellInvocation:
    program: str
    args: list[str]
    env: dict[str, str]


def _utf8_env() -> dict[str, str]:
    return {
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8:surrogateescape",
    }


@lru_cache(maxsize=1)
def detect_windows_shell() -> WindowsShell:
    override = os.environ.get("CODEDOGGY_SHELL", "").strip().lower()
    if override in {"pwsh", "powershell", "bash", "gitbash", "git-bash", "cmd", "cmd.exe"}:
        if override in {"bash", "gitbash", "git-bash"}:
            return WindowsShell.GIT_BASH
        if override in {"cmd", "cmd.exe"}:
            return WindowsShell.CMD
        if override == "pwsh":
            return WindowsShell.PWSH
        return WindowsShell.POWERSHELL

    if shutil.which("pwsh") or shutil.which("pwsh.exe"):
        return WindowsShell.PWSH
    ps = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    if os.path.isfile(ps):
        return WindowsShell.POWERSHELL
    bash = shutil.which("bash") or shutil.which("bash.exe")
    if bash and "git" in bash.lower():
        return WindowsShell.GIT_BASH
    return WindowsShell.POWERSHELL


def shell_command_argv(command: str) -> ShellInvocation:
    """Build (program, args, env) for running ``command`` on this host."""
    env = _utf8_env()
    if sys.platform != "win32":
        shell = os.environ.get("SHELL") or "/bin/bash"
        return ShellInvocation(program=shell, args=["-lc", command], env=env)

    kind = detect_windows_shell()
    if kind is WindowsShell.PWSH:
        return ShellInvocation(
            program="pwsh",
            args=["-NoProfile", "-NonInteractive", "-Command", command],
            env=env,
        )
    if kind is WindowsShell.POWERSHELL:
        return ShellInvocation(
            program="powershell.exe",
            args=["-NoProfile", "-NonInteractive", "-Command", command],
            env=env,
        )
    if kind is WindowsShell.GIT_BASH:
        bash = shutil.which("bash") or "bash"
        env = {
            **env,
            "MSYS_NO_PATHCONV": "1",
            "MSYS2_ARG_CONV_EXCL": "*",
        }
        return ShellInvocation(program=bash, args=["-c", command], env=env)
    return ShellInvocation(program="cmd", args=["/C", command], env=env)


def chain_separator() -> str:
    """Command chain operator for the active shell."""
    if sys.platform != "win32":
        return "&&"
    kind = detect_windows_shell()
    if kind in {WindowsShell.PWSH, WindowsShell.GIT_BASH}:
        return "&&"
    return ";"


def has_unix_utilities() -> bool:
    if sys.platform != "win32":
        return True
    return detect_windows_shell() is WindowsShell.GIT_BASH
