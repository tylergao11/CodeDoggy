"""Background ``&`` operator detection — source port from Grok bash.

Ported from:
  grok-build/.../implementations/grok_build/bash/mod.rs
    has_trailing_background_operator
    contains_background_operator (subset: quotes/escapes/&&/&>/&>>/>&/<&)
    ends_with_wait_builtin
    contains_unwaited_background_operator
    powershell_has_trailing_background
    should_reject_background_op
    background_operator_validation_message
    powershell_background_operator_message

Heredoc body skip is approximated (no full heredoc body consume) — mid-heredoc
``&`` may false-positive; marked A for that edge only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal


class AmpersandSemantics(str, Enum):
    PosixBackground = "posix"
    PowerShellCore = "pwsh"
    WindowsPowerShell = "powershell"
    CmdSeparator = "cmd"


class BackgroundOpKind(Enum):
    Bash = auto()
    PowerShell = auto()


@dataclass(frozen=True)
class BackgroundOpViolation:
    kind: BackgroundOpKind
    trailing_is_syntax_error: bool = False


def has_trailing_background_operator(command: str) -> bool:
    """Trailing ``&`` only; excludes ``&&`` and ``>&``."""
    trimmed = command.strip()
    if not trimmed.endswith("&") or trimmed.endswith("&&") or trimmed.endswith(">&"):
        return False
    return True


def powershell_has_trailing_background(command: str) -> bool:
    stripped = command.rstrip().rstrip(";\n").rstrip()
    return has_trailing_background_operator(stripped)


def ends_with_wait_builtin(command: str) -> bool:
    trimmed = command.rstrip().rstrip(";").rstrip("\n").rstrip()
    return (
        trimmed == "wait"
        or trimmed.endswith(" wait")
        or trimmed.endswith(";wait")
        or trimmed.endswith("\twait")
        or trimmed.endswith("\nwait")
    )


def contains_background_operator(command: str) -> bool:
    """Detect backgrounding ``&`` outside quotes (Grok scan, no full heredoc)."""
    chars = list(command)
    n = len(chars)
    i = 0
    in_single = False
    in_double = False
    while i < n:
        ch = chars[i]
        if ch == "\\" and not in_single:
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if ch == "&" and not in_single and not in_double:
            if i + 1 < n and chars[i + 1] == "&":
                i += 2
                continue
            if i + 1 < n and chars[i + 1] == ">":
                i += 2
                if i < n and chars[i] == ">":
                    i += 1
                continue
            if i > 0 and chars[i - 1] in {">", "<"}:
                i += 1
                continue
            return True
        i += 1
    return False


def contains_unwaited_background_operator(command: str) -> bool:
    return contains_background_operator(command) and not ends_with_wait_builtin(command)


def command_has_bash_background_operator(command: str, *, is_legacy: bool = False) -> bool:
    if is_legacy:
        return has_trailing_background_operator(command)
    return contains_unwaited_background_operator(command)


def detect_background_op_violation(
    semantics: AmpersandSemantics,
    command: str,
    *,
    is_legacy: bool = False,
) -> BackgroundOpViolation | None:
    if semantics is AmpersandSemantics.PosixBackground:
        if command_has_bash_background_operator(command, is_legacy=is_legacy):
            return BackgroundOpViolation(BackgroundOpKind.Bash)
        return None
    if semantics is AmpersandSemantics.PowerShellCore:
        if powershell_has_trailing_background(command):
            return BackgroundOpViolation(
                BackgroundOpKind.PowerShell, trailing_is_syntax_error=False
            )
        return None
    if semantics is AmpersandSemantics.WindowsPowerShell:
        if powershell_has_trailing_background(command):
            return BackgroundOpViolation(
                BackgroundOpKind.PowerShell, trailing_is_syntax_error=True
            )
        return None
    # CmdSeparator — & is sequential separator
    return None


def should_reject_background_op(
    *,
    is_background: bool,
    allow_background_operator: bool,
    background_enabled: bool,
    semantics: AmpersandSemantics,
    command: str,
    is_legacy: bool = False,
) -> BackgroundOpViolation | None:
    """Grok should_reject_background_op.

    When ``is_background`` or (allow AND enabled), never reject.
    """
    if is_background or (allow_background_operator and background_enabled):
        return None
    return detect_background_op_violation(semantics, command, is_legacy=is_legacy)


def background_operator_validation_message(
    *,
    background_enabled: bool,
    is_legacy: bool,
    param_name: str,
) -> str:
    if background_enabled and is_legacy:
        return (
            f"Command must not end with '&'. Remove the '&' and set {param_name}=true "
            "to run the command in the background."
        )
    if background_enabled and not is_legacy:
        return (
            f"Remove the background '&' from your command and set {param_name}=true instead."
        )
    if not background_enabled and is_legacy:
        return (
            "Command must not end with '&' because background execution is disabled. "
            "Remove the '&' and run the command in the foreground."
        )
    return "Remove the background '&' from your command; background execution is disabled."


def powershell_background_operator_message(
    *,
    background_enabled: bool,
    param_name: str,
    trailing_is_syntax_error: bool,
) -> str:
    effect = (
        "is a syntax error in Windows PowerShell 5.1"
        if trailing_is_syntax_error
        else "starts a background job"
    )
    if background_enabled:
        return f"Trailing '&' {effect}. Set {param_name}=true instead."
    return f"Trailing '&' {effect}. Background execution is disabled; remove it."


def ampersand_semantics_for_host() -> AmpersandSemantics:
    """Map CodeDoggy shell detection → AmpersandSemantics."""
    import sys

    if sys.platform != "win32":
        return AmpersandSemantics.PosixBackground
    from codedoggy.tools.util.shell import WindowsShell, detect_windows_shell

    kind = detect_windows_shell()
    if kind is WindowsShell.GIT_BASH:
        return AmpersandSemantics.PosixBackground
    if kind is WindowsShell.PWSH:
        return AmpersandSemantics.PowerShellCore
    if kind is WindowsShell.POWERSHELL:
        return AmpersandSemantics.WindowsPowerShell
    return AmpersandSemantics.CmdSeparator


def rejection_message(
    violation: BackgroundOpViolation,
    *,
    background_enabled: bool,
    param_name: str = "background",
    is_legacy: bool = False,
) -> str:
    if violation.kind is BackgroundOpKind.PowerShell:
        return powershell_background_operator_message(
            background_enabled=background_enabled,
            param_name=param_name,
            trailing_is_syntax_error=violation.trailing_is_syntax_error,
        )
    return background_operator_validation_message(
        background_enabled=background_enabled,
        is_legacy=is_legacy,
        param_name=param_name,
    )
