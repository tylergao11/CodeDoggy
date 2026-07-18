"""Bash model-facing output formatting.

Ported from:
  grok-build/.../grok_build/bash/mod.rs
    format_default_prompt, annotations, KillReason Display

Function map:
  format_default_prompt ↔ format_default_prompt
  annotations           ↔ annotations
  format_bytes          ↔ format_bytes (util)
"""

from __future__ import annotations

import re
from enum import Enum


class KillReason(str, Enum):
    # bash/mod.rs KillReason
    Timeout = "timeout"
    MaxRuntime = "max_runtime"
    Cancelled = "cancelled"
    Killed = "killed"

    @classmethod
    def parse(cls, signal: str | None) -> "KillReason | None":
        if signal is None:
            return None
        for m in cls:
            if m.value == signal:
                return m
        if signal.startswith("signal "):
            return None  # raw signal → annotations path
        return None


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text)


def annotations(
    *,
    truncated: bool = False,
    output_len: int = 0,
    total_bytes: int = 0,
    output_file: str | None = None,
    signal: str | None = None,
) -> str:
    """bash/mod.rs annotations()."""
    s = ""
    if truncated:
        shown = format_bytes(output_len)
        total = format_bytes(total_bytes if total_bytes else output_len)
        if output_file:
            s += (
                f" [truncated: showing first/last {shown} of {total} "
                f"- full output at: {output_file}]"
            )
        else:
            s += f" [truncated: showing {shown} of {total}]"
    if signal is not None:
        # Synthetic kill reasons are in exit header; suppress redundant [signal=…]
        if KillReason.parse(signal) is None:
            s += f" [signal={signal}]"
    return s


def format_default_prompt(
    *,
    exit_code: int | None = None,
    output: str = "",
    signal: str | None = None,
    truncated: bool = False,
    total_bytes: int = 0,
    output_file: str | None = None,
    current_dir: str | None = None,
) -> str:
    """bash/mod.rs format_default_prompt for foreground results."""
    output_str = strip_ansi(output)
    is_backgrounded = signal in {"backgrounded", "auto_backgrounded"}

    if is_backgrounded:
        shown = format_bytes(len(output.encode("utf-8", errors="replace")))
        total = format_bytes(total_bytes or len(output.encode("utf-8", errors="replace")))
        of = output_file or ""
        cwd = current_dir or ""
        return (
            "[Command moved to background]\n\n"
            f"Partial output ({shown} of {total} total):\n\n"
            f"```\n{output_str}\n```\n\n"
            "The command is still running in the background. You can continue with other tasks.\n"
            f"Full output is being written to: {of}\n\n"
            f"On the next terminal tool call, the directory of the shell will be {cwd}."
        )

    reason = KillReason.parse(signal)
    if reason is not None:
        header = f"exit: killed ({reason.value})"
    elif signal == "killed (timeout)":
        header = "exit: killed (timeout)"
    else:
        code = exit_code if exit_code is not None else -1
        header = f"exit: {code}"

    ann = annotations(
        truncated=truncated,
        output_len=len(output_str.encode("utf-8", errors="replace")),
        total_bytes=total_bytes,
        output_file=output_file,
        signal=None if reason is not None else signal,
    )
    return f"{header}{ann}\n{output_str}" if output_str else f"{header}{ann}"
