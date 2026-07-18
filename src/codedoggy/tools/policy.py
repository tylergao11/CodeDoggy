"""Workspace tool policy — Grok-style guardrails at the tool boundary.

Fuses with resident audit:
  - denied writes never set mutation (no false audit)
  - allowed writes can attach policy note for auditor context
  - policy is workspace-scoped (cwd), not a full OS sandbox
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Shared deny table — tools + audit + docs stay aligned
DEFAULT_DENY_WRITE: tuple[str, ...] = (
    ".git",
    ".git/",
    ".env",
    ".env.local",
    ".env.production",
    "node_modules/",
    ".codedoggy/",
    "__pycache__/",
    ".ssh/",
    "id_rsa",
    "id_ed25519",
    "*.pem",
)


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str = ""
    code: str = "ok"


@dataclass
class WorkspacePolicy:
    """Minimal policy for coding-agent tools under a session cwd."""

    cwd: Path
    allow_writes: bool = True
    allow_shell: bool = True
    # Relative path prefixes that must not be written (e.g. .git, .env)
    deny_write_globs: list[str] = field(
        default_factory=lambda: list(DEFAULT_DENY_WRITE)
    )
    # If set, only these relative prefixes may be written (empty = all under cwd)
    allow_write_prefixes: list[str] = field(default_factory=list)
    enabled: bool = True

    def __post_init__(self) -> None:
        self.cwd = Path(self.cwd).resolve()

    @classmethod
    def from_env(cls, cwd: Path | str) -> WorkspacePolicy:
        deny = os.environ.get("CODEDOGGY_DENY_WRITE", "")
        extra = [p.strip() for p in deny.split(",") if p.strip()]
        base = cls(cwd=Path(cwd))
        if extra:
            base.deny_write_globs = list(base.deny_write_globs) + extra
        if os.environ.get("CODEDOGGY_POLICY", "1").strip().lower() in {
            "0",
            "false",
            "off",
            "no",
        }:
            base.enabled = False
        return base

    def check_read(self, path: str) -> PolicyDecision:
        """Grok-style read boundary: no path escape outside cwd."""
        if not self.enabled:
            return PolicyDecision(True)
        rel = self._rel_or_none(path)
        if rel is None:
            return PolicyDecision(
                False, f"path escapes workspace: {path}", "path_escape"
            )
        return PolicyDecision(True)

    def check_write(self, path: str) -> PolicyDecision:
        if not self.enabled:
            return PolicyDecision(True)
        if not self.allow_writes:
            return PolicyDecision(False, "writes disabled by policy", "write_disabled")
        rel = self._rel_or_none(path)
        if rel is None:
            return PolicyDecision(
                False, f"path escapes workspace: {path}", "path_escape"
            )
        norm = _norm_rel(rel)
        for d in self.deny_write_globs:
            d = _norm_rel(d)
            if _path_matches_rule(norm, d):
                return PolicyDecision(
                    False, f"write denied for protected path: {norm}", "deny_path"
                )
        if self.allow_write_prefixes:
            ok = any(
                _path_matches_prefix(norm, _norm_rel(pref))
                for pref in self.allow_write_prefixes
            )
            if not ok:
                return PolicyDecision(
                    False, f"write outside allowed prefixes: {norm}", "allowlist"
                )
        return PolicyDecision(True)

    def check_shell(self, command: str) -> PolicyDecision:
        if not self.enabled:
            return PolicyDecision(True)
        if not self.allow_shell:
            return PolicyDecision(False, "shell disabled by policy", "shell_disabled")
        low = (command or "").lower()
        dangerous = (
            "rm -rf /",
            "remove-item -recurse -force c:\\",
            "format c:",
            "git clean -fdx",
            "git clean -ffdx",
            "git reset --hard",
        )
        for d in dangerous:
            if d in low:
                return PolicyDecision(
                    False, f"destructive shell blocked: {d}", "shell_dangerous"
                )
        # When writes disabled: block interpreters / redirect shells that can
        # write arbitrarily (regex is a signal, not the full sandbox — still gate).
        if not self.allow_writes:
            write_capable = (
                "python",
                "python3",
                "py ",
                "node ",
                "nodejs",
                "ruby ",
                "perl ",
                "php ",
                "bash -c",
                "sh -c",
                "pwsh",
                "powershell",
                "cmd /c",
                "cmd.exe",
                ">",
                ">>",
                "tee ",
                "dd ",
                "cp ",
                "mv ",
                "copy ",
                "move ",
                "set-content",
                "out-file",
                "add-content",
                "new-item",
                "remove-item",
            )
            for w in write_capable:
                if w in low:
                    return PolicyDecision(
                        False,
                        f"shell may write while allow_writes=False ({w.strip()})",
                        "shell_write_capable",
                    )
        try:
            from codedoggy.tools.util.write_detect import detect_shell_write_paths

            for wp in detect_shell_write_paths(command or ""):
                wd = self.check_write(wp)
                if not wd.allowed:
                    return PolicyDecision(
                        False,
                        wd.reason or f"shell would write denied path: {wp}",
                        wd.code or "deny_path",
                    )
        except Exception:  # noqa: BLE001
            pass
        return PolicyDecision(True)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "allow_writes": self.allow_writes,
            "allow_shell": self.allow_shell,
            "deny_write_globs": list(self.deny_write_globs),
            "allow_write_prefixes": list(self.allow_write_prefixes),
            "cwd": str(self.cwd),
        }

    def _rel_or_none(self, path: str) -> str | None:
        try:
            p = Path(path)
            if not p.is_absolute():
                p = (self.cwd / p).resolve()
            else:
                p = p.resolve()
            return str(p.relative_to(self.cwd))
        except (OSError, ValueError):
            return None


def _norm_rel(path: str) -> str:
    """Normalize relative path; casefold for Windows-safe deny matching."""
    s = path.replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    s = s.lstrip("/")  # only leading slashes, NOT dots
    # Case-insensitive compare for deny rules (Windows / mixed tooling)
    return s.casefold()


def _path_matches_prefix(path: str, prefix: str) -> bool:
    if not prefix:
        return False
    p = prefix.rstrip("/")
    if path == p:
        return True
    return path.startswith(p + "/")


def _path_matches_rule(path: str, rule: str) -> bool:
    """Prefix match or simple ``*.ext`` / basename deny rules (case-insensitive)."""
    if not rule:
        return False
    # path and rule already casefold via _norm_rel for callers of check_write
    rule_n = rule.replace("\\", "/").casefold()
    path_n = path.replace("\\", "/").casefold()
    if "/" not in rule_n.rstrip("/") and not rule_n.startswith("*") and not rule_n.endswith("/"):
        base = path_n.rsplit("/", 1)[-1]
        if base == rule_n or path_n == rule_n:
            return True
    if rule_n.startswith("*."):
        return path_n.endswith(rule_n[1:])
    return _path_matches_prefix(path_n, rule_n.rstrip("/"))
