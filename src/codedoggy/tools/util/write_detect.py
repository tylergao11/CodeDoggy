"""Best-effort detect workspace file writes from shell commands.

Used so resident audit can review shell-produced files, not only
search_replace. Covers redirects, PowerShell write cmdlets, and common
``python -c`` / ``pathlib`` one-liners.
"""

from __future__ import annotations

import re
from pathlib import Path

# cmd/bash style redirects:  > path  >> path  (not 2> or &>)
_REDIRECT_RE = re.compile(
    r"(?<![0-9])>{1,2}\s*['\"]?([^\s'\"|&;]+)",
    re.IGNORECASE,
)
# PowerShell: Set-Content/Out-File/Add-Content/New-Item
_PS_WRITE_RE = re.compile(
    r"(?:Set-Content|Out-File|Add-Content|New-Item)\s+"
    r"(?:-Path\s+|-)?"
    r"['\"]?([^\s'\"]+)",
    re.IGNORECASE,
)
# python -c "open('f','w')..." / open("f", "w")
_PY_OPEN_RE = re.compile(
    r"""open\s*\(\s*['"]([^'"]+)['"]\s*,\s*['"]([wax])""",
    re.IGNORECASE,
)
# Path('f').write_text(...)
_PY_PATH_WRITE_RE = re.compile(
    r"""Path\s*\(\s*['"]([^'"]+)['"]\s*\)\s*\.\s*write_(?:text|bytes)\s*\(""",
    re.IGNORECASE,
)
# tee file, install -m ... file
_TEE_RE = re.compile(
    r"(?:tee|install)\s+(?:-a\s+)?['\"]?([^\s'\"]+)",
    re.IGNORECASE,
)


def detect_shell_write_paths(command: str) -> list[str]:
    """Return candidate relative/absolute paths the command may write."""
    if not command or not command.strip():
        return []
    found: list[str] = []
    for m in _REDIRECT_RE.finditer(command):
        p = m.group(1).strip()
        if p and p not in {"$null", "nul", "/dev/null", "NUL"}:
            found.append(p)
    for m in _PS_WRITE_RE.finditer(command):
        p = m.group(1).strip()
        if p and not p.startswith("-"):
            found.append(p)
    for m in _PY_OPEN_RE.finditer(command):
        # only write modes w/a/x
        found.append(m.group(1).strip())
    for m in _PY_PATH_WRITE_RE.finditer(command):
        found.append(m.group(1).strip())
    for m in _TEE_RE.finditer(command):
        p = m.group(1).strip()
        if p and not p.startswith("-"):
            found.append(p)
    # de-dupe preserve order
    out: list[str] = []
    seen: set[str] = set()
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def resolve_writable_under_cwd(cwd: Path, raw_path: str) -> Path | None:
    """Resolve path; only accept files under cwd (no escape)."""
    try:
        cwd = cwd.resolve()
        p = Path(raw_path)
        if not p.is_absolute():
            p = (cwd / p).resolve()
        else:
            p = p.resolve()
        p.relative_to(cwd)
        return p
    except (OSError, ValueError):
        return None


def record_shell_mutations(
    ctx: object,
    command: str,
    *,
    exit_ok: bool,
    tool_name: str = "run_terminal_cmd",
    call_id: str = "",
    max_after_chars: int = 8_000,
) -> bool:
    """If command wrote a workspace file, set first-hand mutation on ctx.

    Returns True when a mutation was recorded.
    """
    if not exit_ok:
        return False
    set_mut = getattr(ctx, "set_mutation", None)
    if not callable(set_mut):
        return False
    cwd = getattr(ctx, "cwd", None)
    if cwd is None:
        return False
    cwd_path = Path(cwd)
    candidates = detect_shell_write_paths(command)
    # Also: scan cwd for files modified in last few seconds? too aggressive — skip.
    for raw in candidates:
        path = resolve_writable_under_cwd(cwd_path, raw)
        if path is None or not path.is_file():
            continue
        try:
            after = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            after = None
        if after is not None and len(after) > max_after_chars:
            after = after[:max_after_chars] + "\n… (truncated for audit)"
        try:
            rel = str(path.relative_to(cwd_path.resolve()))
        except ValueError:
            rel = str(path)
        set_mut(
            path=rel.replace("\\", "/"),
            before=None,
            after=after,
            is_create=True,
            tool_name=tool_name,
            call_id=call_id,
            args={"command": command[:500]},
        )
        return True
    return False
