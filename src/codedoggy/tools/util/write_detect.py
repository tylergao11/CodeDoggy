"""Best-effort detect workspace file writes from shell commands.

Used for policy + shadow. Regex is not a sandbox — it feeds policy checks
before execution. Prefer dedicated file tools for mutations.
"""

from __future__ import annotations

import re
from pathlib import Path

# cmd/bash style redirects:  > path  >> path  (not 2> or &>)
_REDIRECT_RE = re.compile(
    r"(?<![0-9&])>{1,2}\s*['\"]?([^\s'\"|&;]+)",
    re.IGNORECASE,
)
# PowerShell write cmdlets with -Path / -LiteralPath / positional
_PS_WRITE_RE = re.compile(
    r"(?:Set-Content|Add-Content|Out-File|New-Item|Set-Item|Add-Content)\b"
    r"(?:[^;\n]*?(?:-LiteralPath|-Path|-FilePath)\s+['\"]?([^\s'\"]+)"
    r"|[^;\n]*?\s+['\"]?([^\s'\-][^\s'\"]*))",
    re.IGNORECASE,
)
# Remove-Item / del / rm / erase / ri
_PS_REMOVE_RE = re.compile(
    r"(?:Remove-Item|ri\b|del\b|erase\b|rm\b|rmdir\b)"
    r"(?:[^;\n]*?(?:-LiteralPath|-Path)\s+['\"]?([^\s'\"]+)"
    r"|[^;\n]*?\s+['\"]?([^\s'\-][^\s'\"]*))",
    re.IGNORECASE,
)
# Move-Item / mv / move / ren / rename
_PS_MOVE_RE = re.compile(
    r"(?:Move-Item|Copy-Item|Rename-Item|mi\b|mv\b|move\b|ren\b|rename\b|cp\b|copy\b)"
    r"(?:[^;\n]*?(?:-LiteralPath|-Path|-Destination)\s+['\"]?([^\s'\"]+)"
    r"|[^;\n]*?\s+['\"]?([^\s'\-][^\s'\"]*))",
    re.IGNORECASE,
)
# python -c "open('f','w')..." / open("f", "w")
_PY_OPEN_RE = re.compile(
    r"""open\s*\(\s*['"]([^'"]+)['"]\s*,\s*['"]([wax])""",
    re.IGNORECASE,
)
_PY_PATH_WRITE_RE = re.compile(
    r"""Path\s*\(\s*['"]([^'"]+)['"]\s*\)\s*\.\s*write_(?:text|bytes)\s*\(""",
    re.IGNORECASE,
)
_TEE_RE = re.compile(
    r"(?:tee|install)\s+(?:-a\s+)?['\"]?([^\s'\"]+)",
    re.IGNORECASE,
)
# git clean / checkout write
_GIT_WRITE_RE = re.compile(
    r"\bgit\s+(?:clean|checkout|restore|reset\s+--hard)\b",
    re.IGNORECASE,
)


def detect_shell_write_paths(command: str) -> list[str]:
    """Return candidate relative/absolute paths the command may write or delete."""
    if not command or not command.strip():
        return []
    found: list[str] = []

    def _add(p: str | None) -> None:
        if not p:
            return
        p = p.strip().strip("'\"")
        if not p or p.startswith("-"):
            return
        if p.lower() in {"$null", "nul", "/dev/null", "nul:"}:
            return
        # Skip common flag values mistaken as paths
        if p.lower() in {"force", "recurse", "confirm", "whatif", "verbose"}:
            return
        found.append(p)

    for m in _REDIRECT_RE.finditer(command):
        _add(m.group(1))
    for m in _PS_WRITE_RE.finditer(command):
        _add(m.group(1) or m.group(2))
    for m in _PS_REMOVE_RE.finditer(command):
        _add(m.group(1) or m.group(2))
    for m in _PS_MOVE_RE.finditer(command):
        _add(m.group(1) or m.group(2))
    for m in _PY_OPEN_RE.finditer(command):
        _add(m.group(1))
    for m in _PY_PATH_WRITE_RE.finditer(command):
        _add(m.group(1))
    for m in _TEE_RE.finditer(command):
        _add(m.group(1))

    # git clean etc. — treat as touching workspace root markers for policy
    if _GIT_WRITE_RE.search(command):
        # Force policy to evaluate protected prefixes if command is destructive git
        found.append(".git/config")
        found.append(".env")

    # python/script.py invocation may write anything — cannot enumerate;
    # still flag if path ends with known protected names in the command text
    low = command.lower()
    for marker in (".git", ".env", "id_rsa", "id_ed25519"):
        if marker in low and marker not in [f.lower() for f in found]:
            # Extract a token containing the marker
            for tok in re.findall(r"[^\s;'\"|]+", command):
                if marker in tok.lower():
                    _add(tok)

    out: list[str] = []
    seen: set[str] = set()
    for p in found:
        key = p.replace("\\", "/").lower()
        if key not in seen:
            seen.add(key)
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
    """Record first-hand mutations for *all* paths the shell may have written.

    Grok: partial writes after non-zero exit still need Shadow review — do not
    require exit_ok. Multi-file commands record every existing path.
    """
    set_mut = getattr(ctx, "set_mutation", None)
    if not callable(set_mut):
        return False
    cwd = getattr(ctx, "cwd", None)
    if cwd is None:
        return False
    cwd_path = Path(cwd)
    candidates = detect_shell_write_paths(command)
    any_set = False
    for raw in candidates:
        path = resolve_writable_under_cwd(cwd_path, raw)
        if path is None:
            continue
        is_delete = not path.exists()
        after = None
        if path.is_file():
            try:
                after = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                after = None
            if after is not None and len(after) > max_after_chars:
                after = after[:max_after_chars] + "\n… (truncated for audit)"
        elif not is_delete:
            continue
        try:
            rel = str(path.relative_to(cwd_path.resolve()))
        except ValueError:
            rel = str(path)
        set_mut(
            path=rel.replace("\\", "/"),
            before=None,
            after=after,
            is_create=False,
            is_delete=is_delete,
            tool_name=tool_name,
            call_id=call_id,
            args={"command": command, "exit_ok": exit_ok},
        )
        any_set = True
    return any_set
