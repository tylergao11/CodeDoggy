"""Env-driven subagent policy (isolation defaults, depth, agent discovery).

Grok spirit: ``[subagents]`` toggle/models + worktree defaults, without a full
remote config stack. All knobs are optional env vars.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from codedoggy.orchestration.types import IsolationMode
from codedoggy.tools.grok_build.task_format import MAX_SUBAGENT_DEPTH

# Absolute hard ceiling even if env is set too high.
_HARD_MAX_DEPTH = 5


def effective_max_subagent_depth() -> int:
    """Nesting limit: env ``CODEDOGGY_MAX_SUBAGENT_DEPTH`` (default Grok = 1).

    MAIN is depth 0; a child is depth 1. Spawn is refused when
    ``depth >= effective_max_subagent_depth()``.
    """
    raw = (os.environ.get("CODEDOGGY_MAX_SUBAGENT_DEPTH") or "").strip()
    if not raw:
        return max(1, int(MAX_SUBAGENT_DEPTH))
    try:
        n = int(raw)
    except ValueError:
        return max(1, int(MAX_SUBAGENT_DEPTH))
    return max(1, min(_HARD_MAX_DEPTH, n))


def default_isolation_for(subagent_type: str) -> IsolationMode:
    """Default isolation when Task.isolation is omitted.

    Env:
      CODEDOGGY_SUBAGENT_ISOLATION=none|worktree|auto
        none     — always share parent cwd (default, Grok default)
        worktree — always isolated git worktree when possible
        auto     — worktree for write-capable types (general-purpose, plan);
                   none for explore / read-only roles
    """
    mode = (os.environ.get("CODEDOGGY_SUBAGENT_ISOLATION") or "none").strip().lower()
    if mode in {"worktree", "wt", "isolated"}:
        return IsolationMode.WORKTREE
    if mode in {"auto", "smart"}:
        key = (subagent_type or "").strip().lower().replace("_", "-")
        if key in {"explore", "search", "read-only", "readonly"}:
            return IsolationMode.NONE
        # general-purpose / plan / custom writers → isolate when possible
        return IsolationMode.WORKTREE
    return IsolationMode.NONE


def drain_prompt_queue_after_cancel() -> bool:
    """Whether to auto-run queued full prompts after a cancelled turn.

    Grok: do not auto-wake after cancel. Default **false**.
    Set ``CODEDOGGY_DRAIN_AFTER_CANCEL=1`` to restore queue drain on cancel.
    """
    raw = (os.environ.get("CODEDOGGY_DRAIN_AFTER_CANCEL") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def agent_discovery_paths(cwd: Path | str | None = None) -> list[Path]:
    """Directories scanned for custom agent markdown definitions.

    Order (later overrides earlier on name collision among customs; builtins win):
      1. ``CODEDOGGY_AGENTS_PATHS`` (os.pathsep-separated)
      2. ``~/.codedoggy/agents``
      3. ``{cwd}/.codedoggy/agents``
      4. ``{cwd}/agents`` (optional project folder)
    """
    paths: list[Path] = []
    raw = (os.environ.get("CODEDOGGY_AGENTS_PATHS") or "").strip()
    if raw:
        for part in raw.split(os.pathsep):
            p = part.strip()
            if p:
                paths.append(Path(p).expanduser())
    home = Path.home() / ".codedoggy" / "agents"
    paths.append(home)
    if cwd is not None:
        root = Path(cwd).expanduser().resolve()
        paths.append(root / ".codedoggy" / "agents")
        paths.append(root / "agents")
    # de-dupe preserving order
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        try:
            key = str(p.resolve()) if p.exists() else str(p)
        except Exception:  # noqa: BLE001
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def discover_agent_files(cwd: Path | str | None = None) -> list[Path]:
    """Return ``*.md`` agent definition files under discovery paths."""
    files: list[Path] = []
    for root in agent_discovery_paths(cwd):
        if not root.is_dir():
            continue
        try:
            for path in sorted(root.glob("*.md")):
                if path.is_file():
                    files.append(path)
            for path in sorted(root.glob("**/*.md")):
                if path.is_file() and path not in files:
                    # Only one level of subdirs (type folders)
                    if path.parent == root or path.parent.parent == root:
                        files.append(path)
        except OSError:
            continue
    return files


def load_discovered_agents(
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    """Load custom agents as ``{name_lower: AgentDefinition}``."""
    from codedoggy.orchestration.agent_def import load_agent_definition_file

    out: dict[str, Any] = {}
    for path in discover_agent_files(cwd):
        try:
            defn = load_agent_definition_file(path)
        except Exception:  # noqa: BLE001
            continue
        key = (defn.name or path.stem).strip().lower()
        if not key:
            continue
        # Last file wins among customs
        out[key] = defn
        # also register stem alias
        stem = path.stem.strip().lower()
        if stem and stem not in out:
            out[stem] = defn
    return out
