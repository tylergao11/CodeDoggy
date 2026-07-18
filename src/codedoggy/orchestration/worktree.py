"""Git worktree isolation + merge for subagents.

**Contract (ported from Grok, do not invent):**
  ``xai-tool-types/task.rs`` — IsolationMode none|worktree; child edits must not
  affect parent until *explicitly* merged; resume inherits worktree.

**Implementation note (honest):**
  Full Grok stack is ``xai-fast-worktree`` + shell ``session/worktree*`` + pool.
  This module is a **minimal git-cli subset** that satisfies the isolation
  *contract* (separate worktree path, preserve until merge), not a port of the
  btrfs/overlay pool. Paths differ (``.codedoggy/worktrees/<id>``).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class WorktreeError(RuntimeError):
    """Failed to create, merge, or remove a subagent worktree."""


@dataclass(slots=True)
class WorktreeHandle:
    """Live worktree for one subagent run."""

    path: Path
    branch: str
    parent_cwd: Path
    subagent_id: str
    created: bool = True

    def cleanup(self, *, force: bool = True) -> None:
        remove_worktree(self.parent_cwd, self.path, force=force)

    def commit_if_dirty(self, message: str = "subagent work") -> str | None:
        """Commit uncommitted worktree changes; return new HEAD or None if clean."""
        return commit_worktree_changes(self.path, message=message)


@dataclass(slots=True)
class MergeResult:
    """Outcome of merging a subagent branch into the parent HEAD."""

    ok: bool
    strategy: str = "merge"  # merge | squash | ff
    branch: str = ""
    commit: str | None = None
    conflicts: list[str] = field(default_factory=list)
    message: str = ""
    cleaned_worktree: bool = False
    worktree_path: str | None = None


def find_git_root(cwd: Path) -> Path | None:
    """Return the git toplevel for *cwd*, or None if not a repo."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.debug("git rev-parse failed: %s", e)
        return None
    if r.returncode != 0:
        return None
    top = (r.stdout or "").strip()
    if not top:
        return None
    return Path(top).resolve()


def create_worktree(
    parent_cwd: Path,
    *,
    subagent_id: str,
    base_ref: str = "HEAD",
) -> WorktreeHandle:
    """Add a new git worktree for *subagent_id*.

    Layout: ``{git_root}/.codedoggy/worktrees/{subagent_id}``
    Branch: ``codedoggy/sub/{subagent_id}``
    """
    root = find_git_root(parent_cwd)
    if root is None:
        raise WorktreeError(
            "worktree isolation requires a git repository "
            f"(cwd={parent_cwd})"
        )
    safe_id = _safe_id(subagent_id)
    wt_parent = root / ".codedoggy" / "worktrees"
    wt_parent.mkdir(parents=True, exist_ok=True)
    path = wt_parent / safe_id
    if path.exists():
        try:
            remove_worktree(root, path, force=True)
        except WorktreeError:
            shutil.rmtree(path, ignore_errors=True)

    branch = f"codedoggy/sub/{safe_id}"
    cmd = [
        "git",
        "worktree",
        "add",
        "-b",
        branch,
        str(path),
        base_ref,
    ]
    r = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if r.returncode != 0:
        # Branch may already exist — attach path to existing branch
        cmd2 = ["git", "worktree", "add", str(path), branch]
        r2 = subprocess.run(
            cmd2,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if r2.returncode != 0:
            raise WorktreeError(
                f"git worktree add failed: {(r.stderr or r.stdout or r2.stderr or '')[:400]}"
            )
    return WorktreeHandle(
        path=path.resolve(),
        branch=branch,
        parent_cwd=root,
        subagent_id=safe_id,
    )


def reattach_worktree(
    parent_cwd: Path,
    *,
    subagent_id: str,
    branch: str | None = None,
    existing_path: str | Path | None = None,
) -> WorktreeHandle:
    """Reuse an existing worktree path or re-add from branch (resume path)."""
    root = find_git_root(parent_cwd)
    if root is None:
        raise WorktreeError("worktree reattach requires a git repository")
    safe_id = _safe_id(subagent_id)
    branch = branch or f"codedoggy/sub/{safe_id}"
    if existing_path:
        p = Path(existing_path)
        if p.is_dir() and _is_worktree(p):
            return WorktreeHandle(
                path=p.resolve(),
                branch=branch,
                parent_cwd=root,
                subagent_id=safe_id,
                created=False,
            )
    # Preferred path
    path = root / ".codedoggy" / "worktrees" / safe_id
    if path.is_dir() and _is_worktree(path):
        return WorktreeHandle(
            path=path.resolve(),
            branch=branch,
            parent_cwd=root,
            subagent_id=safe_id,
            created=False,
        )
    # Re-add from branch
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    r = subprocess.run(
        ["git", "worktree", "add", str(path), branch],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if r.returncode != 0:
        raise WorktreeError(
            f"git worktree reattach failed for {branch}: {(r.stderr or r.stdout or '')[:400]}"
        )
    return WorktreeHandle(
        path=path.resolve(),
        branch=branch,
        parent_cwd=root,
        subagent_id=safe_id,
        created=True,
    )


def commit_worktree_changes(
    worktree_path: Path,
    *,
    message: str = "subagent work",
) -> str | None:
    """Stage + commit dirty files in the worktree. Returns HEAD sha or None if clean."""
    wt = Path(worktree_path)
    if not wt.is_dir():
        raise WorktreeError(f"worktree path missing: {wt}")
    # Configure identity if missing (local to this repo)
    _ensure_git_identity(wt)
    st = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(wt),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if st.returncode != 0:
        raise WorktreeError(f"git status failed: {(st.stderr or '')[:300]}")
    if not (st.stdout or "").strip():
        return _rev_parse(wt, "HEAD")
    subprocess.run(
        ["git", "add", "-A"],
        cwd=str(wt),
        capture_output=True,
        timeout=60,
        check=False,
    )
    c = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(wt),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if c.returncode != 0:
        # Nothing to commit after add (race) or hook failure
        if "nothing to commit" in (c.stdout or "").lower() + (c.stderr or "").lower():
            return _rev_parse(wt, "HEAD")
        raise WorktreeError(f"git commit failed: {(c.stderr or c.stdout or '')[:400]}")
    return _rev_parse(wt, "HEAD")


def merge_worktree_into_parent(
    parent_cwd: Path,
    *,
    branch: str | None = None,
    worktree_path: str | Path | None = None,
    subagent_id: str | None = None,
    strategy: str = "merge",
    commit_message: str | None = None,
    cleanup_worktree: bool = False,
    delete_branch: bool = False,
    auto_commit_dirty: bool = True,
) -> MergeResult:
    """Merge a subagent branch into the parent's current branch.

    Steps:
      1. Optionally commit dirty files in the worktree
      2. ``git merge`` or ``git merge --squash`` from parent HEAD
      3. Optional worktree remove + branch delete

    On conflict: leaves the index conflicted; returns ``ok=False`` with paths.
    """
    root = find_git_root(parent_cwd)
    if root is None:
        return MergeResult(
            ok=False,
            message="not a git repository",
            strategy=strategy,
        )
    safe_id = _safe_id(subagent_id or "sub")
    branch = branch or f"codedoggy/sub/{safe_id}"
    wt_path = Path(worktree_path) if worktree_path else (root / ".codedoggy" / "worktrees" / safe_id)

    # Ensure worktree changes are committed on the branch
    if auto_commit_dirty and wt_path.is_dir():
        try:
            commit_worktree_changes(
                wt_path,
                message=commit_message or f"subagent {safe_id} changes",
            )
        except WorktreeError as e:
            return MergeResult(
                ok=False,
                strategy=strategy,
                branch=branch,
                message=f"commit dirty worktree failed: {e}",
                worktree_path=str(wt_path),
            )

    # Verify branch exists
    br = subprocess.run(
        ["git", "rev-parse", "--verify", branch],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if br.returncode != 0:
        return MergeResult(
            ok=False,
            strategy=strategy,
            branch=branch,
            message=f"branch not found: {branch}",
            worktree_path=str(wt_path) if wt_path.exists() else None,
        )

    strategy = (strategy or "merge").strip().lower()
    if strategy not in {"merge", "squash", "ff"}:
        strategy = "merge"

    msg = commit_message or f"Merge subagent worktree {branch}"
    if strategy == "ff":
        cmd = ["git", "merge", "--ff-only", branch]
    elif strategy == "squash":
        cmd = ["git", "merge", "--squash", branch]
    else:
        cmd = ["git", "merge", "--no-ff", "-m", msg, branch]

    m = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if m.returncode != 0:
        conflicts = _list_conflicts(root)
        # abort if hard failure without conflicts? leave for host
        return MergeResult(
            ok=False,
            strategy=strategy,
            branch=branch,
            conflicts=conflicts,
            message=(m.stderr or m.stdout or "merge failed")[:500],
            worktree_path=str(wt_path) if wt_path.exists() else None,
        )

    # Squash needs an explicit commit
    if strategy == "squash":
        st = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if (st.stdout or "").strip():
            _ensure_git_identity(root)
            c = subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if c.returncode != 0:
                return MergeResult(
                    ok=False,
                    strategy=strategy,
                    branch=branch,
                    message=f"squash commit failed: {(c.stderr or c.stdout or '')[:400]}",
                    worktree_path=str(wt_path) if wt_path.exists() else None,
                )

    head = _rev_parse(root, "HEAD")
    cleaned = False
    if cleanup_worktree and wt_path.exists():
        try:
            remove_worktree(root, wt_path, force=True)
            cleaned = True
        except WorktreeError as e:
            logger.warning("post-merge worktree cleanup failed: %s", e)

    if delete_branch:
        subprocess.run(
            ["git", "branch", "-D", branch],
            cwd=str(root),
            capture_output=True,
            timeout=15,
            check=False,
        )

    return MergeResult(
        ok=True,
        strategy=strategy,
        branch=branch,
        commit=head,
        message="merged",
        cleaned_worktree=cleaned,
        worktree_path=str(wt_path) if wt_path.exists() else None,
    )


def remove_worktree(
    parent_cwd: Path,
    path: Path,
    *,
    force: bool = True,
) -> None:
    """Remove a worktree registration and directory."""
    root = find_git_root(parent_cwd) or Path(parent_cwd).resolve()
    p = Path(path)
    cmd = ["git", "worktree", "remove"]
    if force:
        cmd.append("--force")
    cmd.append(str(p))
    r = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if r.returncode != 0 and p.exists():
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=str(root),
            capture_output=True,
            timeout=30,
            check=False,
        )
        try:
            shutil.rmtree(p, ignore_errors=True)
        except OSError as e:
            raise WorktreeError(f"failed to remove worktree {p}: {e}") from e


def should_cleanup_worktree() -> bool:
    """Env ``CODEDOGGY_SUBAGENT_WORKTREE_CLEANUP=1`` removes worktree after run."""
    raw = (os.environ.get("CODEDOGGY_SUBAGENT_WORKTREE_CLEANUP") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def branch_for_subagent(subagent_id: str) -> str:
    return f"codedoggy/sub/{_safe_id(subagent_id)}"


def _safe_id(subagent_id: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in (subagent_id or "sub"))
    return (s[:64] or "sub").strip("_") or "sub"


def _rev_parse(cwd: Path, ref: str) -> str | None:
    r = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if r.returncode != 0:
        return None
    return (r.stdout or "").strip() or None


def _list_conflicts(root: Path) -> list[str]:
    r = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if r.returncode != 0:
        return []
    return [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]


def _is_worktree(path: Path) -> bool:
    git = path / ".git"
    if git.is_file():
        return True
    if git.is_dir():
        return True
    return False


def _ensure_git_identity(cwd: Path) -> None:
    """Set local user.name/email if missing so commits can succeed in tests/CI."""
    for key, default in (("user.email", "codedoggy@local"), ("user.name", "CodeDoggy")):
        r = subprocess.run(
            ["git", "config", "--get", key],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if r.returncode != 0 or not (r.stdout or "").strip():
            subprocess.run(
                ["git", "config", key, default],
                cwd=str(cwd),
                capture_output=True,
                timeout=10,
                check=False,
            )
