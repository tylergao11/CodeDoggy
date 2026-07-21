"""merge_subagent_worktree — land a child's worktree branch into the parent repo.

Grok isolation contract: child edits stay isolated until MAIN *explicitly*
merges. This tool is that explicit land step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolError,
    ToolId,
)

_DESC = (
    "Merge a completed subagent's git worktree branch into the parent workspace. "
    "Use after a child ran with isolation=worktree and finished successfully. "
    "Does nothing useful if the child used isolation=none (shared cwd)."
)


class MergeSubagentWorktreeTool(Tool):
    """Explicit worktree land (Grok: merge only on request)."""

    def id(self) -> ToolId:
        return ToolId("merge_subagent_worktree")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Task

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="merge_subagent_worktree", description=_DESC)

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "subagent_id": {
                    "type": "string",
                    "description": "Id returned by spawn_subagent / parallel_tasks.",
                },
                "strategy": {
                    "type": "string",
                    "description": 'Merge strategy: "merge" (default), "squash", or "ff".',
                },
                "commit_message": {
                    "type": "string",
                    "description": "Optional merge/commit message.",
                },
                "cleanup": {
                    "type": "boolean",
                    "description": (
                        "If true, remove the worktree directory after a successful merge."
                    ),
                },
                "delete_branch": {
                    "type": "boolean",
                    "description": "If true, delete the subagent branch after merge.",
                },
            },
            "required": ["subagent_id"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        bag = ctx.extra or {}
        coord = bag.get("subagent_coordinator")
        if coord is None:
            raise ToolError(
                "merge_subagent_worktree requires subagent coordinator "
                "(missing backend).",
                code="missing_resource",
            )
        sid = str(args.get("subagent_id") or "").strip()
        if not sid:
            raise ToolError.invalid_arguments("subagent_id is required")

        strategy = str(args.get("strategy") or "merge").strip().lower() or "merge"
        if strategy not in {"merge", "squash", "ff"}:
            raise ToolError.invalid_arguments(
                'strategy must be "merge", "squash", or "ff"'
            )
        cleanup = bool(args.get("cleanup"))
        delete_branch = bool(args.get("delete_branch"))
        msg = args.get("commit_message")
        commit_message = str(msg).strip() if isinstance(msg, str) and msg.strip() else None

        merge_fn = getattr(coord, "merge_worktree", None)
        if not callable(merge_fn):
            raise ToolError(
                "subagent coordinator cannot merge worktrees.",
                code="missing_resource",
            )

        result = merge_fn(
            sid,
            Path(ctx.cwd),
            strategy=strategy,
            commit_message=commit_message,
            cleanup_worktree=cleanup,
            delete_branch=delete_branch,
        )
        ok = bool(getattr(result, "ok", False))
        lines = [
            f"## Worktree merge {'ok' if ok else 'failed'}",
            f"- subagent_id: {sid}",
            f"- strategy: {getattr(result, 'strategy', strategy)}",
            f"- branch: {getattr(result, 'branch', '') or '(none)'}",
        ]
        commit = getattr(result, "commit", None)
        if commit:
            lines.append(f"- commit: {commit}")
        wt = getattr(result, "worktree_path", None)
        if wt:
            lines.append(f"- worktree: {wt}")
        if getattr(result, "cleaned_worktree", False):
            lines.append("- worktree cleaned: true")
        conflicts = list(getattr(result, "conflicts", None) or [])
        if conflicts:
            lines.append(f"- conflicts: {', '.join(conflicts[:20])}")
        message = str(getattr(result, "message", "") or "").strip()
        if message:
            lines.append("")
            lines.append(message)
        if not ok:
            raise ToolError.invalid_arguments(
                message or f"merge failed for subagent {sid}"
            )
        return "\n".join(lines).rstrip() + "\n"
