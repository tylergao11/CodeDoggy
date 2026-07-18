"""Execute tool calls and build model-facing observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codedoggy.tools.kinds import ToolKind
from codedoggy.tools.registry import FinalizedToolset
from codedoggy.tools.runtime import ToolCallContext, ToolError
from codedoggy.turn.types import FileMutation, ToolCall, ToolResultRecord

# Kinds that change the workspace (quality-gate candidates).
_MUTATING_KINDS = frozenset(
    {
        ToolKind.Edit,
        ToolKind.Write,
        ToolKind.Delete,
        ToolKind.Move,
    }
)


def is_mutating_kind(kind: ToolKind | None) -> bool:
    return kind is not None and kind in _MUTATING_KINDS


def extract_mutation_path(args: dict[str, Any]) -> str | None:
    """Best-effort path from common edit/write tool args."""
    for key in ("file_path", "target_file", "path", "destination"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def format_tool_error_observation(err: ToolError) -> str:
    """Stable observation when a tool raises ToolError."""
    return f"Error ({err.code}): {err.message}"


def execute_tool_call(
    tools: FinalizedToolset,
    call: ToolCall,
    *,
    cwd: Path,
    session_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> ToolResultRecord:
    """Run one tool call; ToolError becomes an observation, not a crash."""
    kind = tools.kind_of(call.name)
    ctx = ToolCallContext(cwd=cwd, session_id=session_id, extra=dict(extra or {}))
    try:
        content = tools.call(call.name, call.arguments, ctx)
        mutation = _resolve_mutation(ctx, call, kind)
        return ToolResultRecord(
            call=call,
            content=content if content is not None else "",
            ok=True,
            kind=kind,
            mutation=mutation,
        )
    except ToolError as e:
        return ToolResultRecord(
            call=call,
            content=format_tool_error_observation(e),
            ok=False,
            error_code=e.code,
            kind=kind,
            mutation=None,
        )
    except Exception as e:  # noqa: BLE001 — observation surface for the model
        return ToolResultRecord(
            call=call,
            content=f"Error (internal): {type(e).__name__}: {e}",
            ok=False,
            error_code="internal",
            kind=kind,
            mutation=None,
        )


def execute_tool_batch(
    tools: FinalizedToolset,
    calls: list[ToolCall],
    *,
    cwd: Path,
    session_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> list[ToolResultRecord]:
    """Execute tool calls in request order (sequential).

    Parallel batching is a future executor concern; keep order stable for
    writeback and same-file conflicts until locks exist.
    """
    return [
        execute_tool_call(
            tools, call, cwd=cwd, session_id=session_id, extra=extra
        )
        for call in calls
    ]


def _resolve_mutation(
    ctx: ToolCallContext,
    call: ToolCall,
    kind: ToolKind | None,
) -> FileMutation | None:
    """Prefer first-hand mutation from the tool; else path-only for mutating kinds."""
    raw = ctx.extra.get("mutation")
    if isinstance(raw, FileMutation):
        return FileMutation(
            path=raw.path,
            tool_name=raw.tool_name or call.name,
            call_id=raw.call_id or call.id,
            args=dict(raw.args or call.arguments or {}),
            before=raw.before,
            after=raw.after,
            is_create=raw.is_create,
        )
    if is_mutating_kind(kind):
        path = extract_mutation_path(call.arguments if isinstance(call.arguments, dict) else {})
        if path is not None:
            return FileMutation(
                path=path,
                tool_name=call.name,
                call_id=call.id,
                args=dict(call.arguments) if isinstance(call.arguments, dict) else {},
            )
    return None


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """Normalize model tool arguments (dict or JSON string) to a dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"_raw": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"_value": parsed}
    return {"_value": raw}
