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
    """Run one tool call; ToolError becomes an observation, not a crash.

    Mutations are collected even when the tool returns non-zero / ToolError
    after partial writes (shell multi-file).
    """
    kind = tools.kind_of(call.name)
    ctx = ToolCallContext(cwd=cwd, session_id=session_id, extra=dict(extra or {}))
    try:
        content = tools.call(call.name, call.arguments, ctx)
        mutations = _resolve_mutations(ctx, call, kind)
        return ToolResultRecord(
            call=call,
            content=content if content is not None else "",
            ok=True,
            kind=kind,
            mutation=mutations[0] if mutations else None,
            mutations=mutations,
        )
    except ToolError as e:
        mutations = _resolve_mutations(ctx, call, kind)
        return ToolResultRecord(
            call=call,
            content=format_tool_error_observation(e),
            ok=False,
            error_code=e.code,
            kind=kind,
            mutation=mutations[0] if mutations else None,
            mutations=mutations,
        )
    except Exception as e:  # noqa: BLE001 — observation surface for the model
        mutations = _resolve_mutations(ctx, call, kind)
        return ToolResultRecord(
            call=call,
            content=f"Error (internal): {type(e).__name__}: {e}",
            ok=False,
            error_code="internal",
            kind=kind,
            mutation=mutations[0] if mutations else None,
            mutations=mutations,
        )


def execute_tool_batch(
    tools: FinalizedToolset,
    calls: list[ToolCall],
    *,
    cwd: Path,
    session_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> list[ToolResultRecord]:
    """Legacy sequential helper. Prefer ``execute_tool_calls_two_phase`` /
    path-lock batch for Grok-aligned dispatch.
    """
    return [
        execute_tool_call(
            tools, call, cwd=cwd, session_id=session_id, extra=extra
        )
        for call in calls
    ]


def _resolve_mutations(
    ctx: ToolCallContext,
    call: ToolCall,
    kind: ToolKind | None,
) -> list[FileMutation]:
    """Collect all first-hand mutations (multi-file); fallback path-only."""
    out: list[FileMutation] = []
    bag = ctx.extra.get("mutations")
    if isinstance(bag, list):
        for raw in bag:
            if isinstance(raw, FileMutation):
                out.append(
                    FileMutation(
                        path=raw.path,
                        tool_name=raw.tool_name or call.name,
                        call_id=raw.call_id or call.id,
                        args=dict(raw.args or call.arguments or {}),
                        before=raw.before,
                        after=raw.after,
                        is_create=raw.is_create,
                        is_delete=raw.is_delete,
                    )
                )
    if not out:
        raw = ctx.extra.get("mutation")
        if isinstance(raw, FileMutation):
            out.append(
                FileMutation(
                    path=raw.path,
                    tool_name=raw.tool_name or call.name,
                    call_id=raw.call_id or call.id,
                    args=dict(raw.args or call.arguments or {}),
                    before=raw.before,
                    after=raw.after,
                    is_create=raw.is_create,
                    is_delete=raw.is_delete,
                )
            )
    if not out and is_mutating_kind(kind):
        path = extract_mutation_path(
            call.arguments if isinstance(call.arguments, dict) else {}
        )
        if path is not None:
            out.append(
                FileMutation(
                    path=path,
                    tool_name=call.name,
                    call_id=call.id,
                    args=dict(call.arguments)
                    if isinstance(call.arguments, dict)
                    else {},
                )
            )
    return out


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
