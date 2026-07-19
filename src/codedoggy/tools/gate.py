"""Central tool gate — Grok permission boundary before any tool.run().

All dispatch must pass FinalizedToolset.call → gate.
- schema validation (required, types, enum, nested, additionalProperties)
- policy check_read / check_write / check_shell
- registration kind is authoritative (config kind cannot downgrade writes)
"""

from __future__ import annotations

import logging
from typing import Any

from codedoggy.tools.kinds import (
    FILE_MUTATING_KINDS,
    HARD_EXECUTE_TOOL_NAMES,
    HARD_WRITE_TOOL_NAMES,
    ToolKind,
    is_registration_authoritative_kind,
)
from codedoggy.tools.runtime import ToolCallContext, ToolError
from codedoggy.tools.util.write_detect import detect_shell_write_paths

logger = logging.getLogger(__name__)

# Re-export alias used by write-policy branch (file mutations only).
_WRITE_KINDS = FILE_MUTATING_KINDS
_WRITE_TOOL_NAMES = HARD_WRITE_TOOL_NAMES


def validate_args_against_schema(args: dict[str, Any], schema: dict[str, Any]) -> None:
    """JSON-schema subset: required, type, enum, nested object/array, additionalProperties."""
    if not isinstance(schema, dict):
        return
    _validate_object(args, schema, path="$")


def _validate_object(obj: Any, schema: dict[str, Any], *, path: str) -> None:
    if not isinstance(obj, dict):
        raise ToolError.invalid_arguments(f"{path} must be an object")
    required = schema.get("required") or []
    if isinstance(required, list):
        missing = [k for k in required if k not in obj or obj[k] is None]
        if missing:
            raise ToolError.invalid_arguments(
                f"missing required arguments: {', '.join(str(m) for m in missing)}"
            )
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        props = {}
    additional = schema.get("additionalProperties", True)
    for key, val in list(obj.items()):
        if key not in props:
            if additional is False:
                raise ToolError.invalid_arguments(
                    f"{path}.{key}: additional property not allowed"
                )
            continue
        spec = props[key]
        if isinstance(spec, dict):
            _validate_value(val, spec, path=f"{path}.{key}")


def _validate_value(val: Any, spec: dict[str, Any], *, path: str) -> None:
    if "enum" in spec:
        if val not in spec["enum"]:
            raise ToolError.invalid_arguments(
                f"{path} must be one of {spec['enum']!r}, got {val!r}"
            )
    expected = spec.get("type")
    if expected is None:
        return
    if isinstance(expected, list):
        # multi-type: pass if any matches
        ok = False
        for t in expected:
            try:
                _check_type(val, t, spec, path=path)
                ok = True
                break
            except ToolError:
                continue
        if not ok:
            raise ToolError.invalid_arguments(
                f"{path} must be one of types {expected}"
            )
        return
    _check_type(val, expected, spec, path=path)


def _check_type(val: Any, expected: str, spec: dict[str, Any], *, path: str) -> None:
    if expected == "string":
        if not isinstance(val, str):
            raise ToolError.invalid_arguments(f"{path} must be a string")
        if "minLength" in spec and len(val) < int(spec["minLength"]):
            raise ToolError.invalid_arguments(f"{path} shorter than minLength")
        if "maxLength" in spec and len(val) > int(spec["maxLength"]):
            raise ToolError.invalid_arguments(f"{path} longer than maxLength")
    elif expected == "integer":
        # Strict: bool is a subclass of int in Python — reject bool
        if isinstance(val, bool) or not isinstance(val, int):
            raise ToolError.invalid_arguments(f"{path} must be an integer")
        if "minimum" in spec and val < int(spec["minimum"]):
            raise ToolError.invalid_arguments(f"{path} below minimum")
        if "maximum" in spec and val > int(spec["maximum"]):
            raise ToolError.invalid_arguments(f"{path} above maximum")
    elif expected == "number":
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ToolError.invalid_arguments(f"{path} must be a number")
        if "minimum" in spec and float(val) < float(spec["minimum"]):
            raise ToolError.invalid_arguments(f"{path} below minimum")
        if "maximum" in spec and float(val) > float(spec["maximum"]):
            raise ToolError.invalid_arguments(f"{path} above maximum")
    elif expected == "boolean":
        if not isinstance(val, bool):
            raise ToolError.invalid_arguments(f"{path} must be a boolean")
    elif expected == "object":
        if not isinstance(val, dict):
            raise ToolError.invalid_arguments(f"{path} must be an object")
        # nested
        nested = dict(spec)
        if "properties" in nested or "required" in nested:
            _validate_object(val, nested, path=path)
    elif expected == "array":
        if not isinstance(val, list):
            raise ToolError.invalid_arguments(f"{path} must be an array")
        if "minItems" in spec and len(val) < int(spec["minItems"]):
            raise ToolError.invalid_arguments(f"{path} fewer than minItems")
        if "maxItems" in spec and len(val) > int(spec["maxItems"]):
            raise ToolError.invalid_arguments(f"{path} more than maxItems")
        items = spec.get("items")
        if isinstance(items, dict):
            for i, el in enumerate(val):
                _validate_value(el, items, path=f"{path}[{i}]")


def effective_kind(
    *,
    tool_name: str,
    registered_kind: ToolKind | None,
    config_kind: ToolKind | None,
) -> ToolKind | None:
    """Registration kind wins for write/execute tools — config cannot downgrade.

    Protects both class rules (Edit/Write/Delete/Move/Execute) and allowlisted
    short-ids / product client names (write, memory, scheduler_*, shell aliases).
    Config Search (or any non-mutating kind) must never mask these.
    """
    # Hard write allowlist + file-mutating registration kinds
    if tool_name in HARD_WRITE_TOOL_NAMES or (
        registered_kind is not None and registered_kind in FILE_MUTATING_KINDS
    ):
        return registered_kind or config_kind
    # Execute: hard names + registration Execute kind
    if tool_name in HARD_EXECUTE_TOOL_NAMES or registered_kind is ToolKind.Execute:
        return registered_kind if registered_kind is ToolKind.Execute else ToolKind.Execute
    if is_registration_authoritative_kind(registered_kind):
        return registered_kind
    return config_kind or registered_kind


def enforce_policy(
    *,
    tool_name: str,
    kind: ToolKind | None,
    args: dict[str, Any],
    ctx: ToolCallContext,
    registered_kind: ToolKind | None = None,
) -> None:
    """Apply WorkspacePolicy. Raises ToolError if denied."""
    policy = (ctx.extra or {}).get("policy")
    if policy is None:
        return

    kind = effective_kind(
        tool_name=tool_name,
        registered_kind=registered_kind if registered_kind is not None else kind,
        config_kind=kind,
    )

    # Shell / execute
    if kind is ToolKind.Execute or tool_name in HARD_EXECUTE_TOOL_NAMES:
        cmd = args.get("command")
        if isinstance(cmd, str):
            check_shell = getattr(policy, "check_shell", None)
            if callable(check_shell):
                d = check_shell(cmd)
                if d is not None and not getattr(d, "allowed", True):
                    raise ToolError(
                        getattr(d, "reason", None) or "shell denied by policy",
                        code=getattr(d, "code", None) or "policy_denied",
                    )
            check_w = getattr(policy, "check_write", None)
            if callable(check_w):
                for wp in detect_shell_write_paths(cmd):
                    wd = check_w(wp)
                    if wd is not None and not getattr(wd, "allowed", True):
                        raise ToolError(
                            getattr(wd, "reason", None)
                            or f"shell write denied for {wp}",
                            code=getattr(wd, "code", None) or "policy_denied",
                        )
        return

    # Graph cache is profile-owned (outside the workspace). Reindex remains a
    # Search operation; CodebaseGraph separately decides whether cache writes
    # are allowed for this Session.
    if tool_name == "code_nav" and (args.get("action") or "").strip() == "reindex":
        check_w = getattr(policy, "check_write", None)
        if callable(check_w):
            from codedoggy.graph.cache import CACHE_FILE_NAME

            wd = check_w(CACHE_FILE_NAME)
            if wd is not None and not getattr(wd, "allowed", True):
                raise ToolError(
                    getattr(wd, "reason", None) or "reindex denied by write policy",
                    code=getattr(wd, "code", None) or "policy_denied",
                )
        return

    # Reads — Grok check_read (workspace boundary)
    if kind in {
        ToolKind.Read,
        ToolKind.ListDir,
        ToolKind.Search,
        ToolKind.Lsp,
    } or tool_name in {"read_file", "list_dir", "grep", "code_nav"}:
        path = None
        for key in (
            "file_path",
            "target_file",
            "path",
            "target_directory",
            "directory",
        ):
            v = args.get(key)
            if isinstance(v, str) and v.strip():
                path = v.strip()
                break
        if path:
            check_r = getattr(policy, "check_read", None)
            if callable(check_r):
                rd = check_r(path)
                if rd is not None and not getattr(rd, "allowed", True):
                    raise ToolError(
                        getattr(rd, "reason", None) or f"read denied for {path}",
                        code=getattr(rd, "code", None) or "policy_denied",
                    )
        return

    # File mutations
    if kind in _WRITE_KINDS or tool_name in _WRITE_TOOL_NAMES:
        path = None
        for key in ("file_path", "target_file", "path", "destination"):
            v = args.get(key)
            if isinstance(v, str) and v.strip():
                path = v.strip()
                break
        if path:
            check_w = getattr(policy, "check_write", None)
            if callable(check_w):
                wd = check_w(path)
                if wd is not None and not getattr(wd, "allowed", True):
                    raise ToolError(
                        getattr(wd, "reason", None) or f"write denied for {path}",
                        code=getattr(wd, "code", None) or "policy_denied",
                    )
        return

    # use_tool: nested tool_input path keys (host MCP glue — not full sandbox)
    if kind is ToolKind.UseTool or tool_name in {"use_tool", "Doggy:use_tool"}:
        nested = args.get("tool_input")
        if isinstance(nested, dict) and nested:
            from codedoggy.tools.builtins.use_tool import enforce_mcp_tool_input_policy

            enforce_mcp_tool_input_policy(nested, ctx)
        return
