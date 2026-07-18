"""use_tool — Grok UseTool wire surface + host-dispatch glue.

Source: implementations/use_tool/mod.rs
Dispatch via host extra['mcp_dispatch'](tool_name, tool_input) only.
No invented MCP client / transport / sandbox — host owns transport.

Prepare-before-execute (gate spirit, not full Grok MCP sandbox):
  1. tool_name format + tool_input object
  2. when catalog (mcp_tools / mcp_tool_index) has a schema → validate args
  3. when policy present → path-like keys via check_read / check_write
  4. after dispatch → set_mutation from structured host return (see below)

Host-owned (not CodeDoggy):
  - MCP transport, auth, process isolation
  - BM25 tool index (optional extra['mcp_tool_index'])

**Host mutation contract (required for write tools):**

If ``mcp_dispatch`` returns only a plain string (or unstructured blob),
Shadow never sees file side effects — the host MUST return structured
mutations for any workspace write, or Shadow is blind.

Accepted return shapes (any combination; relative paths preferred)::

  # preferred — multi path with optional before/after for restore
  {"text": "...", "mutations": [
      {"path": "rel/a.py", "before": "...", "after": "...",
       "is_create": false, "is_delete": false}, ...]}

  # single mutation object
  {"output": "...", "mutation": {"path": "rel/a.py", "after": "..."}}

  # minimal path list (Shadow sees paths; before/after optional)
  {"result": "...", "mutated_paths": ["rel/a.py", "rel/b.py"]}

  # single path shortcut
  {"text": "...", "mutated_path": "rel/a.py", "before": None, "after": "..."}

Policy on *returned* paths: always ``set_mutation`` when ``path`` is present
(Shadow truth). ``ToolCallContext.set_mutation`` attaches ``args["_policy"]``
when policy is present — do not drop escaped/denied paths after the fact
(pre-dispatch tool_input paths are still gated by
``enforce_mcp_tool_input_policy``).

Model observation text is taken from ``text`` / ``output`` / ``result`` /
``content`` when the host returns a dict envelope.
"""

from __future__ import annotations

from typing import Any

from codedoggy.tools.gate import validate_args_against_schema
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolError,
    ToolId,
)

_DESC = """\
Call an MCP integration tool.

The `tool_name` must be the qualified `server__tool` name (e.g., `linear__save_issue`).
The `tool_input` must conform exactly to the input schema returned by `search_tool`.
"""

# Path-like keys scanned in tool_input for workspace policy (glue only).
_WRITE_PATH_KEYS = frozenset(
    {
        "file_path",
        "target_file",
        "path",
        "destination",
        "output_path",
        "filepath",
        "write_path",
        "dest",
    }
)
_READ_PATH_KEYS = frozenset(
    {
        "file_path",
        "target_file",
        "path",
        "target_directory",
        "directory",
        "dir",
        "cwd",
        "working_directory",
        "filepath",
        "source",
    }
)
_PATH_KEYS = _WRITE_PATH_KEYS | _READ_PATH_KEYS
_LIST_PATH_KEYS = frozenset({"paths", "files", "file_paths"})


def lookup_mcp_tool_schema(extra: dict[str, Any], tool_name: str) -> dict[str, Any] | None:
    """Resolve input schema from host catalog when available.

    Sources (first hit wins):
      - extra['mcp_tools']: list of {name, parameters|input_schema}
      - extra['mcp_tool_index']: get_schema/schema_for/get/lookup, or .catalog/.tools
    Returns None when catalog missing or tool has no schema (dispatch still allowed).
    """
    name = tool_name.strip()
    tools = extra.get("mcp_tools")
    if isinstance(tools, list):
        for raw in tools:
            if not isinstance(raw, dict):
                continue
            if str(raw.get("name") or "") != name:
                continue
            schema = raw.get("parameters") or raw.get("input_schema")
            if isinstance(schema, dict) and schema:
                return schema
            return None

    index = extra.get("mcp_tool_index")
    if index is None:
        return None

    for meth in ("get_schema", "schema_for", "get", "lookup"):
        fn = getattr(index, meth, None)
        if not callable(fn):
            continue
        try:
            item = fn(name)
        except Exception:  # noqa: BLE001 — host index shapes vary
            continue
        schema = _schema_from_catalog_item(item)
        if schema is not None:
            return schema
        # Explicit schema lookup returned empty — stop; do not invent schema
        if meth in {"get_schema", "schema_for"}:
            return None

    for attr in ("catalog", "tools", "by_name"):
        bag = getattr(index, attr, None)
        if isinstance(bag, dict):
            item = bag.get(name)
            schema = _schema_from_catalog_item(item)
            if schema is not None:
                return schema
        elif isinstance(bag, list):
            for raw in bag:
                if not isinstance(raw, dict):
                    continue
                if str(raw.get("name") or raw.get("tool_name") or "") != name:
                    continue
                return _schema_from_catalog_item(raw)
    return None


def _schema_from_catalog_item(item: Any) -> dict[str, Any] | None:
    if item is None:
        return None
    if isinstance(item, dict):
        schema = (
            item.get("parameters")
            or item.get("input_schema")
            or item.get("schema")
            or (item if item.get("type") == "object" or "properties" in item else None)
        )
        if isinstance(schema, dict) and schema:
            return schema
        return None
    schema = (
        getattr(item, "parameters", None)
        or getattr(item, "input_schema", None)
        or getattr(item, "schema", None)
    )
    if isinstance(schema, dict) and schema:
        return schema
    return None


def enforce_mcp_tool_input_policy(
    tool_input: dict[str, Any],
    ctx: ToolCallContext,
) -> None:
    """Deny path escapes / protected writes when policy is present.

    Scans path-like keys only — not a full MCP sandbox. Missing policy → no-op.
    """
    policy = (ctx.extra or {}).get("policy")
    if policy is None:
        return

    for key, path, is_write in _iter_path_like(tool_input):
        check_r = getattr(policy, "check_read", None)
        if callable(check_r):
            rd = check_r(path)
            if rd is not None and not getattr(rd, "allowed", True):
                raise ToolError(
                    getattr(rd, "reason", None)
                    or f"MCP tool_input path denied ({key}): {path}",
                    code=getattr(rd, "code", None) or "policy_denied",
                )
        if is_write:
            check_w = getattr(policy, "check_write", None)
            if callable(check_w):
                wd = check_w(path)
                if wd is not None and not getattr(wd, "allowed", True):
                    raise ToolError(
                        getattr(wd, "reason", None)
                        or f"MCP tool_input write denied ({key}): {path}",
                        code=getattr(wd, "code", None) or "policy_denied",
                    )


def _iter_path_like(tool_input: dict[str, Any]) -> list[tuple[str, str, bool]]:
    found: list[tuple[str, str, bool]] = []
    for key, val in tool_input.items():
        if key in _PATH_KEYS and isinstance(val, str) and val.strip():
            found.append((key, val.strip(), key in _WRITE_PATH_KEYS))
        elif key in _LIST_PATH_KEYS and isinstance(val, list):
            for i, el in enumerate(val):
                if isinstance(el, str) and el.strip():
                    found.append((f"{key}[{i}]", el.strip(), True))
                elif isinstance(el, dict):
                    found.extend(_iter_path_like(el))
        elif isinstance(val, dict):
            # one nested level (common MCP arg wrappers)
            for nk, nv in val.items():
                if nk in _PATH_KEYS and isinstance(nv, str) and nv.strip():
                    found.append((f"{key}.{nk}", nv.strip(), nk in _WRITE_PATH_KEYS))
    return found


def prepare_mcp_dispatch(
    tool_name: str,
    tool_input: dict[str, Any],
    ctx: ToolCallContext,
) -> None:
    """Gate-like prepare: catalog schema (if any) + path policy. Raises ToolError."""
    extra = ctx.extra or {}
    schema = lookup_mcp_tool_schema(extra, tool_name)
    if schema is not None:
        try:
            validate_args_against_schema(tool_input, schema)
        except ToolError as e:
            raise ToolError.invalid_arguments(
                f"tool_input does not match schema for {tool_name}: {e.message}"
            ) from e
    enforce_mcp_tool_input_policy(tool_input, ctx)


def _normalize_host_mutation_entry(raw: Any) -> dict[str, Any] | None:
    """Coerce one host mutation entry to a dict with a non-empty path, or None."""
    if isinstance(raw, str) and raw.strip():
        return {"path": raw.strip()}
    if not isinstance(raw, dict):
        return None
    path = raw.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    return raw


def _collect_host_mutation_entries(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract mutation entries from all accepted host return shapes.

    Shapes (combined; order preserved, duplicates allowed):
      - mutations: list of {path, ...} or path strings
      - mutation: single {path, ...}
      - mutated_paths: list of path strings or {path, ...}
      - mutated_path: single path string (+ optional top-level before/after/...)
    """
    entries: list[dict[str, Any]] = []

    raw_list = result.get("mutations")
    if isinstance(raw_list, list):
        for item in raw_list:
            norm = _normalize_host_mutation_entry(item)
            if norm is not None:
                entries.append(norm)

    single = result.get("mutation")
    if isinstance(single, dict):
        norm = _normalize_host_mutation_entry(single)
        if norm is not None:
            entries.append(norm)

    paths = result.get("mutated_paths")
    if isinstance(paths, list):
        for item in paths:
            norm = _normalize_host_mutation_entry(item)
            if norm is not None:
                entries.append(norm)

    mp = result.get("mutated_path")
    if isinstance(mp, str) and mp.strip():
        entries.append(
            {
                "path": mp.strip(),
                "before": result.get("before"),
                "after": result.get("after"),
                "is_create": result.get("is_create", False),
                "is_delete": result.get("is_delete", False),
            }
        )

    return entries


def _model_text_from_host_result(result: dict[str, Any]) -> Any:
    """Prefer model-facing text fields from a host envelope dict."""
    for k in ("text", "output", "result", "content"):
        if k in result and result[k] is not None:
            return result[k]
    return result


def _record_host_mutations(
    ctx: ToolCallContext,
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    result: Any,
) -> Any:
    """Capture host-reported mutations for Shadow; return model text.

    Plain string / non-dict returns produce **no** mutation (no false positives).
    When path is present on a structured entry, always call ``ctx.set_mutation``
    (Shadow truth) — policy notes attach via ``set_mutation`` / ``args["_policy"]``.
    """
    if not isinstance(result, dict):
        return result

    for m in _collect_host_mutation_entries(result):
        path = m.get("path")
        if not isinstance(path, str) or not path.strip():
            continue
        # Always record for Shadow; set_mutation attaches _policy when present.
        ctx.set_mutation(
            path=path.strip(),
            before=m.get("before") if isinstance(m.get("before"), (str, type(None))) else None,
            after=m.get("after") if isinstance(m.get("after"), (str, type(None))) else None,
            is_create=bool(m.get("is_create")),
            is_delete=bool(m.get("is_delete")),
            tool_name=tool_name,
            args=tool_input,
        )

    return _model_text_from_host_result(result)


class UseToolTool(Tool):
    def id(self) -> ToolId:
        return ToolId("use_tool")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.UseTool

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="use_tool", description=_DESC.strip())

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": (
                        "The qualified name of the integration tool to call "
                        '(e.g., "linear__save_issue").'
                    ),
                },
                "tool_input": {
                    "type": "object",
                    "description": "Arguments to pass to the tool as a JSON object.",
                    "additionalProperties": True,
                },
            },
            "required": ["tool_name", "tool_input"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        name = args.get("tool_name")
        if not isinstance(name, str) or not name.strip():
            raise ToolError.invalid_arguments("tool_name is required")
        name = name.strip()
        tool_input = args.get("tool_input")
        if tool_input is None:
            tool_input = {}
        if not isinstance(tool_input, dict):
            raise ToolError.invalid_arguments("tool_input must be a JSON object")

        # Grok: native-tool correction when name looks like a built-in
        if "__" not in name and not name.startswith("MCP:"):
            return (
                f"Error: `{name}` is not a valid MCP tool name. "
                f"If this is a built-in tool, call it directly (not via use_tool)."
            )

        # Prepare (schema + policy) before host dispatch — gate spirit
        prepare_mcp_dispatch(name, tool_input, ctx)

        dispatch = (ctx.extra or {}).get("mcp_dispatch")
        if not callable(dispatch):
            raise ToolError(
                "MCP dispatch is not available (host must provide extra['mcp_dispatch']).",
                code="mcp_dispatch_missing",
            )
        try:
            result = dispatch(name, tool_input)
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"Failed to call {name}: {e}", code="mcp_error") from e

        result = _record_host_mutations(
            ctx, tool_name=name, tool_input=tool_input, result=result
        )

        if result is None:
            return "(empty MCP result)"
        text = result if isinstance(result, str) else str(result)
        if len(text) > 40_000:
            return text[:39_960] + "\n… [MCP output truncated]"
        return text
