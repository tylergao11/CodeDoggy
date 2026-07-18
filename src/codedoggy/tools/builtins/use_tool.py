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

**Host mutation contract (for write tools):**

If ``mcp_dispatch`` returns only a plain string (or unstructured blob),
workspace side effects are not recorded on the tool context. Hosts that
mutate files SHOULD return structured mutations so ``set_mutation`` can run.

Accepted return shapes (any combination; relative paths preferred)::

  # preferred — multi path with optional before/after
  {"text": "...", "mutations": [
      {"path": "rel/a.py", "before": "...", "after": "...",
       "is_create": false, "is_delete": false}, ...]}

  # single mutation object
  {"output": "...", "mutation": {"path": "rel/a.py", "after": "..."}}

  # minimal path list
  {"result": "...", "mutated_paths": ["rel/a.py", "rel/b.py"]}

  # single path shortcut
  {"text": "...", "mutated_path": "rel/a.py", "before": None, "after": "..."}

When path is present on a structured entry, always call ``ctx.set_mutation``
(policy notes attach via ``args["_policy"]`` when policy is present).
Pre-dispatch tool_input paths are still gated by
``enforce_mcp_tool_input_policy``.

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

from codedoggy.tools.grok_build.use_tool_logic import (
    USE_TOOL_DESCRIPTION,
    UseToolInput,
    UseToolParams,
    gateway_result_is_error,
    gateway_result_to_text,
    native_tool_correction_message,
    normalize_mcp_arguments,
    unqualified_mcp_name_message,
)

_DESC = USE_TOOL_DESCRIPTION

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
    """Resolve input schema from the host catalog / Grok ToolIndex."""
    name = tool_name.strip()
    # Auto-build BM25 ToolIndex from catalog (Grok shell injects ToolIndex)
    try:
        from codedoggy.tools.mcp.tool_index import ensure_mcp_tool_index

        ensure_mcp_tool_index(extra)
    except Exception:  # noqa: BLE001
        pass

    tools = extra.get("mcp_tools")
    if isinstance(tools, list):
        for raw in tools:
            if not isinstance(raw, dict):
                continue
            if str(raw.get("name") or raw.get("tool_name") or "") != name:
                continue
            return _schema_from_catalog_item(raw)

    index = extra.get("mcp_tool_index")
    if index is None:
        return None

    for meth in ("get_schema", "schema_for", "get", "lookup"):
        fn = getattr(index, meth, None)
        if not callable(fn):
            inner = getattr(index, "index", None)
            fn = getattr(inner, meth, None) if inner is not None else None
        if not callable(fn):
            continue
        try:
            item = fn(name)
        except Exception:  # noqa: BLE001 — host index shapes vary
            continue
        schema = _schema_from_catalog_item(item)
        if schema is not None:
            return schema
        if meth in {"get_schema", "schema_for"}:
            return None

    for attr in ("catalog", "tools", "by_name"):
        bag = getattr(index, attr, None)
        if isinstance(bag, dict):
            schema = _schema_from_catalog_item(bag.get(name))
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
    """Capture host-reported mutations on the tool context; return model text.

    Plain string / non-dict returns produce **no** mutation (no false positives).
    When path is present on a structured entry, always call ``ctx.set_mutation``.
    """
    if not isinstance(result, dict):
        return result

    for m in _collect_host_mutation_entries(result):
        path = m.get("path")
        if not isinstance(path, str) or not path.strip():
            continue
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
        # Grok UseToolInput doc comments → schemars descriptions (exact).
        return {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": (
                        "The qualified name of the integration tool to call "
                        '(e.g., "linear__save_issue"). '
                        "Must be a tool previously discovered via `search_tool`."
                    ),
                },
                "tool_input": {
                    "type": "object",
                    "description": (
                        "The arguments to pass to the tool, as a JSON object. "
                        "Use the parameter schema returned by `search_tool` "
                        "to construct this."
                    ),
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
        tool_input = normalize_mcp_arguments(tool_input)
        if not isinstance(tool_input, dict):
            raise ToolError.invalid_arguments("tool_input must be a JSON object")
        # Grok UseToolInput surface
        _ = UseToolInput(tool_name=name, tool_input=tool_input)

        extra = ctx.extra or {}
        # Grok UseToolParams.native_tool_correction (default true)
        correction_raw = extra.get("native_tool_correction")
        params = UseToolParams(
            native_tool_correction=True if correction_raw is None else bool(correction_raw)
        )
        correction = params.native_tool_correction
        native_names = extra.get("enabled_native_tool_names") or extra.get(
            "native_tool_names"
        )
        is_native = False
        if correction and isinstance(native_names, (set, list, tuple, frozenset)):
            is_native = name in set(native_names)
        # Also treat known finalized short-ids as native when no catalog
        if correction and not is_native and "__" not in name:
            # soft native guess: host tools without server__ prefix
            is_native = bool(extra.get("treat_unqualified_as_native", False))

        if "__" not in name and not name.startswith("MCP:"):
            if is_native or name in {
                "read_file",
                "search_replace",
                "write",
                "grep",
                "list_dir",
                "run_terminal_command",
                "run_terminal_cmd",
                "spawn_subagent",
                "skill",
            }:
                raise ToolError.invalid_arguments(native_tool_correction_message(name))
            raise ToolError.invalid_arguments(
                unqualified_mcp_name_message(name, "search_tool")
            )

        prepare_mcp_dispatch(name, tool_input, ctx)

        # Ensure BM25 index exists for schema lookup path (same as search_tool)
        try:
            from codedoggy.tools.mcp.tool_index import ensure_mcp_tool_index

            ensure_mcp_tool_index(extra)
        except Exception:  # noqa: BLE001
            pass

        dispatch = extra.get("mcp_dispatch")
        if not callable(dispatch):
            raise ToolError(
                "MCP dispatch is not available (host must provide extra['mcp_dispatch']).",
                code="mcp_dispatch_missing",
            )
        try:
            result = dispatch(name, tool_input)
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"Failed to call {name}: {e}", code="mcp_error") from e

        # Grok gateway content[] shape
        if isinstance(result, dict) and (
            "content" in result or "isError" in result or "is_error" in result
        ):
            text = gateway_result_to_text(result)
            if gateway_result_is_error(result):
                return f"Error from MCP tool {name}: {text}"
            result = text

        result = _record_host_mutations(
            ctx, tool_name=name, tool_input=tool_input, result=result
        )

        if result is None:
            return "(empty MCP result)"
        text = result if isinstance(result, str) else str(result)
        if len(text) > 40_000:
            return text[:39_960] + "\n… [MCP output truncated]"
        return text
