"""memory_get — Grok MemoryGetImpl contract.

Ported from:
  grok-build/crates/codegen/xai-grok-tools/src/implementations/memory/get_tool.rs
  grok-build/crates/codegen/xai-grok-tools/src/implementations/memory/types.rs
  (line-slice semantics mirror xai-grok-memory storage::read_file)

Function map:
  MemoryGetImpl::run              → MemoryGetTool.run
  format_with_line_numbers        → format_with_line_numbers
  MemoryGetInput {path,from,lines}→ parameters_schema + arg parse
  MemoryStorage::read_file slice  → _slice_memory_content

Wire schema (Grok MemoryGetInput / types.rs):
  - path: string (required in Grok)
  - from: optional 0-based start line (default: beginning)
  - lines: optional max lines (default: all — no read_file soft cap)

Doggy vs Grok MemoryBackend:
  - Grok: host injects Arc<dyn MemoryBackend>; tool calls memory.get(path, from, lines).
  - Doggy: host injects MemoryStore via extra['memory_store']; this tool reads files
    from disk (store-relative or absolute path). We do NOT invent a MemoryBackend
    trait or fake search-as-get. memory_search is out of scope here.
  - Doggy convenience: target=memory|user when path is omitted.

Soft disable (exact Grok get_tool.rs when backend missing):
  "Memory is not enabled. Use --experimental-memory to enable."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codedoggy.memory.store import MemoryStore
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolError,
    ToolId,
)

# Exact Grok description_template (get_tool.rs MemoryGetImpl)
_DESC = """\
Read a memory file by path. Returns the file content with line numbers, optionally \
limited to a range of lines.

Use after `memory_search` returns a relevant result and you need the full context \
around a snippet, or to read a specific MEMORY.md file in full.

Line numbers are 1-based and match the line offsets accepted by the `from` parameter, \
so targeted follow-up reads or edits can reference exact positions.
"""

# Exact Grok soft text when MemoryBackend / memory is not available
_DISABLED = "Memory is not enabled. Use --experimental-memory to enable."


class MemoryGetTool(Tool):
    def id(self) -> ToolId:
        return ToolId("memory_get")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.MemoryGet

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="memory_get", description=_DESC.strip())

    def parameters_schema(self) -> dict[str, Any]:
        # Grok MemoryGetInput (+ Doggy target convenience)
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the memory file to read.",
                },
                "from": {
                    "type": "integer",
                    "description": "0-based start line (default: beginning of file).",
                    "minimum": 0,
                },
                "lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to return (default: all).",
                    "minimum": 1,
                },
                "target": {
                    "type": "string",
                    "enum": ["memory", "user"],
                    "description": "Curated store shortcut when path is omitted (Doggy).",
                },
            },
            # Grok requires path; Doggy allows target=memory|user instead
            "required": [],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        store = (ctx.extra or {}).get("memory_store")
        if not isinstance(store, MemoryStore):
            # Grok get_tool.rs: soft text when MemoryBackend absent (not ToolError)
            return _DISABLED

        path = _resolve_path(store, args)
        # Grok formats input.path as given by the model
        display_path = str(args.get("path") or path)
        if not path.is_file():
            raise ToolError(f"Memory file not found: {path}", code="not_found")

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            # Grok: "memory get failed: {e}"
            raise ToolError(f"memory get failed: {e}", code="io_error") from e

        from_idx, from_label = _parse_from(args.get("from"))
        lines_n, limit_label = _parse_lines(args.get("lines"))

        # Backend returns the (optionally sliced) body; tool numbers it
        content = _slice_memory_content(text, from_idx, lines_n)

        # Grok: total_lines = content.lines().count() on returned body
        total_lines = _rust_lines_count(content)
        first_line_num = from_idx + 1
        numbered = format_with_line_numbers(content, first_line_num)

        # Grok get_tool.rs output format exactly:
        #   **File:** {path}
        #   **Lines:** {total_lines} (from: {from|start}, limit: {lines|all})
        #
        #   {numbered}
        return (
            f"**File:** {display_path}\n"
            f"**Lines:** {total_lines} (from: {from_label}, limit: {limit_label})\n\n"
            f"{numbered}"
        )


def format_with_line_numbers(content: str, first_line_num: int) -> str:
    """Grok format_with_line_numbers (get_tool.rs).

    Uses split('\\n') rather than lines()/splitlines so content ending with a
    newline (``"a\\n"``) emits a trailing blank numbered line, matching read_file.
    ``first_line_num`` is 1-based (accounts for ``from`` offset).
    """
    if content == "":
        return ""
    parts = content.split("\n")
    return "\n".join(f"{first_line_num + i}→{line}" for i, line in enumerate(parts))


def _slice_memory_content(text: str, from_idx: int, lines_n: int | None) -> str:
    """Mirror MemoryStorage::read_file line selection (xai-grok-memory storage.rs).

    - from: 0-based start (default 0)
    - lines: max lines (default all)
    - Rust ``str::lines()`` for ranged reads (trailing final newline does not
      produce an extra empty element); full file with from=0 and no limit returns
      the raw content as-is (preserving trailing newline).
    """
    if lines_n is None and from_idx == 0:
        return text
    # Rust lines() ≈ splitlines(): no trailing empty for final \\n
    file_lines = text.splitlines()
    if lines_n is None:
        selected = file_lines[from_idx:]
    else:
        selected = file_lines[from_idx : from_idx + lines_n]
    return "\n".join(selected)


def _rust_lines_count(content: str) -> int:
    """Rust ``content.lines().count()`` — trailing final newline does not add a line."""
    if content == "":
        return 0
    return len(content.splitlines())


def _parse_from(from_raw: Any) -> tuple[int, str]:
    """Return (0-based index, label for header: 'start' | str(n))."""
    if from_raw is None:
        return 0, "start"
    try:
        from_idx = max(0, int(from_raw))
    except (TypeError, ValueError) as e:
        raise ToolError.invalid_arguments(f"invalid from: {from_raw}") from e
    return from_idx, str(from_idx)


def _parse_lines(lines_raw: Any) -> tuple[int | None, str]:
    """Return (max lines or None=all, label for header: 'all' | str(n))."""
    if lines_raw is None:
        return None, "all"
    try:
        n = max(1, int(lines_raw))
    except (TypeError, ValueError) as e:
        raise ToolError.invalid_arguments(f"invalid lines: {lines_raw}") from e
    return n, str(n)


def _resolve_path(store: MemoryStore, args: dict[str, Any]) -> Path:
    """Resolve path under MemoryStore (Doggy host injection, not Grok backend)."""
    raw_path = args.get("path")
    target = args.get("target")
    if isinstance(raw_path, str) and raw_path.strip():
        p = Path(raw_path.strip())
        if p.is_absolute():
            return p.resolve()
        return (store.memory_dir / p).resolve()
    if target is None:
        target = "memory"
    if target not in {"memory", "user"}:
        raise ToolError.invalid_arguments(
            "path is required (or target=memory|user as Doggy convenience)"
        )
    name = "MEMORY.md" if target == "memory" else "USER.md"
    return (store.memory_dir / name).resolve()
