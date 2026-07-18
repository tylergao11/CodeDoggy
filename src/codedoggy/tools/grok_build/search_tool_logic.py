"""search_tool pure helpers — source port from Grok.

Ported from:
  implementations/search_tool/mod.rs
    MAX_MCP_DESCRIPTION_LENGTH, truncate_description, sanitize_description
    fingerprint_servers (FNV-1a), build_server_reminder, build_delta_reminder
    format_compaction_server_line
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from codedoggy.tools.mcp.types import ServerSummary as ServerSummary  # Grok tool_index

MAX_MCP_DESCRIPTION_LENGTH: int = 2048
TRUNCATION_SUFFIX = "\u2026 [truncated]"

__all__ = [
    "MAX_MCP_DESCRIPTION_LENGTH",
    "NO_MCP_CONFIGURED_JSON",
    "NO_MCP_CONFIGURED_NOTE",
    "SEARCH_TOOL_DESCRIPTION",
    "SearchToolInput",
    "ServerSummary",
    "build_delta_reminder",
    "build_server_reminder",
    "fingerprint_servers",
    "format_compaction_server_line",
    "format_search_results_json",
    "format_search_snapshot_response",
    "sanitize_description",
    "truncate_description",
]

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x00000100000001B3


def truncate_description(s: str) -> str:
    """Grok ``truncate_description`` — char-count budget + suffix."""
    if not s:
        return s
    n = sum(1 for _ in s)
    if len(s) <= MAX_MCP_DESCRIPTION_LENGTH or n <= MAX_MCP_DESCRIPTION_LENGTH:
        return s
    budget = MAX_MCP_DESCRIPTION_LENGTH - len(TRUNCATION_SUFFIX)
    return "".join(list(s)[:budget]) + TRUNCATION_SUFFIX


def sanitize_description(s: str) -> str:
    """Grok ``sanitize_description`` — collapse whitespace / newlines."""
    parts: list[str] = []
    for line in s.replace("\r", "\n").split("\n"):
        parts.extend(line.split())
    return " ".join(parts)


def fnv1a_hash(data: bytes | str) -> int:
    """Grok portable FNV-1a (u64)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    h = _FNV_OFFSET
    for byte in data:
        h ^= byte
        h = (h * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def hash_value(val: Any) -> int:
    if isinstance(val, str):
        return fnv1a_hash(val)
    if isinstance(val, (list, tuple)):
        return fnv1a_hash("\0".join(str(x) for x in val))
    return fnv1a_hash(str(val))


@dataclass
class SearchToolInput:
    """Grok ``SearchToolInput`` — wire schema for ``search_tool``."""

    query: str
    limit: int | None = 5


def fingerprint_servers(
    servers: list[ServerSummary],
) -> dict[str, tuple[int, int, int]]:
    """Grok ``fingerprint_servers`` → name → (count, desc_hash, names_hash)."""
    out: dict[str, tuple[int, int, int]] = {}
    for s in servers:
        out[s.name] = (
            s.tool_count,
            hash_value(s.description or ""),
            hash_value(s.tool_names or []),
        )
    return out


def format_compaction_server_line(name: str, count: int, desc: str | None) -> str:
    tool_word = "tool" if count == 1 else "tools"
    if desc and desc.strip():
        return f"- {name} ({count} {tool_word}): {desc}\n"
    return f"- {name} ({count} {tool_word})\n"


def build_server_reminder(servers: list[ServerSummary]) -> str | None:
    if not servers:
        return None
    text = "Connected MCP servers:\n"
    for server in servers:
        desc = None
        if server.description:
            desc = truncate_description(sanitize_description(server.description))
        text += format_compaction_server_line(server.name, server.tool_count, desc)
    return text


def build_delta_reminder(
    old: dict[str, tuple[int, int, int]],
    new_summaries: list[ServerSummary],
) -> str | None:
    new_map = fingerprint_servers(new_summaries)
    added = [s for s in new_summaries if s.name not in old]
    updated = [
        s
        for s in new_summaries
        if s.name in old and old[s.name] != new_map.get(s.name)
    ]
    removed = [name for name in old if name not in new_map]
    if not added and not updated and not removed:
        return None
    parts: list[str] = []
    if added:
        parts.append("MCP server(s) connected:")
        for s in added:
            d = (
                truncate_description(sanitize_description(s.description))
                if s.description
                else None
            )
            parts.append(format_compaction_server_line(s.name, s.tool_count, d).rstrip())
    if updated:
        parts.append("MCP server(s) updated:")
        for s in updated:
            d = (
                truncate_description(sanitize_description(s.description))
                if s.description
                else None
            )
            parts.append(format_compaction_server_line(s.name, s.tool_count, d).rstrip())
    if removed:
        s_word = "s" if len(removed) != 1 else ""
        parts.append(f"MCP server{s_word} disconnected: {', '.join(removed)}")
    return "\n".join(parts)


def format_search_snapshot_response(
    results: list[Any],
    *,
    total_hidden_tools: int = 0,
    is_ready: bool = True,
) -> str:
    """Grok ``search_tool`` run() response — group by server, BM25 order.

    Shape from search_tool/mod.rs::run after grouping::

      {
        "results": [ {"server": "...", "tools": [
            {"tool_name", "description", "score", "input_schema"}, ...
        ]}, ...],
        "total_hidden_tools": N,
        "status": "ready" | "partial",
        "note": null | "Some MCP servers are still connecting..."
      }
    """
    # Group by server_name, preserve BM25 order within group; group score = first tool score
    groups: list[tuple[str, float, list[dict[str, Any]]]] = []
    for r in results:
        if isinstance(r, dict):
            tool_name = r.get("tool_name") or r.get("name") or ""
            server = str(r.get("server_name") or r.get("server") or "mcp")
            desc = str(r.get("description") or "")
            score = float(r.get("score") or 0)
            schema = r.get("input_schema") or r.get("parameters") or {}
        else:
            tool_name = getattr(r, "tool_name", getattr(r, "name", "")) or ""
            server = str(getattr(r, "server_name", getattr(r, "server", "")) or "mcp")
            desc = str(getattr(r, "description", "") or "")
            score = float(getattr(r, "score", 0) or 0)
            schema = getattr(r, "input_schema", {}) or {}
        tool_json = {
            "tool_name": tool_name,
            "description": truncate_description(desc),
            "score": score,
            "input_schema": schema if isinstance(schema, dict) else {},
        }
        found = False
        for g in groups:
            if g[0] == server:
                g[2].append(tool_json)
                found = True
                break
        if not found:
            groups.append((server, score, [tool_json]))
    groups.sort(key=lambda g: -g[1])
    result_groups = [{"server": s, "tools": tools} for s, _, tools in groups]
    status = "ready" if is_ready else "partial"
    note = (
        None
        if is_ready
        else "Some MCP servers are still connecting. Results may be incomplete."
    )
    response = {
        "results": result_groups,
        "total_hidden_tools": int(total_hidden_tools),
        "status": status,
        "note": note,
    }
    return json.dumps(response, ensure_ascii=False, indent=2)


def format_search_results_json(
    results: list[dict[str, Any]],
    *,
    total_hidden_tools: int = 0,
    note: str | None = None,
    is_ready: bool = True,
) -> str:
    """Back-compat wrapper → Grok grouped snapshot shape."""
    if note and not is_ready:
        is_ready = False
    # Flat list path used by catalog filter
    if results and "server" in (results[0] if results else {}) and "tools" in (
        results[0] if results else {}
    ):
        # already grouped
        payload = {
            "results": results,
            "total_hidden_tools": total_hidden_tools,
            "status": "ready" if is_ready else "partial",
            "note": note
            if note is not None
            else (
                None
                if is_ready
                else "Some MCP servers are still connecting. Results may be incomplete."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return format_search_snapshot_response(
        results,
        total_hidden_tools=total_hidden_tools,
        is_ready=is_ready if note is None else (note is None),
    )


SEARCH_TOOL_DESCRIPTION = (
    "Search for MCP tools by keyword and retrieve their input schemas.\n\n"
    'If status is "partial", some servers may still be connecting.'
)

NO_MCP_CONFIGURED_NOTE = (
    "No integration tools are configured. MCP servers are not connected."
)

NO_MCP_CONFIGURED_JSON = {
    "results": [],
    "total_hidden_tools": 0,
    "note": NO_MCP_CONFIGURED_NOTE,
}
