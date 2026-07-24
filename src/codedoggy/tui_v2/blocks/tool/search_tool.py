"""SearchToolCallBlock — port of ``blocks/tool/search_tool.rs``.

Header: ``Search Tools `` + query + `` (N results)``.
Expanded: numbered discovered MCP tools with description.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from codedoggy.tui_v2.blocks.tool.common import (
    S_BOLD,
    S_COMMAND,
    S_DIM,
    S_ERROR,
    S_MUTED,
    S_PRIMARY,
    Rows,
    arg_str,
    empty_row,
    is_running,
    row,
    truncate_str,
)

HEADER = "Search Tools "


@dataclass
class DiscoveredTool:
    name: str
    server: str
    description: str
    score: float = 0.0


def _titleize(seg: str) -> str:
    """Light title-case for MCP segments (Grok ``mcp_titleize_segment`` spirit)."""
    if not seg:
        return seg
    parts = seg.replace("-", " ").replace("_", " ").split()
    return " ".join(p[:1].upper() + p[1:] if p else "" for p in parts)


def discovered_tool_action(tool: DiscoveredTool) -> str:
    if tool.server and tool.name.startswith(tool.server + "__"):
        return tool.name[len(tool.server) + 2 :]
    if "__" in tool.name:
        return tool.name.split("__", 1)[1]
    return tool.name


def parse_discovered_tools(result: str) -> list[DiscoveredTool]:
    text = (result or "").strip()
    if not text:
        return []
    data = None
    if text.startswith("[") or text.startswith("{"):
        try:
            data = json.loads(text)
        except Exception:  # noqa: BLE001
            data = None
    if isinstance(data, dict):
        data = data.get("tools") or data.get("results") or data.get("items") or []
    if isinstance(data, list):
        out: list[DiscoveredTool] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("tool") or item.get("id") or "")
            if not name:
                continue
            server = str(item.get("server") or "")
            if not server and "__" in name:
                server = name.split("__", 1)[0]
            desc = str(item.get("description") or item.get("desc") or "")
            try:
                score = float(item.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            out.append(
                DiscoveredTool(
                    name=name, server=server, description=desc, score=score
                )
            )
        return out
    # Line fallback: "server__tool — description"
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if " — " in ln:
            name, desc = ln.split(" — ", 1)
        elif " - " in ln:
            name, desc = ln.split(" - ", 1)
        else:
            name, desc = ln, ""
        server = name.split("__", 1)[0] if "__" in name else ""
        out.append(DiscoveredTool(name=name.strip(), server=server, description=desc.strip()))
    return out[:30]


def paint_search_tool(
    arguments: dict,
    result: str,
    *,
    width: int,
    collapsed: bool,
    status: str,
    selected: bool = False,
) -> Rows:
    query = arg_str(arguments, "query", "q", "search", default="") or "…"
    running = is_running(status)
    muted = collapsed and not running
    tools = [] if running else parse_discovered_tools(result)
    failed = status.lower() in {"failed", "error"}
    count = len(tools)

    prefix = HEADER
    p_style = S_MUTED + " bold" if muted else S_BOLD
    q_style = S_MUTED if muted else S_COMMAND
    if selected:
        p_style = f"{p_style} reverse"
        q_style = f"{q_style} reverse"

    s = "" if count == 1 else "s"
    suffix = f" ({count} result{s})" if count or not running else ""
    budget = max(1, width - len(prefix) - len(suffix))
    shown_q = truncate_str(query, budget)
    parts: list[tuple[str, str]] = [(p_style, prefix), (q_style, shown_q)]
    if suffix:
        parts.append((S_DIM if not muted else S_MUTED, suffix))
    rows: Rows = [row(*parts)]

    if collapsed:
        return rows

    if failed and result:
        rows.append(empty_row())
        for err_line in result.splitlines()[:8]:
            rows.append(row((S_ERROR, err_line)))
        return rows

    if not tools:
        rows.append(empty_row())
        rows.append(row((S_MUTED, "  (no tools found)")))
        return rows

    for i, tool in enumerate(tools[:20]):
        rows.append(empty_row())
        action = _titleize(discovered_tool_action(tool))
        server = _titleize(tool.server) if tool.server else ""
        label = f"{action}  {server}".rstrip() if server else action
        rows.append(
            row(
                (S_MUTED, f"  {i + 1}. "),
                (S_PRIMARY, truncate_str(label, max(8, width - 6))),
            )
        )
        if tool.description:
            rows.append(
                row(
                    (S_MUTED, "     "),
                    (
                        S_MUTED,
                        truncate_str(tool.description.replace("\n", " "), max(8, width - 6)),
                    ),
                )
            )

    return rows


__all__ = [
    "DiscoveredTool",
    "discovered_tool_action",
    "paint_search_tool",
    "parse_discovered_tools",
]
