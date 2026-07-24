"""UseToolCallBlock — port of ``blocks/tool/use_tool.rs``.

Header: ``Server Action`` (titleized MCP segments) or bare tool name.
Expanded: key/value inputs + truncated output panel.
"""

from __future__ import annotations

import json

from codedoggy.tui_v2.blocks.tool.common import (
    S_BOLD,
    S_COMMAND,
    S_DIM,
    S_ERROR,
    S_MUTED,
    S_PANEL,
    S_PRIMARY,
    Rows,
    arg_str,
    empty_row,
    is_running,
    result_lines,
    row,
    truncate_str,
)

MAX_INLINE = 10
TRUNCATED_INLINE = 3
DELIMITER = "__"


def _titleize(seg: str) -> str:
    parts = seg.replace("-", " ").replace("_", " ").split()
    return " ".join(p[:1].upper() + p[1:] if p else "" for p in parts)


def split_mcp_name(tool_name: str) -> tuple[str, str]:
    """Return (server_title, action_title). Empty server if unqualified."""
    name = (tool_name or "").strip()
    if DELIMITER in name:
        server, action = name.split(DELIMITER, 1)
        return _titleize(server), _titleize(action)
    return "", _titleize(name) if name else "Tool"


def _flatten_args(arguments: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    # Prefer nested tool_input
    raw = arguments.get("tool_input") or arguments.get("input") or arguments
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001
            raw = {"input": raw}
    if not isinstance(raw, dict):
        return [("input", str(raw))]
    for k, v in raw.items():
        if k in {"tool_name", "name"}:
            continue
        if isinstance(v, (dict, list)):
            try:
                s = json.dumps(v, ensure_ascii=False)
            except Exception:  # noqa: BLE001
                s = str(v)
        else:
            s = str(v)
        if len(s) > 200:
            s = s[:199] + "…"
        out.append((str(k), s))
    return out


def paint_use_tool(
    arguments: dict,
    result: str,
    *,
    width: int,
    collapsed: bool,
    status: str,
    selected: bool = False,
    tool_name: str | None = None,
) -> Rows:
    name = (
        tool_name
        or arg_str(arguments, "tool_name", "name", default="")
        or "tool"
    )
    # Qualified name may be the dispatch key itself (linear__save_issue)
    if DELIMITER not in name and DELIMITER in (tool_name or ""):
        name = tool_name or name

    running = is_running(status)
    muted = collapsed and not running
    failed = status.lower() in {"failed", "error"}
    server, action = split_mcp_name(name)

    p_style = S_MUTED + " bold" if muted else S_BOLD
    a_style = S_MUTED if muted else S_COMMAND
    if selected:
        p_style = f"{p_style} reverse"
        a_style = f"{a_style} reverse"

    if not server:
        display = truncate_str(action, max(1, width))
        rows: Rows = [row((p_style, display))]
    else:
        prefix = f"{server} "
        budget = max(1, width - len(prefix))
        rows = [row((p_style, prefix), (a_style, truncate_str(action, budget)))]

    if collapsed:
        return rows

    args = _flatten_args(arguments if isinstance(arguments, dict) else {})
    if args:
        rows.append(empty_row())
        for k, v in args[:12]:
            rows.append(
                row(
                    (S_MUTED, f"  {k}: "),
                    (S_PRIMARY, truncate_str(v, max(4, width - len(k) - 4))),
                )
            )

    if failed and result:
        rows.append(empty_row())
        for err_line in result.splitlines()[:8]:
            rows.append(row((S_ERROR, err_line)))
        return rows

    lines = result_lines(result)
    if not lines:
        return rows

    rows.append(empty_row())
    max_n = TRUNCATED_INLINE if collapsed else MAX_INLINE
    shown = lines[:max_n]
    for ln in shown:
        text = truncate_str(ln, max(4, width - 2))
        rows.append(row((f"{S_PANEL} {S_PRIMARY}", f"  {text}")))
    if len(lines) > max_n:
        rem = len(lines) - max_n
        rows.append(
            row(
                (
                    f"{S_PANEL} {S_DIM}",
                    f"  ... ({rem} more lines, press Enter to view)",
                )
            )
        )
    return rows


__all__ = ["paint_use_tool", "split_mcp_name"]
