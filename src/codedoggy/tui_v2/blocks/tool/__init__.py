"""Tool-call scrollback block painters (Grok pager port).

Dispatch entry::

    paint_tool(
        name, arguments, result, *, width, collapsed, status,
        selected=False, truncated=False,
    ) -> list[Row]   # each Row is StyleAndTextTuples + trailing newline
"""

from __future__ import annotations

from codedoggy.tui_v2.blocks.tool.common import Rows, text_of
from codedoggy.tui_v2.blocks.tool.edit import (
    DiffLine,
    DiffRenderConfig,
    ChangeTag,
    build_hunks_from_strings,
    diff_outputs_to_string,
    paint_edit,
    render_diff_hunk,
    render_diff_hunks,
)
from codedoggy.tui_v2.blocks.tool.execute import paint_execute
from codedoggy.tui_v2.blocks.tool.hook import (
    ToolCallHookData,
    hooks_inline_suffix,
    paint_hook_body,
    parse_hooks_from_meta,
)
from codedoggy.tui_v2.blocks.tool.lifecycle import paint_lifecycle
from codedoggy.tui_v2.blocks.tool.list_dir import paint_list_dir
from codedoggy.tui_v2.blocks.tool.memory_search import paint_memory_search
from codedoggy.tui_v2.blocks.tool.other import paint_other
from codedoggy.tui_v2.blocks.tool.read import paint_read
from codedoggy.tui_v2.blocks.tool.search import paint_search
from codedoggy.tui_v2.blocks.tool.search_tool import paint_search_tool
from codedoggy.tui_v2.blocks.tool.use_tool import paint_use_tool
from codedoggy.tui_v2.blocks.tool.web_fetch import paint_web_fetch
from codedoggy.tui_v2.blocks.tool.web_search import paint_web_search

# VerbGroupKind-style classification for fold/label use.
# Keep in sync with verb_group.classify_verb (execute/shell not groupable).
VERB_GROUPABLE = frozenset(
    {
        # file
        "read",
        "read_file",
        # dir
        "list_dir",
        "ls",
        # search
        "grep",
        "search",
        "glob",
        "rg",
        "memory_search",
        "search_tool",
        # web_fetch
        "web_fetch",
        "fetch",
        # web_search
        "web_search",
        "x_keyword_search",
        "x_semantic_search",
    }
)


_LIFECYCLE_EVENTS = frozenset(
    {
        "session_start",
        "session_end",
        "user_prompt_submit",
        "stop",
        "subagent_start",
        "subagent_stop",
        "pre_tool_use",
        "post_tool_use",
    }
)


def paint_tool(
    name: str,
    arguments: dict,
    result: str,
    *,
    width: int,
    collapsed: bool,
    status: str,
    selected: bool = False,
    truncated: bool = False,
    hooks: ToolCallHookData | None = None,
    meta: dict | None = None,
) -> Rows:
    """Paint a tool call block by Grok tool kind.

    Parameters
    ----------
    name:
        Tool name (e.g. ``read_file``, ``run_terminal_cmd``, ``search_replace``).
    arguments:
        Tool input dict from Doggy / ACP projection.
    result:
        Tool result text (stdout, file content, grep output, …).
    width:
        Content width in columns.
    collapsed:
        True → header-only (DisplayMode::Collapsed).
    status:
        ``running`` / ``completed`` / ``failed`` / …
    selected:
        Selection highlight on the header.
    truncated:
        When not collapsed, for read/execute: first 5 + last 3 body lines.
        Default False → full expanded body (Grok Expanded).
    hooks / meta:
        Optional hook data (inline ``[hooks: N]`` + expanded body).
    """
    key = (name or "").strip().lower()
    args = arguments if isinstance(arguments, dict) else {}
    res = result if isinstance(result, str) else ("" if result is None else str(result))
    w = max(1, int(width))
    hook_data = hooks or parse_hooks_from_meta(meta)

    if key in _LIFECYCLE_EVENTS or key.startswith("lifecycle:"):
        label = name[len("lifecycle:") :] if key.startswith("lifecycle:") else (name or key)
        return paint_lifecycle(
            label, width=w, collapsed=collapsed, status=status, selected=selected
        )

    if key in {"run_terminal_command", "run_terminal_cmd", "bash", "shell", "execute"}:
        rows = paint_execute(
            args,
            res,
            width=w,
            collapsed=collapsed,
            status=status,
            selected=selected,
            truncated=truncated,
        )
    elif key in {"read_file", "read"}:
        rows = paint_read(
            args,
            res,
            width=w,
            collapsed=collapsed,
            status=status,
            selected=selected,
            truncated=truncated,
        )
    elif key in {"search_replace", "edit", "apply_patch", "strreplace"}:
        rows = paint_edit(
            args, res, width=w, collapsed=collapsed, status=status, selected=selected
        )
    elif key == "write":
        rows = paint_edit(
            args,
            res,
            width=w,
            collapsed=collapsed,
            status=status,
            selected=selected,
            is_write=True,
        )
    elif key in {"list_dir", "ls"}:
        rows = paint_list_dir(
            args, res, width=w, collapsed=collapsed, status=status, selected=selected
        )
    elif key in {"grep", "search", "glob"}:
        rows = paint_search(
            args, res, width=w, collapsed=collapsed, status=status, selected=selected
        )
    elif key in {"web_fetch", "fetch"}:
        rows = paint_web_fetch(
            args, res, width=w, collapsed=collapsed, status=status, selected=selected
        )
    elif key in {"web_search", "x_keyword_search", "x_semantic_search"}:
        rows = paint_web_search(
            args, res, width=w, collapsed=collapsed, status=status, selected=selected
        )
    elif key in {"memory_search"}:
        rows = paint_memory_search(
            args, res, width=w, collapsed=collapsed, status=status, selected=selected
        )
    elif key in {"search_tool"}:
        rows = paint_search_tool(
            args, res, width=w, collapsed=collapsed, status=status, selected=selected
        )
    elif key in {"use_tool"} or "__" in key:
        rows = paint_use_tool(
            args,
            res,
            width=w,
            collapsed=collapsed,
            status=status,
            selected=selected,
            tool_name=name or key,
        )
    else:
        rows = paint_other(
            name or key,
            args,
            res,
            width=w,
            collapsed=collapsed,
            status=status,
            selected=selected,
        )

    return _attach_hooks(rows, hook_data, collapsed=collapsed, width=w)


def _attach_hooks(
    rows: Rows,
    hook_data: ToolCallHookData | None,
    *,
    collapsed: bool,
    width: int,
) -> Rows:
    if not hook_data or hook_data.is_empty():
        return rows
    suffix = hooks_inline_suffix(hook_data)
    if suffix and rows:
        # Append to first row before trailing newline fragment.
        head = list(rows[0])
        if head and head[-1][1] == "\n":
            head = head[:-1] + list(suffix) + [("", "\n")]
        else:
            head = head + list(suffix)
        rows = [head] + list(rows[1:])
    if not collapsed:
        body = paint_hook_body(hook_data, width=width)
        if body:
            rows = list(rows) + body
    return rows


__all__ = [
    "ChangeTag",
    "DiffLine",
    "DiffRenderConfig",
    "ToolCallHookData",
    "VERB_GROUPABLE",
    "build_hunks_from_strings",
    "diff_outputs_to_string",
    "paint_edit",
    "paint_execute",
    "paint_lifecycle",
    "paint_list_dir",
    "paint_memory_search",
    "paint_other",
    "paint_read",
    "paint_search",
    "paint_search_tool",
    "paint_tool",
    "paint_use_tool",
    "paint_web_fetch",
    "paint_web_search",
    "render_diff_hunk",
    "render_diff_hunks",
    "text_of",
]
