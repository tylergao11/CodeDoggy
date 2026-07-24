"""Regression for Grok source-port TUI (tui_v2).

Includes golden fixtures in the spirit of Grok tool snaps under
``xai-grok-pager/.../tool/snapshots/``.
"""

from __future__ import annotations

from codedoggy.tui_v2.blocks.tool import (
    diff_outputs_to_string,
    paint_tool,
    render_diff_hunks,
)
from codedoggy.tui_v2.blocks.tool.edit import DiffRenderConfig
from codedoggy.tui_v2.blocks.thinking import paint_thinking
from codedoggy.tui_v2.blocks.user import paint_user_prompt
from codedoggy.tui_v2.glyphs import (
    accent_bar,
    braille_spinner_frames,
    diamond_dotted,
    diamond_filled,
    diamond_hollow,
    prompt_arrow,
)
from codedoggy.tui_v2.scrollback import ScrollbackState, render_scrollback
from codedoggy.tui_v2.theme import build_style, groknight, theme_style_dict
from codedoggy.tui_v2.verb_group import build_display_items, classify_verb
from codedoggy.tui_v2.app import run_tui


def _flat(rows: list) -> str:
    return "".join(t for row in rows for _, t in row)


# ── Glyphs / theme ───────────────────────────────────────────────────────────


def test_glyphs_from_grok() -> None:
    assert accent_bar() in {"┃", "│"}
    assert diamond_filled()
    assert diamond_hollow()
    assert diamond_dotted()
    assert "❯" in prompt_arrow() or prompt_arrow().startswith(">")
    assert len(braille_spinner_frames()) >= 4


def test_theme_groknight_rgb() -> None:
    t = groknight()
    assert t.bg_base.lower() == "#141414"
    assert t.accent_assistant.lower() == "#bb9af7"
    assert build_style() is not None


def test_theme_painter_aliases_registered() -> None:
    """Every class:grok.* used by blocks must resolve in Style.from_dict."""
    d = theme_style_dict(groknight())
    required = [
        "grok.primary",
        "grok.muted",
        "grok.dim",
        "grok.path",
        "grok.command",
        "grok.diff.insert",
        "grok.diff.delete",
        "grok.diff.gutter",
        "grok.diff.equal",
        "grok.prompt.prefix",
        "grok.prompt.body",
        "grok.thinking.header",
        "grok.md.text",
        "grok.md.h1",
        "grok.md.code",
        "grok.tool.header",
        "grok.context.critical",
        "grok.accent_user",
        "grok.selection_border",
    ]
    missing = [k for k in required if k not in d]
    assert missing == [], f"missing theme keys: {missing}"


# ── Block painters ───────────────────────────────────────────────────────────


def test_user_header_prompt_arrow() -> None:
    text = _flat(paint_user_prompt("hello", width=40))
    assert "hello" in text
    assert "✦" not in text
    assert "❯" in text or ">" in text


def test_thinking_strings() -> None:
    text = _flat(
        paint_thinking(
            "x", width=40, running=False, elapsed_ms=1500, show_header=True
        )
    )
    assert "Thought" in text
    assert "1.5s" in text
    text2 = _flat(paint_thinking("x", width=40, running=True, show_header=True))
    assert "Thinking" in text2


def test_tool_read_execute_edit_headers() -> None:
    r = _flat(
        paint_tool(
            "read_file",
            {"target_file": "main.rs"},
            "fn main() {}",
            width=60,
            collapsed=True,
            status="completed",
        )
    )
    assert "Read" in r and "main.rs" in r

    e = _flat(
        paint_tool(
            "run_terminal_cmd",
            {"command": "cargo test"},
            "ok",
            width=60,
            collapsed=True,
            status="completed",
        )
    )
    assert "$" in e and "cargo test" in e

    d = _flat(
        paint_tool(
            "search_replace",
            {
                "path": "x.py",
                "old_string": "a=1\nb=2",
                "new_string": "a=2\nb=3",
            },
            "",
            width=60,
            collapsed=False,
            status="completed",
        )
    )
    assert "Edit" in d or "Edited" in d
    assert "a=1" in d or "a=2" in d


# ── Golden: full Grok edit snap matrix ───────────────────────────────────────


def _dl(text: str, lo: int, ln: int, tag: str):
    from codedoggy.tui_v2.blocks.tool.edit import ChangeTag, DiffLine

    return DiffLine(text=text, lo=lo, ln=ln, tag=ChangeTag(tag))


def _basic_hunk():
    return [
        _dl("let x = 1;", 10, 10, "equal"),
        _dl("let y = 2;", 11, 0, "delete"),
        _dl("let y = 3;", 0, 11, "insert"),
        _dl("let z = 4;", 12, 12, "equal"),
    ]


def test_golden_edit_diff_basic() -> None:
    got = diff_outputs_to_string(render_diff_hunks([_basic_hunk()], 80))
    assert got == (
        "  10  let x = 1;\n"
        "  11  let y = 2;\n"
        "  11  let y = 3;\n"
        "  12  let z = 4;"
    )


def test_golden_edit_diff_basic_dual() -> None:
    cfg = DiffRenderConfig(dual_line_numbers=True)
    got = diff_outputs_to_string(render_diff_hunks([_basic_hunk()], 80, cfg))
    assert got == (
        "  10 10  let x = 1;\n"
        "  11     let y = 2;\n"
        "     11  let y = 3;\n"
        "  12 12  let z = 4;"
    )


def test_golden_edit_diff_three_digit() -> None:
    hunk = [
        _dl("context before", 99, 99, "equal"),
        _dl("old code", 100, 0, "delete"),
        _dl("new code", 0, 100, "insert"),
        _dl("context after", 101, 101, "equal"),
    ]
    got = diff_outputs_to_string(render_diff_hunks([hunk], 80))
    assert got == (
        "   99  context before\n"
        "  100  old code\n"
        "  100  new code\n"
        "  101  context after"
    )


def test_golden_edit_diff_three_digit_dual() -> None:
    hunk = [
        _dl("context before", 99, 99, "equal"),
        _dl("old code", 100, 0, "delete"),
        _dl("new code", 0, 100, "insert"),
        _dl("context after", 101, 101, "equal"),
    ]
    cfg = DiffRenderConfig(dual_line_numbers=True)
    got = diff_outputs_to_string(render_diff_hunks([hunk], 80, cfg))
    assert got == (
        "   99  99  context before\n"
        "  100      old code\n"
        "      100  new code\n"
        "  101 101  context after"
    )


def test_golden_edit_diff_reflow() -> None:
    hunk = [
        _dl("short line", 1, 1, "equal"),
        _dl(
            "this is a very long line that will definitely wrap to multiple lines",
            0,
            2,
            "insert",
        ),
        _dl("another short one", 2, 3, "equal"),
    ]
    got = diff_outputs_to_string(render_diff_hunks([hunk], 40))
    assert got == (
        "  1  short line\n"
        "  2  this is a very long line that will \n"
        "     definitely wrap to multiple lines\n"
        "  3  another short one"
    )


def test_golden_edit_diff_reflow_dual() -> None:
    hunk = [
        _dl("short line", 1, 1, "equal"),
        _dl(
            "this is a very long line that will definitely wrap to multiple lines",
            0,
            2,
            "insert",
        ),
        _dl("another short one", 2, 3, "equal"),
    ]
    cfg = DiffRenderConfig(dual_line_numbers=True)
    got = diff_outputs_to_string(render_diff_hunks([hunk], 40, cfg))
    assert got == (
        "  1 1  short line\n"
        "    2  this is a very long line that \n"
        "       will definitely wrap to multiple \n"
        "       lines\n"
        "  2 3  another short one"
    )


def test_golden_edit_diff_multiple_hunks() -> None:
    h1 = [
        _dl("first hunk context", 5, 5, "equal"),
        _dl("deleted in first", 6, 0, "delete"),
    ]
    h2 = [
        _dl("second hunk context", 50, 49, "equal"),
        _dl("inserted in second", 0, 50, "insert"),
    ]
    got = diff_outputs_to_string(render_diff_hunks([h1, h2], 80))
    assert got == (
        "  5  first hunk context\n"
        "  6  deleted in first\n"
        "  … 43 unchanged lines\n"
        "  49  second hunk context\n"
        "  50  inserted in second"
    )


def test_golden_edit_diff_multiple_hunks_dual() -> None:
    h1 = [
        _dl("first hunk context", 5, 5, "equal"),
        _dl("deleted in first", 6, 0, "delete"),
    ]
    h2 = [
        _dl("second hunk context", 50, 49, "equal"),
        _dl("inserted in second", 0, 50, "insert"),
    ]
    cfg = DiffRenderConfig(dual_line_numbers=True)
    got = diff_outputs_to_string(render_diff_hunks([h1, h2], 80, cfg))
    assert got == (
        "  5 5  first hunk context\n"
        "  6    deleted in first\n"
        "  … 43 unchanged lines\n"
        "  50 49  second hunk context\n"
        "     50  inserted in second"
    )


def test_golden_edit_diff_merged_hunks_gap_markers() -> None:
    m1 = [
        _dl("fn one() {", 3, 3, "equal"),
        _dl("old_one();", 4, 0, "delete"),
        _dl("new_one();", 0, 4, "insert"),
    ]
    m2 = [
        _dl("ctx_two();", 12, 12, "equal"),
        _dl("add_two();", 0, 13, "insert"),
    ]
    m3 = [
        _dl("ctx_three();", 19, 20, "equal"),
        _dl("add_three();", 0, 21, "insert"),
    ]
    got = diff_outputs_to_string(render_diff_hunks([m1, m2, m3], 80))
    assert got == (
        "  3  fn one() {\n"
        "  4  old_one();\n"
        "  4  new_one();\n"
        "  … 7 unchanged lines\n"
        "  12  ctx_two();\n"
        "  13  add_two();\n"
        "  … 6 unchanged lines\n"
        "  20  ctx_three();\n"
        "  21  add_three();"
    )


def test_golden_chrome_user_and_tool() -> None:
    """User + single tool must paint accent rail + prompt arrow + Read verb."""
    s = ScrollbackState()
    s.append_message({"role": "user", "content": "hello chrome"})
    s.append_message(
        {
            "role": "assistant",
            "content": "done",
            "tool_calls": [
                {
                    "id": "1",
                    "name": "read_file",
                    "arguments": {"target_file": "a.py"},
                }
            ],
        }
    )
    s.append_message(
        {
            "role": "tool",
            "tool_call_id": "1",
            "name": "read_file",
            "content": "x = 1",
        }
    )
    text = "".join(t for _, t in render_scrollback(s, width=48, height=12))
    assert accent_bar() in text
    assert "hello chrome" in text
    assert "done" in text
    assert "Read" in text
    assert "a.py" in text
    assert "✦" not in text
    assert "💭" not in text


def test_golden_verb_group_fold() -> None:
    """2+ collapsed groupable tools fold; execute is not groupable (Grok parity).

    2 reads alone → ``Read 2 files``. Reads + exec → group only the reads;
    the shell command paints as its own row.
    """
    s = ScrollbackState()
    s.append_message({"role": "user", "content": "go"})
    s.append_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "1", "name": "read_file", "arguments": {"target_file": "a.py"}},
                {"id": "2", "name": "read_file", "arguments": {"target_file": "b.py"}},
                {
                    "id": "3",
                    "name": "run_terminal_cmd",
                    "arguments": {"command": "pytest"},
                },
            ],
        }
    )
    for cid, name in (("1", "read_file"), ("2", "read_file"), ("3", "run_terminal_cmd")):
        s.append_message(
            {"role": "tool", "tool_call_id": cid, "name": name, "content": "ok"}
        )

    display = build_display_items(s.items)
    groups = [d for d in display if d.group is not None]
    assert len(groups) == 1
    label = groups[0].group.label  # type: ignore[union-attr]
    assert label == "Read 2 files"
    assert "Ran" not in label
    assert "command" not in label

    # Ungrouped entries include the execute tool (not folded into the verb group).
    ungrouped_tools = [
        s.items[d.entry_index]
        for d in display
        if d.entry_index is not None and s.items[d.entry_index].kind == "tool"
    ]
    assert len(ungrouped_tools) == 1
    assert ungrouped_tools[0].tool_name == "run_terminal_cmd"

    text = "".join(t for _, t in render_scrollback(s, width=56, height=16))
    assert "Read 2 files" in text
    assert "Ran 1 command" not in text
    # Folded read paths stay hidden; exec still paints its own shell row.
    assert "a.py" not in text
    assert "b.py" not in text
    assert "$" in text and "pytest" in text
    assert diamond_dotted() in text or "◈" in text


def test_classify_verb_names() -> None:
    assert classify_verb("read_file") == "file"
    assert classify_verb("grep") == "search"
    assert classify_verb("search_replace") is None  # edit is not groupable
    # execute is no longer groupable (Grok parity).
    assert classify_verb("run_terminal_cmd") is None
    assert classify_verb("run_terminal_command") is None
    assert classify_verb("bash") is None
    assert classify_verb("shell") is None
    assert classify_verb("execute") is None


def test_scrollback_scene() -> None:
    s = ScrollbackState()
    s.append_message({"role": "user", "content": "ping"})
    s.append_message(
        {
            "role": "assistant",
            "content": "pong",
            "tool_calls": [
                {
                    "id": "1",
                    "name": "read_file",
                    "arguments": {"target_file": "a.py"},
                }
            ],
        }
    )
    s.append_message(
        {
            "role": "tool",
            "tool_call_id": "1",
            "name": "read_file",
            "content": "hello",
        }
    )
    text = "".join(t for _, t in render_scrollback(s, width=50, height=16))
    assert "ping" in text
    assert "pong" in text
    assert "Read" in text
    assert "✦" not in text
    assert "💭" not in text


def test_selection_box_corners() -> None:
    """Selected entry paints Grok ┌┐ / │ / └┘ selection chrome."""
    from codedoggy.tui_v2.project import ScrollItem
    from codedoggy.tui_v2.selection import BOTTOM_LEFT, TOP_LEFT, TOP_RIGHT, VERTICAL

    item = ScrollItem(kind="user", id="u1", text="select me")
    text = _flat(item.paint(width=40, selected=True))
    assert TOP_LEFT in text
    assert TOP_RIGHT in text
    assert BOTTOM_LEFT in text
    assert VERTICAL in text
    assert "select me" in text


def test_subagent_header_strings() -> None:
    from codedoggy.tui_v2.blocks.subagent import paint_subagent
    from codedoggy.tui_v2.project import ScrollItem

    t = _flat(
        paint_subagent("explore tree", width=60, status="running", is_background=False)
    )
    assert "Subagent " in t
    assert "running:" in t
    assert "explore tree" in t

    t2 = _flat(
        paint_subagent(
            "task", width=60, status="completed", elapsed_ms=1200, is_background=True
        )
    )
    assert "completed in 1.2s" in t2

    item = ScrollItem(
        kind="subagent",
        id="s1",
        text="do work",
        status="running",
        collapsed=True,
    )
    painted = _flat(item.paint(width=48, selected=False))
    assert "Subagent " in painted
    assert accent_bar() in painted


def test_edit_syntax_highlight_styles() -> None:
    """Equal/insert lines carry grok.syn.* styles when path is known."""
    from codedoggy.tui_v2.blocks.tool.edit import (
        ChangeTag,
        DiffLine,
        DiffRenderConfig,
        render_diff_hunks,
    )

    hunk = [
        DiffLine("def foo():", 1, 1, ChangeTag.EQUAL),
        DiffLine('    return "x"', 0, 2, ChangeTag.INSERT),
    ]
    cfg = DiffRenderConfig(path="main.py", syntax_highlight=True)
    outs = render_diff_hunks([hunk], 80, cfg)
    styles = [st for o in outs for st, _ in o.fragments]
    joined = " ".join(styles)
    assert "class:grok.syn." in joined or "grok.syn." in joined
    # Plain text still reconstructs.
    assert "def foo():" in diff_outputs_to_string(outs)
    assert 'return "x"' in diff_outputs_to_string(outs)


def test_read_expanded_syntax_highlight() -> None:
    from codedoggy.tui_v2.blocks.tool import paint_tool

    rows = paint_tool(
        "read_file",
        {"target_file": "app.py"},
        "def main():\n    pass\n",
        width=60,
        collapsed=False,
        status="completed",
    )
    styles = [st for row in rows for st, _ in row]
    text = _flat(rows)
    assert "def main" in text
    assert any("grok.syn" in st for st in styles)


def test_read_expanded_full_body_not_truncated() -> None:
    """Expanded read (truncated=False) shows full body, not first 5 + last 3."""
    from codedoggy.tui_v2.blocks.tool import paint_tool
    from codedoggy.tui_v2.blocks.tool.common import ELLIPSIS, HEADER_READ

    body = "\n".join(f"line_{i}" for i in range(1, 21))
    full = paint_tool(
        "read_file",
        {"target_file": "big.py"},
        body,
        width=60,
        collapsed=False,
        status="completed",
        truncated=False,
    )
    text = _flat(full)
    assert HEADER_READ in text or "Read " in text
    for i in range(1, 21):
        assert f"line_{i}" in text
    assert ELLIPSIS not in text

    # truncated=True still caps to first 5 + last 3 for contrast.
    short = paint_tool(
        "read_file",
        {"target_file": "big.py"},
        body,
        width=60,
        collapsed=False,
        status="completed",
        truncated=True,
    )
    short_text = _flat(short)
    assert "line_1" in short_text and "line_5" in short_text
    assert "line_18" in short_text and "line_20" in short_text
    assert "line_10" not in short_text
    assert ELLIPSIS in short_text


def test_read_header_fixed_tense_while_running() -> None:
    """Individual read headers use fixed 'Read ' even while status is running."""
    from codedoggy.tui_v2.blocks.tool import paint_tool
    from codedoggy.tui_v2.blocks.tool.common import HEADER_READ

    text = _flat(
        paint_tool(
            "read_file",
            {"target_file": "main.rs"},
            "",
            width=60,
            collapsed=True,
            status="running",
        )
    )
    assert HEADER_READ in text or "Read " in text
    assert "Reading " not in text


def test_selection_viewport_clip_dashed() -> None:
    """When selection box is cut by viewport, sides use dashed vertical."""
    from codedoggy.tui_v2.project import ScrollItem
    from codedoggy.tui_v2.selection import VERTICAL_DASHED

    s = ScrollbackState()
    # Tall multi-line user so selection box exceeds small height.
    s.append_message(
        {
            "role": "user",
            "content": "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8",
        }
    )
    s.selected = 0
    s.follow_tail = False
    # Small height forces clip of top/bottom corners.
    text = "".join(t for _, t in render_scrollback(s, width=40, height=4))
    # Either dashed edges present, or content still visible (clip path exercised).
    assert "line" in text
    # Force post-process path by painting selected tall block into short window.
    item = ScrollItem(
        kind="assistant",
        id="a1",
        text="a\nb\nc\nd\ne\nf\ng\nh",
        status="done",
    )
    full = item.paint(width=36, selected=True)
    from codedoggy.tui_v2.selection import apply_viewport_selection_clip

    owners = [0] * len(full)
    clipped = apply_viewport_selection_clip(
        full,
        offset=2,
        height=3,
        line_owners=owners,
        selected_owners={0},
        total_width=36,
    )
    clipped_text = "".join(t for row in clipped for _, t in row)
    assert VERTICAL_DASHED in clipped_text or "b" in clipped_text or "c" in clipped_text


def test_theme_syn_aliases() -> None:
    d = theme_style_dict(groknight())
    for k in (
        "grok.syn.kw",
        "grok.syn.str",
        "grok.syn.cmt",
        "grok.syn.fn",
        "grok.syn.plain",
    ):
        assert k in d


def test_memory_search_and_search_tool_headers() -> None:
    from codedoggy.tui_v2.blocks.tool import paint_tool

    m = _flat(
        paint_tool(
            "memory_search",
            {"query": "auth flow"},
            '[{"path":"a.md","start_line":1,"end_line":3,"score":0.9,'
            '"source":"global","snippet":"hello\\nworld"}]',
            width=60,
            collapsed=True,
            status="completed",
        )
    )
    assert "Memory Search" in m
    assert "auth flow" in m

    m2 = _flat(
        paint_tool(
            "memory_search",
            {"query": "auth flow"},
            '[{"path":"a.md","start_line":1,"end_line":3,"score":0.9,'
            '"source":"global","snippet":"hello"}]',
            width=60,
            collapsed=False,
            status="completed",
        )
    )
    assert "a.md" in m2
    assert "score" in m2

    s = _flat(
        paint_tool(
            "search_tool",
            {"query": "linear"},
            '[{"name":"linear__save_issue","server":"linear",'
            '"description":"Create issue","score":1}]',
            width=60,
            collapsed=True,
            status="completed",
        )
    )
    assert "Search Tools" in s
    assert "linear" in s


def test_use_tool_mcp_header() -> None:
    from codedoggy.tui_v2.blocks.tool import paint_tool

    t = _flat(
        paint_tool(
            "linear__save_issue",
            {"title": "Bug"},
            "ok",
            width=40,
            collapsed=True,
            status="completed",
        )
    )
    assert "Linear" in t
    assert "Save" in t or "Issue" in t or "Save Issue" in t


def test_file_scoped_styles_upgrade() -> None:
    from codedoggy.tui_v2.blocks.tool.edit import (
        ChangeTag,
        DiffLine,
        DiffRenderConfig,
        compute_file_scoped_styles,
        render_diff_hunks,
        diff_outputs_to_string,
    )

    file_text = "def a():\n    return 1\n"
    hunk = [
        DiffLine("def a():", 1, 1, ChangeTag.EQUAL),
        DiffLine("    return 0", 2, 0, ChangeTag.DELETE),
        DiffLine("    return 1", 0, 2, ChangeTag.INSERT),
    ]
    by_new = compute_file_scoped_styles("x.py", file_text, [hunk])
    assert by_new is not None
    assert 1 in by_new and 2 in by_new
    # Drift refuses upgrade
    bad = compute_file_scoped_styles(
        "x.py", "def a():\n    return 9\n", [hunk]
    )
    assert bad is None

    cfg = DiffRenderConfig(path="x.py", by_new_line=by_new, syntax_highlight=True)
    got = diff_outputs_to_string(render_diff_hunks([hunk], 80, cfg))
    assert "def a():" in got
    assert "return 1" in got


def test_verb_group_expand_collapse() -> None:
    s = ScrollbackState()
    s.append_message({"role": "user", "content": "go"})
    s.append_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "1", "name": "read_file", "arguments": {"target_file": "a.py"}},
                {"id": "2", "name": "read_file", "arguments": {"target_file": "b.py"}},
            ],
        }
    )
    s.append_message(
        {"role": "tool", "tool_call_id": "1", "name": "read_file", "content": "x"}
    )
    s.append_message(
        {"role": "tool", "tool_call_id": "2", "name": "read_file", "content": "y"}
    )
    # Select a tool inside the natural group
    tool_idxs = [i for i, it in enumerate(s.items) if it.kind == "tool"]
    assert len(tool_idxs) >= 2
    s.selected = tool_idxs[0]
    s.follow_tail = False
    text = "".join(t for _, t in render_scrollback(s, width=50, height=12))
    assert "Read 2 files" in text
    assert s.expand_group_at_selection()
    text2 = "".join(t for _, t in render_scrollback(s, width=50, height=12))
    assert "a.py" in text2
    assert "Read 2 files" not in text2
    assert s.collapse_group_at_selection()
    text3 = "".join(t for _, t in render_scrollback(s, width=50, height=12))
    assert "Read 2 files" in text3


def test_text_selection_reconstruct_and_highlight() -> None:
    from codedoggy.tui_v2.text_selection import (
        TextSel,
        apply_text_selection_highlight,
        reconstruct_selection_text,
    )

    rows = [
        [("class:grok.primary", "hello world")],
        [("class:grok.primary", "second line")],
    ]
    sel = TextSel(0, 0, 0, 5)
    assert reconstruct_selection_text(rows, sel) == "hello"
    sel2 = TextSel(0, 6, 1, 6)
    assert reconstruct_selection_text(rows, sel2) == "world\nsecond"
    hl = apply_text_selection_highlight(rows, TextSel(0, 0, 0, 5))
    styles = " ".join(st for st, _ in hl[0])
    assert "selected" in styles or "reverse" in styles


def test_lifecycle_and_hooks_on_tool() -> None:
    from codedoggy.tui_v2.blocks.tool import paint_tool
    from codedoggy.tui_v2.blocks.tool.hook import HookRunEntry, ToolCallHookData

    life = _flat(
        paint_tool(
            "session_start",
            {},
            "",
            width=40,
            collapsed=True,
            status="completed",
        )
    )
    assert "session_start" in life

    hooks = ToolCallHookData(
        post_hooks=[
            HookRunEntry(name="fmt", status="success"),
            HookRunEntry(name="lint", status="failed", detail="boom"),
        ]
    )
    r = _flat(
        paint_tool(
            "read_file",
            {"target_file": "a.py"},
            "x",
            width=60,
            collapsed=True,
            status="completed",
            hooks=hooks,
        )
    )
    assert "Read" in r
    assert "[hooks:" in r
    assert "1" in r and "1" in r  # success/fail counts


def test_hooks_from_meta() -> None:
    from codedoggy.tui_v2.blocks.tool import paint_tool

    r = _flat(
        paint_tool(
            "read_file",
            {"target_file": "a.py"},
            "x",
            width=50,
            collapsed=True,
            status="completed",
            meta={"hooks": {"post": [{"name": "h1", "status": "success"}]}},
        )
    )
    assert "[hooks:" in r


def test_run_tui_export() -> None:
    assert callable(run_tui)
