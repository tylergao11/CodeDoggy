"""Contracts for the isolated full-fidelity Agent detail surface."""

from __future__ import annotations

import re

from prompt_toolkit.utils import get_cwidth

from codedoggy.tui.agent_detail import (
    AgentDetailSnapshot,
    DetailBlock,
    DetailRecord,
    THINKING_PREVIEW_LINES,
    block_collapse_key,
    default_collapsed_keys,
    filter_detail_records,
    render_agent_detail,
    render_detail_body,
    snapshot_from_messages,
    thinking_collapse_keys,
)
from codedoggy.turn.types import Message, Role, ToolCall


def _plain(fragments: list[tuple]) -> str:
    return "".join(fragment[1] for fragment in fragments)


def test_message_adapter_keeps_prose_tool_arguments_and_tool_results() -> None:
    messages = [
        Message(role=Role.USER, content="实现详情页"),
        Message(role=Role.USER, content="补充：必须显示完整工具记录"),
        Message(role=Role.ASSISTANT, content="我先读取入口，再修改模型。"),
        Message(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[
                ToolCall(
                    id="read-1",
                    name="read_file",
                    arguments={"path": "src/codedoggy/tui/app.py", "offset": 330},
                ),
                ToolCall(
                    id="test-1",
                    name="shell",
                    arguments={"command": "pytest tests/test_tui.py -q"},
                ),
            ],
        ),
        Message(
            role=Role.TOOL,
            name="read_file",
            tool_call_id="read-1",
            content="369 def _start_task(self, prompt: str) -> None:",
        ),
        Message(
            role=Role.TOOL,
            name="shell",
            tool_call_id="test-1",
            content="12 passed in 0.84s",
        ),
    ]

    snapshot = snapshot_from_messages(
        messages,
        task_id="task_001",
        agent_id="builder",
        agent_label="builder",
        task_title="实现详情页",
    )
    text = "\n".join(
        block.text for record in snapshot.records for block in record.blocks
    )

    assert [record.category for record in snapshot.records] == [
        "message",
        "message",
        "file",
        "test",
    ]
    assert snapshot.records[0].actor == "USER"
    assert "补充：必须显示完整工具记录" in text
    assert "我先读取入口" in text
    assert "path: src/codedoggy/tui/app.py" in text
    assert "369 def _start_task" in text
    assert "$ pytest tests/test_tui.py -q" in text
    assert "12 passed in 0.84s" in text
    assert "实现详情页" not in text


def test_message_adapter_accepts_serialized_subagent_transcripts() -> None:
    snapshot = snapshot_from_messages(
        [
            {"role": "assistant", "content": "正在读取子 Agent 文件。"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "child-read",
                        "name": "read_file",
                        "arguments": {"path": "src/child.py", "limit": 80},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "read_file",
                "tool_call_id": "child-read",
                "content": "1  def child():\n2      return True",
            },
        ],
        task_id="task",
        agent_id="child",
        agent_label="BUILDER",
        task_title="子任务",
    )
    messages = _plain(render_detail_body(snapshot, 72, active_filter="message"))
    tools = _plain(render_detail_body(snapshot, 72, active_filter="tool"))
    assert "正在读取子 Agent 文件" in messages
    assert "path: src/child.py" in tools
    assert "def child" in tools


def test_filters_keep_file_and_test_records_under_tool_tab() -> None:
    snapshot = snapshot_from_messages(
        [
            Message(role=Role.ASSISTANT, content="开始"),
            Message(
                role=Role.ASSISTANT,
                tool_calls=[
                    ToolCall(id="f", name="apply_patch", arguments={"patch": "+x"}),
                    ToolCall(id="t", name="shell", arguments={"command": "pytest -q"}),
                    ToolCall(id="g", name="grep", arguments={"pattern": "x"}),
                ],
            ),
        ],
        task_id="task",
        agent_id="main",
        agent_label="MAIN",
        task_title="筛选",
    )

    assert len(filter_detail_records(snapshot.records, "message")) == 1
    # file + test + generic tool all live under the 工具 tab
    assert len(filter_detail_records(snapshot.records, "tool")) == 3


def test_renderer_shows_full_wrapped_content_and_never_overflows_width() -> None:
    long_text = "完整细节不能被摘要。" * 18
    snapshot = AgentDetailSnapshot(
        task_id="task",
        agent_id="builder",
        agent_label="BUILDER",
        task_title="详情页",
        records=(
            DetailRecord(
                id="message",
                sequence=1,
                actor="BUILDER",
                category="message",
                title="当前进度",
                blocks=(DetailBlock("text", long_text),),
                timestamp="14:32:07",
            ),
            DetailRecord(
                id="patch",
                sequence=2,
                actor="TOOL",
                category="file",
                title="TOOL · apply_patch",
                blocks=(
                    DetailBlock(
                        "diff",
                        "@@ -1,2 +1,3 @@\n-output: str = ''\n+records: list[AgentRecord]",
                    ),
                ),
            ),
        ),
    )

    message_frags = render_agent_detail(
        snapshot, 56, active_filter="message", elapsed_seconds=266
    )
    message_text = _plain(message_frags)
    assert long_text.replace("\n", "") == "".join(
        line.strip() for line in message_text.splitlines() if "完整细节" in line
    )
    assert all(get_cwidth(line) <= 56 for line in message_text.splitlines())

    tool_frags = render_agent_detail(snapshot, 56, active_filter="tool")
    tool_text = _plain(tool_frags)
    assert "TOOL · apply_patch" in tool_text
    assert "records: list[AgentRecord]" in tool_text
    assert all(get_cwidth(line) <= 56 for line in tool_text.splitlines())

    styles = " ".join(fragment[0] for fragment in tool_frags)
    assert "class:detail.diff.remove" in styles
    assert "class:detail.diff.add" in styles


def test_header_and_body_fit_terminals_narrower_than_filter_row() -> None:
    snapshot = snapshot_from_messages(
        [Message(role=Role.ASSISTANT, content="窄终端仍显示完整正文")],
        task_id="task",
        agent_id="main",
        agent_label="MAIN",
        task_title="详情",
    )
    for width in (12, 16, 20, 24, 32, 36):
        rendered = _plain(render_agent_detail(snapshot, width))
        assert all(get_cwidth(line) <= width for line in rendered.splitlines())


def test_empty_filter_is_explicit_instead_of_falling_back_to_summary() -> None:
    snapshot = snapshot_from_messages(
        [Message(role=Role.ASSISTANT, content="只有消息")],
        task_id="task",
        agent_id="main",
        agent_label="MAIN",
        task_title="空筛选",
    )
    rendered = _plain(render_detail_body(snapshot, 72, active_filter="tool"))
    assert "当前分类没有记录" in rendered
    assert "只有消息" not in rendered


def test_tool_blocks_can_collapse_arguments_and_results() -> None:
    long_result = "line\n" * 12 + "tail payload"
    snapshot = snapshot_from_messages(
        [
            Message(
                role=Role.ASSISTANT,
                tool_calls=[
                    ToolCall(
                        id="read-1",
                        name="read_file",
                        arguments={"path": "src/app.py"},
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                name="read_file",
                tool_call_id="read-1",
                content=long_result,
            ),
        ],
        task_id="task",
        agent_id="main",
        agent_label="MAIN",
        task_title="折叠",
    )
    defaults = default_collapsed_keys(snapshot.records)
    # Multi-line result starts collapsed by default.
    assert any(key.endswith(":1") for key in defaults)

    def _fold_handler(_key: str):
        return None

    # Grok collapsed default: one-line tool headline, no body dump.
    collapsed = _plain(
        render_detail_body(
            snapshot,
            72,
            active_filter="tool",
            collapsed_keys=defaults,
            fold_mouse=_fold_handler,
        )
    )
    assert "Read" in collapsed and "app.py" in collapsed
    assert "tail payload" not in collapsed
    assert "调用参数" not in collapsed

    # Expand: drop collapsed keys for this tool record.
    expanded_keys = {
        k for k in defaults if not k.startswith("read-1")
    }
    expanded = _plain(
        render_detail_body(
            snapshot,
            72,
            active_filter="tool",
            collapsed_keys=expanded_keys,
            fold_mouse=_fold_handler,
        )
    )
    assert "Read" in expanded
    assert "tail payload" in expanded


def test_system_reminder_hidden_from_detail_snapshot() -> None:
    """Plan-mode <system-reminder> injects must not paint in the TUI."""
    from codedoggy.tui.agent_detail import strip_system_reminders

    raw = (
        "<system-reminder>\nPlan mode is active. Do not make edits.\n"
        "</system-reminder>\n\n你好，继续规划"
    )
    assert "Plan mode" not in strip_system_reminders(raw)
    assert "你好，继续规划" in strip_system_reminders(raw)

    snapshot = snapshot_from_messages(
        [
            Message(role=Role.USER, content=raw),
            Message(role=Role.ASSISTANT, content="好的，我先写 plan。"),
        ],
        task_id="t",
        agent_id="main",
        agent_label="MAIN",
        task_title="计划",
    )
    plain = "\n".join(
        b.text for r in snapshot.records for b in r.blocks
    )
    assert "system-reminder" not in plain.lower()
    assert "Plan mode is active" not in plain
    assert "你好，继续规划" in plain
    assert "好的，我先写 plan" in plain

    # Pure reminder-only user turn → no USER record at all.
    only = snapshot_from_messages(
        [
            Message(
                role=Role.USER,
                content="<system-reminder>\nPlan mode is still active.\n</system-reminder>",
            )
        ],
        task_id="t2",
        agent_id="main",
        agent_label="MAIN",
        task_title="空",
    )
    assert all(r.actor != "USER" for r in only.records)


def test_inline_md_bold_italic_code() -> None:
    from codedoggy.tui.agent_detail import _render_inline_md

    frags = _render_inline_md("见 **重点** 与 *斜体* 和 `x = 1`")
    styles = " ".join(f[0] for f in frags)
    plain = "".join(f[1] for f in frags)
    assert "重点" in plain and "斜体" in plain and "x = 1" in plain
    assert "class:detail.md.bold" in styles
    assert "class:detail.md.italic" in styles
    assert "class:detail.md.inline" in styles


def test_code_blocks_use_token_styles_not_single_white() -> None:
    snapshot = AgentDetailSnapshot(
        task_id="task",
        agent_id="main",
        agent_label="MAIN",
        task_title="code",
        records=(
            DetailRecord(
                id="code-1",
                sequence=1,
                actor="TOOL",
                category="file",
                title="TOOL · read_file",
                blocks=(
                    DetailBlock(
                        "code",
                        'def hello():\n    return "world"  # note',
                        label="返回结果",
                    ),
                ),
            ),
        ),
    )
    fragments = render_detail_body(snapshot, 72, active_filter="tool")
    styles = " ".join(fragment[0] for fragment in fragments)
    assert "class:detail.code.kw" in styles
    assert "class:detail.code.str" in styles
    assert "class:detail.code.cmt" in styles
    assert "class:detail.code.gutter" in styles or "class:detail.code.gutter.mark" in styles
    assert "class:detail.code.rail" in styles


def test_grok_line_prefix_uses_colored_gutter_not_inline_blob() -> None:
    snapshot = AgentDetailSnapshot(
        task_id="task",
        agent_id="main",
        agent_label="MAIN",
        task_title="gutter",
        records=(
            DetailRecord(
                id="read",
                sequence=1,
                actor="TOOL",
                category="file",
                title="TOOL · read_file",
                blocks=(
                    DetailBlock(
                        "code",
                        "1→def alpha():\n"
                        "    return 1\n"
                        "10→def beta():\n"
                        "    return 2",
                        label="返回结果",
                    ),
                ),
            ),
        ),
    )
    fragments = render_detail_body(snapshot, 72, active_filter="tool")
    plain = _plain(fragments)
    styles = " ".join(fragment[0] for fragment in fragments)
    # Line numbers live in gutter column, not glued into code as "10→def"
    assert "10→def" not in plain
    assert "def alpha" in plain
    assert "def beta" in plain
    assert "class:detail.code.gutter" in styles
    assert "class:detail.code.gutter.mark" in styles  # line 10


def test_code_soft_wrap_hangs_under_gutter() -> None:
    long = "x" * 80
    snapshot = AgentDetailSnapshot(
        task_id="task",
        agent_id="main",
        agent_label="MAIN",
        task_title="wrap",
        records=(
            DetailRecord(
                id="c",
                sequence=1,
                actor="TOOL",
                category="file",
                title="TOOL · read_file",
                blocks=(DetailBlock("code", f"1→{long}", label="返回结果"),),
            ),
        ),
    )
    width = 40
    fragments = render_detail_body(snapshot, width, active_filter="tool")
    lines = _plain(fragments).splitlines()
    # At least one continuation line from soft-wrap
    code_lines = [ln for ln in lines if "┃" in ln]
    assert len(code_lines) >= 2
    # First has a digit gutter; second hangs with blank gutter before │
    first = code_lines[0]
    second = code_lines[1]
    assert re.search(r"┃\s*\d+│", first.replace(" ", "") or first) or "│" in first
    assert "│" in second
    assert all(get_cwidth(ln) <= width for ln in lines)


def test_reasoning_content_appears_in_message_tab_before_prose() -> None:
    long_think = "\n".join(f"step {i}: plan" for i in range(1, 8))
    snapshot = snapshot_from_messages(
        [
            Message(
                role=Role.ASSISTANT,
                content="我先读取入口。",
                reasoning_content=long_think,
            )
        ],
        task_id="task",
        agent_id="main",
        agent_label="MAIN",
        task_title="思考",
    )
    assert [r.title for r in snapshot.records] == ["思考", "进度"]
    assert snapshot.records[0].blocks[0].kind == "thinking"
    assert "step 1" in snapshot.records[0].blocks[0].text

    defaults = default_collapsed_keys(snapshot.records)
    think_keys = thinking_collapse_keys(snapshot.records)
    assert think_keys
    # Thinking starts expanded in the data model (not in default collapse set).
    assert not (think_keys & defaults)

    # Product: once assistant prose exists, message tab drops thinking from paint.
    plain = _plain(
        render_detail_body(
            snapshot, 72, active_filter="message", collapsed_keys=defaults
        )
    )
    assert "我先读取入口" in plain
    assert "思考过程" not in plain
    assert "step 7" not in plain


def test_message_tab_hides_thinking_once_assistant_prose_exists() -> None:
    """真实回答出现后，消息页不再画思考过程（transcript 仍保留）。"""
    snapshot = snapshot_from_messages(
        [
            Message(
                role=Role.ASSISTANT,
                content="最终方案用 JWT。",
                reasoning_content="长篇思考……" * 20,
            ),
        ],
        task_id="task",
        agent_id="main",
        agent_label="MAIN",
        task_title="有答",
    )
    filtered = filter_detail_records(snapshot.records, "message")
    assert any(r.actor == "THINK" for r in snapshot.records)
    assert all(r.actor != "THINK" for r in filtered)
    plain = _plain(render_detail_body(snapshot, 72, active_filter="message"))
    assert "思考过程" not in plain
    assert "最终方案用 JWT" in plain


def test_message_tab_keeps_thinking_without_assistant_prose() -> None:
    snapshot = snapshot_from_messages(
        [
            Message(
                role=Role.ASSISTANT,
                content=None,
                reasoning_content="还在想……",
            ),
        ],
        task_id="task",
        agent_id="main",
        agent_label="MAIN",
        task_title="只思考",
    )
    filtered = filter_detail_records(snapshot.records, "message")
    assert any(r.actor == "THINK" for r in filtered)


def test_redacted_thinking_placeholder_from_provider_blocks() -> None:
    snapshot = snapshot_from_messages(
        [
            {
                "role": "assistant",
                "content": "done",
                "provider_data": {
                    "anthropic_content_blocks": [
                        {"type": "redacted_thinking", "data": "x"}
                    ]
                },
            }
        ],
        task_id="task",
        agent_id="main",
        agent_label="MAIN",
        task_title="t",
    )
    assert snapshot.records[0].title == "思考"
    assert "隐藏" in snapshot.records[0].blocks[0].text


def test_message_lists_style_ordered_markers() -> None:
    snapshot = AgentDetailSnapshot(
        task_id="task",
        agent_id="main",
        agent_label="MAIN",
        task_title="lists",
        records=(
            DetailRecord(
                id="m",
                sequence=1,
                actor="MAIN",
                category="message",
                title="进度",
                blocks=(
                    DetailBlock(
                        "text",
                        "计划如下：\n1. 读取入口\n2. 修改模型\n- 补充说明\n\n## 标题",
                    ),
                ),
            ),
        ),
    )
    fragments = render_detail_body(snapshot, 72, active_filter="message")
    styles = " ".join(fragment[0] for fragment in fragments)
    plain = _plain(fragments)
    assert "1." in plain and "读取入口" in plain
    assert "class:detail.md.ol" in styles
    assert "class:detail.md.ul" in styles
    assert "class:detail.md.h2" in styles
