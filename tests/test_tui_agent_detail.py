"""Contracts for the isolated full-fidelity Agent detail surface."""

from __future__ import annotations

from prompt_toolkit.utils import get_cwidth

from codedoggy.tui.agent_detail import (
    AgentDetailSnapshot,
    DetailBlock,
    DetailRecord,
    filter_detail_records,
    render_agent_detail,
    render_detail_body,
    snapshot_from_messages,
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
    rendered = _plain(render_detail_body(snapshot, 72))
    assert "正在读取子 Agent 文件" in rendered
    assert "path: src/child.py" in rendered
    assert "def child" in rendered


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

    assert len(filter_detail_records(snapshot.records, "all")) == 4
    assert len(filter_detail_records(snapshot.records, "message")) == 1
    assert len(filter_detail_records(snapshot.records, "tool")) == 3
    assert len(filter_detail_records(snapshot.records, "file")) == 1
    assert len(filter_detail_records(snapshot.records, "test")) == 1


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

    fragments = render_agent_detail(snapshot, 56, elapsed_seconds=266)
    rendered = _plain(fragments)
    assert long_text.replace("\n", "") == "".join(
        line.strip() for line in rendered.splitlines() if "完整细节" in line
    )
    assert "TOOL · apply_patch" in rendered
    assert "records: list[AgentRecord]" in rendered
    assert all(get_cwidth(line) <= 56 for line in rendered.splitlines())

    styles = " ".join(fragment[0] for fragment in fragments)
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
    rendered = _plain(render_detail_body(snapshot, 72, active_filter="test"))
    assert "当前分类没有记录" in rendered
    assert "只有消息" not in rendered
