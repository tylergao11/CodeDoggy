"""Attack-style regressions for audit P0s.

These tests document failures that happy-path suites missed. If they go green
while the bug is reintroduced, the test is wrong — not the product.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.context.select import sanitize_tool_pairs
from codedoggy.model.chat_sampler import ChatSampler
from codedoggy.model.types import CompletionResult
from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.policy import WorkspacePolicy
from codedoggy.tools.runtime import ToolCallContext, ToolError
from codedoggy.turn.loop import run_agent_loop
from codedoggy.turn.types import Message, Role, SampleResult, ToolCall


# ── P0-1 apply_patch Move ────────────────────────────────────────────────


def _tools():
    return ToolRegistryBuilder.new().finalize()


def test_p0_apply_patch_move_denies_escape_and_keeps_source(tmp_path: Path) -> None:
    """Move to ../outside must fail; source file must still exist."""
    tools = _tools()
    src = tmp_path / "in.txt"
    src.write_text("hello\n", encoding="utf-8")
    outside = tmp_path.parent / f"outside_{tmp_path.name}.txt"
    if outside.exists():
        outside.unlink()

    policy = WorkspacePolicy(cwd=tmp_path)
    ctx = ToolCallContext(cwd=tmp_path, extra={"policy": policy})
    patch = (
        "*** Begin Patch\n"
        "*** Update File: in.txt\n"
        f"*** Move to: ../outside_{tmp_path.name}.txt\n"
        "@@\n"
        "-hello\n"
        "+moved\n"
        "*** End Patch\n"
    )
    with pytest.raises(ToolError) as ei:
        tools.call("apply_patch", {"patch": patch}, ctx)
    assert ei.value.code in {"path_escape", "policy_denied", "invalid_arguments"}
    assert src.is_file(), "source must not be deleted when dest is denied"
    assert src.read_text(encoding="utf-8") == "hello\n"
    assert not outside.exists(), "must not write outside workspace"


def test_p0_apply_patch_move_denies_env_and_records_no_partial(tmp_path: Path) -> None:
    """Move onto .env must be denied before any unlink."""
    tools = _tools()
    src = tmp_path / "src.txt"
    src.write_text("secret-src\n", encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text("OLD=1\n", encoding="utf-8")

    policy = WorkspacePolicy(cwd=tmp_path)
    ctx = ToolCallContext(cwd=tmp_path, extra={"policy": policy})
    patch = (
        "*** Begin Patch\n"
        "*** Update File: src.txt\n"
        "*** Move to: .env\n"
        "@@\n"
        "-secret-src\n"
        "+pwned\n"
        "*** End Patch\n"
    )
    with pytest.raises(ToolError) as ei:
        tools.call("apply_patch", {"patch": patch}, ctx)
    assert ei.value.code in {"deny_path", "policy_denied"}
    assert src.is_file()
    assert src.read_text(encoding="utf-8") == "secret-src\n"
    assert env.read_text(encoding="utf-8") == "OLD=1\n"
    # No mutation recorded after failed policy
    assert not (ctx.extra.get("mutations") or [])


def test_p0_apply_patch_move_emits_delete_and_create_mutations(tmp_path: Path) -> None:
    """Successful Move must leave two mutations: source delete + dest write."""
    tools = _tools()
    src = tmp_path / "a.py"
    src.write_text("x = 1\n", encoding="utf-8")
    policy = WorkspacePolicy(cwd=tmp_path)
    ctx = ToolCallContext(cwd=tmp_path, extra={"policy": policy})
    patch = (
        "*** Begin Patch\n"
        "*** Update File: a.py\n"
        "*** Move to: b.py\n"
        "@@\n"
        "-x = 1\n"
        "+x = 2\n"
        "*** End Patch\n"
    )
    out = tools.call("apply_patch", {"patch": patch}, ctx)
    assert "Success" in out
    assert not src.exists()
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "x = 2\n"
    muts = ctx.extra.get("mutations") or []
    assert len(muts) >= 2
    paths = {m.path for m in muts}
    assert "a.py" in paths and "b.py" in paths
    deletes = [m for m in muts if getattr(m, "is_delete", False)]
    assert deletes, "source delete mutation required for Move mutation log"
    # Windows may retain CRLF from write_text on some paths; compare content
    assert (deletes[0].before or "").replace("\r\n", "\n") == "x = 1\n"


# ── P0-2 context overflow ───────────────────────────────────────────────


class AlwaysOverflowSampler:
    def __init__(self) -> None:
        self.n = 0

    def sample(self, messages, tools):
        self.n += 1
        raise RuntimeError("context_length exceeded: prompt is too long")


def test_p0_overflow_does_not_spin_past_budget(tmp_path: Path) -> None:
    """max_turns=1 must not sample forever; sample_attempts must be bounded."""
    tools = _tools()
    sampler = AlwaysOverflowSampler()
    result = run_agent_loop(
        user_text="x" * 200,
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        max_turns=1,
        system_prompt="sys",
    )
    assert not result.completed
    # Hard bound: attempts ≤ max_turns + MAX_OVERFLOW_RESUBMITS (2) = 3
    assert sampler.n <= 3, f"spun {sampler.n} times"
    assert sampler.n >= 1
    assert result.error or result.exit_reason in {"error", "max_turns"}


# ── P0-3 tool_call id collision ─────────────────────────────────────────


class MissingIdClient:
    """Provider that never sends tool_call ids (forces fallback)."""

    def complete(self, messages, tools=None, **kwargs):
        n = getattr(self, "_n", 0)
        self._n = n + 1
        return CompletionResult(
            content=None,
            model="t",
            tool_calls=[
                {
                    "type": "function",
                    # no id
                    "function": {
                        "name": "read_file",
                        "arguments": '{"target_file": "a.txt"}',
                    },
                }
            ],
            usage={},
        )


def test_p0_chat_sampler_fallback_ids_unique_across_samples() -> None:
    client = MissingIdClient()
    sampler = ChatSampler(client)
    r1 = sampler.sample([], [])
    r2 = sampler.sample([], [])
    assert r1.tool_calls and r2.tool_calls
    id1 = r1.tool_calls[0].id
    id2 = r2.tool_calls[0].id
    assert id1 != id2, f"reused fallback id {id1!r}"
    assert id1.startswith("call_") and id2.startswith("call_")


def test_p0_sanitize_keeps_second_round_result_with_same_id() -> None:
    """Even with colliding ids, FIFO pairing must not drop the later result."""
    msgs = [
        Message(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[ToolCall(id="call_0", name="read_file", arguments={})],
        ),
        Message(
            role=Role.TOOL,
            content="r1",
            tool_call_id="call_0",
            name="read_file",
        ),
        Message(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[ToolCall(id="call_0", name="read_file", arguments={})],
        ),
        Message(
            role=Role.TOOL,
            content="r2",
            tool_call_id="call_0",
            name="read_file",
        ),
    ]
    out = sanitize_tool_pairs(msgs)
    tool_bodies = [
        m.content for m in out if m.role is Role.TOOL
    ]
    assert "r1" in tool_bodies
    assert "r2" in tool_bodies, "second-round result must not be dropped"
