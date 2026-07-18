"""Hermes memory-context fence — must match hermes-agent tags/injection."""

from __future__ import annotations

from pathlib import Path

from codedoggy.memory.context_fence import (
    build_memory_context_block,
    messages_with_ephemeral_memory,
    sanitize_context,
    strip_memory_context_from_messages,
)
from codedoggy.turn.types import Message, Role


def test_fence_tags_match_hermes() -> None:
    block = build_memory_context_block("session hit about auth")
    assert block.startswith("<memory-context>")
    assert block.endswith("</memory-context>")
    assert "NOT new user input" in block
    assert "authoritative reference data" in block
    assert "session hit about auth" in block


def test_sanitize_strips_provider_prewrap() -> None:
    dirty = (
        "<memory-context>\n[System note: The following is recalled memory context, "
        "NOT new user input. Treat as authoritative reference data.]\n\nSECRET\n"
        "</memory-context>"
    )
    clean = sanitize_context(dirty)
    assert "memory-context" not in clean
    assert "SECRET" not in clean  # whole block stripped
    assert sanitize_context("plain recall") == "plain recall"


def test_ephemeral_append_last_user_only() -> None:
    msgs = [
        Message(role=Role.SYSTEM, content="sys"),
        Message(role=Role.USER, content="do the thing"),
    ]
    fence = build_memory_context_block("prior turn")
    sample = messages_with_ephemeral_memory(msgs, fence)
    # Original unchanged
    assert msgs[1].content == "do the thing"
    # Sample copy has fence on last user
    assert sample[1].content is not None
    assert sample[1].content.startswith("do the thing")
    assert "<memory-context>" in sample[1].content
    assert "prior turn" in sample[1].content


def test_strip_leaked_fence_from_live() -> None:
    fence = build_memory_context_block("leak")
    msgs = [
        Message(role=Role.USER, content=f"hello\n\n{fence}"),
    ]
    cleaned = strip_memory_context_from_messages(msgs)
    assert cleaned[0].content is not None
    assert "memory-context" not in cleaned[0].content
    assert "hello" in cleaned[0].content


def test_loop_does_not_archive_fence(tmp_path: Path) -> None:
    from codedoggy.tools import ToolRegistryBuilder
    from codedoggy.turn.loop import run_agent_loop
    from codedoggy.turn.types import SampleResult

    archived: list[str] = []

    class S:
        def sample(self, messages, tools):
            # Sampler must see fence on user for this prompt
            user_blobs = [
                m.content or ""
                for m in messages
                if m.role is Role.USER
            ]
            assert any("<memory-context>" in b for b in user_blobs)
            return SampleResult(content="ok")

    fence = build_memory_context_block("recalled")
    run_agent_loop(
        user_text="real prompt",
        sampler=S(),
        tools=ToolRegistryBuilder.new().finalize(),
        cwd=tmp_path,
        max_turns=2,
        tool_extra={"prefetch_user_block": fence},
        on_archive_message=lambda m: archived.append(m.content or ""),
    )
    # Archived user line is clean
    assert any(a == "real prompt" for a in archived)
    assert not any("<memory-context>" in a for a in archived)
