"""Model sampler protocol for the turn loop."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from codedoggy.tools.runtime import ToolSpec
from codedoggy.turn.types import Message, SampleResult


@runtime_checkable
class Sampler(Protocol):
    """Produces the next assistant message (text and/or tool calls)."""

    def sample(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> SampleResult:
        ...
