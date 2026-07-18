"""Monitor line processor + XML wrap — source port from Grok.

Ported from:
  grok-build/crates/codegen/xai-grok-tools/src/implementations/grok_build/monitor/event.rs

Function map:
  LineProcessor / push / flush
  truncate_line / batch_lines
  sanitize_monitor_description / wrap_monitor_event
"""

from __future__ import annotations

from codedoggy.tools.grok_build.monitor_types import (
    BATCH_TRUNCATION_LIMIT,
    BUFFER_CAP_BYTES,
    LINE_TRUNCATION_LIMIT,
)


def _floor_char_boundary(s: str, index: int) -> int:
    """Grok floor_char_boundary — never split a UTF-8 codepoint (str slice is codepoint-safe in Python)."""
    if index <= 0:
        return 0
    if index >= len(s):
        return len(s)
    return index


def truncate_line(line: str) -> str:
    if len(line) > LINE_TRUNCATION_LIMIT:
        boundary = _floor_char_boundary(line, LINE_TRUNCATION_LIMIT)
        return f"{line[:boundary]}...(truncated)"
    return line


class LineProcessor:
    """Processes raw stdout chunks into complete lines (event.rs LineProcessor)."""

    def __init__(self) -> None:
        self.buffer = bytearray()

    def push(self, chunk: bytes) -> list[str]:
        self.buffer.extend(chunk)
        if len(self.buffer) > BUFFER_CAP_BYTES:
            self.buffer = self.buffer[-BUFFER_CAP_BYTES:]

        lines: list[str] = []
        while True:
            try:
                nl_pos = self.buffer.index(0x0A)  # \n
            except ValueError:
                break
            raw = bytes(self.buffer[: nl_pos + 1])
            del self.buffer[: nl_pos + 1]
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            lines.append(truncate_line(text))
        return lines

    def flush(self) -> str | None:
        if not self.buffer:
            return None
        raw = bytes(self.buffer)
        self.buffer.clear()
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return None
        return truncate_line(text)


def batch_lines(lines: list[str]) -> str:
    """Batch multiple lines into a single event string."""
    joined = "\n".join(lines)
    if len(joined) > BATCH_TRUNCATION_LIMIT:
        boundary = _floor_char_boundary(joined, BATCH_TRUNCATION_LIMIT)
        return f"{joined[:boundary]}\n...(truncated)"
    return joined


def sanitize_monitor_description(description: str) -> str:
    """Neutralize quotes/newlines for <monitor-event …> attributes."""
    return description.replace('"', "'").replace("\n", " ").replace("\r", " ")


def wrap_monitor_event(description: str, event_text: str, task_id: str) -> str:
    """Wrap event text in XML tags for the LLM conversation."""
    description = sanitize_monitor_description(description)
    return (
        f'<monitor-event description="{description}" task_id="{task_id}">\n'
        f"{event_text}\n"
        f"</monitor-event>"
    )
