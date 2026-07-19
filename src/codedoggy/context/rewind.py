"""Compaction checkpoint rewind — restore pre-fold middle into live window.

Grok-style recovery: when a fold went wrong or the model needs verbatim
history, re-inject the last written segment as *reference* messages without
wiping the current tail (latest user + recent tools).

Not a full UI rewind product — a harness API:
  ContextCompactor.rewind_from_checkpoint(live_messages) -> messages
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from codedoggy.turn.types import Message, Role, ToolCall

logger = logging.getLogger(__name__)

_ROLE_RE = re.compile(r"^##\s+(system|user|assistant|tool)\s*$", re.I)


def parse_segment_file(path: Path | str) -> list[Message]:
    """Parse a segment_*.md written by ``write_segment`` into Messages."""
    p = Path(path)
    if not p.is_file():
        return []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("rewind read failed: %s", e)
        return []
    messages: list[Message] = []
    current_role: Role | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal current_role, name, tool_call_id, tool_calls, buf
        if current_role is None:
            buf = []
            return
        content = "\n".join(buf).strip()
        messages.append(
            Message(
                role=current_role,
                content=content or None,
                name=name,
                tool_call_id=tool_call_id,
                tool_calls=tool_calls,
            )
        )
        current_role = None
        name = None
        tool_call_id = None
        tool_calls = None
        buf = []

    for line in text.splitlines():
        m = _ROLE_RE.match(line.strip())
        if m:
            flush()
            try:
                current_role = Role(m.group(1).lower())
            except ValueError:
                current_role = Role.USER
            continue
        if line.startswith("name:") and current_role is not None and not buf:
            name = line.split(":", 1)[1].strip() or None
            continue
        if line.startswith("tool_call_id:") and current_role is not None and not buf:
            tool_call_id = line.split(":", 1)[1].strip() or None
            continue
        if line.startswith("tool_calls_json:") and current_role is not None and not buf:
            try:
                raw_calls = json.loads(line.split(":", 1)[1].strip())
                tool_calls = [
                    ToolCall(
                        id=str(item.get("id") or ""),
                        name=str(item.get("name") or ""),
                        arguments=item.get("arguments") or {},
                        provider_data=(
                            dict(item["provider_data"])
                            if isinstance(item.get("provider_data"), dict)
                            else None
                        ),
                    )
                    for item in raw_calls
                    if isinstance(item, dict)
                ]
            except (json.JSONDecodeError, TypeError, ValueError):
                tool_calls = None
            continue
        if line.startswith("- tool_call") or line.startswith("time:") or line.startswith(
            "messages:"
        ):
            continue
        if line.startswith("# Compaction") or line.startswith("pre-fold"):
            continue
        if current_role is not None:
            buf.append(line)
    flush()
    return messages


def inject_checkpoint_into_live(
    live: list[Message],
    checkpoint_messages: list[Message],
    *,
    as_reference: bool = True,
) -> list[Message]:
    """Merge checkpoint middle back into live transcript.

    Strategy:
      - Keep all SYSTEM messages from live
      - Insert a USER reference block listing recovered turns
      - Keep non-system live tail (so latest user/tools stay)

    Does not delete the fold summary; recovery is additive.
    """
    if not checkpoint_messages:
        return list(live)
    system = [m for m in live if m.role is Role.SYSTEM]
    rest = [m for m in live if m.role is not Role.SYSTEM]
    if as_reference:
        sketch_lines = ["[CHECKPOINT REWIND — recovered pre-fold middle]"]
        sketch_lines.append(
            "The following was restored from a compaction segment. "
            "It is historical reference; the latest user message still wins."
        )
        for m in checkpoint_messages[:40]:
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            body = (m.content or "").replace("\n", " ")
            if len(body) > 200:
                body = body[:197] + "…"
            sketch_lines.append(f"- {role}: {body}")
        if len(checkpoint_messages) > 40:
            sketch_lines.append(f"… ({len(checkpoint_messages) - 40} more)")
        ref = Message(role=Role.USER, content="\n".join(sketch_lines))
        return system + [ref] + rest
    # Full inject (rare): system + checkpoint + rest
    return system + list(checkpoint_messages) + rest


def rewind_from_path(
    live: list[Message],
    checkpoint_path: str | Path | None,
    **kwargs: Any,
) -> list[Message]:
    if not checkpoint_path:
        return list(live)
    recovered = parse_segment_file(checkpoint_path)
    return inject_checkpoint_into_live(live, recovered, **kwargs)
