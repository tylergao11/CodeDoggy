"""Persist compaction segments (Grok Segments mode) under CODEDOGGY_HOME/compaction/."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from pathlib import Path

from codedoggy.memory.paths import default_memory_home
from codedoggy.model.redact import redact_sensitive_text, redact_tool_arguments
from codedoggy.turn.types import Message, Role

_INDEX_LOCK = threading.RLock()


def compaction_dir(
    home: Path | None = None,
    *,
    workspace: Path | str | None = None,
    session_id: str | None = None,
) -> Path:
    root = home if home is not None else default_memory_home()
    d = Path(root) / "compaction"
    if workspace is not None or session_id:
        ws = Path(workspace or ".").expanduser().resolve()
        digest = hashlib.sha256(str(ws).casefold().encode("utf-8")).hexdigest()[:16]
        sid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(session_id or "default"))[:80]
        d = d / digest / (sid or "default")
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_segment(
    messages: list[Message],
    *,
    home: Path | None = None,
    note: str = "",
    workspace: Path | str | None = None,
    session_id: str | None = None,
) -> Path:
    """Write one segment markdown file; update INDEX.md. Returns segment path."""
    d = compaction_dir(home, workspace=workspace, session_id=session_id)
    ts = time.strftime("%Y%m%d_%H%M%S")
    unique = f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
    path = d / f"segment_{unique}.md"
    lines = [
        f"# Compaction segment {unique}",
        f"",
        f"time: {ts}",
        f"messages: {len(messages)}",
        f"",
    ]
    if note:
        lines.extend([note, ""])
    for m in messages:
        role = m.role.value if isinstance(m.role, Role) else str(m.role)
        lines.append(f"## {role}")
        if m.name:
            lines.append(f"name: {m.name}")
        if m.tool_call_id:
            lines.append(f"tool_call_id: {m.tool_call_id}")
        if m.tool_calls:
            calls: list[dict[str, object]] = []
            for tc in m.tool_calls:
                calls.append(
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": redact_tool_arguments(tc.arguments),
                        **(
                            {"provider_data": dict(tc.provider_data)}
                            if isinstance(getattr(tc, "provider_data", None), dict)
                            else {}
                        ),
                    }
                )
            lines.append(
                "tool_calls_json: "
                + json.dumps(calls, ensure_ascii=False, separators=(",", ":"))
            )
        lines.append(redact_sensitive_text(m.content or "", force=True) or "")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    _update_index(d, path, len(messages), ts)
    return path


def _update_index(d: Path, segment: Path, n_messages: int, ts: str) -> None:
    index = d / "INDEX.md"
    row = f"| {segment.name} | {ts} | {n_messages} |\n"
    with _INDEX_LOCK:
        if not index.exists():
            index.write_text(
                "# Compaction segment index\n\n"
                "| File | Time | Messages |\n"
                "|------|------|----------|\n"
                + row,
                encoding="utf-8",
            )
        else:
            with index.open("a", encoding="utf-8") as f:
                f.write(row)
