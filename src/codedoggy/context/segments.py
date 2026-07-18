"""Persist compaction segments (Grok Segments mode) under CODEDOGGY_HOME/compaction/."""

from __future__ import annotations

import time
from pathlib import Path

from codedoggy.memory.paths import default_memory_home
from codedoggy.turn.types import Message, Role


def compaction_dir(home: Path | None = None) -> Path:
    root = home if home is not None else default_memory_home()
    d = Path(root) / "compaction"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_segment(
    messages: list[Message],
    *,
    home: Path | None = None,
    note: str = "",
) -> Path:
    """Write one segment markdown file; update INDEX.md. Returns segment path."""
    d = compaction_dir(home)
    ts = time.strftime("%Y%m%d_%H%M%S")
    # sequential index
    existing = sorted(d.glob("segment_*.md"))
    idx = len(existing)
    path = d / f"segment_{idx:03d}.md"
    lines = [
        f"# Compaction segment {idx}",
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
        if m.tool_calls:
            for tc in m.tool_calls:
                lines.append(f"- tool_call {tc.name}({tc.arguments!r})")
        lines.append(m.content or "")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    _update_index(d, path, len(messages), ts)
    return path


def _update_index(d: Path, segment: Path, n_messages: int, ts: str) -> None:
    index = d / "INDEX.md"
    row = f"| {segment.name} | {ts} | {n_messages} |\n"
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
