"""Optional external-style memory provider example (plugin slot demo).

Not registered by default. Use::

    mm = MemoryManager.create_default(curated=..., session_store=...)
    mm.add_provider(FileNotesProvider(path))

Hermes allows one non-builtin external; this is that slot's reference shape.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from codedoggy.memory.provider import BaseMemoryProvider


class FileNotesProvider(BaseMemoryProvider):
    """Simple file-backed notes: one markdown file of free-form durable notes.

    ``prefetch`` returns lines that share tokens with the query (cheap grep).
    """

    name = "file_notes"

    def __init__(self, path: Path | str, *, max_lines: int = 40) -> None:
        self.path = Path(path)
        self.max_lines = max_lines

    def system_prompt_block(self) -> str:
        if not self.path.is_file():
            return ""
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""
        if not text:
            return ""
        # Cap always-on injection
        if len(text) > 800:
            text = text[:797] + "…"
        return f"### File notes ({self.path.name})\n{text}"

    def prefetch(
        self, query: str, *, session_id: str = "", cwd: str = ""
    ) -> str:
        if not self.path.is_file() or not (query or "").strip():
            return ""
        try:
            lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        tokens = [t.lower() for t in re.split(r"\s+", query) if len(t) > 2][:12]
        if not tokens:
            return ""
        hits: list[str] = []
        for line in lines:
            low = line.lower()
            if any(t in low for t in tokens):
                hits.append(line.strip())
            if len(hits) >= self.max_lines:
                break
        if not hits:
            return ""
        return "### File notes matches\n" + "\n".join(f"- {h}" for h in hits)

    def sync_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        session_id: str = "",
        cwd: str = "",
    ) -> None:
        # Optional auto-append of durable-looking decisions (very conservative)
        blob = f"{user_text or ''}\n{assistant_text or ''}"
        if "DECIDE:" not in blob and "决策:" not in blob:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                for line in blob.splitlines():
                    if "DECIDE:" in line or "决策:" in line:
                        f.write(line.strip() + "\n")
        except OSError:
            return

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return []
