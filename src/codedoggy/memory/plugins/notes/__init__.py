"""Example external memory provider (Hermes plugin shape).

Hermes: plugins/memory/<name>/ with register_memory_provider().
Activate via CODEDOGGY_MEMORY_PROVIDER=notes when this package is discoverable
(bundled under codedoggy.memory.plugins.notes).

Simple file-backed notes under CODEDOGGY_HOME/memories/notes.md — not Honcho.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from codedoggy.memory.paths import get_memory_dir
from codedoggy.memory.provider import BaseMemoryProvider


class NotesMemoryProvider(BaseMemoryProvider):
    """External-slot demo: durable free-form notes file + grep prefetch."""

    name = "notes"

    def __init__(self, path: Path | str | None = None, *, max_hits: int = 24) -> None:
        self.path = Path(path) if path else (get_memory_dir() / "notes.md")
        self.max_hits = max_hits
        self._session_id = ""

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str = "", **kwargs: Any) -> None:
        self._session_id = session_id or ""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(
                "# CodeDoggy notes (external memory provider demo)\n\n",
                encoding="utf-8",
            )

    def system_prompt_block(self) -> str:
        # External static block stays small — heavy recall is via prefetch
        return (
            "### Notes memory provider\n"
            "Durable free-form notes live in notes.md. "
            "Relevant lines are prefetched each turn; "
            "use tool notes_append to store durable facts."
        )

    def prefetch(self, query: str, *, session_id: str = "", cwd: str = "") -> str:
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
            if line.startswith("#") or not line.strip():
                continue
            low = line.lower()
            if any(t in low for t in tokens):
                hits.append(line.strip())
            if len(hits) >= self.max_hits:
                break
        if not hits:
            return ""
        return "### Notes matches\n" + "\n".join(f"- {h}" for h in hits)

    def on_pre_compress(self, messages: list[Any] | None = None) -> str:
        """Extract short user lines before fold (Hermes on_pre_compress)."""
        if not messages:
            return ""
        bits: list[str] = []
        for m in messages:
            if isinstance(m, dict):
                role_s = str(m.get("role") or "")
                content = m.get("content")
            else:
                role = getattr(m, "role", None)
                role_s = str(getattr(role, "value", role) or "")
                content = getattr(m, "content", None)
            if role_s.lower() != "user":
                continue
            text = (content or "").strip() if isinstance(content, str) else ""
            if not text or "<memory-context>" in text:
                continue
            if len(text) > 200:
                text = text[:197] + "…"
            bits.append(text)
            if len(bits) >= 5:
                break
        if not bits:
            return ""
        return "### Notes pre-compress user excerpts\n" + "\n".join(f"- {b}" for b in bits)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "notes_append",
                "description": "Append a durable note to notes.md (external memory provider).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Short durable fact to store.",
                        }
                    },
                    "required": ["content"],
                },
            }
        ]

    def handle_tool_call(
        self, tool_name: str, args: dict[str, Any], **kwargs: Any
    ) -> str:
        import json

        if tool_name != "notes_append":
            return json.dumps({"success": False, "error": f"unknown tool {tool_name}"})
        content = str((args or {}).get("content") or "").strip()
        if not content:
            return json.dumps({"success": False, "error": "content required"})
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(content.rstrip() + "\n")
            return json.dumps({"success": True, "message": "note appended"})
        except OSError as e:
            return json.dumps({"success": False, "error": str(e)})


def register_memory_provider() -> NotesMemoryProvider:
    """Hermes plugin entrypoint."""
    return NotesMemoryProvider()
