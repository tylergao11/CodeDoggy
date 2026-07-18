"""Codex apply_patch engine — ported from Grok.

Source: crates/codegen/xai-grok-tools/src/implementations/codex/apply_patch/
"""

from codedoggy.tools.codex.apply_patch.apply_logic import derive_new_contents
from codedoggy.tools.codex.apply_patch.parser import (
    Hunk,
    ParseError,
    ParsedPatch,
    UpdateFileChunk,
    parse_patch,
)

__all__ = [
    "Hunk",
    "ParseError",
    "ParsedPatch",
    "UpdateFileChunk",
    "derive_new_contents",
    "parse_patch",
]
