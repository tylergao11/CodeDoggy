"""Pure (I/O-free) patch application logic.

Ported from:
  grok-build/.../codex/apply_patch/apply.rs

Function map:
  derive_new_contents   ↔ derive_new_contents
  compute_replacements  ↔ compute_replacements
  apply_replacements    ↔ apply_replacements
"""

from __future__ import annotations

from codedoggy.tools.codex.apply_patch.parser import UpdateFileChunk
from codedoggy.tools.codex.apply_patch.seek_sequence import seek_sequence


class ApplyPatchError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def derive_new_contents(
    original_content: str,
    path: str,
    chunks: list[UpdateFileChunk],
) -> str:
    original_lines = original_content.split("\n")
    if original_lines and original_lines[-1] == "":
        original_lines.pop()

    replacements = compute_replacements(original_lines, path, chunks)
    new_lines = apply_replacements(original_lines, replacements)
    if not new_lines or new_lines[-1] != "":
        new_lines.append("")
    return "\n".join(new_lines)


def compute_replacements(
    original_lines: list[str],
    path: str,
    chunks: list[UpdateFileChunk],
) -> list[tuple[int, int, list[str]]]:
    replacements: list[tuple[int, int, list[str]]] = []
    line_index = 0

    for chunk in chunks:
        if chunk.change_context is not None:
            idx = seek_sequence(
                original_lines, [chunk.change_context], line_index, False
            )
            if idx is None:
                raise ApplyPatchError(
                    f"Failed to find context '{chunk.change_context}' in {path}"
                )
            line_index = idx + 1

        if not chunk.old_lines:
            insertion_idx = (
                len(original_lines) - 1
                if original_lines and original_lines[-1] == ""
                else len(original_lines)
            )
            replacements.append((insertion_idx, 0, list(chunk.new_lines)))
            continue

        pattern = list(chunk.old_lines)
        found = seek_sequence(
            original_lines, pattern, line_index, chunk.is_end_of_file
        )
        new_slice = list(chunk.new_lines)

        if found is None and pattern and pattern[-1] == "":
            pattern = pattern[:-1]
            if new_slice and new_slice[-1] == "":
                new_slice = new_slice[:-1]
            found = seek_sequence(
                original_lines, pattern, line_index, chunk.is_end_of_file
            )

        if found is not None:
            replacements.append((found, len(pattern), new_slice))
            line_index = found + len(pattern)
        else:
            raise ApplyPatchError(
                f"Failed to find expected lines in {path}:\n"
                + "\n".join(chunk.old_lines)
            )

    replacements.sort(key=lambda r: r[0])
    return replacements


def apply_replacements(
    lines: list[str],
    replacements: list[tuple[int, int, list[str]]],
) -> list[str]:
    out = list(lines)
    for start_idx, old_len, new_segment in reversed(replacements):
        for _ in range(old_len):
            if start_idx < len(out):
                out.pop(start_idx)
        for offset, new_line in enumerate(new_segment):
            out.insert(start_idx + offset, new_line)
    return out
