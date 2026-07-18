"""Patch parser for the codex apply-patch format.

Ported from:
  grok-build/.../codex/apply_patch/parser.rs

Function map:
  parse_patch                    ↔ parse_patch
  parse_patch_text               ↔ parse_patch_text
  check_patch_boundaries_strict  ↔ check_patch_boundaries_strict
  check_patch_boundaries_lenient ↔ check_patch_boundaries_lenient
  check_start_and_end_lines_strict ↔ check_start_and_end_lines_strict
  parse_one_hunk                 ↔ parse_one_hunk
  parse_update_file_chunk        ↔ parse_update_file_chunk
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Union

# ─── Marker constants (parser.rs) ─────────────────────────────────────
BEGIN_PATCH_MARKER = "*** Begin Patch"
END_PATCH_MARKER = "*** End Patch"
ADD_FILE_MARKER = "*** Add File: "
DELETE_FILE_MARKER = "*** Delete File: "
UPDATE_FILE_MARKER = "*** Update File: "
MOVE_TO_MARKER = "*** Move to: "
EOF_MARKER = "*** End of File"
CHANGE_CONTEXT_MARKER = "@@ "
EMPTY_CHANGE_CONTEXT_MARKER = "@@"

# We always use lenient mode (matching the codex / Grok default).
PARSE_IN_STRICT_MODE = False


class ParseError(Exception):
    """InvalidPatchError | InvalidHunkError."""

    def __init__(self, message: str, *, line_number: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.line_number = line_number


@dataclass
class UpdateFileChunk:
    change_context: str | None
    old_lines: list[str] = field(default_factory=list)
    new_lines: list[str] = field(default_factory=list)
    is_end_of_file: bool = False


@dataclass
class AddFile:
    path: str
    contents: str


@dataclass
class DeleteFile:
    path: str


@dataclass
class UpdateFile:
    path: str
    move_path: str | None
    chunks: list[UpdateFileChunk]


Hunk = Union[AddFile, DeleteFile, UpdateFile]


@dataclass
class ParsedPatch:
    hunks: list[Hunk]
    patch: str


class ParseMode(Enum):
    Strict = auto()
    Lenient = auto()


def parse_patch(patch: str) -> ParsedPatch:
    mode = ParseMode.Strict if PARSE_IN_STRICT_MODE else ParseMode.Lenient
    return parse_patch_text(patch, mode)


def parse_patch_text(patch: str, mode: ParseMode) -> ParsedPatch:
    lines = patch.strip().splitlines()
    try:
        check_patch_boundaries_strict(lines)
        work = lines
    except ParseError as e:
        if mode is ParseMode.Strict:
            raise
        work = check_patch_boundaries_lenient(lines, e)

    hunks: list[Hunk] = []
    # boundary checks guarantee len >= 2
    last_line_index = max(0, len(work) - 1)
    remaining = work[1:last_line_index]
    line_number = 2
    while remaining:
        hunk, hunk_lines = parse_one_hunk(remaining, line_number)
        hunks.append(hunk)
        line_number += hunk_lines
        remaining = remaining[hunk_lines:]
    return ParsedPatch(hunks=hunks, patch="\n".join(work))


def check_patch_boundaries_strict(lines: list[str]) -> None:
    if not lines:
        first = last = None
    elif len(lines) == 1:
        first = last = lines[0]
    else:
        first, last = lines[0], lines[-1]
    check_start_and_end_lines_strict(first, last)


def check_patch_boundaries_lenient(
    original_lines: list[str], original_parse_error: ParseError
) -> list[str]:
    if len(original_lines) >= 2:
        first, last = original_lines[0], original_lines[-1]
        if (
            first in {"<<EOF", "<<'EOF'", '<<"EOF"'}
            and last.endswith("EOF")
            and len(original_lines) >= 4
        ):
            inner = original_lines[1:-1]
            check_patch_boundaries_strict(inner)
            return inner
    raise original_parse_error


def check_start_and_end_lines_strict(
    first_line: str | None, last_line: str | None
) -> None:
    first = first_line.strip() if first_line is not None else None
    last = last_line.strip() if last_line is not None else None
    if first == BEGIN_PATCH_MARKER and last == END_PATCH_MARKER:
        return
    if first != BEGIN_PATCH_MARKER:
        raise ParseError("The first line of the patch must be '*** Begin Patch'")
    raise ParseError("The last line of the patch must be '*** End Patch'")


def parse_one_hunk(lines: list[str], line_number: int) -> tuple[Hunk, int]:
    first_line = lines[0].strip()
    if first_line.startswith(ADD_FILE_MARKER):
        path = first_line[len(ADD_FILE_MARKER) :]
        contents = ""
        parsed_lines = 1
        for add_line in lines[1:]:
            if add_line.startswith("+"):
                contents += add_line[1:] + "\n"
                parsed_lines += 1
            else:
                break
        return AddFile(path=path, contents=contents), parsed_lines

    if first_line.startswith(DELETE_FILE_MARKER):
        path = first_line[len(DELETE_FILE_MARKER) :]
        return DeleteFile(path=path), 1

    if first_line.startswith(UPDATE_FILE_MARKER):
        path = first_line[len(UPDATE_FILE_MARKER) :]
        remaining = lines[1:]
        parsed_lines = 1
        move_path = None
        if remaining and remaining[0].startswith(MOVE_TO_MARKER):
            move_path = remaining[0][len(MOVE_TO_MARKER) :]
            remaining = remaining[1:]
            parsed_lines += 1

        chunks: list[UpdateFileChunk] = []
        while remaining:
            if remaining[0].strip() == "":
                parsed_lines += 1
                remaining = remaining[1:]
                continue
            if remaining[0].startswith("***"):
                break
            chunk, chunk_lines = parse_update_file_chunk(
                remaining, line_number + parsed_lines, not chunks
            )
            chunks.append(chunk)
            parsed_lines += chunk_lines
            remaining = remaining[chunk_lines:]

        if not chunks:
            raise ParseError(
                f"Update file hunk for path '{path}' is empty",
                line_number=line_number,
            )
        return (
            UpdateFile(path=path, move_path=move_path, chunks=chunks),
            parsed_lines,
        )

    raise ParseError(
        f"'{first_line}' is not a valid hunk header. Valid hunk headers: "
        "'*** Add File: {path}', '*** Delete File: {path}', '*** Update File: {path}'",
        line_number=line_number,
    )


def parse_update_file_chunk(
    lines: list[str],
    line_number: int,
    allow_missing_context: bool,
) -> tuple[UpdateFileChunk, int]:
    if not lines:
        raise ParseError(
            "Update hunk does not contain any lines", line_number=line_number
        )

    if lines[0] == EMPTY_CHANGE_CONTEXT_MARKER:
        change_context: str | None = None
        start_index = 1
    elif lines[0].startswith(CHANGE_CONTEXT_MARKER):
        change_context = lines[0][len(CHANGE_CONTEXT_MARKER) :]
        start_index = 1
    else:
        if not allow_missing_context:
            raise ParseError(
                f"Expected update hunk to start with a @@ context marker, got: '{lines[0]}'",
                line_number=line_number,
            )
        change_context = None
        start_index = 0

    if start_index >= len(lines):
        raise ParseError(
            "Update hunk does not contain any lines",
            line_number=line_number + 1,
        )

    chunk = UpdateFileChunk(
        change_context=change_context,
        old_lines=[],
        new_lines=[],
        is_end_of_file=False,
    )
    parsed_lines = 0
    for line in lines[start_index:]:
        if line == EOF_MARKER:
            if parsed_lines == 0:
                raise ParseError(
                    "Update hunk does not contain any lines",
                    line_number=line_number + 1,
                )
            chunk.is_end_of_file = True
            parsed_lines += 1
            break
        if line == "":
            chunk.old_lines.append("")
            chunk.new_lines.append("")
        else:
            ch = line[0]
            if ch == " ":
                chunk.old_lines.append(line[1:])
                chunk.new_lines.append(line[1:])
            elif ch == "+":
                chunk.new_lines.append(line[1:])
            elif ch == "-":
                chunk.old_lines.append(line[1:])
            else:
                if parsed_lines == 0:
                    raise ParseError(
                        f"Unexpected line found in update hunk: '{line}'. "
                        "Every line should start with ' ' (context line), "
                        "'+' (added line), or '-' (removed line)",
                        line_number=line_number + 1,
                    )
                break
        parsed_lines += 1

    return chunk, parsed_lines + start_index
