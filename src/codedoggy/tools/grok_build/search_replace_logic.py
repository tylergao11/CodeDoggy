"""search_replace match helpers — source port from Grok helpers.rs.

Ported from:
  implementations/grok_build/search_replace/helpers.rs
    find_normalized_match_positions, replace_normalized_matches
    replace_using_positions
  util/unicode_confusables.rs::build_offset_map
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from codedoggy.tools.util.unicode_confusables import normalize_confusables


@dataclass(frozen=True)
class NormalizedMatch:
    original_start: int
    original_len: int


class NormalizedMatchResultKind(Enum):
    NoMatch = auto()
    Matches = auto()
    Ambiguous = auto()


@dataclass
class NormalizedMatchResult:
    kind: NormalizedMatchResultKind
    matches: list[NormalizedMatch] | None = None


def build_offset_map(s: str) -> tuple[str, list[int]]:
    """Grok build_offset_map: (normalized_text, offset_map).

    offset_map[i] = original byte index for normalized byte i;
    offset_map[len(norm)] = len(s) sentinel.
    Python uses character indices (str is Unicode); same algorithm on codepoints.
    """
    from codedoggy.tools.util.unicode_confusables import CONFUSABLE_MAP

    norm_chars: list[str] = []
    # Map each norm index → original char index
    offset_map: list[int] = []
    for i, ch in enumerate(s):
        rep = CONFUSABLE_MAP.get(ch, ch)
        for _ in rep:
            offset_map.append(i)
            norm_chars.append(_)
    norm = "".join(norm_chars)
    offset_map.append(len(s))  # sentinel
    return norm, offset_map


def find_normalized_match_positions(text: str, pattern: str) -> NormalizedMatchResult:
    """Grok find_normalized_match_positions with roundtrip + overlap checks."""
    norm_text, offset_map = build_offset_map(text)
    norm_pattern = normalize_confusables(pattern)
    if not norm_pattern:
        return NormalizedMatchResult(NormalizedMatchResultKind.NoMatch)

    validated: list[NormalizedMatch] = []
    had_rejected = False
    start = 0
    while True:
        idx = norm_text.find(norm_pattern, start)
        if idx < 0:
            break
        norm_end = idx + len(norm_pattern)
        if norm_end >= len(offset_map):
            had_rejected = True
            break
        orig_start = offset_map[idx]
        orig_end = offset_map[norm_end]
        if orig_end <= orig_start:
            had_rejected = True
            start = idx + 1
            continue
        orig_slice = text[orig_start:orig_end]
        if normalize_confusables(orig_slice) != norm_pattern:
            had_rejected = True
            start = idx + 1
            continue
        validated.append(
            NormalizedMatch(original_start=orig_start, original_len=orig_end - orig_start)
        )
        start = idx + 1  # Grok match_indices are non-overlapping for same pattern?
        # Grok uses match_indices which finds non-overlapping. Use +len for non-overlap
        # Actually Rust match_indices is non-overlapping successive. Use:
        start = idx + len(norm_pattern)

    if not validated:
        if had_rejected:
            return NormalizedMatchResult(NormalizedMatchResultKind.Ambiguous)
        return NormalizedMatchResult(NormalizedMatchResultKind.NoMatch)

    for a, b in zip(validated, validated[1:]):
        if a.original_start + a.original_len > b.original_start:
            return NormalizedMatchResult(NormalizedMatchResultKind.Ambiguous)

    return NormalizedMatchResult(NormalizedMatchResultKind.Matches, validated)


def replace_using_positions(
    text: str,
    match_positions: list[int],
    old_string: str,
    new_string: str,
) -> tuple[str, list[int]]:
    new_text_parts: list[str] = []
    new_positions: list[int] = []
    last_end = 0
    out_len = 0
    for pos in match_positions:
        chunk = text[last_end:pos]
        new_text_parts.append(chunk)
        out_len += len(chunk)
        new_positions.append(out_len)
        new_text_parts.append(new_string)
        out_len += len(new_string)
        last_end = pos + len(old_string)
    new_text_parts.append(text[last_end:])
    return "".join(new_text_parts), new_positions


def replace_normalized_matches(
    text: str,
    matches: list[NormalizedMatch],
    new_string: str,
) -> tuple[str, list[int]]:
    result_parts: list[str] = []
    new_positions: list[int] = []
    last_end = 0
    out_len = 0
    for m in matches:
        chunk = text[last_end : m.original_start]
        result_parts.append(chunk)
        out_len += len(chunk)
        new_positions.append(out_len)
        result_parts.append(new_string)
        out_len += len(new_string)
        last_end = m.original_start + m.original_len
    result_parts.append(text[last_end:])
    return "".join(result_parts), new_positions


def build_nearest_match_hint(file_text: str, old_string: str) -> str:
    """Grok build_nearest_match_hint."""
    keyword = ""
    for tok in old_string.split():
        if len(tok) > len(keyword):
            keyword = tok
    if not keyword:
        first = old_string.strip().split("\n", 1)[0][:40]
        keyword = first
    if not keyword:
        return ""
    for i, line in enumerate(file_text.splitlines(), start=1):
        if keyword in line:
            snippet = line.strip()
            if len(snippet) > 120:
                snippet = snippet[:120] + "…"
            return f"\n\nNearest match: line {i}: {snippet}"
    return ""
