"""Fuzzy line-sequence matcher.

Ported from:
  grok-build/crates/codegen/xai-grok-tools/src/implementations/codex/apply_patch/seek_sequence.rs
  (itself ported from codex-rs/apply-patch/src/seek_sequence.rs)

Function map:
  seek_sequence ↔ seek_sequence
  normalise    ↔ normalise (inline in Rust)
"""

from __future__ import annotations


def seek_sequence(
    lines: list[str],
    pattern: list[str],
    start: int,
    eof: bool,
) -> int | None:
    """Find ``pattern`` within ``lines`` starting at ``start``.

    When ``eof`` is True, prefer matching at end of file first.
    Returns start index or None.
    """
    if not pattern:
        return start
    if len(pattern) > len(lines):
        return None

    search_start = (
        len(lines) - len(pattern) if eof and len(lines) >= len(pattern) else start
    )

    # Pass 1: exact
    last = len(lines) - len(pattern)
    for i in range(search_start, last + 1):
        if lines[i : i + len(pattern)] == pattern:
            return i

    # Pass 2: rstrip
    for i in range(search_start, last + 1):
        if all(
            lines[i + p_idx].rstrip() == pat.rstrip()
            for p_idx, pat in enumerate(pattern)
        ):
            return i

    # Pass 3: trim both
    for i in range(search_start, last + 1):
        if all(
            lines[i + p_idx].strip() == pat.strip()
            for p_idx, pat in enumerate(pattern)
        ):
            return i

    # Pass 4: Unicode normalise
    for i in range(search_start, last + 1):
        if all(
            _normalise(lines[i + p_idx]) == _normalise(pat)
            for p_idx, pat in enumerate(pattern)
        ):
            return i

    return None


def _normalise(s: str) -> str:
    # seek_sequence.rs normalise()
    out: list[str] = []
    for c in s.strip():
        if c in {
            "\u2010",
            "\u2011",
            "\u2012",
            "\u2013",
            "\u2014",
            "\u2015",
            "\u2212",
        }:
            out.append("-")
        elif c in {"\u2018", "\u2019", "\u201a", "\u201b"}:
            out.append("'")
        elif c in {"\u201c", "\u201d", "\u201e", "\u201f"}:
            out.append('"')
        elif c in {
            "\u00a0",
            "\u2002",
            "\u2003",
            "\u2004",
            "\u2005",
            "\u2006",
            "\u2007",
            "\u2008",
            "\u2009",
            "\u200a",
            "\u202f",
            "\u205f",
            "\u3000",
        }:
            out.append(" ")
        else:
            out.append(c)
    return "".join(out)
