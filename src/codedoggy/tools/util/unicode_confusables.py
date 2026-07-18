"""Typography confusable Unicode → ASCII (Grok unicode_confusables.rs).

Used when exact search_replace fails: normalize for comparison only; apply
edits against the original file text when a unique normalized match exists.
"""

from __future__ import annotations

# Narrow map: high-confidence typography only (smart quotes, dashes, …, NBSP).
CONFUSABLE_MAP: dict[str, str] = {
    "\u201c": '"',  # left double quotation
    "\u201d": '"',  # right double quotation
    "\u2018": "'",  # left single quotation
    "\u2019": "'",  # right single quotation
    "\u2014": "--",  # em dash
    "\u2013": "-",  # en dash
    "\u2026": "...",  # ellipsis
    "\u00a0": " ",  # non-breaking space
}


def has_confusables(s: str) -> bool:
    return any(ch in CONFUSABLE_MAP for ch in s)


def normalize_confusables(s: str) -> str:
    """Replace confusable characters with ASCII equivalents."""
    if not s:
        return s
    return "".join(CONFUSABLE_MAP.get(ch, ch) for ch in s)


def find_normalized_spans(
    original: str,
    needle_normalized: str,
) -> list[tuple[int, int]]:
    """Find (start, end) exclusive spans in *original* matching normalized needle."""
    if not needle_normalized:
        return []

    # Parallel arrays: each char of normalized form maps to an original index
    norm_chars: list[str] = []
    orig_index_for_norm: list[int] = []
    for i, ch in enumerate(original):
        rep = CONFUSABLE_MAP.get(ch, ch)
        for j in range(len(rep)):
            norm_chars.append(rep[j])
            orig_index_for_norm.append(i)

    norm = "".join(norm_chars)
    spans: list[tuple[int, int]] = []
    start = 0
    nlen = len(needle_normalized)
    while True:
        idx = norm.find(needle_normalized, start)
        if idx < 0:
            break
        end_norm = idx + nlen
        if end_norm <= 0 or end_norm > len(orig_index_for_norm):
            break
        orig_start = orig_index_for_norm[idx]
        orig_end = orig_index_for_norm[end_norm - 1] + 1
        spans.append((orig_start, orig_end))
        start = idx + 1
    return spans
