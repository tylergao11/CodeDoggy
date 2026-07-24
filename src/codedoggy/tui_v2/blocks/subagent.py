"""SubagentBlock — port of ``scrollback/blocks/subagent.rs``.

Always collapsed one-line header::

    Subagent running: "description"
    Subagent started: "description"
    Subagent completed in 1.2s: "description"
    Subagent failed in 0.5s: "description"
    Subagent cancelled in 0.3s: "description"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_toolkit.formatted_text import StyleAndTextTuples
else:
    StyleAndTextTuples = list  # type: ignore[misc, assignment]

S_BOLD = "class:grok.muted bold"
S_BOLD_SELECTED = "class:grok.primary bold"
S_MUTED = "class:grok.muted"

# Curly quotes Grok uses (U+201C / U+201D)
_LQ = "\u201c"
_RQ = "\u201d"
_EM = "\u2014"  # —
_ELLIPSIS = "\u2026"


def _format_elapsed_ms(ms: int | None) -> str:
    if ms is None:
        return "0.0s"
    secs = max(0.0, ms / 1000.0)
    if secs < 60.0:
        return f"{secs:.1f}s"
    mins = int(secs // 60.0)
    rem = secs - mins * 60.0
    return f"{mins}m{rem:.0f}s"


def quoted_desc(desc: str, max_width: int) -> str:
    if max_width <= 2:
        return f"{_LQ}{_ELLIPSIS}{_RQ}"
    inner = desc or ""
    budget = max_width - 2
    if len(inner) > budget:
        if budget <= 1:
            inner = _ELLIPSIS[:budget]
        else:
            inner = inner[: budget - 1] + _ELLIPSIS
    return f"{_LQ}{inner}{_RQ}"


def paint_subagent(
    description: str,
    *,
    width: int,
    status: str = "running",
    is_background: bool = False,
    elapsed_ms: int | None = None,
    error: str | None = None,
    activity_label: str | None = None,
    selected: bool = False,
) -> list[StyleAndTextTuples]:
    """Paint one-line subagent lifecycle header (content only; chrome wraps)."""
    st = (status or "").lower()
    bold = S_BOLD_SELECTED if selected else S_BOLD
    muted = S_MUTED
    w = max(8, int(width))

    if st in {"running", "pending", "started"}:
        verb = "started: " if is_background else "running: "
        activity = ""
        if activity_label and str(activity_label).strip():
            activity = f" {_EM} {activity_label.strip()}"
        # "Subagent " + verb ≈ 18 with "running: " / "started: "
        overhead = 9 + len(verb) + len(activity)
        desc = quoted_desc(description, w - overhead)
        spans: StyleAndTextTuples = [
            (bold, "Subagent "),
            (muted, verb),
            (muted, desc),
        ]
        if activity:
            spans.append((muted, activity))
        return [spans]

    if st in {"completed", "done", "success"}:
        time_str = _format_elapsed_ms(elapsed_ms)
        verb = f"completed in {time_str}: "
        desc = quoted_desc(description, w - 9 - len(verb))
        return [[(bold, "Subagent "), (muted, verb), (muted, desc)]]

    if st in {"failed", "error"}:
        time_str = _format_elapsed_ms(elapsed_ms)
        detail = f" ({error})" if error else ""
        verb = f"failed in {time_str}{detail}: "
        desc = quoted_desc(description, w - 9 - len(verb))
        return [[(bold, "Subagent "), (muted, verb), (muted, desc)]]

    if st in {"cancelled", "canceled"}:
        time_str = _format_elapsed_ms(elapsed_ms)
        verb = f"cancelled in {time_str}: "
        desc = quoted_desc(description, w - 9 - len(verb))
        return [[(bold, "Subagent "), (muted, verb), (muted, desc)]]

    # Fallback
    desc = quoted_desc(description, w - 10)
    return [[(bold, "Subagent "), (muted, desc)]]


__all__ = ["paint_subagent", "quoted_desc"]
