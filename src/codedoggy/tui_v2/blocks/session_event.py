"""SessionEventBlock — port spirit of ``scrollback/blocks/session_event.rs``.

Concise session-level markers in scrollback (turn end, compact, reauth, …).
Content-only rows; chrome is applied by the host.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codedoggy.tui_v2.blocks.tool.common import wrap_text

if TYPE_CHECKING:
    from prompt_toolkit.formatted_text import StyleAndTextTuples
else:
    StyleAndTextTuples = list  # type: ignore[misc, assignment]

S_BODY = "class:grok.muted"
S_ERROR = "class:grok.accent_error"

_ELLIPSIS = "\u2026"  # …


def _label_for(kind: str, detail: str) -> str:
    """Map kind (+ optional detail) to the Doggy display string."""
    k = (kind or "").strip().lower().replace("-", "_")
    d = (detail or "").strip()

    if k in {"turn_completed", "worked"}:
        if d:
            return f"Worked for {d}"
        return "Turn completed"

    if k == "turn_cancelled":
        return "Turn cancelled"

    if k == "turn_failed":
        if d:
            return f"Turn failed: {d}"
        return "Turn failed"

    if k == "turn_queued":
        return "Turn queued"

    if k == "compact_started":
        return f"Compacting context{_ELLIPSIS}"

    if k == "compact_done":
        return "Context compacted"

    if k == "compact_failed":
        if d:
            return f"Compaction failed: {d}"
        return "Compaction failed"

    if k == "reauth":
        return "Re-authentication required"

    if k == "context_too_large":
        return "Context too large"

    if k == "memory_saved":
        return "Memory saved"

    if k == "model_unavailable":
        if d:
            return f"Model unavailable: {d}"
        return "Model unavailable"

    if k == "hook_annotation":
        return d or "Hook"

    if k == "goal_completed":
        if d:
            return f"Goal completed: {d}"
        return "Goal completed"

    if k == "retry_failed":
        if d:
            return f"Retry failed: {d}"
        return "Retry failed"

    if k == "turn_halted":
        if d:
            return f"Turn halted: {d}"
        return "Turn halted"

    if k == "max_turns":
        return "Max turns reached"

    # generic — prefer detail, else the raw kind token
    if k == "generic" or not k:
        return d or "generic"
    return d or kind


def paint_session_event(
    kind: str,
    *,
    detail: str = "",
    width: int,
    selected: bool = False,
) -> list[StyleAndTextTuples]:
    """Paint a session-event marker as content-only wrapped rows.

    Kinds (labels):
    - ``turn_completed`` / ``worked`` → ``Worked for {detail}`` or ``Turn completed``
    - ``turn_cancelled`` → ``Turn cancelled``
    - ``turn_failed`` → ``Turn failed`` (+ detail)
    - ``turn_queued`` → ``Turn queued``
    - ``compact_started`` → ``Compacting context…``
    - ``compact_done`` → ``Context compacted``
    - ``compact_failed`` → ``Compaction failed`` (+ detail)
    - ``reauth`` → ``Re-authentication required``
    - ``context_too_large`` → ``Context too large``
    - ``memory_saved`` → ``Memory saved``
    - ``model_unavailable`` → ``Model unavailable`` (+ detail)
    - ``hook_annotation`` → detail or ``Hook``
    - ``goal_completed`` → ``Goal completed`` (+ detail)
    - ``retry_failed`` → ``Retry failed`` (+ detail)
    - ``turn_halted`` → ``Turn halted`` (+ detail)
    - ``max_turns`` → ``Max turns reached``
    - ``generic`` / unknown → ``detail`` or kind
    """
    text = _label_for(kind, detail)
    k = (kind or "").strip().lower().replace("-", "_")
    # Actionable failures / reauth use error accent (Grok warning); rest muted.
    failed = k in {
        "turn_failed",
        "retry_failed",
        "compact_failed",
        "model_unavailable",
        "turn_halted",
        "max_turns",
        "reauth",
        "context_too_large",
    }
    style = S_ERROR if failed else S_BODY
    if selected:
        style = f"{style} reverse"

    w = max(8, int(width))
    if not text:
        return [[("", "")]]

    rows: list[StyleAndTextTuples] = []
    for logical in text.splitlines() or [text]:
        for piece in wrap_text(logical, w):
            rows.append([(style, piece)])
    return rows or [[(style, text)]]


__all__ = ["paint_session_event"]
