"""Shortcuts bar — port of Grok ``views/shortcuts_bar.rs``.

Accepts ``HintItem`` lists from any source. Each view builds its own hints
dynamically. Doggy wiring: :func:`hints_for` for focus/running, :func:`render`
for paint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.utils import get_cwidth

# Style classes — Grok key bold secondary, action gray (theme.py fields).
STYLE_KEY = "class:grok.text_secondary"
STYLE_ACTION = "class:grok.gray"
STYLE_SEP = "class:grok.gray_dim"
STYLE_PENDING = "class:grok.warning"
STYLE_RIGHT = "class:grok.gray"

# Grok separator between hints: two spaces, box drawing, two spaces.
HINT_SEPARATOR = "  │  "


@dataclass(slots=True)
class HintItem:
    """A single hint for the shortcuts bar (Grok ``HintItem``).

    Carries semantic key data — the bar handles rendering.
    """

    keys: list[str]
    label: str
    custom_display: str | None = None
    description: str | None = None
    pinned: bool = False

    @classmethod
    def new(cls, key: str, label: str) -> HintItem:
        """Single-key hint."""
        return cls(keys=[key], label=label)

    @classmethod
    def paired(cls, a: str, b: str, label: str) -> HintItem:
        """Paired-key hint (e.g. Up/Down for nav, Left/Right for fold)."""
        return cls(keys=[a, b], label=label)

    def with_pinned(self) -> HintItem:
        """Mark this hint as pinned (always shown in compact mode)."""
        self.pinned = True
        return self

    def key_display(self) -> str:
        """Render the keys portion (e.g. ``"Up/Down"``, ``"Enter"``, ``"Ctrl+J"``)."""
        if self.custom_display is not None:
            return self.custom_display
        return "/".join(self.keys)


@dataclass(slots=True)
class CompactConfig:
    """Compact-mode configuration for the shortcuts bar."""

    max_visible: int
    help_hint: HintItem | None = None


@dataclass(slots=True)
class PendingHint:
    """Info needed to render the "press again" confirmation hint."""

    shortcut: str
    label: str


def compute_effective_hints(
    hints: Sequence[HintItem],
    compact: CompactConfig | None = None,
) -> list[HintItem]:
    """Compute the hint list the bar will actually render.

    Without ``compact``: every hint.
    With ``compact``: pinned always included; remaining
    ``max_visible − pinned_count`` slots filled with unpinned in order;
    trailing ``help_hint`` always appended.
    """
    if compact is None:
        return list(hints)

    pinned_count = sum(1 for h in hints if h.pinned)
    unpinned_budget = max(0, compact.max_visible - pinned_count)
    unpinned_used = 0
    out: list[HintItem] = []
    for h in hints:
        if h.pinned:
            out.append(h)
        elif unpinned_used < unpinned_budget:
            unpinned_used += 1
            out.append(h)
    if compact.help_hint is not None:
        out.append(compact.help_hint)
    return out


def hints_for(focus: str, running: bool) -> list[HintItem]:
    """Doggy wiring: build hints that match real ``app.py`` key bindings only.

    ``focus`` is a coarse pane id (``"prompt"``, ``"scrollback"``, …).
    Only advertise keys that tui_v2 actually binds — no invented panes or
    phantom shortcuts (no S-Tab, Ctrl+Q, Space, j/k).
    """
    focus = (focus or "prompt").lower()
    hints: list[HintItem] = []

    if focus in {"prompt", "input", "composer"}:
        # Enter submit (queues while a turn runs), Ctrl+J newline, Ctrl+L login.
        submit = "queue" if running else "send"
        hints.append(HintItem.new("Enter", submit))
        hints.append(HintItem.new("Ctrl+J", "newline"))
        hints.append(HintItem.new("Ctrl+L", "login"))
        if running:
            hints.append(HintItem.new("Esc", "cancel"))
        return hints

    if focus in {"scrollback", "tasks", "stream", "reading"}:
        # Tab ↔ prompt, ↑/↓ navigate, ←/→ fold/expand, Ctrl+L login.
        hints.append(HintItem.new("Tab", "prompt"))
        hints.append(HintItem.paired("Up", "Down", "nav").with_pinned())
        hints.append(HintItem.paired("Left", "Right", "fold"))
        hints.append(HintItem.new("Ctrl+L", "login"))
        if running:
            hints.append(HintItem.new("Esc", "cancel"))
        return hints

    # Fallback: keys that always apply (no fake quit binding).
    if running:
        hints.append(HintItem.new("Esc", "cancel"))
    hints.append(HintItem.new("Ctrl+L", "login"))
    return hints


def render(
    hints: Sequence[HintItem],
    *,
    pending: PendingHint | None = None,
    right_text: str | None = None,
    compact: CompactConfig | None = None,
    width: int | None = None,
) -> StyleAndTextTuples:
    """Render the shortcuts bar as styled fragments (Grok ``ShortcutsBar``).

    Layout: ``key:label  │  key:label …`` with optional right-aligned text.
    When ``pending`` is set, replaces all hints with ``key:press again to {label}``.
    """
    if width is None:
        try:
            import shutil

            width = shutil.get_terminal_size(fallback=(80, 24)).columns
        except Exception:  # noqa: BLE001
            width = 80

    if width <= 0:
        return [("", "")]

    fragments: StyleAndTextTuples = []

    if pending is not None:
        key_text = pending.shortcut
        label = f"press again to {pending.label}"
        fragments.append((STYLE_KEY, key_text))
        fragments.append((STYLE_ACTION, ":"))
        fragments.append((STYLE_PENDING, label))
        used = get_cwidth(key_text) + 1 + get_cwidth(label)
        if used < width:
            fragments.append((STYLE_ACTION, " " * (width - used)))
        return fragments

    effective = compute_effective_hints(hints, compact)
    x = 0
    for i, hint in enumerate(effective):
        if i > 0:
            sep_w = get_cwidth(HINT_SEPARATOR)
            if x + sep_w > width:
                break
            fragments.append((STYLE_SEP, HINT_SEPARATOR))
            x += sep_w

        key_text = hint.key_display()
        key_w = get_cwidth(key_text)
        if x + key_w > width:
            break
        fragments.append((STYLE_KEY, key_text))
        x += key_w

        if x + 1 > width:
            break
        fragments.append((STYLE_ACTION, ":"))
        x += 1

        label = hint.label
        label_w = get_cwidth(label)
        if x + label_w > width:
            break
        fragments.append((STYLE_ACTION, label))
        x += label_w

    if right_text:
        display = f"{right_text} "
        rw = get_cwidth(display)
        if 0 < rw < width:
            rx = width - rw
            if rx > x + 1:
                fragments.append((STYLE_RIGHT, " " * (rx - x)))
                fragments.append((STYLE_RIGHT, display))
                x = width

    if x < width:
        fragments.append((STYLE_ACTION, " " * (width - x)))

    return fragments


def render_hints(
    items: Iterable[tuple[str, str]],
    **kwargs: object,
) -> StyleAndTextTuples:
    """Convenience: render ``(key, label)`` pairs as a shortcuts bar."""
    hints = [HintItem.new(k, lab) for k, lab in items]
    return render(hints, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "HINT_SEPARATOR",
    "CompactConfig",
    "HintItem",
    "PendingHint",
    "compute_effective_hints",
    "hints_for",
    "render",
    "render_hints",
]
