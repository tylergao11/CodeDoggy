"""Turn status line — port of Grok ``views/turn_status.rs``.

Layout (running)::

    ⠧ Run command 0.2s              1m20s ⇣12k [stop]

Hidden when idle with no watchers (0 height / empty fragments). Appears
between scrollback and prompt.

Doggy wiring::

    render(running, started_at, tools_running, subagents_running, tick)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.utils import get_cwidth

# ---------------------------------------------------------------------------
# Glyphs — prefer codedoggy.tui_v2.glyphs when the glyphs agent has landed
# ---------------------------------------------------------------------------

try:
    from codedoggy.tui_v2.glyphs import (
        braille_spinner_frames as _glyph_braille,
        diamond_filled as _glyph_diamond,
        monitor_icon_frames as _glyph_monitor,
        token_arrow as _glyph_token_arrow,
    )
except Exception:  # noqa: BLE001
    _glyph_braille = None  # type: ignore[assignment]
    _glyph_diamond = None  # type: ignore[assignment]
    _glyph_monitor = None  # type: ignore[assignment]
    _glyph_token_arrow = None  # type: ignore[assignment]


def _braille_spinner_frames() -> Sequence[str]:
    if _glyph_braille is not None:
        return _glyph_braille()
    # Grok fancy frames: ⠋⠙⠹⠸⠼⠴⠦⠧
    return (
        "\u280b",
        "\u2819",
        "\u2839",
        "\u2838",
        "\u283c",
        "\u2834",
        "\u2826",
        "\u2827",
    )


def _monitor_icon_frames() -> Sequence[str]:
    if _glyph_monitor is not None:
        return _glyph_monitor()
    # ○ ◎ ◉ ◎
    return ("\u25cb", "\u25ce", "\u25c9", "\u25ce")


def _diamond_filled() -> str:
    if _glyph_diamond is not None:
        return _glyph_diamond()
    return "\u25c6"  # ◆


def _token_arrow() -> str:
    if _glyph_token_arrow is not None:
        return _glyph_token_arrow()
    return "\u21e3"  # ⇣


# ---------------------------------------------------------------------------
# Animation constants (Grok)
# ---------------------------------------------------------------------------

# Show each spinner frame for this many animation ticks.
# At ~30fps, 4 ticks = ~133ms per frame = ~7.5 spinner fps.
SPINNER_DIVISOR = 4

# Monitor-pulse dwell — twice SPINNER_DIVISOR (~3.75 fps). Calm breath for
# the idle still-running cue (○ ◎ ◉ ◎).
MONITOR_PULSE_DIVISOR = 8

# Pulse speed for every "waiting on you" diamond. sin²(tick*speed) has
# period π; at ~30fps ≈ 1.3s per cycle.
USER_WAITING_PULSE_SPEED = 0.08

# Style classes — map onto theme.py semantic fields.
STYLE_SPINNER = "class:grok.text_secondary"
STYLE_LABEL = "class:grok.text_secondary"
STYLE_TOOL = "class:grok.accent_success"
STYLE_TIMER = "class:grok.gray"
STYLE_STILL = "class:grok.accent_system"
STYLE_STOP = "class:grok.accent_error"
STYLE_DIAMOND = "class:grok.accent_user"
STYLE_DIM = "class:grok.gray"


# ---------------------------------------------------------------------------
# Watchers / still-running (Grok Watchers + format_still_running)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Watchers:
    """Counts of idle-surviving background work (Grok ``Watchers``)."""

    commands: int = 0
    monitors: int = 0
    loops: int = 0
    subagents: int = 0
    workflows: int = 0

    def total(self) -> int:
        return (
            self.commands
            + self.monitors
            + self.loops
            + self.subagents
            + self.workflows
        )

    def awaitable_work(self) -> int:
        return self.commands + self.monitors + self.subagents


def format_still_running(kinds: Sequence[tuple[int, str]]) -> str | None:
    """Counts-first ``… still running`` cue from ``(count, noun)`` pairs.

    e.g. ``"1 command · 2 monitors still running"``. ``None`` when every
    count is zero. Single owner of the format so idle cue and dashboard
    labels cannot drift.
    """
    parts: list[str] = []
    for count, noun in kinds:
        if count <= 0:
            continue
        plural = "" if count == 1 else "s"
        parts.append(f"{count} {noun}{plural}")
    if not parts:
        return None
    return " \u00b7 ".join(parts) + " still running"


def still_running_label(watchers: Watchers) -> str | None:
    """Idle watcher cue label. ``None`` when no watchers are live."""
    return format_still_running(
        [
            (watchers.commands, "command"),
            (watchers.monitors, "monitor"),
            (watchers.loops, "loop"),
            (watchers.subagents, "subagent"),
            (watchers.workflows, "workflow"),
        ]
    )


# ---------------------------------------------------------------------------
# Timers / tokens
# ---------------------------------------------------------------------------


def format_turn_timer(seconds: float) -> str:
    """Format a duration as a compact human-friendly string (Grok ``format_duration``).

    - Under 10s: ``"5.2s"``
    - 10–59s: ``"32s"``
    - 1m–59m: ``"2m5s"``
    - 1h+: ``"1h2m"``
    """
    seconds = max(0.0, float(seconds))
    total_secs = int(seconds)
    if total_secs < 10:
        return f"{seconds:.1f}s"
    if total_secs < 60:
        return f"{total_secs}s"
    mins = total_secs // 60
    secs = total_secs % 60
    if mins < 60:
        return f"{mins}m{secs}s"
    hours = mins // 60
    remaining_mins = mins % 60
    return f"{hours}h{remaining_mins}m"


def format_tokens_short(tokens: int) -> str:
    """Compact token count for the turn-status right side (Grok)."""
    tokens = int(tokens)
    if tokens < 1000:
        return str(tokens)
    if tokens < 100_000:
        k = tokens / 1000.0
        if tokens < 10_000:
            return f"{k:.2f}k"
        return f"{k:.1f}k"
    if tokens < 1_000_000:
        return f"{tokens // 1000}k"
    m = tokens / 1_000_000.0
    if tokens < 10_000_000:
        return f"{m:.2f}m"
    return f"{m:.1f}m"


def pulse_brightness(tick: int, speed: float = USER_WAITING_PULSE_SPEED) -> float:
    """``sin²(tick * speed)`` — Grok ``pulse_brightness``."""
    return math.sin(tick * speed) ** 2


def should_show(
    *,
    running: bool,
    tools_running: int = 0,
    subagents_running: int = 0,
    drain_blocked: bool = False,
    watchers: Watchers | None = None,
) -> bool:
    """Whether the turn status line should be visible (Grok ``should_show``)."""
    if running or drain_blocked:
        return True
    if watchers is not None and watchers.total() > 0:
        return True
    return (tools_running + subagents_running) > 0


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _truncate(text: str, max_width: int) -> str:
    if max_width <= 0:
        return ""
    if get_cwidth(text) <= max_width:
        return text
    if max_width <= 1:
        return "…"[:max_width]
    # Walk code points until width fits.
    out: list[str] = []
    w = 0
    for ch in text:
        cw = get_cwidth(ch)
        if w + cw > max_width - 1:
            break
        out.append(ch)
        w += cw
    return "".join(out) + "…"


def _spinner_frame(tick: int) -> str:
    frames = _braille_spinner_frames()
    idx = (int(tick) // SPINNER_DIVISOR) % len(frames)
    return frames[idx]


def _monitor_frame(tick: int) -> str:
    frames = _monitor_icon_frames()
    idx = (int(tick) // MONITOR_PULSE_DIVISOR) % len(frames)
    return frames[idx]


def render(
    running: bool,
    started_at: float | None,
    tools_running: int = 0,
    subagents_running: int = 0,
    tick: int = 0,
    *,
    label: str | None = None,
    total_tokens: int | None = None,
    pending_user_input: bool = False,
    drain_blocked: bool = False,
    show_stop: bool = True,
    width: int | None = None,
    now: float | None = None,
    monitors_running: int = 0,
    loops_running: int = 0,
    workflows_running: int = 0,
) -> StyleAndTextTuples:
    """Render the turn status line (Doggy hook + Grok still-running cues).

    Parameters
    ----------
    running:
        Whether a foreground turn is active.
    started_at:
        ``time.monotonic()`` timestamp when the turn (or phase) began.
    tools_running / subagents_running:
        Background watcher counts for the idle ``still running`` cue.
    tick:
        Animation tick (host increments ~30/s).
    label:
        Optional activity label override (e.g. ``"Thinking…"``, tool title).
    """
    if width is None:
        try:
            import shutil

            width = shutil.get_terminal_size(fallback=(80, 24)).columns
        except Exception:  # noqa: BLE001
            width = 80

    if width < 10:
        return [("", "")]

    now = time.monotonic() if now is None else now
    watchers = Watchers(
        commands=max(0, int(tools_running)),
        monitors=max(0, int(monitors_running)),
        loops=max(0, int(loops_running)),
        subagents=max(0, int(subagents_running)),
        workflows=max(0, int(workflows_running)),
    )

    # Drain-blocked: pulsing diamond + "agent idle ~ waiting on your edit"
    if drain_blocked and not running:
        diamond = _diamond_filled()
        return [
            (STYLE_DIAMOND, f"{diamond} "),
            (STYLE_DIM, "agent idle ~ waiting on your edit"),
        ]

    # Idle / not running with watchers: still-running cue
    if not running:
        cue = still_running_label(watchers)
        if cue is None:
            return [("", "")]
        icon = _monitor_frame(tick)
        return [
            (STYLE_STILL, f"{icon} "),
            (STYLE_DIM, cue),
        ]

    # ── Running turn ──────────────────────────────────────────────────
    elapsed = 0.0
    if started_at is not None:
        elapsed = max(0.0, now - float(started_at))

    # Spinner / diamond
    if pending_user_input:
        spinner_str = f"{_diamond_filled()} "
        spinner_style = STYLE_DIAMOND
    else:
        spinner_str = f"{_spinner_frame(tick)} "
        spinner_style = STYLE_SPINNER

    # Activity label
    if label:
        activity = label
    elif tools_running > 0:
        activity = "Running…"
    else:
        activity = "Thinking…"

    # Phase timer (Grok never truncates this)
    phase_timer = f" {format_turn_timer(elapsed)}"

    # Right side: turn timer + optional tokens + [stop]
    turn_timer = format_turn_timer(elapsed)
    if total_tokens and total_tokens > 0:
        turn_timer_str = f"{turn_timer} {_token_arrow()}{format_tokens_short(total_tokens)}"
    else:
        turn_timer_str = turn_timer

    cancel_str = " [stop]" if show_stop else ""
    right = f"{turn_timer_str}{cancel_str}"
    right_w = get_cwidth(right)
    spinner_w = get_cwidth(spinner_str)
    phase_w = get_cwidth(phase_timer)
    min_gap = 1
    available = max(
        0,
        width - spinner_w - phase_w - min_gap - right_w,
    )
    display_label = _truncate(activity, available)

    left = f"{spinner_str}{display_label}{phase_timer}"
    left_w = get_cwidth(left)
    gap = max(min_gap, width - left_w - right_w)

    fragments: StyleAndTextTuples = [
        (spinner_style, spinner_str),
        (STYLE_TOOL if tools_running > 0 else STYLE_LABEL, display_label),
        (STYLE_TIMER, phase_timer),
        (STYLE_TIMER, " " * gap),
        (STYLE_TIMER, turn_timer_str),
    ]
    if show_stop:
        fragments.append((STYLE_STOP, cancel_str))
    return fragments


__all__ = [
    "MONITOR_PULSE_DIVISOR",
    "SPINNER_DIVISOR",
    "USER_WAITING_PULSE_SPEED",
    "Watchers",
    "format_still_running",
    "format_tokens_short",
    "format_turn_timer",
    "pulse_brightness",
    "render",
    "should_show",
    "still_running_label",
]
