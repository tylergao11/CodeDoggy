"""Scrollback state + paint using ported Grok painters + layout chrome."""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field
from typing import Any

from prompt_toolkit.formatted_text import StyleAndTextTuples

from codedoggy.tui_v2.project import ScrollItem, project_message
from codedoggy.tui_v2.text_selection import TextSel
from codedoggy.tui_v2.verb_group import build_display_items


@dataclass
class ScrollbackState:
    items: list[ScrollItem] = field(default_factory=list)
    selected: int = -1
    follow_tail: bool = True
    tool_open: dict[str, ScrollItem] = field(default_factory=dict)
    line_owners: list[int] = field(default_factory=list)
    # Verb-group start indices the user expanded (show members).
    expanded_groups: set[int] = field(default_factory=set)
    # Last painted viewport rows (for text selection reconstruct).
    viewport_rows: list[StyleAndTextTuples] = field(default_factory=list)
    text_sel: TextSel | None = None
    # Stable viewport top (line index into full paint). Wheel scrolls this —
    # NOT block selection — so content does not jump when selection chrome
    # appears/disappears on a different item.
    scroll_offset: int = 0
    # Last full paint height (lines); used to clamp scroll_offset.
    content_lines: int = 0
    # Last *viewport* height used by render_scrollback (must match wheel clamp).
    viewport_h: int = 0
    # After keyboard select, render should bring selected item into view.
    scroll_to_selected: bool = False
    _id_seq: itertools.count = field(default_factory=lambda: itertools.count(1))
    _last_click_ms: float = 0.0
    _last_click_owner: int = -1

    def new_id(self, prefix: str) -> str:
        return f"{prefix}_{next(self._id_seq)}_{uuid.uuid4().hex[:6]}"

    def clear(self) -> None:
        self.items.clear()
        self.tool_open.clear()
        self.selected = -1
        self.follow_tail = True
        self.line_owners.clear()
        self.expanded_groups.clear()
        self.viewport_rows.clear()
        self.text_sel = None
        self.scroll_offset = 0
        self.content_lines = 0
        self.viewport_h = 0
        self.scroll_to_selected = False
        self._last_click_ms = 0.0
        self._last_click_owner = -1

    def expand_group_at_selection(self) -> bool:
        """Expand folded verb group covering selection. True if expanded."""
        from codedoggy.tui_v2.verb_group import find_group_at

        if self.selected < 0 or self.selected >= len(self.items):
            return False
        g = find_group_at(self.items, self.selected)
        if g is None or g.start in self.expanded_groups:
            return False
        self.expanded_groups.add(g.start)
        return True

    def collapse_group_at_selection(self) -> bool:
        """Re-fold expanded verb group covering selection. True if collapsed."""
        from codedoggy.tui_v2.verb_group import find_group_at

        if self.selected < 0 or self.selected >= len(self.items):
            return False
        g = find_group_at(self.items, self.selected)
        if g is None or g.start not in self.expanded_groups:
            return False
        self.expanded_groups.discard(g.start)
        return True

    def append_message(self, message: Any) -> None:
        for item in project_message(
            message, tool_open=self.tool_open, id_factory=self.new_id
        ):
            self.items.append(item)
        if self.follow_tail and self.items:
            self.selected = len(self.items) - 1

    def seed_from_messages(self, messages: Any) -> None:
        """Clear and re-project a full transcript (history restore / resume)."""
        seed_scrollback(self, messages)

    def set_draft(self, text: str) -> None:
        if self.items and self.items[-1].meta.get("draft"):
            self.items[-1].text = text
            self.items[-1].status = "running"
            return
        self.items.append(
            ScrollItem(
                kind="assistant",
                id=self.new_id("draft"),
                text=text,
                status="running",
                meta={"draft": True},
            )
        )
        if self.follow_tail:
            self.selected = len(self.items) - 1

    def clear_draft(self) -> None:
        self.items = [i for i in self.items if not i.meta.get("draft")]
        if self.selected >= len(self.items):
            self.selected = len(self.items) - 1 if self.items else -1

    def select_delta(self, d: int) -> None:
        """Move block selection (keyboard). Does not itself scroll by lines."""
        if not self.items:
            self.selected = -1
            return
        if self.selected < 0:
            self.selected = len(self.items) - 1
        self.selected = max(0, min(len(self.items) - 1, self.selected + d))
        self.follow_tail = self.selected >= len(self.items) - 1
        self.scroll_to_selected = True

    def scroll_by_lines(self, delta: int, *, viewport_h: int | None = None) -> None:
        """Move the viewport by painted lines (mouse wheel). Leaves selection alone.

        Clamping uses the **last rendered** ``viewport_h`` / ``content_lines`` so
        the wheel and ``render_scrollback`` agree on max_offset — mismatch here
        caused a one-frame jump at the top/bottom edges.
        """
        if delta == 0:
            return
        self.follow_tail = False
        self.scroll_to_selected = False
        h = max(1, int(viewport_h if viewport_h is not None else self.viewport_h or 20))
        total = max(0, int(self.content_lines))
        max_off = max(0, total - h) if total > h else 0
        before = self.scroll_offset
        self.scroll_offset = max(0, min(max_off, before + int(delta)))
        # Do **not** flip follow_tail here — wait until render with true h.
        # Premature follow_tail + different body_h estimate snapped offset.

    def selected_item(self) -> ScrollItem | None:
        if 0 <= self.selected < len(self.items):
            return self.items[self.selected]
        return None


def seed_scrollback(state: ScrollbackState, messages: Any) -> None:
    """Populate ``state`` from a message list (clears first).

    Pure orchestration over :func:`project_message` / :meth:`append_message`
    so resume / history wiring can be unit-tested without a TUI Application.
    """
    state.clear()
    for message in messages or ():
        state.append_message(message)


def _msg_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "").lower()
    role = getattr(message, "role", None)
    return str(getattr(role, "value", role) or "").lower()


def _msg_content(message: Any) -> str:
    if isinstance(message, dict):
        raw = message.get("content", "")
    else:
        raw = getattr(message, "content", "")
    if raw is None:
        return ""
    if isinstance(raw, list):
        # Rare multi-part content — join text-ish parts.
        parts: list[str] = []
        for p in raw:
            if isinstance(p, dict) and p.get("type") in {None, "text", "input_text"}:
                parts.append(str(p.get("text") or p.get("content") or ""))
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(raw)


def _msg_tool_calls(message: Any) -> list[Any]:
    if isinstance(message, dict):
        return list(message.get("tool_calls") or [])
    return list(getattr(message, "tool_calls", None) or [])


def _tc_id(tc: Any) -> str:
    if isinstance(tc, dict):
        return str(tc.get("id") or "")
    return str(getattr(tc, "id", "") or "")


def _has_assistant_text_since(
    state: ScrollbackState, since: int, *, prefer: str = ""
) -> bool:
    """True when scroll already shows assistant prose for this turn."""
    prefer = (prefer or "").strip()
    for it in state.items[max(0, since) :]:
        if it.kind != "assistant":
            continue
        text = (it.text or "").strip()
        if not text:
            continue
        if not prefer:
            return True
        if text == prefer or prefer in text or text in prefer:
            return True
    return False


def _message_represented_in_scroll(
    state: ScrollbackState, message: Any, *, since: int
) -> bool:
    """Best-effort duplicate check before projecting a live message.

    Intentionally conservative: when unsure whether a tool result still needs
    to land on an open call, returns False so :meth:`append_message` /
    :func:`project_message` can update ``tool_open`` in place.
    """
    role = _msg_role(message)
    items = state.items[max(0, since) :]

    if role == "user":
        content = _msg_content(message).strip()
        if not content:
            return True
        return any(i.kind == "user" and i.text.strip() == content for i in items)

    if role == "assistant":
        content = _msg_content(message).strip()
        reasoning = (
            message.get("reasoning_content")
            if isinstance(message, dict)
            else getattr(message, "reasoning_content", None)
        )
        tcs = _msg_tool_calls(message)

        content_ok = True
        if content:
            content_ok = any(
                i.kind == "assistant"
                and (i.text or "").strip()
                and (
                    (i.text or "").strip() == content
                    or content in (i.text or "").strip()
                    or (i.text or "").strip() in content
                )
                for i in items
            )

        reasoning_ok = True
        if isinstance(reasoning, str) and reasoning.strip():
            r = reasoning.strip()
            reasoning_ok = any(
                i.kind == "thinking" and (i.text or "").strip() == r for i in items
            )

        tools_ok = True
        for tc in tcs:
            cid = _tc_id(tc)
            if not cid:
                continue
            found = cid in state.tool_open or any(
                i.kind == "tool" and i.meta.get("call_id") == cid for i in items
            )
            if not found:
                tools_ok = False
                break

        if not content and not (
            isinstance(reasoning, str) and reasoning.strip()
        ) and not tcs:
            return True
        return content_ok and reasoning_ok and tools_ok

    if role == "tool":
        if isinstance(message, dict):
            cid = str(message.get("tool_call_id") or "")
        else:
            cid = str(getattr(message, "tool_call_id", "") or "")
        content = _msg_content(message)
        if cid:
            # Open call → project_message will attach the result; not "done".
            if cid in state.tool_open:
                existing = state.tool_open[cid]
                if existing.tool_result:
                    return True
                return False
            for i in items:
                if i.kind != "tool" or i.meta.get("call_id") != cid:
                    continue
                if i.tool_result or i.status in {"completed", "failed"}:
                    return True
                # Running shell without result still needs projection.
                return False
            # No matching tool row — project as orphan result.
            return False
        # No call id: match exact result text if present.
        if content:
            return any(
                i.kind == "tool" and (i.tool_result or "") == content for i in items
            )
        return True

    # system / unknown — skip re-projection
    return True


def reconcile_turn_finish(
    state: ScrollbackState,
    *,
    result_status: Any = None,
    result_error: str | None = None,
    final_text: str | None = None,
    live_messages: list[Any] | None = None,
    turn_scroll_start: int = 0,
) -> None:
    """Finish-of-turn scrollback reconciliation.

    - Drops streaming draft rows
    - Projects any *missing* ``live_messages`` for this turn (deduped carefully)
    - Marks still-``running`` tools as ``completed`` (orphan tool_call without
      a matching tool result); settles running assistant/thinking rows
    - If assistant prose is still absent, appends ``final_text`` as assistant
    - On failed / cancelled / error status, appends a system error row when
      ``result_error`` is non-empty
    """
    state.clear_draft()
    since = max(0, int(turn_scroll_start))

    # Prefer live transcript fill-in before final_text fallback.
    if live_messages:
        for msg in live_messages:
            try:
                if _message_represented_in_scroll(state, msg, since=since):
                    continue
                state.append_message(msg)
            except Exception:  # noqa: BLE001
                # Never let a bad message block the rest of finish.
                continue

    for it in state.items[since:]:
        if it.kind == "tool" and it.status == "running":
            it.status = "completed"
        elif it.kind in {"assistant", "thinking"} and it.status == "running":
            it.status = "done"

    text = (final_text or "").strip()
    if text and not _has_assistant_text_since(state, since, prefer=text):
        state.items.append(
            ScrollItem(
                kind="assistant",
                id=state.new_id("assistant"),
                text=text,
                status="done",
            )
        )
        if state.follow_tail:
            state.selected = len(state.items) - 1

    st = str(getattr(result_status, "value", result_status) or "").lower()
    err = (result_error or "").strip()
    if st in {"failed", "cancelled", "error"} and err:
        state.items.append(
            ScrollItem(
                kind="system",
                id=state.new_id("err"),
                text=err,
            )
        )
        if state.follow_tail:
            state.selected = len(state.items) - 1


def render_scrollback(
    state: ScrollbackState,
    *,
    width: int,
    height: int,
    welcome: StyleAndTextTuples | None = None,
) -> StyleAndTextTuples:
    w = max(20, width)
    h = max(4, height)
    state.line_owners = []

    if not state.items:
        if welcome:
            return _pad(welcome, h)
        return _pad([("class:grok.gray", "  (empty)\n")], h)

    lines: list[StyleAndTextTuples] = []
    # (first_line_index, owner_item_index) for each display entry
    entry_starts: list[tuple[int, int]] = []
    full_owners: list[int] = []
    selected_owners: set[int] = set()
    display = build_display_items(
        state.items, expanded_groups=state.expanded_groups
    )
    first = True
    selected_start_line: int | None = None

    for di in display:
        if di.group is not None:
            g = di.group
            owner = g.start
            selected = (
                state.selected >= 0 and g.start <= state.selected < g.end
            ) or (
                state.follow_tail
                and state.selected < 0
                and g.end >= len(state.items)
            )
            if selected:
                selected_owners.add(owner)
            entry_line = len(lines) + (0 if first else 1)
            entry_starts.append((entry_line, owner))
            if selected and selected_start_line is None:
                selected_start_line = entry_line
            if not first:
                lines.append([("", "\n")])
                full_owners.append(owner)
            first = False
            vg = ScrollItem(
                kind="verb_group",
                id=f"vg_{g.start}",
                text=g.label,
                status="running" if g.running else "done",
            )
            for row in vg.paint(width=w, selected=selected):
                lines.append(list(row))
                full_owners.append(owner)
            continue

        i = int(di.entry_index or 0)
        item = state.items[i]
        selected = i == state.selected or (
            state.follow_tail and state.selected < 0 and i == len(state.items) - 1
        )
        if selected:
            selected_owners.add(i)
        entry_line = len(lines) + (0 if first else 1)
        entry_starts.append((entry_line, i))
        if selected and selected_start_line is None:
            selected_start_line = entry_line
        if not first:
            lines.append([("", "\n")])
            full_owners.append(i)
        first = False
        try:
            rows = item.paint(width=w, selected=selected)
        except Exception as exc:  # noqa: BLE001
            rows = [
                [
                    ("class:grok.accent_error", f"[paint error {item.kind}: {exc}]"),
                    ("", "\n"),
                ]
            ]
        for row in rows:
            lines.append(list(row))
            full_owners.append(i)

    total = len(lines)
    state.content_lines = total
    state.viewport_h = h
    max_off = max(0, total - h) if total > h else 0

    if total <= h:
        offset = 0
        state.scroll_offset = 0
        state.follow_tail = True
        state.scroll_to_selected = False
    elif state.follow_tail:
        # Stick to true bottom for this paint height (no stale offset).
        offset = max_off
        state.scroll_offset = offset
        state.scroll_to_selected = False
    else:
        offset = max(0, min(int(state.scroll_offset), max_off))
        # Keyboard block selection: keep selected entry's first paint line visible.
        if state.scroll_to_selected and selected_start_line is not None:
            target = selected_start_line
            if target < offset:
                offset = target
            elif target >= offset + h:
                offset = min(max_off, max(0, target - h + 1))
            state.scroll_offset = offset
            state.scroll_to_selected = False
        else:
            # Soft-clamp only — never yank offset when already valid.
            state.scroll_offset = offset
            state.scroll_to_selected = False
        # At true bottom (same h as paint) → resume follow for live stream.
        if max_off > 0 and offset >= max_off:
            state.follow_tail = True
            offset = max_off
            state.scroll_offset = offset

    try:
        from codedoggy.tui_v2.selection import apply_viewport_selection_clip

        window = apply_viewport_selection_clip(
            lines,
            offset=offset,
            height=h,
            line_owners=full_owners,
            selected_owners=selected_owners,
            total_width=w,
        )
    except Exception:  # noqa: BLE001
        window = lines[offset : offset + h]

    owners = full_owners[offset : offset + h]
    state.line_owners = list(owners)

    # Text-drag highlight over viewport rows
    try:
        from codedoggy.tui_v2.text_selection import apply_text_selection_highlight

        painted = apply_text_selection_highlight(list(window), state.text_sel)
    except Exception:  # noqa: BLE001
        painted = list(window)

    state.viewport_rows = [list(r) for r in painted]

    out: StyleAndTextTuples = []
    for row in painted:
        out.extend(row)
        if out and not str(out[-1][1]).endswith("\n"):
            out.append(("", "\n"))
    for _ in range(h - len(painted)):
        out.append(("", "\n"))
        state.line_owners.append(-1)
        state.viewport_rows.append([("", "\n")])
    return out


def copy_text_selection(state: ScrollbackState) -> str:
    """Plain text for active text selection, or empty."""
    if state.text_sel is None or not state.viewport_rows:
        return ""
    try:
        from codedoggy.tui_v2.text_selection import reconstruct_selection_text

        return reconstruct_selection_text(state.viewport_rows, state.text_sel)
    except Exception:  # noqa: BLE001
        return ""


def entry_at_line(state: ScrollbackState, line: int) -> int | None:
    if 0 <= line < len(state.line_owners):
        idx = state.line_owners[line]
        return idx if idx >= 0 else None
    return None


def _pad(frags: StyleAndTextTuples, height: int) -> StyleAndTextTuples:
    text = "".join(t for _, t in frags)
    n = text.count("\n")
    out = list(frags)
    for _ in range(max(0, height - n)):
        out.append(("", "\n"))
    return out
