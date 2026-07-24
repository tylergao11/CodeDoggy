"""Verb-group fold — port spirit of ``scrollback/state/verb_group.rs``.

Header labels like ``Read 3 files, Searched 1 pattern``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

VerbKind = Literal[
    "file",
    "search",
    "dir",
    "web_fetch",
    "web_search",
    "subagent",
]

# Tools that join a verb run when collapsed (Grok VerbGroupKind / non-destructive).
# Note: execute no longer groupable (Grok parity)
_READ = frozenset({"read", "read_file"})
_SEARCH = frozenset(
    {"grep", "search", "glob", "rg", "memory_search", "search_tool"}
)
_DIR = frozenset({"list_dir", "ls"})
_FETCH = frozenset({"web_fetch", "fetch"})
_WEB = frozenset({"web_search", "x_keyword_search", "x_semantic_search"})


def classify_verb(tool_name: str) -> VerbKind | None:
    n = (tool_name or "").strip().lower()
    if n in _READ:
        return "file"
    if n in _SEARCH:
        return "search"
    if n in _DIR:
        return "dir"
    if n in _FETCH:
        return "web_fetch"
    if n in _WEB:
        return "web_search"
    return None


def _pl(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def format_bucket(kind: VerbKind, n: int, *, running: bool) -> str:
    if kind == "file":
        return f"{'Reading' if running else 'Read'} {n} {_pl(n, 'file', 'files')}"
    if kind == "search":
        return (
            f"{'Searching' if running else 'Searched'} {n} "
            f"{_pl(n, 'pattern', 'patterns')}"
        )
    if kind == "dir":
        return f"{'Listing' if running else 'Listed'} {n} {_pl(n, 'dir', 'dirs')}"
    if kind == "web_fetch":
        return f"{'Fetching' if running else 'Fetched'} {n} {_pl(n, 'url', 'urls')}"
    if kind == "web_search":
        return f"{'Searching' if running else 'Searched'} {n} {_pl(n, 'query', 'queries')}"
    if kind == "subagent":
        return (
            f"{'Running' if running else 'Ran'} {n} "
            f"{_pl(n, 'subagent', 'subagents')}"
        )
    return f"{n} tools"


@dataclass
class VerbGroup:
    start: int
    end: int  # exclusive
    label: str
    running: bool = False
    failed: int = 0


@dataclass
class DisplayItem:
    entry_index: int | None = None
    group: VerbGroup | None = None


def item_verb_kind(item: Any) -> VerbKind | None:
    kind = str(getattr(item, "kind", "") or "")
    if kind == "subagent" and bool(getattr(item, "collapsed", True)):
        return "subagent"
    if kind != "tool":
        return None
    if not bool(getattr(item, "collapsed", True)):
        return None
    return classify_verb(str(getattr(item, "tool_name", "") or ""))


def build_display_items(
    items: list[Any],
    *,
    expanded_groups: set[int] | None = None,
) -> list[DisplayItem]:
    """Fold consecutive collapsed groupable tools when members >= 2.

    ``expanded_groups`` holds group *start* indices that the user expanded
    (Grok expand-in-place): those ranges emit individual entries instead of
    a single verb-group header.
    """
    expanded = expanded_groups or set()
    out: list[DisplayItem] = []
    i = 0
    n = len(items)
    while i < n:
        vk = item_verb_kind(items[i])
        if vk is None:
            out.append(DisplayItem(entry_index=i))
            i += 1
            continue
        j = i
        buckets: dict[VerbKind, int] = {}
        order: list[VerbKind] = []
        running = False
        failed = 0
        while j < n:
            ek = item_verb_kind(items[j])
            if ek is None:
                # collapsed thinking is transparent
                if (
                    str(getattr(items[j], "kind", "")) == "thinking"
                    and getattr(items[j], "collapsed", True)
                ):
                    j += 1
                    continue
                break
            if ek not in buckets:
                order.append(ek)
                buckets[ek] = 0
            buckets[ek] += 1
            st = str(getattr(items[j], "status", "") or "")
            if st in {"running", "pending"}:
                running = True
            if st in {"failed", "error"}:
                failed += 1
            j += 1
        members = sum(buckets.values())
        if members >= 2:
            if i in expanded:
                # Expanded: show each member (and transparent thinking) individually.
                for k in range(i, j):
                    out.append(DisplayItem(entry_index=k))
            else:
                parts = [
                    format_bucket(k, buckets[k], running=running) for k in order
                ]
                label = ", ".join(parts)
                if failed:
                    label = f"{label} — {failed} failed"
                out.append(
                    DisplayItem(
                        group=VerbGroup(
                            start=i,
                            end=j,
                            label=label,
                            running=running,
                            failed=failed,
                        )
                    )
                )
            i = j
        else:
            out.append(DisplayItem(entry_index=i))
            i += 1
    return out


def find_group_at(
    items: list[Any],
    index: int,
    *,
    expanded_groups: set[int] | None = None,  # noqa: ARG001 — API symmetry
) -> VerbGroup | None:
    """Return the natural verb group covering ``index`` (ignores expand state)."""
    for di in build_display_items(items, expanded_groups=None):
        if di.group is not None and di.group.start <= index < di.group.end:
            return di.group
    return None
