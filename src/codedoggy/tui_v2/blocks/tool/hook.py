"""Hook run display helpers — port of ``blocks/tool/hook.rs``.

Hooks attach to tool blocks as:
- inline suffix on header: ``  [hooks: 2]`` or ``  [hooks: 1/1]``
- optional expanded body lines under the tool
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from codedoggy.tui_v2.blocks.tool.common import (
    S_DIM,
    S_ERROR,
    S_MUTED,
    S_SUCCESS,
    Fragment,
    Rows,
    row,
)

HookStatusKind = Literal["success", "skipped", "blocked", "failed"]


@dataclass
class HookRunEntry:
    name: str
    status: HookStatusKind = "success"
    detail: str = ""
    elapsed_ms: int | None = None
    output: str | None = None


@dataclass
class ToolCallHookData:
    pre_hooks: list[HookRunEntry] = field(default_factory=list)
    post_hooks: list[HookRunEntry] = field(default_factory=list)
    # (event_name, runs)
    lifecycle: list[tuple[str, list[HookRunEntry]]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.pre_hooks and not self.post_hooks and not self.lifecycle

    def all_runs(self) -> list[HookRunEntry]:
        runs = list(self.pre_hooks) + list(self.post_hooks)
        for _, rs in self.lifecycle:
            runs.extend(rs)
        return runs


def count_hooks(entries: list[HookRunEntry]) -> tuple[int, int]:
    success = failed = 0
    for r in entries:
        if r.status in {"success", "blocked"}:
            success += 1
        elif r.status == "failed":
            failed += 1
    return success, failed


def hooks_inline_suffix(data: ToolCallHookData) -> list[Fragment] | None:
    """``  [hooks: N]`` or ``  [hooks: ok/fail]`` spans for tool headers."""
    success, failed = count_hooks(data.all_runs())
    if success == 0 and failed == 0:
        return None
    spans: list[Fragment] = [(S_MUTED, "  [hooks: ")]
    if success > 0:
        spans.append((f"{S_SUCCESS} dim", str(success)))
    if success > 0 and failed > 0:
        spans.append((S_MUTED, "/"))
    if failed > 0:
        spans.append((f"{S_ERROR} dim", str(failed)))
    spans.append((S_MUTED, "]"))
    return spans


def paint_hook_body(data: ToolCallHookData, *, width: int = 60) -> Rows:
    """Expanded hook detail lines (pre/post lists)."""
    if data.is_empty():
        return []
    rows: Rows = []
    for phase, runs in (("pre", data.pre_hooks), ("post", data.post_hooks)):
        for r in runs:
            if r.status == "skipped":
                continue
            mark = {
                "success": "ok",
                "blocked": "blocked",
                "failed": "fail",
            }.get(r.status, r.status)
            style = S_ERROR if r.status == "failed" else S_DIM
            label = f"    {phase}:{r.name} ({mark})"
            if r.elapsed_ms is not None:
                label += f" {r.elapsed_ms}ms"
            rows.append(row((style, label[: max(8, width)])))
            if r.detail:
                rows.append(row((S_MUTED, f"      {r.detail[: max(4, width - 6)]}")))
    for event, runs in data.lifecycle:
        for r in runs:
            if r.status == "skipped":
                continue
            rows.append(
                row((S_DIM, f"    {event}:{r.name}"[: max(8, width)]))
            )
    return rows


def parse_hooks_from_meta(meta: dict | None) -> ToolCallHookData | None:
    """Optional Doggy projection: meta['hooks'] structured or simple counts."""
    if not meta:
        return None
    raw = meta.get("hooks")
    if raw is None:
        return None
    data = ToolCallHookData()
    if isinstance(raw, dict):
        for key, bucket in (("pre", "pre_hooks"), ("post", "post_hooks")):
            items = raw.get(key) or raw.get(bucket) or []
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, str):
                    getattr(data, bucket).append(HookRunEntry(name=it))
                elif isinstance(it, dict):
                    getattr(data, bucket).append(
                        HookRunEntry(
                            name=str(it.get("name") or "hook"),
                            status=str(it.get("status") or "success"),  # type: ignore[arg-type]
                            detail=str(it.get("detail") or it.get("error") or ""),
                            elapsed_ms=it.get("elapsed_ms"),
                            output=it.get("output"),
                        )
                    )
        return data if not data.is_empty() else None
    if isinstance(raw, list):
        for it in raw:
            if isinstance(it, str):
                data.post_hooks.append(HookRunEntry(name=it))
            elif isinstance(it, dict):
                data.post_hooks.append(
                    HookRunEntry(
                        name=str(it.get("name") or "hook"),
                        status=str(it.get("status") or "success"),  # type: ignore[arg-type]
                    )
                )
        return data if not data.is_empty() else None
    return None


__all__ = [
    "HookRunEntry",
    "ToolCallHookData",
    "count_hooks",
    "hooks_inline_suffix",
    "paint_hook_body",
    "parse_hooks_from_meta",
]
