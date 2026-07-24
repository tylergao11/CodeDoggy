"""Project Doggy Message → Grok block paint (via layout chrome)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prompt_toolkit.formatted_text import StyleAndTextTuples

from codedoggy.tui_v2.blocks.bg_task import paint_bg_task
from codedoggy.tui_v2.blocks.markdown import render_markdown
from codedoggy.tui_v2.blocks.session_event import paint_session_event
from codedoggy.tui_v2.blocks.subagent import paint_subagent
from codedoggy.tui_v2.blocks.system import paint_system
from codedoggy.tui_v2.blocks.thinking import paint_thinking
from codedoggy.tui_v2.blocks.tool import paint_tool
from codedoggy.tui_v2.blocks.user import paint_user_prompt
from codedoggy.tui_v2.chrome import (
    accent_for_kind,
    bullet_for_kind,
    with_chrome,
)

def _role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "").lower()
    role = getattr(message, "role", None)
    return str(getattr(role, "value", role) or "").lower()


def _get(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            import json

            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _ensure_rows(rows: Any) -> list[StyleAndTextTuples]:
    """Normalize painter output to list of rows."""
    if not rows:
        return [[("", "")]]
    # Already list of rows
    first = rows[0]
    if isinstance(first, tuple) and len(first) >= 2 and isinstance(first[0], str):
        # single row of fragments
        return [list(rows)]  # type: ignore[list-item]
    out: list[StyleAndTextTuples] = []
    for row in rows:
        if isinstance(row, list):
            out.append(row)
        else:
            out.append([("class:grok.gray", str(row))])
    return out


@dataclass
class ScrollItem:
    kind: str
    id: str
    collapsed: bool = False
    truncated: bool = False  # when True and not collapsed → first/last body (read/execute)
    status: str = "done"
    text: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: str = ""
    elapsed_ms: int | None = None
    meta: dict = field(default_factory=dict)

    def tool_display_collapsed(self) -> bool:
        return self.collapsed

    def tool_display_truncated(self) -> bool:
        return (not self.collapsed) and self.truncated

    def __post_init__(self) -> None:
        # bg_task rows default collapsed (three-state fold like tools).
        if self.kind == "bg_task":
            self.collapsed = True

    def cycle_fold(self) -> None:
        """Collapsed → Truncated → Expanded → Collapsed (tools/bg_task foldable)."""
        if self.kind in {"tool", "bg_task"}:
            if self.collapsed:
                self.collapsed = False
                self.truncated = True  # open truncated first (Grok default for long tools)
            elif self.truncated:
                self.truncated = False  # full expanded
            else:
                self.collapsed = True
                self.truncated = False
        else:
            # binary for user/thinking/assistant
            self.collapsed = not self.collapsed

    def paint(self, *, width: int, selected: bool = False) -> list[StyleAndTextTuples]:
        running = self.status in {"running", "pending"}
        failed = self.status in {"failed", "error"}
        content_w = max(8, width - 6)  # leave room for chrome

        if self.kind == "user":
            raw = paint_user_prompt(
                self.text,
                width=content_w,
                collapsed=self.collapsed,
                selected=False,  # selection via chrome
            )
            return with_chrome(
                _ensure_rows(raw),
                width=width,
                accent=accent_for_kind("user"),
                bullet=None,
                selected=selected,
            )

        if self.kind == "thinking":
            raw = paint_thinking(
                self.text,
                width=content_w,
                collapsed=self.collapsed,
                running=running,
                elapsed_ms=self.elapsed_ms,
                selected=False,
                show_header=True,
            )
            return with_chrome(
                _ensure_rows(raw),
                width=width,
                accent=accent_for_kind("thinking", running=running),
                bullet=bullet_for_kind("thinking", running=running),
                selected=selected,
                animated=running,
            )

        if self.kind == "assistant":
            rows_md = render_markdown(self.text, width=content_w)
            raw: list[StyleAndTextTuples] = []
            for line in rows_md:
                row = [
                    (
                        st
                        if str(st).startswith("class:")
                        else f"class:{st}" if st else "class:grok.md.text",
                        tx,
                    )
                    for st, tx in line
                ]
                raw.append(row)
            if not raw:
                if running:
                    raw = [[("class:grok.gray", "…")]]
                else:
                    return []
            return with_chrome(
                raw,
                width=width,
                accent=accent_for_kind("assistant", running=running),
                bullet=bullet_for_kind("assistant", running=running),
                selected=selected,
                animated=running,
            )

        if self.kind == "tool":
            raw = paint_tool(
                self.tool_name,
                self.tool_args,
                self.tool_result,
                width=content_w,
                collapsed=self.collapsed,
                truncated=self.truncated and not self.collapsed,
                status=self.status,
                selected=False,
                meta=self.meta,
            )
            return with_chrome(
                _ensure_rows(raw),
                width=width,
                accent=accent_for_kind("tool", running=running, failed=failed),
                bullet=bullet_for_kind("tool", running=running),
                selected=selected,
                animated=running,
            )

        if self.kind == "lifecycle":
            raw = paint_tool(
                self.tool_name or self.text or "lifecycle",
                {},
                "",
                width=content_w,
                collapsed=self.collapsed,
                status=self.status,
                selected=False,
            )
            return with_chrome(
                _ensure_rows(raw),
                width=width,
                accent=accent_for_kind("system", running=running, failed=failed),
                bullet=bullet_for_kind("system"),
                selected=selected,
            )

        if self.kind == "verb_group":
            raw = [[("class:grok.tool.header", self.text or "")]]
            return with_chrome(
                raw,
                width=width,
                accent=accent_for_kind("verb_group", running=running),
                bullet=bullet_for_kind("verb_group", running=running),
                selected=selected,
                animated=running,
            )

        if self.kind == "subagent":
            raw = paint_subagent(
                self.text or "subagent",
                width=content_w,
                status=self.status,
                is_background=bool(self.meta.get("background")),
                elapsed_ms=self.elapsed_ms,
                error=str(self.meta.get("error") or "") or None,
                activity_label=str(self.meta.get("activity") or "") or None,
                selected=selected,
            )
            return with_chrome(
                _ensure_rows(raw),
                width=width,
                accent=accent_for_kind("subagent", running=running, failed=failed),
                bullet=bullet_for_kind("subagent", running=running),
                selected=selected,
                animated=running,
            )

        if self.kind == "bg_task":
            raw = paint_bg_task(
                self.text or "",
                width=content_w,
                status=self.status,
                elapsed_ms=self.elapsed_ms,
                selected=selected,
                exit_code=self.meta.get("exit_code"),
                signal=self.meta.get("signal"),
                collapsed=self.collapsed,
                truncated=self.truncated and not self.collapsed,
                output=self.tool_result or str(self.meta.get("output") or ""),
            )
            return with_chrome(
                _ensure_rows(raw),
                width=width,
                accent=accent_for_kind("tool", running=running, failed=failed),
                bullet=bullet_for_kind("tool", running=running),
                selected=selected,
                animated=running,
            )

        if self.kind == "session_event":
            raw = paint_session_event(
                kind=self.meta.get("event") or self.tool_name or "generic",
                detail=self.text,
                width=content_w,
                selected=selected,
            )
            return with_chrome(
                _ensure_rows(raw),
                width=width,
                accent=accent_for_kind("system"),
                bullet=bullet_for_kind("system"),
                selected=selected,
            )

        # system / error fallback
        raw = paint_system(
            self.text or "", width=content_w, failed=failed
        )
        return with_chrome(
            _ensure_rows(raw),
            width=width,
            accent=accent_for_kind(
                "error" if failed else "system", failed=failed
            ),
            bullet=bullet_for_kind("system"),
            selected=selected,
        )


def project_message(
    message: Any,
    *,
    tool_open: dict[str, ScrollItem],
    id_factory,
) -> list[ScrollItem]:
    role = _role(message)
    out: list[ScrollItem] = []

    if role == "user":
        content = str(_get(message, "content", "") or "").strip()
        if content:
            out.append(
                ScrollItem(kind="user", id=id_factory("user"), text=content)
            )
        return out

    if role == "assistant":
        reasoning = _get(message, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning.strip():
            out.append(
                ScrollItem(
                    kind="thinking",
                    id=id_factory("thinking"),
                    text=reasoning.strip(),
                    collapsed=True,
                    status="done",
                )
            )
        content = str(_get(message, "content", "") or "").strip()
        tcs = _get(message, "tool_calls", None) or []
        if content:
            out.append(
                ScrollItem(
                    kind="assistant",
                    id=id_factory("assistant"),
                    text=content,
                    status="running" if tcs else "done",
                )
            )
        for tc in tcs:
            if isinstance(tc, dict):
                cid = str(tc.get("id") or "")
                name = str(
                    (tc.get("function") or {}).get("name")
                    or tc.get("name")
                    or "tool"
                )
                args = _args(tc.get("arguments"))
            else:
                cid = str(getattr(tc, "id", "") or "")
                name = str(getattr(tc, "name", None) or "tool")
                args = _args(getattr(tc, "arguments", None))
            item = ScrollItem(
                kind="tool",
                id=id_factory("tool"),
                tool_name=name,
                tool_args=args,
                tool_result="",
                status="running",
                collapsed=True,
                meta={"call_id": cid},
            )
            if cid:
                tool_open[cid] = item
            out.append(item)
        return out

    if role == "tool":
        cid = str(_get(message, "tool_call_id", "") or "")
        name = str(_get(message, "name", "") or "tool")
        content = str(_get(message, "content", "") or "")
        head = content[:400].lower()
        failed = head.startswith("error") or "traceback" in head
        status = "failed" if failed else "completed"
        existing = tool_open.pop(cid, None) if cid else None
        if existing is not None:
            existing.tool_result = content
            existing.status = status
            existing.collapsed = True
            if name and name != "tool":
                existing.tool_name = name
            return []
        out.append(
            ScrollItem(
                kind="tool",
                id=id_factory("tool"),
                tool_name=name,
                tool_args={},
                tool_result=content,
                status=status,
                collapsed=True,
            )
        )
        return out

    return out
