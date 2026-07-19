"""Project codex app-server events into OpenAI-shaped messages (Hermes).

Converts Codex ``item/*`` notifications into
``{role, content, tool_calls, tool_call_id}`` entries for history / memory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional


def _deterministic_call_id(item_type: str, item_id: str) -> str:
    if item_id:
        return f"codex_{item_type}_{item_id}"
    digest = hashlib.sha256(f"{item_type}".encode()).hexdigest()[:16]
    return f"codex_{item_type}_{digest}"


def _format_tool_args(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, sort_keys=True)


@dataclass
class ProjectionResult:
    messages: list[dict] = field(default_factory=list)
    is_tool_iteration: bool = False
    final_text: Optional[str] = None


class CodexEventProjector:
    """Stateful projector; only ``item/completed`` materializes messages."""

    def __init__(self) -> None:
        self._pending_reasoning: list[str] = []

    def project(self, notification: dict) -> ProjectionResult:
        method = notification.get("method", "")
        params = notification.get("params", {}) or {}
        if method != "item/completed":
            return ProjectionResult()

        item = params.get("item") or {}
        item_type = item.get("type") or ""
        item_id = item.get("id") or ""

        if item_type == "agentMessage":
            return self._project_agent_message(item)
        if item_type == "reasoning":
            self._pending_reasoning.extend(item.get("summary") or [])
            self._pending_reasoning.extend(item.get("content") or [])
            return ProjectionResult()
        if item_type == "commandExecution":
            return self._project_command(item, item_id)
        if item_type == "fileChange":
            return self._project_file_change(item, item_id)
        if item_type == "mcpToolCall":
            return self._project_mcp_tool_call(item, item_id)
        if item_type == "dynamicToolCall":
            return self._project_dynamic_tool_call(item, item_id)
        if item_type == "userMessage":
            return self._project_user_message(item)
        return self._project_opaque(item, item_type)

    def _project_agent_message(self, item: dict) -> ProjectionResult:
        text = item.get("text") or ""
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if self._pending_reasoning:
            msg["reasoning"] = "\n".join(
                str(x) for x in self._pending_reasoning if x is not None
            )
            self._pending_reasoning = []
        return ProjectionResult(messages=[msg], final_text=text)

    def _project_user_message(self, item: dict) -> ProjectionResult:
        text_parts: list[str] = []
        for fragment in item.get("content") or []:
            if isinstance(fragment, dict):
                if fragment.get("type") == "text":
                    text_parts.append(fragment.get("text") or "")
                elif "text" in fragment:
                    text_parts.append(str(fragment["text"]))
        return ProjectionResult(
            messages=[{"role": "user", "content": "\n".join(text_parts)}]
        )

    def _project_command(self, item: dict, item_id: str) -> ProjectionResult:
        call_id = _deterministic_call_id("exec", item_id)
        args = {
            "command": item.get("command") or "",
            "cwd": item.get("cwd") or "",
        }
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": _format_tool_args(args),
                    },
                }
            ],
        }
        if self._pending_reasoning:
            assistant_msg["reasoning"] = "\n".join(
                str(x) for x in self._pending_reasoning if x is not None
            )
            self._pending_reasoning = []
        output = item.get("aggregatedOutput") or ""
        exit_code = item.get("exitCode")
        if exit_code is not None and exit_code != 0:
            output = f"[exit {exit_code}]\n{output}"
        tool_msg = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": output,
        }
        return ProjectionResult(
            messages=[assistant_msg, tool_msg], is_tool_iteration=True
        )

    def _project_file_change(self, item: dict, item_id: str) -> ProjectionResult:
        call_id = _deterministic_call_id("apply_patch", item_id)
        changes_summary = []
        for change in item.get("changes") or []:
            kind = (change.get("kind") or {}).get("type") or "update"
            path = change.get("path") or ""
            changes_summary.append({"kind": kind, "path": path})
        args = {"changes": changes_summary}
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "arguments": _format_tool_args(args),
                    },
                }
            ],
        }
        if self._pending_reasoning:
            assistant_msg["reasoning"] = "\n".join(
                str(x) for x in self._pending_reasoning if x is not None
            )
            self._pending_reasoning = []
        status = item.get("status") or "unknown"
        n = len(changes_summary)
        tool_msg = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"apply_patch status={status}, {n} change(s)",
        }
        return ProjectionResult(
            messages=[assistant_msg, tool_msg], is_tool_iteration=True
        )

    def _project_mcp_tool_call(self, item: dict, item_id: str) -> ProjectionResult:
        server = item.get("server") or "mcp"
        tool = item.get("tool") or "unknown"
        call_id = _deterministic_call_id(f"mcp__{server}__{tool}", item_id)
        args = item.get("arguments") or {}
        if not isinstance(args, dict):
            args = {"arguments": args}
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": f"mcp.{server}.{tool}",
                        "arguments": _format_tool_args(args),
                    },
                }
            ],
        }
        if self._pending_reasoning:
            assistant_msg["reasoning"] = "\n".join(
                str(x) for x in self._pending_reasoning if x is not None
            )
            self._pending_reasoning = []
        result = item.get("result")
        error = item.get("error")
        if error:
            content = f"[error] {json.dumps(error, ensure_ascii=False)[:1000]}"
        elif result is not None:
            content = json.dumps(result, ensure_ascii=False)[:4000]
        else:
            content = ""
        tool_msg = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        }
        return ProjectionResult(
            messages=[assistant_msg, tool_msg], is_tool_iteration=True
        )

    def _project_dynamic_tool_call(
        self, item: dict, item_id: str
    ) -> ProjectionResult:
        tool = item.get("tool") or "unknown"
        call_id = _deterministic_call_id(f"dyn_{tool}", item_id)
        args = item.get("arguments") or {}
        if not isinstance(args, dict):
            args = {"arguments": args}
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool,
                        "arguments": _format_tool_args(args),
                    },
                }
            ],
        }
        if self._pending_reasoning:
            assistant_msg["reasoning"] = "\n".join(
                str(x) for x in self._pending_reasoning if x is not None
            )
            self._pending_reasoning = []
        content_items = item.get("contentItems") or []
        if isinstance(content_items, list) and content_items:
            content = json.dumps(content_items, ensure_ascii=False)[:4000]
        else:
            content = f"success={item.get('success')}"
        tool_msg = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        }
        return ProjectionResult(
            messages=[assistant_msg, tool_msg], is_tool_iteration=True
        )

    def _project_opaque(self, item: dict, item_type: str) -> ProjectionResult:
        try:
            payload = json.dumps(item, ensure_ascii=False)[:1500]
        except (TypeError, ValueError):
            payload = repr(item)[:1500]
        return ProjectionResult(
            messages=[
                {
                    "role": "assistant",
                    "content": f"[codex {item_type}] {payload}",
                }
            ]
        )
