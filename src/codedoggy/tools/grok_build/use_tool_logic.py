"""use_tool pure helpers — source port from Grok use_tool/mod.rs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UseToolInput:
    """Grok ``UseToolInput`` — wire schema for ``use_tool``."""

    tool_name: str
    tool_input: dict[str, Any] = field(default_factory=dict)


@dataclass
class UseToolParams:
    """Grok ``UseToolParams`` — native-tool correction gate."""

    native_tool_correction: bool = True


def normalize_mcp_arguments(tool_input: Any) -> Any:
    """Grok ``normalize_mcp_arguments``."""
    if isinstance(tool_input, str):
        try:
            parsed = json.loads(tool_input)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return tool_input
        return tool_input
    if tool_input is None:
        return {}
    return tool_input


def gateway_result_is_error(result: Any) -> bool:
    """Grok ``gateway_result_is_error``."""
    if not isinstance(result, dict):
        return False
    v = result.get("isError")
    if v is None:
        v = result.get("is_error")
    return bool(v) if isinstance(v, bool) else False


def gateway_result_to_text(result: Any) -> str:
    """Grok ``gateway_result_to_text`` — content[] text/image/resource."""
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        try:
            return json.dumps(result, ensure_ascii=False, indent=2)
        except TypeError:
            return str(result)
    content = result.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            typ = item.get("type")
            if typ == "text":
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
            elif typ == "image":
                mime = (
                    item.get("mimeType")
                    or item.get("mime_type")
                    or "image/png"
                )
                data = item.get("data")
                if isinstance(data, str):
                    parts.append(f"data:{mime};base64,{data}")
            elif typ == "resource":
                try:
                    parts.append(json.dumps(item, ensure_ascii=False))
                except TypeError:
                    parts.append(str(item))
        if parts:
            return "\n".join(parts)
    for key in ("text", "output", "result", "content"):
        v = result.get(key)
        if isinstance(v, str):
            return v
    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except TypeError:
        return str(result)


def native_tool_correction_message(tool_name: str) -> str:
    """Grok use_tool native-tool corrective error (exact shape)."""
    return (
        f"`{tool_name}` is a native tool, not an MCP integration tool. "
        f"Call `{tool_name}` directly as its own tool call instead of "
        f"routing it through `use_tool`."
    )


def unqualified_mcp_name_message(
    tool_name: str,
    search_tool_name: str = "search_tool",
) -> str:
    """Grok use_tool unqualified name steer."""
    return (
        f"'{tool_name}' is not a valid MCP tool name. "
        f"Tool names must be qualified as `server__tool` "
        f"(e.g., `linear__save_issue`). "
        f"Use `{search_tool_name}` to discover available tools."
    )


def use_tool_description(search_tool_name: str = "search_tool") -> str:
    return (
        "Call an MCP integration tool.\n\n"
        "The `tool_name` must be the qualified `server__tool` name "
        f"(e.g., `linear__save_issue`). "
        f"The `tool_input` must conform exactly to the input schema returned by "
        f"`{search_tool_name}`."
    )


USE_TOOL_DESCRIPTION = use_tool_description()
