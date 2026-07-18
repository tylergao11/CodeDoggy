"""web_fetch — thin shell over source-ported Grok WebFetchClient.

Ported from:
  grok-build/.../implementations/grok_build/web_fetch/mod.rs
    WebFetchTool description_template, input schema, run wiring

Pure logic:
  tools/util/ssrf.py
  tools/grok_build/web_fetch_*.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codedoggy.tools.grok_build.web_fetch_client import get_default_client
from codedoggy.tools.grok_build.web_fetch_error import WebFetchError
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolError,
    ToolId,
)

# Grok WebFetchTool::description_template (mod.rs) — product name untemplated.
_DESC = """\
Fetch the content of a specific URL and return it as markdown.

IMPORTANT: web_fetch WILL FAIL for authenticated or private URLs (e.g. Google Docs, Confluence, Jira, GitHub private repos). Use specialized MCP tools for those instead.

Usage notes:
  - HTTP URLs will be automatically upgraded to HTTPS
  - Long pages will be truncated to fit your context window"""


class WebFetchTool(Tool):
    def id(self) -> ToolId:
        return ToolId("web_fetch")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.WebFetch

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="web_fetch", description=_DESC.strip())

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from.",
                },
            },
            "required": ["url"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ToolError.invalid_arguments("url is required")
        url = url.strip()

        session_folder = _session_folder(ctx)
        read_name = "read_file"
        execute_name = "run_terminal_command"

        client = get_default_client()
        try:
            return client.fetch(
                url,
                session_folder=session_folder,
                read_tool_name=read_name,
                execute_tool_name=execute_name,
            )
        except WebFetchError as e:
            raise ToolError(str(e), code=e.code) from e
        except ValueError as e:
            # SSRF / validation raised as ValueError
            msg = str(e)
            code = "ssrf_blocked" if "SSRF blocked" in msg else "web_fetch"
            raise ToolError(msg, code=code) from e


def _session_folder(ctx: ToolCallContext) -> Path | None:
    """Prefer host-provided session dir; fall back to cwd."""
    extra = ctx.extra or {}
    for key in ("session_folder", "session_dir", "session_path"):
        val = extra.get(key)
        if val:
            return Path(val)
    return Path(ctx.cwd)
