"""Grok Bm25ToolSearchIndex source-level alignment tests."""

from __future__ import annotations

from codedoggy.tools.mcp.tool_index import (
    Bm25ToolSearchIndex,
    ToolMetadata,
    ensure_mcp_tool_index,
    normalize_query,
    split_identifier,
    tools_from_mcp_catalog,
)
from codedoggy.tools.builtins.search_tool import SearchToolTool
from codedoggy.tools.runtime import ToolCallContext


def test_split_identifier_snake_camel_kebab() -> None:
    assert "Search" in split_identifier("SearchDashboards")
    assert "Dashboards" in split_identifier("SearchDashboards")
    words = split_identifier("linear__save_issue")
    assert "linear" in words
    assert "save" in words
    assert "issue" in words


def test_normalize_query_expands_compounds() -> None:
    q = normalize_query("linear__save_issue")
    assert "linear" in q
    assert "save" in q
    assert normalize_query("plain words") == "plain words"


def test_exact_match_fast_path() -> None:
    tools = [
        ToolMetadata(
            qualified_name="linear__save_issue",
            server_name="linear",
            tool_name="save_issue",
            description="Create an issue",
            parameters=["title", "team"],
            input_schema={"type": "object", "properties": {"title": {}, "team": {}}},
        ),
        ToolMetadata(
            qualified_name="gh__list_prs",
            server_name="github",
            tool_name="list_prs",
            description="List pull requests",
            parameters=["repo"],
            input_schema={"type": "object"},
        ),
    ]
    idx = Bm25ToolSearchIndex(tools)
    snap = idx.search_snapshot("linear__save_issue", 5)
    assert len(snap.results) == 1
    assert snap.results[0].tool_name == "linear__save_issue"
    assert snap.results[0].score == 1.0


def test_bm25_ranks_relevant_tool() -> None:
    tools = [
        ToolMetadata(
            qualified_name="linear__save_issue",
            server_name="linear",
            tool_name="save_issue",
            description="Create Linear issue in a team",
            parameters=["title"],
            input_schema={"type": "object", "properties": {"title": {}}},
        ),
        ToolMetadata(
            qualified_name="slack__post_message",
            server_name="slack",
            tool_name="post_message",
            description="Post chat message to channel",
            parameters=["channel", "text"],
            input_schema={"type": "object"},
        ),
    ]
    idx = Bm25ToolSearchIndex(tools)
    snap = idx.search_snapshot("create issue linear", 5)
    assert snap.results
    assert snap.results[0].tool_name == "linear__save_issue"
    assert snap.results[0].score > 0


def test_list_server_summaries() -> None:
    tools = tools_from_mcp_catalog(
        [
            {"name": "linear__a", "description": "a", "server": "linear"},
            {"name": "linear__b", "description": "b", "server": "linear"},
            {"name": "gh__c", "description": "c", "server": "github"},
        ]
    )
    idx = Bm25ToolSearchIndex(tools)
    servers = idx.list_server_summaries()
    by = {s.name: s for s in servers}
    assert by["linear"].tool_count == 2
    assert by["github"].tool_count == 1


def test_ensure_index_from_mcp_tools_and_search_tool() -> None:
    import json

    extra = {
        "mcp_tools": [
            {
                "name": "linear__save_issue",
                "description": "Create issue",
                "server": "linear",
                "parameters": {"type": "object", "properties": {"title": {}}},
            }
        ]
    }
    idx = ensure_mcp_tool_index(extra)
    assert idx is not None
    assert extra["mcp_tool_index"] is idx
    tool = SearchToolTool()
    ctx = ToolCallContext(cwd=".", extra=extra)
    out = tool.run(ctx, {"query": "save issue", "limit": 5})
    assert "linear__save_issue" in out
    # Grok grouped JSON: results[].server + tools[]
    data = json.loads(out)
    assert data["status"] == "ready"
    assert isinstance(data["results"], list)
    assert data["results"][0]["server"] == "linear"
    assert data["results"][0]["tools"][0]["tool_name"] == "linear__save_issue"


def test_search_tool_empty_matches_grok_note() -> None:
    import json

    tool = SearchToolTool()
    out = tool.run(ToolCallContext(cwd=".", extra={}), {"query": "x"})
    data = json.loads(out)
    assert data["results"] == []
    assert "No integration tools are configured" in data["note"]
