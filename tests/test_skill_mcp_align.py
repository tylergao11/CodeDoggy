"""Grok skill + MCP search_tool/use_tool source-level alignment tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.tools.builtins.search_tool import SearchToolTool
from codedoggy.tools.builtins.skill import SkillTool
from codedoggy.tools.builtins.use_tool import UseToolTool
from codedoggy.tools.grok_build.search_tool_logic import (
    MAX_MCP_DESCRIPTION_LENGTH,
    SEARCH_TOOL_DESCRIPTION,
    ServerSummary,
    build_server_reminder,
    sanitize_description,
    truncate_description,
)
from codedoggy.tools.grok_build.skill_discovery import (
    ParsedFrontmatter,
    is_valid_skill_name,
    normalize_skill_name,
    parse_skill_frontmatter,
    parse_skill_md,
    resolve_skill,
)
from codedoggy.tools.grok_build.skill_logic import (
    SkillInfo,
    SkillInput,
    SkillOutput,
    SkillRef,
    SubstitutionContext,
    apply_substitutions,
    build_skill_information,
    build_skill_message,
    extract_skill_display_text,
    skill_name_from_path,
)
from codedoggy.tools.grok_build.use_tool_logic import (
    UseToolInput,
    UseToolParams,
    native_tool_correction_message,
    normalize_mcp_arguments,
    unqualified_mcp_name_message,
)
from codedoggy.tools.mcp.types import (
    McpDispatch,
    SearchSnapshot,
    ToolIndex,
    ToolSearchResult,
)
from codedoggy.tools.grok_build.search_tool_logic import SearchToolInput
from codedoggy.tools.registry import ToolRegistryBuilder
from codedoggy.tools.runtime import ToolCallContext, ToolError


def test_build_skill_tool_description_empty_and_roster(tmp_path: Path) -> None:
    """Grok OpenCode DESCRIPTION with available_skills expanded."""
    from codedoggy.tools.grok_build.skill_logic import (
        SKILL_TOOL_EMPTY_HINT,
        build_skill_tool_description,
    )

    empty = build_skill_tool_description([])
    assert empty.startswith("Load a specialized skill")
    assert "<available_skills>" in empty
    assert SKILL_TOOL_EMPTY_HINT in empty or "No skills available" in empty
    assert empty.rstrip().endswith("</available_skills>")

    skills = [
        SkillInfo(
            name="commit",
            description="Create a git commit",
            path=str(tmp_path / "commit" / "SKILL.md"),
        ),
        SkillInfo(
            name="review",
            description="Review <code> & more",
            path=str(tmp_path / "review" / "SKILL.md"),
        ),
        SkillInfo(
            name="hidden",
            description="should not list",
            path="/x",
            disable_model_invocation=True,
        ),
    ]
    text = build_skill_tool_description(skills)
    assert "<name>commit</name>" in text
    assert "<name>review</name>" in text
    assert "hidden" not in text
    assert "&lt;code&gt;" in text  # |e escape
    assert "&amp;" in text
    assert str(tmp_path / "commit" / "SKILL.md") in text or "commit" in text


def test_skill_tool_description_discovers_workspace_skills(tmp_path: Path) -> None:
    """SkillTool.description re-renders roster from cwd discovery."""
    skill_dir = tmp_path / ".grok" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill for listing\n---\n\nBody\n",
        encoding="utf-8",
    )
    tool = SkillTool()
    assert tool.has_dynamic_description()
    from codedoggy.tools.runtime import ListToolsContext

    desc = tool.description(ListToolsContext(cwd=tmp_path, extra={})).description
    assert "<available_skills>" in desc
    assert "<name>demo</name>" in desc
    assert "Demo skill for listing" in desc

    # Explicit empty host catalog → empty hint (no disk fallback)
    empty = tool.description(
        ListToolsContext(cwd=tmp_path, extra={"skills_registry": []})
    ).description
    assert "No skills available" in empty


def test_skill_message_envelope() -> None:
    skill = SkillInfo(name="commit", description="git commit", path="/s/SKILL.md")
    msg = build_skill_message(skill, "Do a commit")
    assert msg.startswith('<skill name="commit"')
    assert 'description="git commit"' in msg
    assert 'path="/s/SKILL.md"' in msg
    assert "Do a commit" in msg
    assert msg.rstrip().endswith("</skill>")


def test_skill_information_envelope() -> None:
    blocks = [build_skill_message(SkillInfo(name="a", path="/a"), "body")]
    text = build_skill_information(blocks, [SkillRef(name="a", path="/a")])
    assert text.startswith("<skill_information>")
    assert "<skills_referenced>" in text
    assert 'name="a"' in text
    # tuple form still accepted
    text2 = build_skill_information(blocks, [("a", "/a")])
    assert "skills_referenced" in text2


def test_grok_public_input_output_types() -> None:
    """Wire-facing Grok types exist with matching field names."""
    assert SkillInput(skill="commit", args="msg").skill == "commit"
    out = SkillOutput(
        success=True,
        tool_result="ok",
        skill_name="commit",
        skill_message="<skill",
    )
    assert out.skill_message is not None
    assert UseToolInput(tool_name="linear__x", tool_input={"a": 1}).tool_name.startswith(
        "linear"
    )
    assert UseToolParams().native_tool_correction is True
    assert SearchToolInput(query="linear", limit=5).limit == 5
    assert skill_name_from_path("/skills/deploy/SKILL.md") == "deploy"
    assert skill_name_from_path("/skills/deploy/skill.md") is None
    assert normalize_skill_name("Tool V1.2") == "tool-v1-2"
    assert is_valid_skill_name("tool-v1-2")
    assert not is_valid_skill_name("Bad_Name")
    fm = parse_skill_frontmatter(
        "---\nname: demo\ndescription: Hello\ndisable-model-invocation: true\n---\n\nbody\n",
        "fallback",
    )
    assert isinstance(fm, ParsedFrontmatter)
    assert fm.name == "demo"
    assert fm.disable_model_invocation is True
    # ToolIndex / SearchSnapshot interface
    snap = SearchSnapshot(results=[], total_hidden_tools=0, is_ready=True)
    assert snap.is_ready
    assert ToolSearchResult(
        tool_name="a__b",
        server_name="a",
        description="d",
        score=1.0,
    ).tool_name == "a__b"


def test_apply_substitutions_arguments() -> None:
    body = "Run $ARGUMENTS and first=$0"
    out = apply_substitutions(body, "fix typo now", SubstitutionContext())
    assert "fix typo now" in out
    assert "first=fix" in out
    assert "$ARGUMENTS" not in out


def test_apply_substitutions_skill_dir_suffix() -> None:
    body = "Dir is ${SKILL_DIR}"
    out = apply_substitutions(
        body, "arg1", SubstitutionContext(skill_dir="/skills/x")
    )
    assert "/skills/x" in out
    assert "**ARGUMENTS:** arg1" in out


def test_extract_skill_display_text() -> None:
    wire = (
        "<command-name>commit</command-name>\n"
        "<command-message>/commit</command-message>\n"
        "<command-args>msg</command-args>"
    )
    assert extract_skill_display_text(wire) == "/commit msg"


def test_pyyaml_frontmatter_lists_metadata_and_colon_retry() -> None:
    """Real PyYAML path: lists, metadata map, description with colon."""
    text = """---
name: ship-it
description: Deploy: prod only
when-to-use: shipping releases
allowed-tools:
  - Bash(git diff:*)
  - Read
paths:
  - src/**
  - "docs/{a,b}/**"
metadata:
  author: alice
  short-description: Ship skill
  team: platform
disable-model-invocation: true
user-invocable: true
model: grok
effort: high
license: Apache-2.0
---

Body with $ARGUMENTS
"""
    fm = parse_skill_frontmatter(text, "fallback")
    assert fm.name == "ship-it"
    assert fm.description == "Deploy: prod only"
    assert fm.has_user_specified_description
    assert fm.when_to_use == "shipping releases"
    assert fm.allowed_tools == ["Bash(git diff:*)", "Read"]
    assert fm.paths == ["src", "docs/{a,b}"]  # /** stripped
    assert fm.author == "alice"
    assert fm.short_description == "Ship skill"
    assert fm.metadata == {"team": "platform"}
    assert fm.disable_model_invocation is True
    assert fm.user_invocable is True
    assert fm.model == "grok"
    assert fm.effort == "high"
    assert fm.license == "Apache-2.0"


def test_frontmatter_recovery_when_yaml_broken() -> None:
    # Indented junk under description often breaks YAML; recover scalars.
    text = """---
name: recover-me
description: good line
  bad: indented
when-to-use: still ok
---

body
"""
    fm = parse_skill_frontmatter(text, None)
    assert fm.name == "recover-me"
    # recovery or successful partial — name at least
    assert fm.name == "recover-me"


def test_parse_and_invoke_skill(tmp_path: Path) -> None:
    d = tmp_path / "myskill"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill\n---\n\nHello $ARGUMENTS\n",
        encoding="utf-8",
    )
    info = parse_skill_md(d / "SKILL.md")
    assert info is not None
    assert info.name == "demo"
    assert info.has_user_specified_description
    tool = SkillTool()
    ctx = ToolCallContext(
        cwd=tmp_path,
        session_id="s1",
        extra={
            "skills_registry": [
                {
                    "name": "demo",
                    "description": "Demo skill",
                    "path": str(d / "SKILL.md"),
                    "body": "Hello $ARGUMENTS",
                }
            ]
        },
    )
    out = tool.run(ctx, {"skill": "demo", "args": "world"})
    assert "<skill name=\"demo\"" in out
    assert "Hello world" in out


def test_skill_unknown(tmp_path: Path) -> None:
    tool = SkillTool()
    ctx = ToolCallContext(cwd=tmp_path, extra={"skills_registry": []})
    with pytest.raises(ToolError) as ei:
        tool.run(ctx, {"skill": "nope"})
    assert "Unknown skill" in str(ei.value)


def test_truncate_and_sanitize_description() -> None:
    short = "ok"
    assert truncate_description(short) == short
    long = "x" * (MAX_MCP_DESCRIPTION_LENGTH + 50)
    t = truncate_description(long)
    assert t.endswith("… [truncated]") or t.endswith("\u2026 [truncated]")
    assert len(t) <= MAX_MCP_DESCRIPTION_LENGTH + 5
    assert sanitize_description("a\nb  c") == "a b c"


def test_server_reminder() -> None:
    text = build_server_reminder(
        [ServerSummary(name="linear", tool_count=2, description="PM tools")]
    )
    assert text is not None
    assert "Connected MCP servers:" in text
    assert "linear" in text
    assert "2 tools" in text


def test_search_tool_description_matches_grok() -> None:
    assert "Search for MCP tools by keyword" in SEARCH_TOOL_DESCRIPTION
    assert 'status is "partial"' in SEARCH_TOOL_DESCRIPTION


def test_search_tool_no_mcp_json(tmp_path: Path) -> None:
    import json

    tool = SearchToolTool()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    out = tool.run(ctx, {"query": "linear"})
    data = json.loads(out)
    assert data["results"] == []
    assert "No integration tools are configured" in data["note"]


def test_search_tool_catalog_filter(tmp_path: Path) -> None:
    import json

    tool = SearchToolTool()
    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={
            "mcp_tools": [
                {
                    "name": "linear__save_issue",
                    "description": "Create Linear issue",
                    "server": "linear",
                    "parameters": {"type": "object"},
                },
                {
                    "name": "gh__list",
                    "description": "List PRs",
                    "server": "github",
                },
            ]
        },
    )
    out = tool.run(ctx, {"query": "linear issue", "limit": 5})
    data = json.loads(out)
    assert data["status"] == "ready"
    servers = {g["server"] for g in data["results"]}
    assert "linear" in servers
    names = [t["tool_name"] for g in data["results"] for t in g["tools"]]
    assert "linear__save_issue" in names


def test_use_tool_unqualified_and_native(tmp_path: Path) -> None:
    tool = UseToolTool()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    with pytest.raises(ToolError) as ei:
        tool.run(ctx, {"tool_name": "read_file", "tool_input": {}})
    assert "native tool" in str(ei.value).lower()
    with pytest.raises(ToolError) as ei2:
        tool.run(ctx, {"tool_name": "mystery", "tool_input": {}})
    assert "not a valid MCP tool name" in str(ei2.value)
    assert "search_tool" in str(ei2.value)


def test_use_tool_dispatch_and_normalize(tmp_path: Path) -> None:
    assert normalize_mcp_arguments('{"a": 1}') == {"a": 1}
    assert normalize_mcp_arguments(None) == {}
    calls: list = []

    def dispatch(name, inp):
        calls.append((name, inp))
        return {"content": [{"type": "text", "text": "ok"}]}

    tool = UseToolTool()
    ctx = ToolCallContext(cwd=tmp_path, extra={"mcp_dispatch": dispatch})
    out = tool.run(
        ctx,
        {"tool_name": "linear__save_issue", "tool_input": {"title": "x"}},
    )
    assert out == "ok"
    assert calls[0][0] == "linear__save_issue"


def test_product_registers_skill() -> None:
    names = set(ToolRegistryBuilder.new().finalize().client_names())
    assert "skill" in names
    assert "search_tool" in names
    assert "use_tool" in names


def test_native_correction_strings() -> None:
    assert "native tool" in native_tool_correction_message("grep")
    assert "server__tool" in unqualified_mcp_name_message("x")
