"""Tool-layer: confusable edit, rich read_file, MCP search/use."""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.runtime import ToolCallContext
from codedoggy.tools.util.unicode_confusables import normalize_confusables


def _tools():
    return ToolRegistryBuilder.new().finalize()


def test_confusable_search_replace(tmp_path: Path) -> None:
    # File has smart quotes; model sends ASCII
    path = tmp_path / "notes.txt"
    path.write_text("He said \u201chello\u201d there.\n", encoding="utf-8")
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path)
    out = tools.call(
        "search_replace",
        {
            "file_path": "notes.txt",
            "old_string": 'He said "hello" there.',
            "new_string": 'He said "hi" there.',
        },
        ctx,
    )
    assert "updated successfully" in out or "confusable" in out.lower()
    text = path.read_text(encoding="utf-8")
    assert "hi" in text
    assert "Edit context" in out


def test_normalize_confusables() -> None:
    s = "\u201cquote\u201d \u2014 dash \u2026"
    assert normalize_confusables(s) == '"quote" -- dash ...'


def test_read_image_meta(tmp_path: Path) -> None:
    # Minimal 1x1 PNG
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
        "de0000000c4944415408d763f8ffff3f0005fe02fe"
        "a75b2a150000000049454e44ae426082"
    )
    # Fix: use a known tiny PNG
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff"
        b"\xff?\x00\x05\xfe\x02\xfe\xa7[*\x15\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    p = tmp_path / "dot.png"
    p.write_bytes(png)
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path)
    out = tools.call("read_file", {"target_file": "dot.png"}, ctx)
    assert "[image]" in out
    assert "dimensions" in out or "size_bytes" in out


def test_read_pptx_text(tmp_path: Path) -> None:
    pptx = tmp_path / "deck.pptx"
    # Minimal pptx zip with one slide
    slide_xml = (
        b'<?xml version="1.0"?>'
        b'<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        b'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        b"<p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>HelloSlide</a:t>"
        b"</a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
    )
    with zipfile.ZipFile(pptx, "w") as zf:
        zf.writestr("ppt/slides/slide1.xml", slide_xml)
        zf.writestr("[Content_Types].xml", b"<Types></Types>")
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path)
    out = tools.call("read_file", {"target_file": "deck.pptx"}, ctx)
    assert "HelloSlide" in out
    assert "slide" in out.lower()


def test_mcp_search_and_use(tmp_path: Path) -> None:
    tools = _tools()
    calls: list[tuple[str, dict]] = []

    def dispatch(name: str, args: dict) -> str:
        calls.append((name, args))
        return f"ok:{name}:{args.get('x')}"

    mcp_tools = [
        {
            "name": "linear__save_issue",
            "description": "Create or update a Linear issue",
            "server": "linear",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "x": {"type": "integer"}},
            },
        },
        {
            "name": "github__list_prs",
            "description": "List pull requests on GitHub",
            "server": "github",
        },
    ]
    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={"mcp_tools": mcp_tools, "mcp_dispatch": dispatch},
    )
    found = tools.call("search_tool", {"query": "linear issue"}, ctx)
    assert "linear__save_issue" in found

    out = tools.call(
        "use_tool",
        {"tool_name": "linear__save_issue", "tool_input": {"x": 1, "title": "t"}},
        ctx,
    )
    assert "ok:linear__save_issue:1" in out
    assert calls and calls[0][0] == "linear__save_issue"


def test_search_tool_empty_catalog(tmp_path: Path) -> None:
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    out = tools.call("search_tool", {"query": "anything"}, ctx)
    # Grok search_tool/mod.rs — exact empty-index note
    assert "No integration tools are configured" in out


def test_builtins_include_mcp_and_memory() -> None:
    b = ToolRegistryBuilder.new()
    assert b.has_tool_id("Doggy:search_tool")
    assert b.has_tool_id("Doggy:use_tool")
    assert b.has_tool_id("Doggy:memory")
    assert b.has_tool_id("Doggy:session_search")
    assert not b.has_tool_id("Doggy:memory_search")
