"""Grok product surface fidelity tests (names, write/apply_patch/lsp)."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.runtime import ToolCallContext, ToolError


def test_product_names_include_workspace_extras() -> None:
    set_ = ToolRegistryBuilder.new().finalize()
    names = set(set_.client_names())
    for n in (
        "run_terminal_command",
        "write",
        "apply_patch",
        "lsp",
        "image_gen",
        "image_edit",
        "image_to_video",
        "reference_to_video",
        "get_command_or_subagent_output",
        "kill_command_or_subagent",
        "wait_commands_or_subagents",
        "spawn_subagent",
    ):
        assert n in names, n


def test_write_tool(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    out = tools.call("write", {"file_path": "a.txt", "content": "hello\n"}, ctx)
    # OpenCode write (Grok): "The file {} has been created." / "Wrote file successfully to {}."
    assert "has been created." in out or out.startswith("Wrote file successfully to ")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hello\n"


def test_apply_patch_add_update(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    patch = """\
*** Begin Patch
*** Add File: hello.txt
+Hello world
*** End Patch
"""
    out = tools.call("apply_patch", {"patch": patch}, ctx)
    assert "Success" in out
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8").startswith("Hello world")

    patch2 = """\
*** Begin Patch
*** Update File: hello.txt
@@
-Hello world
+Hello universe
*** End Patch
"""
    out2 = tools.call("apply_patch", {"patch": patch2}, ctx)
    assert "Success" in out2
    assert "universe" in (tmp_path / "hello.txt").read_text(encoding="utf-8")


def test_apply_patch_rejects_absolute(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    abs_path = str((tmp_path / "x.txt").resolve())
    patch = f"""\
*** Begin Patch
*** Add File: {abs_path}
+nope
*** End Patch
"""
    with pytest.raises(ToolError, match="relative"):
        tools.call("apply_patch", {"patch": patch}, ctx)


def test_grep_description_is_grok_style() -> None:
    tools = ToolRegistryBuilder.new().finalize()
    defs = {d.name: d for d in tools.tool_definitions()}
    desc = defs["grep"].description or ""
    assert "ripgrep" in desc
    assert "Respects .gitignore" in desc or "gitignore" in desc


def test_read_file_description_mentions_pdf_pptx() -> None:
    tools = ToolRegistryBuilder.new().finalize()
    defs = {d.name: d for d in tools.tool_definitions()}
    desc = defs["read_file"].description or ""
    assert "PDF" in desc
    assert "PowerPoint" in desc or "pptx" in desc.lower()


def test_lsp_schema_has_camel_operations() -> None:
    tools = ToolRegistryBuilder.new().finalize()
    defs = {d.name: d for d in tools.tool_definitions()}
    props = defs["lsp"].parameters.get("properties") or {}
    assert "operation" in props
    enum = props["operation"].get("enum") or []
    assert "goToDefinition" in enum
    assert "workspaceSymbol" in enum


def test_lsp_unavailable_without_backend(tmp_path: Path) -> None:
    """Grok: no LspBackend → ToolError code process_manager + fixed message."""
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    with pytest.raises(ToolError) as ei:
        tools.call(
            "lsp",
            {"operation": "goToDefinition", "file_path": "a.py", "line": 0, "character": 0},
            ctx,
        )
    assert ei.value.code == "process_manager"
    assert "LSP tool is unavailable" in str(ei.value)


def test_bash_description_grok_timeout_enforcement() -> None:
    tools = ToolRegistryBuilder.new().finalize()
    defs = {d.name: d for d in tools.tool_definitions()}
    desc = defs["run_terminal_command"].description or ""
    assert "Timeout enforcement" in desc or "timeout" in desc.lower()
    # product param rename
    props = defs["run_terminal_command"].parameters.get("properties") or {}
    assert "background" in props


def test_memory_get_zero_based_from(tmp_path: Path) -> None:
    from codedoggy.memory.store import MemoryStore

    mem = MemoryStore(memory_dir=tmp_path / "m")
    mem.load_from_disk()
    (mem.memory_dir / "MEMORY.md").write_text("lineA\nlineB\nlineC\n", encoding="utf-8")
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path, extra={"memory_store": mem})
    out = tools.call(
        "memory_get",
        {"path": str(mem.memory_dir / "MEMORY.md"), "from": 1, "lines": 1},
        ctx,
    )
    # from=1 (0-based) → display line 2
    assert "2→lineB" in out
    assert "lineA" not in out
    assert "**File:**" in out
    assert "from: 1" in out


def test_memory_get_disabled_soft_message(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    out = tools.call("memory_get", {"path": "MEMORY.md"}, ctx)
    assert "not enabled" in out.lower()


def test_bash_exit_card_format(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    out = tools.call(
        "run_terminal_command",
        {"command": 'python -c "print(42)"', "description": "print"},
        ctx,
    )
    assert out.startswith("exit: 0")
    assert "42" in out


def test_image_gen_reports_not_supported_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for k in (
        "CODEDOGGY_IMAGINE_API_KEY",
        "XAI_API_KEY",
        "CODEDOGGY_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CODEDOGGY_IMAGINE_ENABLED", "1")
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    with pytest.raises(ToolError) as ei:
        tools.call("image_gen", {"prompt": "a red cube"}, ctx)
    assert ei.value.code == "not_supported"
    assert "API key" in ei.value.message or "not supported" in ei.value.message.lower()


def test_image_gen_with_mock_client(tmp_path: Path) -> None:
    """Override still works for unit tests without network."""
    import json

    class Client:
        def generate(self, prompt: str, aspect: str):
            # bytes → tool writes images/1.jpg (Grok SessionFileWriter)
            return b"\xff\xd8\xff" + b"fakejpeg"

    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path, extra={"image_gen_client": Client()})
    out = tools.call("image_gen", {"prompt": "cube", "aspect_ratio": "1:1"}, ctx)
    data = json.loads(out)
    assert data["filename"] == "1.jpg"
    assert data["session_folder"] == "images"
    assert "Image generated and saved" in data["message"]
    assert (tmp_path / "images" / "1.jpg").is_file()


def test_imagine_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from codedoggy.tools.util.imagine_api import ImagineConfig, DEFAULT_MODEL

    monkeypatch.setenv("CODEDOGGY_IMAGINE_API_KEY", "sk-test")
    monkeypatch.setenv("CODEDOGGY_IMAGINE_BASE_URL", "https://api.x.ai/v1")
    monkeypatch.setenv("CODEDOGGY_IMAGINE_MODEL", "grok-imagine-image-quality")
    cfg = ImagineConfig.from_env()
    assert cfg.enabled
    assert cfg.api_key == "sk-test"
    assert cfg.base_url.endswith("/v1")
    assert cfg.model == DEFAULT_MODEL


def test_apply_patch_requires_begin_marker(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    with pytest.raises(ToolError, match="Begin Patch"):
        tools.call("apply_patch", {"patch": "not a patch"}, ctx)


def test_video_tools_not_supported_without_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for k in (
        "CODEDOGGY_IMAGINE_API_KEY",
        "XAI_API_KEY",
        "CODEDOGGY_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    (tmp_path / "a.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    )
    with pytest.raises(ToolError) as ei:
        tools.call(
            "image_to_video",
            {"image": "a.png", "prompt": "zoom in"},
            ctx,
        )
    assert ei.value.code == "not_supported"

    with pytest.raises(ToolError) as ei2:
        tools.call(
            "reference_to_video",
            {
                "prompt": "blend",
                "images": ["a.png", "a.png"],
                "aspect_ratio": "16:9",
            },
            ctx,
        )
    assert ei2.value.code == "not_supported"


def test_reference_to_video_requires_two_images(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    ctx = ToolCallContext(cwd=tmp_path)
    with pytest.raises(ToolError, match="at least two"):
        tools.call(
            "reference_to_video",
            {"prompt": "x", "images": ["only.png"], "aspect_ratio": "16:9"},
            ctx,
        )
