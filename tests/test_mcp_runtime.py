"""Focused probes for the Grok-aligned MCP connection/runtime chain."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from codedoggy.bootstrap import build_session
from codedoggy.mcp.config import McpServerConfig, McpTransport, load_mcp_server_configs
from codedoggy.mcp.runtime import McpRuntime
from codedoggy.tools.runtime import ToolCallContext


_STDIO_SERVER = r'''
from mcp.server.fastmcp import FastMCP

server = FastMCP("codedoggy-test", log_level="ERROR")

@server.tool()
def echo(text: str) -> str:
    """Echo a value through a real MCP stdio transport."""
    return f"echo:{text}"

if __name__ == "__main__":
    server.run(transport="stdio")
'''

_HTTP_SERVER = r'''
import sys
from mcp.server.fastmcp import FastMCP

server = FastMCP(
    "codedoggy-http-test",
    host="127.0.0.1",
    port=int(sys.argv[1]),
    log_level="ERROR",
)

@server.tool()
def echo(text: str) -> str:
    return f"http:{text}"

if __name__ == "__main__":
    server.run(transport="streamable-http")
'''


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"HTTP MCP fixture did not bind port {port}")


def test_stdio_runtime_lists_indexes_calls_and_closes(tmp_path: Path) -> None:
    script = tmp_path / "mcp_fixture.py"
    script.write_text(_STDIO_SERVER, encoding="utf-8")
    config = McpServerConfig(
        name="demo",
        transport=McpTransport.STDIO,
        command=sys.executable,
        args=(str(script),),
        cwd=str(tmp_path),
        startup_timeout_sec=10,
        tool_timeout_sec=10,
        source="test",
    )
    runtime = McpRuntime(
        tmp_path,
        session_id="mcp-e2e",
        configs=[config],
        watch=False,
        auto_restart=False,
    ).start()
    try:
        assert runtime.wait_initialized(15)
        assert [tool["name"] for tool in runtime.tools] == ["demo__echo"]
        snapshot = runtime.tool_index.search_snapshot("demo echo", 5)
        assert snapshot.is_ready is True
        assert snapshot.results[0].tool_name == "demo__echo"

        result = runtime("demo__echo", {"text": "ok"}, ToolCallContext(cwd=tmp_path))
        assert result["isError"] is False
        assert result["content"][0]["text"] == "echo:ok"

        diff = runtime.apply_configs([])
        assert diff is not None
        assert diff.removed == ["demo"]
        assert runtime.tools == []
    finally:
        runtime.close()
    assert runtime._thread is not None and not runtime._thread.is_alive()


def test_build_session_wires_runtime_into_grok_use_tool(tmp_path: Path) -> None:
    script = tmp_path / "mcp_fixture.py"
    script.write_text(_STDIO_SERVER, encoding="utf-8")
    session = build_session(
        tmp_path,
        main_client=object(),
        enable_memory=False,
        enable_session_store=False,
        enable_graph=False,
        enable_mcp=True,
        mcp_watch=False,
        mcp_auto_restart=False,
        mcp_servers={
            "demo": {
                "command": sys.executable,
                "args": [str(script)],
                "cwd": str(tmp_path),
                "startup_timeout_sec": 10,
                "tool_timeout_sec": 10,
            }
        },
    )
    try:
        kernel = session.extensions.kernel
        assert kernel is not None
        runtime = kernel.mcp_runtime
        assert runtime.wait_initialized(15)
        extra = kernel.tool_extra
        assert extra["mcp_inner_dispatch"] is runtime
        output = session.extensions.tools.call(
            "use_tool",
            {"tool_name": "demo__echo", "tool_input": {"text": "session"}},
            ToolCallContext(cwd=tmp_path, session_id=str(session.id), extra=extra),
        )
        assert output == "echo:session"
    finally:
        session.close()
    assert runtime._thread is not None and not runtime._thread.is_alive()


def test_streamable_http_runtime_calls_real_server(tmp_path: Path) -> None:
    script = tmp_path / "mcp_http_fixture.py"
    script.write_text(_HTTP_SERVER, encoding="utf-8")
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, str(script), str(port)],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    runtime = None
    try:
        _wait_port(port)
        runtime = McpRuntime(
            tmp_path,
            session_id="mcp-http-e2e",
            configs=[
                McpServerConfig(
                    name="remote",
                    transport=McpTransport.STREAMABLE_HTTP,
                    url=f"http://127.0.0.1:{port}/mcp",
                    startup_timeout_sec=10,
                    tool_timeout_sec=10,
                    source="test",
                )
            ],
            watch=False,
            auto_restart=False,
        ).start()
        assert runtime.wait_initialized(15)
        result = runtime("remote__echo", {"text": "ok"})
        assert result["isError"] is False
        assert result["content"][0]["text"] == "http:ok"
    finally:
        if runtime is not None:
            runtime.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_grok_config_priority_and_env_expansion(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    grok_home = home / ".grok"
    workspace = tmp_path / "repo"
    (workspace / ".git").mkdir(parents=True)
    grok_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("GROK_HOME", str(grok_home))
    monkeypatch.setenv("MCP_TOKEN", "secret")

    (workspace / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "demo": {"command": "low", "args": []},
                    "json-only": {"command": "${MCP_TOKEN}", "args": []},
                }
            }
        ),
        encoding="utf-8",
    )
    (workspace / ".grok").mkdir()
    (workspace / ".grok" / "config.toml").write_text(
        '[mcp_servers.demo]\ncommand = "high"\nargs = ["${MCP_TOKEN}"]\n',
        encoding="utf-8",
    )

    configs = {config.name: config for config in load_mcp_server_configs(workspace)}
    assert configs["demo"].command == "high"
    assert configs["demo"].args == ("secret",)
    assert configs["json-only"].command == "secret"
