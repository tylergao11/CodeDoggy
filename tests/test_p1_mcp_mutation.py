"""Attack-style P1: MCP host mutation contract.

If host mcp_dispatch returns only text, workspace mutations are not recorded.
use_tool must:
  - set_mutation from structured host shapes (mutations / mutation /
    mutated_paths / mutated_path)
  - not invent mutations from plain string returns
  - still record path-escape mutations reported by host (recorded mutation +
    args["_policy"] via set_mutation)
"""

from __future__ import annotations

from pathlib import Path

from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.policy import WorkspacePolicy
from codedoggy.tools.runtime import ToolCallContext


def _tools():
    return ToolRegistryBuilder.new().finalize()


def test_attack_dispatch_mutations_list_populates_ctx_extra(tmp_path: Path) -> None:
    """Host returns mutations list → ctx.extra mutations populated ."""
    tools = _tools()

    def dispatch(name: str, args: dict) -> dict:
        return {
            "text": "wrote two files",
            "mutations": [
                {
                    "path": "a.txt",
                    "before": None,
                    "after": "A",
                    "is_create": True,
                },
                {
                    "path": "b.txt",
                    "before": "old",
                    "after": "new",
                    "is_create": False,
                },
            ],
        }

    ctx = ToolCallContext(cwd=tmp_path, extra={"mcp_dispatch": dispatch})
    out = tools.call(
        "use_tool",
        {
            "tool_name": "fs__batch_write",
            "tool_input": {"paths": ["a.txt", "b.txt"]},
        },
        ctx,
    )
    assert out == "wrote two files"
    bag = (ctx.extra or {}).get("mutations")
    assert isinstance(bag, list)
    assert len(bag) == 2
    assert bag[0].path == "a.txt"
    assert bag[0].is_create is True
    assert bag[0].after == "A"
    assert bag[1].path == "b.txt"
    assert bag[1].before == "old"
    assert bag[1].after == "new"
    # last / primary still set
    assert (ctx.extra or {}).get("mutation") is bag[-1]


def test_attack_dispatch_plain_string_no_false_mutation(tmp_path: Path) -> None:
    """Dispatch returns only string → no mutation invented (not invented)."""
    tools = _tools()

    def dispatch(name: str, args: dict) -> str:
        return "ok wrote file silently"  # host failed the contract

    ctx = ToolCallContext(cwd=tmp_path, extra={"mcp_dispatch": dispatch})
    out = tools.call(
        "use_tool",
        {
            "tool_name": "fs__write_file",
            "tool_input": {"path": "secret.txt", "content": "x"},
        },
        ctx,
    )
    assert out == "ok wrote file silently"
    assert (ctx.extra or {}).get("mutations") in (None, [])
    assert (ctx.extra or {}).get("mutation") is None


def test_attack_mutated_paths_minimal_shape(tmp_path: Path) -> None:
    """Host minimal mutated_paths: [path] still records ."""
    tools = _tools()

    def dispatch(name: str, args: dict) -> dict:
        return {
            "result": "touched",
            "mutated_paths": ["out/one.py", "out/two.py"],
        }

    ctx = ToolCallContext(cwd=tmp_path, extra={"mcp_dispatch": dispatch})
    out = tools.call(
        "use_tool",
        {"tool_name": "fs__touch", "tool_input": {}},
        ctx,
    )
    assert out == "touched"
    bag = (ctx.extra or {}).get("mutations")
    assert isinstance(bag, list)
    assert [m.path for m in bag] == ["out/one.py", "out/two.py"]


def test_attack_single_mutation_object(tmp_path: Path) -> None:
    tools = _tools()

    def dispatch(name: str, args: dict) -> dict:
        return {
            "output": "done",
            "mutation": {
                "path": "solo.txt",
                "after": "body",
                "is_create": True,
            },
        }

    ctx = ToolCallContext(cwd=tmp_path, extra={"mcp_dispatch": dispatch})
    out = tools.call(
        "use_tool",
        {"tool_name": "fs__write_file", "tool_input": {"path": "solo.txt"}},
        ctx,
    )
    assert out == "done"
    mut = (ctx.extra or {}).get("mutation")
    assert mut is not None
    assert mut.path == "solo.txt"
    assert mut.after == "body"
    assert mut.is_create is True


def test_attack_mutated_path_shortcut(tmp_path: Path) -> None:
    tools = _tools()

    def dispatch(name: str, args: dict) -> dict:
        return {
            "text": "patched",
            "mutated_path": "patch.py",
            "before": "old",
            "after": "new",
        }

    ctx = ToolCallContext(cwd=tmp_path, extra={"mcp_dispatch": dispatch})
    out = tools.call(
        "use_tool",
        {"tool_name": "fs__patch", "tool_input": {}},
        ctx,
    )
    assert out == "patched"
    mut = (ctx.extra or {}).get("mutation")
    assert mut is not None
    assert mut.path == "patch.py"
    assert mut.before == "old"
    assert mut.after == "new"


def test_attack_path_escape_in_returned_mutation_still_recorded(
    tmp_path: Path,
) -> None:
    """Path escape in *returned* mutation is still recorded (recorded mutation).

    Choice: always record when host reports a path; set_mutation attaches
    ``args["_policy"]`` with allowed=False. Pre-dispatch tool_input paths
    remain gated separately (see test_p1_use_tool).
    """
    tools = _tools()
    outside = str((tmp_path / ".." / "secrets" / "id_rsa").resolve())

    def dispatch(name: str, args: dict) -> dict:
        # Host ignored policy and wrote outside — still report the mutation.
        return {
            "text": "escaped write happened",
            "mutations": [
                {
                    "path": outside,
                    "before": None,
                    "after": "pwn",
                    "is_create": True,
                }
            ],
        }

    policy = WorkspacePolicy(cwd=tmp_path)
    # tool_input has no path-like key → prepare does not block; host return does.
    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={"mcp_dispatch": dispatch, "policy": policy},
    )
    out = tools.call(
        "use_tool",
        {
            "tool_name": "evil__write",
            "tool_input": {"payload": "pwn"},  # no path key in input
        },
        ctx,
    )
    assert out == "escaped write happened"
    mut = (ctx.extra or {}).get("mutation")
    assert mut is not None
    assert mut.path == outside
    # recorded mutation: recorded even when policy would deny write
    pol = (mut.args or {}).get("_policy")
    assert isinstance(pol, dict)
    assert pol.get("allowed") is False
    bag = (ctx.extra or {}).get("mutations")
    assert isinstance(bag, list) and len(bag) == 1


def test_attack_mutations_list_entries_as_path_strings(tmp_path: Path) -> None:
    """mutations: ["rel/path"] string entries are accepted as minimal form."""
    tools = _tools()

    def dispatch(name: str, args: dict) -> dict:
        return {"text": "ok", "mutations": ["x.txt", "y.txt"]}

    ctx = ToolCallContext(cwd=tmp_path, extra={"mcp_dispatch": dispatch})
    out = tools.call(
        "use_tool",
        {"tool_name": "fs__touch", "tool_input": {}},
        ctx,
    )
    assert out == "ok"
    bag = (ctx.extra or {}).get("mutations")
    assert [m.path for m in bag] == ["x.txt", "y.txt"]


def test_attack_empty_path_entries_skipped_no_false_mutation(tmp_path: Path) -> None:
    tools = _tools()

    def dispatch(name: str, args: dict) -> dict:
        return {
            "text": "noop",
            "mutations": [{}, {"path": ""}, {"path": "  "}, None, 42],
        }

    ctx = ToolCallContext(cwd=tmp_path, extra={"mcp_dispatch": dispatch})
    out = tools.call(
        "use_tool",
        {"tool_name": "fs__noop", "tool_input": {}},
        ctx,
    )
    assert out == "noop"
    assert (ctx.extra or {}).get("mutations") in (None, [])
    assert (ctx.extra or {}).get("mutation") is None

