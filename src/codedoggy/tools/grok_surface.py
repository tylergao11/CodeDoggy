"""Grok product-facing tool surface (xai-grok-agent config.rs renames).

Wire ids stay GrokBuild-faithful (`run_terminal_cmd`, `task`, `get_task_output`,
`kill_task`, `wait_tasks`). Model-facing client names match production Grok:

  run_terminal_cmd     → run_terminal_command   (is_background → background)
  task                 → spawn_subagent         (run_in_background → background)
  get_task_output      → get_command_or_subagent_output
  wait_tasks           → wait_commands_or_subagents
  kill_task            → kill_command_or_subagent

``ToolServerConfig.grok_build_product()`` is the default finalize surface.
"""

from __future__ import annotations

from codedoggy.tools.config import ToolConfig, ToolServerConfig

# qualified id → (client_name, param_renames internal→client)
_GROK_PRODUCT_RENAMES: dict[str, tuple[str, dict[str, str] | None]] = {
    "Doggy:run_terminal_cmd": ("run_terminal_command", {"is_background": "background"}),
    "Doggy:task": ("spawn_subagent", {"run_in_background": "background"}),
    "Doggy:get_task_output": ("get_command_or_subagent_output", None),
    "Doggy:wait_tasks": ("wait_commands_or_subagents", None),
    "Doggy:kill_task": ("kill_command_or_subagent", None),
}

# Client name → wire short-id (for alias call resolution)
CLIENT_ALIASES: dict[str, str] = {
    "run_terminal_command": "run_terminal_cmd",
    "run_terminal_cmd": "run_terminal_cmd",
    "spawn_subagent": "task",
    "task": "task",
    "get_command_or_subagent_output": "get_task_output",
    "get_task_output": "get_task_output",
    "get_subagent_output": "get_task_output",  # legacy Doggy alias
    "wait_commands_or_subagents": "wait_tasks",
    "wait_tasks": "wait_tasks",
    "kill_command_or_subagent": "kill_task",
    "kill_task": "kill_task",
    "kill_terminal_command": "kill_task",
    "get_terminal_command_output": "get_task_output",
}

# Param client-name → internal name (global reverse map for shell/task)
PARAM_CLIENT_TO_INTERNAL: dict[str, str] = {
    "background": "is_background",  # shell; task uses run_in_background via per-tool map
}


def apply_product_rename(tc: ToolConfig) -> ToolConfig:
    """Apply Grok product name/param renames if this id is known."""
    entry = _GROK_PRODUCT_RENAMES.get(tc.id)
    if entry is None:
        return tc
    client_name, renames = entry
    if tc.name_override is None:
        tc.name_override = client_name
    if renames and tc.params_name_overrides is None:
        tc.params_name_overrides = dict(renames)
    return tc


def remap_schema_properties(
    schema: dict,
    internal_to_client: dict[str, str] | None,
) -> dict:
    """Rename property keys in JSON schema for the model-facing surface."""
    if not internal_to_client or not isinstance(schema, dict):
        return schema
    import copy

    out = copy.deepcopy(schema)
    props = out.get("properties")
    if not isinstance(props, dict):
        return out
    new_props: dict = {}
    for key, val in props.items():
        new_key = internal_to_client.get(key, key)
        new_props[new_key] = val
    out["properties"] = new_props
    req = out.get("required")
    if isinstance(req, list):
        out["required"] = [internal_to_client.get(str(r), r) for r in req]
    return out


def remap_args_client_to_internal(
    args: dict,
    internal_to_client: dict[str, str] | None,
    *,
    short_id: str = "",
) -> dict:
    """Map model-facing param names back to tool implementation names."""
    if not args:
        return args
    # Build reverse map
    rev: dict[str, str] = {}
    if internal_to_client:
        rev = {v: k for k, v in internal_to_client.items()}
    # Task tool: product `background` → run_in_background
    if short_id == "task":
        rev.setdefault("background", "run_in_background")
    elif short_id in {"run_terminal_cmd", "monitor"}:
        rev.setdefault("background", "is_background")

    if not rev:
        # still accept both for shell
        if short_id == "run_terminal_cmd" and "background" in args and "is_background" not in args:
            out = dict(args)
            out["is_background"] = out.pop("background")
            return out
        return args

    out: dict = {}
    for k, v in args.items():
        out[rev.get(k, k)] = v
    return out


# Grok product wire ids only (default_grok_build + workspace_grok_build subset).
# Not in this list: Doggy:memory, Doggy:session_search, Doggy:code_nav,
# Doggy:get_subagent_output (legacy).
_GROK_PRODUCT_IDS: list[str] = [
    "Doggy:run_terminal_cmd",
    "Doggy:read_file",
    "Doggy:search_replace",
    "Doggy:write",  # OpenCode write (workspace_grok_build_toolset)
    "Doggy:list_dir",
    "Doggy:grep",
    "Doggy:kill_task",
    "Doggy:todo_write",
    "Doggy:get_task_output",
    "Doggy:wait_tasks",
    "Doggy:task",
    "Doggy:scheduler_create",
    "Doggy:scheduler_delete",
    "Doggy:scheduler_list",
    "Doggy:monitor",
    "Doggy:search_tool",
    "Doggy:use_tool",
    "Doggy:skill",
    "Doggy:update_goal",
    # workspace extras
    "Doggy:enter_plan_mode",
    "Doggy:exit_plan_mode",
    "Doggy:ask_user_question",
    "Doggy:web_search",
    "Doggy:web_fetch",
    "Doggy:memory_search",
    "Doggy:memory_get",
    "Doggy:lsp",
    "Doggy:apply_patch",
    "Doggy:image_gen",
    "Doggy:image_edit",
    "Doggy:image_to_video",
    "Doggy:reference_to_video",
]

# CodeDoggy-only tools — never claim these are Grok product surface.
_DOGGY_ENHANCEMENT_IDS: list[str] = [
    "Doggy:memory",  # Hermes write surface
    "Doggy:session_search",
    "Doggy:code_nav",
    "Doggy:parallel_tasks",  # MAIN aggressive parallel fan-out + aggregate
]


def grok_build_product_config(*, include_doggy_enhancements: bool = False) -> ToolServerConfig:
    """Grok product toolset only (xai-grok-agent default + workspace extras).

    Doggy enhancements are off by default. Use ``codedoggy_product_config()``
    for the CodeDoggy agent pack (Grok + Doggy extras).
    """
    ids = list(_GROK_PRODUCT_IDS)
    if include_doggy_enhancements:
        ids.extend(_DOGGY_ENHANCEMENT_IDS)
    tools = [apply_product_rename(ToolConfig.from_id(i)) for i in ids]
    return ToolServerConfig(tools=tools, behavior_preset="grok-build")


def codedoggy_product_config() -> ToolServerConfig:
    """CodeDoggy agent pack: Grok product surface + Doggy enhancements.

    Doggy tools remain registered builtins even on pure Grok config; this only
    controls the finalized client-facing list.
    """
    return grok_build_product_config(include_doggy_enhancements=True)
