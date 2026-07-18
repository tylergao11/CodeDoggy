"""Task / spawn_subagent pure helpers — source port from Grok.

Ported from grok-build:
  crates/common/xai-tool-types/src/task.rs
    is_not_sentinel, sanitize_optional_arg
    format_subagent_started_background, format_subagent_completed
    format_resume_footer
    default_subagent_type, build_task_description, TaskToolNaming
    BUILTIN_SUBAGENTS catalog descriptions/tools_template fragments
  crates/codegen/xai-grok-tools/src/implementations/grok_build/task/types.rs
    sanitize_cwd_value, is_valid_resume_id
  crates/codegen/xai-grok-tools/src/implementations/grok_build/task/mod.rs
    MAX_SUBAGENT_DEPTH, auto-backgrounded notice text
  crates/codegen/xai-grok-agent/src/builder.rs
    task_model_guidance (product-facing model constraints messaging)

No host dispatch / coordinator here — pure strings + sanitizers only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# Grok task/mod.rs
MAX_SUBAGENT_DEPTH: int = 1

# Grok default_subagent_type()
DEFAULT_SUBAGENT_TYPE: str = "general-purpose"

# Grok TaskToolInput.model schemars description (exact).
MODEL_PARAM_DESCRIPTION = (
    "Optional model slug for this agent. If provided, it must resolve to one "
    "of the available model slugs. If omitted, the subagent uses the same model "
    "as the parent agent. Do not pass if resume_from is set (prior model will be "
    "used). Only choose an explicit model when the user directly requests it."
)

# ── sanitize (task.rs / types.rs) ────────────────────────────────────


def is_not_sentinel(s: str) -> bool:
    """True when *s* is not a model-emitted placeholder.

    Grok ``is_not_sentinel``: reject empty / whitespace / null / none / undefined
    (case-insensitive) after trim.
    """
    t = (s or "").strip()
    if not t:
        return False
    low = t.lower()
    return low not in {"null", "none", "undefined"}


def sanitize_optional_arg(value: str | None) -> str | None:
    """Drop sentinels and trim (Grok ``sanitize_optional_arg``)."""
    if value is None:
        return None
    if not is_not_sentinel(value):
        return None
    trimmed = value.strip()
    return trimmed


def is_valid_resume_id(s: str) -> bool:
    """Grok ``is_valid_resume_id`` — same as is_not_sentinel."""
    return is_not_sentinel(s)


def sanitize_cwd_value(s: str | None) -> str | None:
    """Sanitize model-emitted ``cwd`` (Grok ``sanitize_cwd_value``).

    Strips surrounding quote/backtick characters, trims, expands leading ``~``,
    rejects sentinels. Returns cleaned path or None.
    """
    if s is None:
        return None
    unquoted = (s or "").strip().strip("\"'`")
    cleaned = unquoted.strip()
    if not is_not_sentinel(cleaned):
        return None
    if cleaned.startswith("~"):
        home = Path.home()
        if cleaned == "~":
            return str(home)
        if cleaned.startswith("~/") or cleaned.startswith("~\\"):
            return str(home / cleaned[2:])
    return cleaned


# ── formatters (task.rs) ─────────────────────────────────────────────


def format_subagent_started_background(
    subagent_id: str,
    subagent_type: str,
    description: str,
    task_output_tool_name: str = "get_command_or_subagent_output",
) -> str:
    """Grok ``format_subagent_started_background`` (exact text)."""
    return (
        f"Subagent started in background.\n"
        f"subagent_id: {subagent_id}\n"
        f"type: {subagent_type}\n"
        f"description: {description}\n\n"
        f"Use {task_output_tool_name} with task_ids=[\"{subagent_id}\"] "
        f"and timeout_ms to wait for results."
    )


def format_resume_footer(
    subagent_id: str,
    subagent_type: str,
    persona: str | None = None,
) -> str:
    """Grok ``format_resume_footer``."""
    footer = (
        f"<subagent_result>\n"
        f"subagent_id: {subagent_id}\n"
        f"subagent_type: {subagent_type}\n"
        f'To continue this subagent\'s conversation, use resume_from="{subagent_id}".'
    )
    if persona:
        footer += (
            f'\nThe subagent used persona="{persona}". '
            f"Pass the same persona when resuming."
        )
    footer += "\n</subagent_result>"
    return footer


def format_subagent_completed(
    output: str,
    subagent_id: str,
    subagent_type: str,
    tool_calls: int,
    turns: int,
    duration_ms: int,
    persona: str | None = None,
) -> str:
    """Grok ``format_subagent_completed`` (exact meta + resume footer)."""
    footer = format_resume_footer(subagent_id, subagent_type, persona)
    return (
        f"{output}\n\n"
        f"<subagent_meta>id={subagent_id}, type={subagent_type}, "
        f"tool_calls={tool_calls}, turns={turns}, "
        f"duration_ms={duration_ms}</subagent_meta>\n\n"
        f"{footer}"
    )


def format_auto_backgrounded_notice(
    subagent_id: str,
    subagent_type: str,
    description: str,
    task_output_tool_name: str = "get_command_or_subagent_output",
) -> str:
    """Grok task/mod.rs blocking→backgrounded notice (exact text)."""
    return (
        "Subagent took longer than the foreground budget and was moved to the "
        "background to keep the conversation responsive. It is still running — you "
        "will be notified when it completes.\n"
        f"subagent_id: {subagent_id}\n"
        f"type: {subagent_type}\n"
        f"description: {description}\n\n"
        f"Use {task_output_tool_name} with task_ids=[\"{subagent_id}\"] "
        f"and timeout_ms to wait for results."
    )


def depth_limit_error_message(depth: int, max_depth: int = MAX_SUBAGENT_DEPTH) -> str:
    """Grok depth-limit ``invalid_arguments`` message (exact)."""
    return (
        f"Subagent depth limit exceeded (current depth: {depth}, max: {max_depth}). "
        f"Cannot spawn further nested subagents."
    )


def unknown_subagent_type_message(subagent_type: str, available: Sequence[str]) -> str:
    """Grok validate Unknown outcome (exact)."""
    if available:
        return (
            f"Unknown subagent type: {subagent_type}. "
            f"Available types: {', '.join(available)}"
        )
    return f"Unknown subagent type: {subagent_type}"


def disabled_subagent_type_message(subagent_type: str) -> str:
    """Grok validate Disabled outcome (exact)."""
    return (
        f"Subagent '{subagent_type}' is disabled via [subagents.toggle] in config.toml"
    )


def not_allowed_subagent_type_message(subagent_type: str, allowed: Sequence[str]) -> str:
    """Grok validate NotAllowed outcome (exact)."""
    return (
        f"agent can only spawn: {', '.join(allowed)}; "
        f"'{subagent_type}' not allowed"
    )


def validation_unavailable_type_message(subagent_type: str) -> str:
    """Grok validate ValidationUnavailable (exact)."""
    return (
        f"Cannot validate subagent type '{subagent_type}': the subagent coordinator is "
        f"unreachable. Retry shortly or notify ops."
    )


def missing_backend_message() -> str:
    """Grok missing SubagentBackendResource (adapted to Doggy extra hooks)."""
    return (
        "SubagentBackendResource (subagent support not initialized)"
    )


def cwd_worktree_mutex_message() -> str:
    """Grok cwd ↔ isolation=worktree mutual exclusion (exact)."""
    return (
        'cwd and isolation="worktree" are mutually exclusive. '
        "Use cwd to point the subagent at an existing directory, "
        'or isolation="worktree" to create a new isolated worktree, '
        "but not both."
    )


def cwd_not_directory_message(cwd_path: str) -> str:
    return f'cwd "{cwd_path}" exists but is not a directory'


def cwd_does_not_exist_message(cwd_path: str) -> str:
    return f'cwd "{cwd_path}" does not exist'


# ── description builder (task.rs + agent builder) ────────────────────


@dataclass(frozen=True)
class TaskToolNaming:
    """Grok ``TaskToolNaming`` — product vs wire name substitution."""

    task_tool: str = "task"
    subagent_type_param: str = "subagent_type"
    run_in_background_param: str = "run_in_background"
    resume_from_param: str = "resume_from"
    background_retrieval_tool: str = "get_task_output"
    isolation_param: str = "isolation"


# Product surface (config.rs renames): task→spawn_subagent, run_in_background→background
PRODUCT_TASK_NAMING = TaskToolNaming(
    task_tool="spawn_subagent",
    subagent_type_param="subagent_type",
    run_in_background_param="background",
    resume_from_param="resume_from",
    background_retrieval_tool="get_command_or_subagent_output",
    isolation_param="isolation",
)

WIRE_TASK_NAMING = TaskToolNaming(
    task_tool="task",
    subagent_type_param="subagent_type",
    run_in_background_param="run_in_background",
    resume_from_param="resume_from",
    background_retrieval_tool="get_task_output",
    isolation_param="isolation",
)


@dataclass(frozen=True)
class SubagentDescriptor:
    """Grok ``SubagentDescriptor`` for description roster lines."""

    name: str
    description: str
    tools: str | None = None


# Grok BUILTIN_SUBAGENTS with product tool names already substituted
# (Grok keeps ${{ tools.by_kind.* }} placeholders; Doggy has no TemplateRenderer).
_GENERAL_PURPOSE_TOOLS = (
    "Has access to all tools: "
    "run_terminal_command, read_file, search_replace, "
    "list_dir, grep, web_search, "
    "and enter_plan_mode."
)
_EXPLORE_TOOLS = (
    "Read-only \u2014 has access to: "
    "read_file, list_dir, "
    "grep."
)
_PLAN_TOOLS = (
    "Read-only \u2014 has access to all tools except file editing "
    "(search_replace is not available): "
    "read_file, list_dir, grep, "
    "web_search, and enter_plan_mode."
)

BUILTIN_SUBAGENT_DESCRIPTORS: tuple[SubagentDescriptor, ...] = (
    SubagentDescriptor(
        name="general-purpose",
        description="General purpose agent for multi-step tasks.",
        tools=_GENERAL_PURPOSE_TOOLS,
    ),
    SubagentDescriptor(
        name="explore",
        description="Fast, read-only agent specialized for codebase exploration.",
        tools=_EXPLORE_TOOLS,
    ),
    SubagentDescriptor(
        name="plan",
        description="Software architect for planning implementation strategies.",
        tools=_PLAN_TOOLS,
    ),
)


def build_task_description(
    subagents: Sequence[SubagentDescriptor],
    naming: TaskToolNaming | None = None,
) -> str:
    """Grok ``build_task_description`` (exact usage-notes body)."""
    naming = naming or PRODUCT_TASK_NAMING
    agent_lines: list[str] = []
    for s in subagents:
        if s.tools:
            agent_lines.append(f"- **{s.name}**: {s.description} {s.tools}")
        else:
            agent_lines.append(f"- **{s.name}**: {s.description}")
    agents_block = "\n".join(agent_lines)

    task_tool = naming.task_tool
    subagent_type_param = naming.subagent_type_param
    run_in_background_param = naming.run_in_background_param
    resume_from_param = naming.resume_from_param
    background_retrieval_tool = naming.background_retrieval_tool
    isolation_param = naming.isolation_param

    return (
        "Start a subagent that works on a task independently and reports back.\n\n"
        "Agent types:\n\n"
        f"{agents_block}\n\n"
        "## Usage notes\n"
        "- When the agent is done, it returns a single message with its agent ID. "
        "Use that ID to resume the agent later for follow-up work.\n"
        f"- {run_in_background_param}: Returns immediately with a subagent_id. "
        f"Use {background_retrieval_tool} to retrieve results. "
        "This is set to true by default.\n"
        "- Subagents receive a compacted version of project instructions (AGENTS.md). "
        "If the task requires detailed conventions (e.g., build rules, testing patterns), "
        "include the relevant rules directly in the prompt.\n"
        f"- When using the {task_tool} tool, you must specify a {subagent_type_param} "
        "parameter to select which agent type to use.\n\n"
        "Resuming a previous agent (resume_from):\n"
        f"- Use {resume_from_param} to continue a previously completed subagent's "
        f"conversation. Pass the subagent_id returned by a prior {task_tool} call. "
        "A resumed agent keeps its full transcript and tool state, so you only need "
        "to describe what changed since the last run — don't re-explain the original "
        "task.\n"
        "- The resumed agent must use the same subagent_type as the source.\n\n"
        "Isolation mode:\n"
        f"- Use {isolation_param} to control the child's execution environment. With "
        '"worktree", the child runs in an isolated git worktree whose edits don\'t '
        "affect the parent workspace; the worktree is preserved after completion and "
        "its path is returned in the output."
    )


def task_model_guidance(
    model_slugs: Iterable[str] | None = None,
    *,
    model_param: str = "model",
) -> str:
    """Grok agent ``task_model_guidance`` (product model-constraints messaging)."""
    slugs = sorted({str(s).strip() for s in (model_slugs or []) if str(s).strip()})
    if not slugs:
        return (
            f"\n\nNo explicit model slugs are currently available. "
            f"Omit `{model_param}` to inherit the parent model."
        )
    model_list = "\n".join(f"- {s}" for s in slugs)
    return (
        "\n\nIf the user explicitly asks for the model of a subagent/task, you may "
        f"ONLY use model slugs from this list:\n"
        f"{model_list}\n\n"
        f"If the user does not explicitly request a model, omit `{model_param}` "
        f"to inherit the parent model."
    )


def default_task_description(
    *,
    naming: TaskToolNaming | None = None,
    model_slugs: Iterable[str] | None = None,
) -> str:
    """Header + built-in roster + model guidance (Grok agent builder assembly)."""
    base = build_task_description(BUILTIN_SUBAGENT_DESCRIPTORS, naming)
    return base + task_model_guidance(model_slugs)


def parse_lenient_bool(raw: object, *, default: bool) -> bool:
    """Grok ``deserialize_lenient_bool`` spirit for run_in_background."""
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"true", "1", "yes", "on"}:
            return True
        if s in {"false", "0", "no", "off"}:
            return False
    return default
