"""Agent as a *config package* — not a loop (Grok xai-grok-agent).

An Agent bundles tools whitelist, system prompt, capability mode, and optional
session mode. The loop lives in shell/turn; Agent is portable definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codedoggy.orchestration.capability import kinds_for_capability
from codedoggy.orchestration.types import CapabilityMode, IsolationMode, SessionMode
from codedoggy.tools.kinds import ToolKind
from codedoggy.tools.registry import FinalizedToolset


@dataclass
class AgentDefinition:
    """Parsed agent definition (Markdown frontmatter spirit of Grok)."""

    name: str
    description: str = ""
    # Client-facing tool names; empty = all tools allowed by capability.
    tools: list[str] = field(default_factory=list)
    capability_mode: CapabilityMode = CapabilityMode.ALL
    isolation: IsolationMode = IsolationMode.NONE
    # extend = append body to base; full = body is the system prompt.
    prompt_mode: str = "extend"
    system_prompt_body: str = ""
    session_mode: SessionMode = SessionMode.NORMAL
    # Max sampling rounds for this agent (None = inherit parent).
    max_turns: int | None = None
    # Default run_in_background when spawned as subagent.
    background: bool = True
    color: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_system_prompt(self, base: str | None = None) -> str:
        body = (self.system_prompt_body or "").strip()
        mode = (self.prompt_mode or "extend").lower()
        if mode == "full":
            return body
        base_s = (base or "").strip()
        if base_s and body:
            return f"{base_s}\n\n{body}"
        return body or base_s

    def filter_toolset(self, parent: FinalizedToolset) -> FinalizedToolset:
        """Apply capability mode + optional name whitelist (Grok resolve_subagent_toolset)."""
        return filter_toolset(
            parent,
            capability=self.capability_mode,
            allow_names=set(self.tools) if self.tools else None,
        )


@dataclass
class Agent:
    """Built agent — ready for a host Session to consume."""

    definition: AgentDefinition
    system_prompt: str
    tools: FinalizedToolset

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def capability_mode(self) -> CapabilityMode:
        return self.definition.capability_mode

    def tool_definitions(self) -> list:
        return self.tools.tool_definitions()


def filter_toolset(
    parent: FinalizedToolset,
    *,
    capability: CapabilityMode = CapabilityMode.ALL,
    allow_names: set[str] | None = None,
    deny_names: set[str] | None = None,
) -> FinalizedToolset:
    """Return a new FinalizedToolset with tools filtered like Grok capability + whitelist."""
    allowed_kinds = kinds_for_capability(capability)
    by: dict[str, Any] = {}
    for name, ft in parent.by_client_name.items():
        if allow_names is not None and name not in allow_names:
            # Also allow short-id match if allow list used short ids
            short = getattr(ft, "short_id", None)
            if short not in allow_names and name not in allow_names:
                continue
        if deny_names and name in deny_names:
            continue
        kind = ft.kind
        if allowed_kinds is not None and kind not in allowed_kinds:
            continue
        by[name] = ft
    return FinalizedToolset(by_client_name=by)


def build_agent(
    definition: AgentDefinition,
    *,
    parent_tools: FinalizedToolset,
    base_system_prompt: str | None = None,
) -> Agent:
    tools = definition.filter_toolset(parent_tools)
    prompt = definition.resolve_system_prompt(base_system_prompt)
    return Agent(definition=definition, system_prompt=prompt, tools=tools)


# ── Built-in definitions (Grok explore / plan spirit) ───────────────────

EXPLORE_PROMPT = """\
You are an explore subagent (read-only codebase reconnaissance).

Hard constraints:
- Do not edit, write, delete, or run shell commands that change state.
- Prefer code_nav / grep / read_file over broad listing.
- Do not spawn further subagents.

Method:
1. Clarify the question into 1–3 concrete search targets.
2. Use code_nav for symbols when names are known; else grep then read_file.
3. Check session_search / memory_search only when prior context may help.

Return to the parent a concise factual report with:
- Findings (bullet list, with file:line when known)
- Open questions / gaps
- Suggested next tools for the parent (if any)
No implementation — exploration only.
"""

PLAN_PROMPT = """\
You are a plan-mode agent. Produce an actionable plan only.

Hard constraints (enforced by the plan gate, not optional):
- You may only edit the plan file (default plan.md). Other workspace writes are rejected.
- Do not run shell or spawn subagents to implement.
- Prefer read/search tools to ground the plan in the actual codebase.

Plan quality:
- Ordered, testable steps with clear done criteria
- Note risks, migrations, and verification commands
- Keep the plan file updated as your single deliverable

When the plan is ready, stop — the parent will implement.
"""


def builtin_explore() -> AgentDefinition:
    return AgentDefinition(
        name="explore",
        description="Read-only codebase exploration",
        tools=[
            "read_file",
            "grep",
            "list_dir",
            "code_nav",
            "lsp",
            "session_search",
            "memory_search",
            "memory_get",
            "web_search",
            "web_fetch",
        ],
        capability_mode=CapabilityMode.READ_ONLY,
        system_prompt_body=EXPLORE_PROMPT,
        prompt_mode="extend",
        background=True,
        max_turns=16,
    )


def builtin_plan() -> AgentDefinition:
    return AgentDefinition(
        name="plan",
        description="Plan-only agent (plan file edits only)",
        tools=[
            "read_file",
            "grep",
            "list_dir",
            "code_nav",
            "lsp",
            "session_search",
            "memory_search",
            "search_replace",
            "write",
            "todo_write",
        ],
        capability_mode=CapabilityMode.READ_WRITE,
        session_mode=SessionMode.PLAN,
        system_prompt_body=PLAN_PROMPT,
        prompt_mode="extend",
        background=False,
        max_turns=12,
    )


GENERAL_PURPOSE_PROMPT = """\
You are a general-purpose subagent working a focused slice of a larger task.

Hard constraints:
- Do not spawn further subagents (including parallel_tasks).
- Stay inside the prompt's scope; return a concise report to MAIN.
- Prefer dedicated tools over shell when possible.

Method:
1. Do the assigned work thoroughly within your slice.
2. Surface failures and open questions clearly.
3. End with a short summary MAIN can synthesise with sibling reports.

MAIN owns final aggregation — do not assume other children's results.
"""


def builtin_general_purpose() -> AgentDefinition:
    """Default child for parallel fan-out (Grok general-purpose catalog name)."""
    return AgentDefinition(
        name="general-purpose",
        description="Focused worker for a slice of MAIN's parallel fan-out",
        # Empty tools = all tools allowed by capability (minus nested spawn strip).
        tools=[],
        capability_mode=CapabilityMode.ALL,
        system_prompt_body=GENERAL_PURPOSE_PROMPT,
        prompt_mode="extend",
        background=True,
        max_turns=24,
    )


BUILTIN_AGENTS: dict[str, AgentDefinition] = {
    "explore": builtin_explore(),
    "plan": builtin_plan(),
    "general-purpose": builtin_general_purpose(),
    "general_purpose": builtin_general_purpose(),  # underscore alias
}


def resolve_agent_definition(name: str) -> AgentDefinition | None:
    key = (name or "").strip().lower()
    if not key:
        return None
    return BUILTIN_AGENTS.get(key)


def load_agent_definition_file(path: Path) -> AgentDefinition:
    """Minimal Markdown+YAML frontmatter loader (Grok agent definition files)."""
    text = Path(path).read_text(encoding="utf-8")
    if not text.startswith("---"):
        return AgentDefinition(
            name=Path(path).stem,
            system_prompt_body=text,
            prompt_mode="full",
        )
    parts = text.split("---", 2)
    if len(parts) < 3:
        return AgentDefinition(name=Path(path).stem, system_prompt_body=text)
    front = parts[1].strip()
    body = parts[2].lstrip("\n")
    meta = _parse_simple_yaml(front)
    name = str(meta.get("name") or Path(path).stem)
    tools_raw = meta.get("tools") or []
    tools: list[str] = []
    if isinstance(tools_raw, list):
        tools = [str(t) for t in tools_raw]
    elif isinstance(tools_raw, str):
        tools = [t.strip() for t in tools_raw.split(",") if t.strip()]
    cap = CapabilityMode.parse(str(meta.get("capability_mode") or meta.get("capabilityMode") or "all"))
    mode_s = str(meta.get("permissionMode") or meta.get("session_mode") or "normal").lower()
    session_mode = {
        "plan": SessionMode.PLAN,
        "goal": SessionMode.GOAL,
        "normal": SessionMode.NORMAL,
        "agent": SessionMode.NORMAL,
    }.get(mode_s, SessionMode.NORMAL)
    return AgentDefinition(
        name=name,
        description=str(meta.get("description") or ""),
        tools=tools,
        capability_mode=cap,
        prompt_mode=str(meta.get("promptMode") or meta.get("prompt_mode") or "extend"),
        system_prompt_body=body,
        session_mode=session_mode,
        max_turns=_as_int(meta.get("max_turns") or meta.get("maxTurns")),
        background=_as_bool(meta.get("background"), default=True),
    )


def _as_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_bool(v: Any, *, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Tiny subset parser for agent frontmatter (no PyYAML dependency)."""
    out: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("  - ") or line.startswith("- "):
            if current_list_key is None:
                continue
            item = line.lstrip()[2:].strip().strip("\"'")
            out.setdefault(current_list_key, []).append(item)
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if not val:
            current_list_key = key
            out[key] = out.get(key) or []
            continue
        current_list_key = None
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            out[key] = [x.strip().strip("\"'") for x in inner.split(",") if x.strip()]
        else:
            out[key] = val.strip("\"'")
    return out
