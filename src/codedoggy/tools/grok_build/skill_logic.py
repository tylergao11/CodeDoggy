"""Skill pure helpers — source port from Grok.

Ported from:
  crates/codegen/xai-grok-tools/src/implementations/skills/skill.rs
    build_skill_message, build_skill_block, build_skill_information
    extract_skill_display_text, apply_substitutions, SubstitutionContext
  types.rs SkillScope enum values
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

# Grok SkillScope string values (lowercase)
SCOPE_LOCAL = "local"
SCOPE_REPO = "repo"
SCOPE_USER = "user"
SCOPE_SERVER = "server"
SCOPE_BUNDLED = "bundled"
SCOPE_PLUGIN = "plugin"


@dataclass
class SkillInfo:
    """Grok ``SkillInfo`` (types.rs) — field names match source."""

    name: str
    description: str = ""
    path: str = ""
    scope: str = SCOPE_USER
    display_name: str | None = None
    has_user_specified_description: bool = False
    paths: list[str] | None = None
    when_to_use: str | None = None
    short_description: str | None = None
    author: str | None = None
    argument_hint: str | None = None
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] | None = None
    config_source: str | None = None
    plugin_name: str | None = None
    plugin_version: str | None = None
    plugin_root: str | None = None
    plugin_data: str | None = None
    allowed_tools: list[str] | None = None
    model: str | None = None
    effort: str | None = None
    user_invocable: bool = True
    disable_model_invocation: bool = False
    enabled: bool = True
    # Grok: Option<String>; empty string treated as None on wire
    body: str | None = None

    def dedup_key(self) -> str:
        """Grok ``SkillInfo::dedup_key``."""
        if self.plugin_name:
            return f"{self.plugin_name}:{self.name}"
        return self.name

    def label(self) -> str:
        """Grok ``SkillInfo::label``."""
        return self.display_name or self.name

    def body_text(self) -> str:
        """Body content or empty string."""
        return self.body or ""


@dataclass
class SkillInput:
    """Grok ``SkillInput`` (skill.rs) — wire schema for the skill tool."""

    skill: str
    args: str | None = None


@dataclass
class SkillOutput:
    """Grok ``SkillOutput`` (skill.rs)."""

    success: bool
    tool_result: str
    skill_name: str
    skill_message: str | None = None
    error: str | None = None


@dataclass
class SkillRef:
    """Grok ``SkillRef`` — entry in ``<skills_referenced>``."""

    name: str
    path: str


@dataclass
class SubstitutionContext:
    skill_dir: str | None = None
    session_id: str | None = None
    plugin_root: str | None = None
    plugin_data: str | None = None


# Grok OpenCode skill/mod.rs DESCRIPTION (static prologue; skills block rendered).
SKILL_TOOL_DESCRIPTION_PROLOGUE = (
    "Load a specialized skill that provides domain-specific instructions and workflows.\n"
    "\n"
    "When you recognize that a task matches one of the available skills listed below, "
    "use this tool to load the full skill instructions.\n"
    "\n"
    "The skill will inject detailed instructions, workflows, and access to bundled "
    "resources into the conversation via a `<skill_content name=\"...\">` block with "
    "the loaded content.\n"
    "\n"
    "The following skills provide specialized sets of instructions for particular tasks.\n"
    "Invoke this tool to load a skill when a task matches one of the available skills "
    "listed below:\n"
    "\n"
    "<available_skills>\n"
)

# Grok empty branch (template else). Doggy also discovers .codedoggy/skills.
SKILL_TOOL_EMPTY_HINT = (
    "(No skills available. Skills can be added in ~/.grok/skills/ or .grok/skills/ "
    "or ~/.codedoggy/skills/ or .codedoggy/skills/)"
)

SKILL_TOOL_DESCRIPTION_EPILOGUE = "</available_skills>"


def _xml_escape(text: str) -> str:
    """MiniJinja ``|e`` spirit for skill description fields."""
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def skill_location(skill: SkillInfo) -> str:
    """Path shown as ``<location>`` (Grok template ``skill.location`` ≈ path)."""
    return skill.path or ""


def is_skill_listable_for_tool(skill: SkillInfo) -> bool:
    """Model-facing skill tool listing filter (AvailableSkills spirit).

    Exclude disabled, non-user-invocable, and disable_model_invocation skills.
    """
    if not skill.enabled:
        return False
    if not skill.user_invocable:
        return False
    if skill.disable_model_invocation:
        return False
    if not (skill.name or "").strip():
        return False
    return True


def build_skill_tool_description(
    skills: Sequence[SkillInfo] | None = None,
    *,
    empty_hint: str = SKILL_TOOL_EMPTY_HINT,
) -> str:
    """Grok OpenCode skill ``DESCRIPTION`` with ``skills`` loop expanded.

    Renders the static prologue + ``<available_skills>`` roster (or empty
    hint) without MiniJinja. Skills are sorted by name for stable listings.
    """
    rows: list[SkillInfo] = [
        s for s in (skills or []) if is_skill_listable_for_tool(s)
    ]
    rows.sort(key=lambda s: (s.name or "").lower())
    body_parts: list[str] = [SKILL_TOOL_DESCRIPTION_PROLOGUE]
    if rows:
        for s in rows:
            name = _xml_escape(s.name)
            desc = _xml_escape(s.description or "")
            loc = _xml_escape(skill_location(s))
            body_parts.append(
                "  <skill>\n"
                f"    <name>{name}</name>\n"
                f"    <description>{desc}</description>\n"
                f"    <location>{loc}</location>\n"
                "  </skill>\n"
            )
    else:
        body_parts.append(f"{empty_hint}\n")
    body_parts.append(SKILL_TOOL_DESCRIPTION_EPILOGUE)
    return "".join(body_parts)


# Back-compat alias used by some builtins/tests.
SKILL_TOOL_DESCRIPTION = build_skill_tool_description(None)


def build_skill_message(skill: SkillInfo, content: str) -> str:
    """Grok ``build_skill_message`` — exact envelope shape."""
    return (
        f'<skill name="{skill.name}" description="{skill.description}" '
        f'path="{skill.path}">\n{content}\n</skill>'
    )


def build_skill_block(name: str, args: str, content: str) -> str:
    """Grok ``build_skill_block``."""
    if not args:
        return f'<skill name="{name}">\n{content}\n</skill>'
    return f'<skill name="{name}" args="{args}">\n{content}\n</skill>'


def build_skill_information(
    skill_blocks: list[str],
    refs: list[SkillRef | tuple[str, str]],
) -> str:
    """Grok ``build_skill_information`` — refs are ``SkillRef`` or (name, path)."""
    if not skill_blocks:
        return ""
    out = ["<skill_information>"]
    if refs:
        seen: list[tuple[str, str]] = []
        out.append("<skills_referenced>")
        for ref in refs:
            if isinstance(ref, SkillRef):
                name, path = ref.name, ref.path
            else:
                name, path = ref[0], ref[1]
            key = (name, path)
            if key in seen:
                continue
            seen.append(key)
            out.append(f'<skill name="{name}" path="{path}"/>')
        out.append("</skills_referenced>")
    out.append("\n".join(skill_blocks))
    out.append("</skill_information>")
    return "\n".join(out)


def format_skill_name(skill: SkillInfo) -> str:
    """Grok ``format_skill_name``."""
    if skill.plugin_name:
        return f"{skill.plugin_name}:{skill.name}"
    return f"{skill.scope}:{skill.name}"


def skill_name_from_path(path: str) -> str | None:
    """Grok ``skill_name_from_path`` — parent dir of exact ``SKILL.md`` only."""
    p = Path(path)
    # Grok: file_name must be exactly "SKILL.md" (case-sensitive)
    if p.name != "SKILL.md":
        return None
    parent = p.parent
    if parent.name in {"", ".", "/"}:
        return None
    return parent.name


def extract_skill_body(content: str) -> str:
    """Grok ``extract_skill_body`` — strip YAML frontmatter if present."""
    text = content or ""
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    if len(parts) >= 3:
        return parts[2].lstrip("\n")
    return text


def extract_skill_display_text(text: str) -> str | None:
    """Grok ``extract_skill_display_text``."""
    name_open = "<command-name>"
    name_close = "</command-name>"
    if name_open not in text:
        return None

    cmd: str | None = None
    cmd_open = "<command-message>"
    cmd_close = "</command-message>"
    start = text.find(cmd_open)
    if start >= 0:
        start += len(cmd_open)
        end = text.find(cmd_close, start)
        if end >= 0:
            cand = text[start:end]
            if cand:
                cmd = cand

    if cmd:
        args = _extract_command_args(text)
        return f"{cmd} {args}" if args else cmd

    inner = text.find(name_open)
    if inner < 0:
        return None
    inner += len(name_open)
    end = text.find(name_close, inner)
    if end < 0:
        return None
    name = text[inner:end]
    if not name:
        return None
    args = _extract_command_args(text)
    return f"/{name} {args}" if args else f"/{name}"


def _extract_command_args(text: str) -> str | None:
    open_t = "<command-args>"
    close_t = "</command-args>"
    start = text.find(open_t)
    if start < 0:
        return None
    start += len(open_t)
    end = text.find(close_t, start)
    if end < 0:
        end = len(text)
    args = text[start:end].strip()
    return args or None


def apply_substitutions(
    content: str,
    args: str | None = None,
    ctx: SubstitutionContext | None = None,
) -> str:
    """Grok ``apply_substitutions`` — argument + path tokens."""
    ctx = ctx or SubstitutionContext()
    args_str = args or ""
    argv = [] if not args_str else args_str.split()
    text = content
    args_substituted = False
    max_idx = max(len(argv), 1)

    for i in range(max_idx + 19, -1, -1):
        pattern = f"$ARGUMENTS[{i}]"
        if pattern in text:
            replacement = argv[i] if i < len(argv) else ""
            text = text.replace(pattern, replacement)
            args_substituted = True

    for i in range(max_idx + 19, -1, -1):
        pattern = f"${i}"
        pat_len = len(pattern)
        replacement = argv[i] if i < len(argv) else ""
        result: list[str] = []
        rest = text
        while True:
            pos = rest.find(pattern)
            if pos < 0:
                result.append(rest)
                break
            result.append(rest[:pos])
            after = rest[pos + pat_len :]
            if after and after[0].isdigit():
                result.append(pattern)
            else:
                result.append(replacement)
                args_substituted = True
            rest = after
        text = "".join(result)

    if "$ARGUMENTS" in text:
        text = text.replace("$ARGUMENTS", args_str)
        args_substituted = True

    if ctx.skill_dir:
        text = text.replace("${SKILL_DIR}", ctx.skill_dir)
        text = text.replace("${CLAUDE_SKILL_DIR}", ctx.skill_dir)
    if ctx.session_id:
        text = text.replace("${SESSION_ID}", ctx.session_id)
        text = text.replace("${CLAUDE_SESSION_ID}", ctx.session_id)
    if ctx.plugin_root:
        text = text.replace("${GROK_PLUGIN_ROOT}", ctx.plugin_root)
        text = text.replace("${CLAUDE_PLUGIN_ROOT}", ctx.plugin_root)
    if ctx.plugin_data:
        text = text.replace("${GROK_PLUGIN_DATA}", ctx.plugin_data)
        text = text.replace("${CLAUDE_PLUGIN_DATA}", ctx.plugin_data)

    if not args_substituted and args_str:
        text = f"{text.rstrip()}\n\n**ARGUMENTS:** {args_str}"
    return text


def resolve_skill_internal_links(body: str, skill_dir: str) -> str:
    """Grok ``resolve_skill_internal_links`` (simplified, no pulldown-cmark).

    Rewrite relative markdown links ``](rel/path)`` to absolute paths when the
    target exists under skill_dir (and stays inside skill_dir).
    """
    from pathlib import Path

    root = Path(skill_dir).resolve()
    if not root.is_dir() or not body:
        return body

    # Inline links: [text](url) — skip http(s), anchors, absolute paths
    pattern = re.compile(r"\]\(([^)]+)\)")

    def repl(m: re.Match[str]) -> str:
        url = m.group(1).strip()
        if not url or url.startswith(("#", "http://", "https://", "mailto:")):
            return m.group(0)
        # Windows drive or POSIX absolute
        p = Path(url)
        if p.is_absolute():
            return m.group(0)
        resolved = (root / url).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            return m.group(0)
        if not resolved.exists():
            return m.group(0)
        return f"]({resolved})"

    return pattern.sub(repl, body)
