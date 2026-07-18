"""skill tool — Grok / OpenCode SkillTool (invoke user-defined SKILL.md).

Ported from:
  implementations/opencode/skill/mod.rs
    DESCRIPTION template + <available_skills> roster
  implementations/skills/skill.rs formatters + apply_substitutions
  implementations/skills/discovery.rs frontmatter + scopes

Host optional:
  extra['skills_registry'] — list[SkillInfo|dict]
  extra['skill_paths'] — extra directories
  else filesystem discovery under .codedoggy/skills and .grok/skills

Description is **dynamic** (Grok AvailableSkills): each list-tools pass
re-renders the roster from discovery / host registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codedoggy.tools.grok_build.skill_discovery import (
    load_skills_from_extra,
    resolve_skill,
)
from codedoggy.tools.grok_build.skill_logic import (
    SkillInfo,
    SkillInput,
    SkillOutput,
    SubstitutionContext,
    apply_substitutions,
    build_skill_message,
    build_skill_tool_description,
    format_skill_name,
    resolve_skill_internal_links,
)
from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolError,
    ToolId,
)


class SkillTool(Tool):
    def id(self) -> ToolId:
        return ToolId("skill")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.Skill

    def has_dynamic_description(self) -> bool:
        return True

    def description(self, ctx: ListToolsContext | None = None) -> ToolDescription:
        # Grok OpenCode DESCRIPTION with skills loop expanded.
        skills = self._skills_for_list(ctx)
        text = build_skill_tool_description(skills)
        return ToolDescription(name="skill", description=text)

    @staticmethod
    def _skills_for_list(ctx: ListToolsContext | None) -> list[SkillInfo]:
        cwd: Path | str | None = None
        extra: dict[str, Any] | None = None
        if ctx is not None:
            cwd = ctx.cwd
            extra = ctx.extra
        if cwd is None and extra and extra.get("cwd"):
            cwd = extra.get("cwd")  # type: ignore[assignment]
        try:
            return load_skills_from_extra(extra, cwd)
        except Exception:  # noqa: BLE001 — listing must not fail finalize
            return []

    def parameters_schema(self) -> dict[str, Any]:
        # Grok skills/skill.rs SkillInput schemars (product wire uses "skill"
        # for the name field; OpenCode variant uses "name" — Doggy keeps "skill").
        return {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "The name of the skill to invoke",
                },
                "args": {
                    "type": "string",
                    "description": "Optional arguments to pass to the skill",
                },
            },
            "required": ["skill"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        out = self.invoke(ctx, args)
        if out.skill_message:
            return out.skill_message
        if out.success:
            return out.tool_result
        raise ToolError(out.error or out.tool_result, code="not_found")

    def invoke(self, ctx: ToolCallContext, args: dict[str, Any]) -> SkillOutput:
        """Grok skill run → ``SkillOutput`` (model text is skill_message)."""
        name = str(args.get("skill") or "").strip()
        if not name:
            raise ToolError.invalid_arguments("skill is required")
        # Grok SkillInput
        _inp = SkillInput(skill=name, args=str(args["args"]) if args.get("args") is not None else None)
        arg_str = _inp.args

        skills = load_skills_from_extra(ctx.extra, ctx.cwd)
        # Ambiguous short-name across scopes (OpenCode/Grok find_skill spirit)
        enabled = [s for s in skills if s.enabled]
        qualified_hits = [s for s in enabled if format_skill_name(s) == name]
        bare_hits = [s for s in enabled if s.name == name]
        if qualified_hits:
            skill = qualified_hits[0]
        elif len(bare_hits) > 1:
            qnames = ", ".join(format_skill_name(s) for s in bare_hits)
            msg = (
                f"Skill '{name}' is ambiguous -- multiple skills share this name. "
                f"Use a qualified name: {qnames}"
            )
            return SkillOutput(
                success=False,
                tool_result=msg,
                skill_name=name,
                skill_message=None,
                error=f"Ambiguous skill '{name}': use one of {qnames}",
            )
        else:
            skill = resolve_skill(name, skills)
        if skill is None:
            available = ", ".join(sorted({s.name for s in skills})) or "(none)"
            msg = f"Unknown skill {name!r}. Available: {available}"
            return SkillOutput(
                success=False,
                tool_result=msg,
                skill_name=name,
                skill_message=None,
                error=msg,
            )

        body = skill.body_text()
        if not body and skill.path:
            try:
                raw = Path(skill.path).read_text(encoding="utf-8")
                from codedoggy.tools.grok_build.skill_logic import extract_skill_body

                body = extract_skill_body(raw)
            except OSError as e:
                err = f"Failed to load skill {name!r}: {e}"
                return SkillOutput(
                    success=False,
                    tool_result=err,
                    skill_name=name,
                    skill_message=None,
                    error=err,
                )

        skill_dir = str(Path(skill.path).parent) if skill.path else None
        session_id = ctx.session_id or ""
        content = apply_substitutions(
            body or "",
            arg_str,
            SubstitutionContext(
                skill_dir=skill_dir,
                session_id=session_id or None,
                plugin_root=skill.plugin_root,
                plugin_data=skill.plugin_data,
            ),
        )
        if skill_dir:
            content = resolve_skill_internal_links(content, skill_dir)
        if not content.strip():
            fallback = f"Skill {format_skill_name(skill)} loaded but has empty body."
            return SkillOutput(
                success=True,
                tool_result=fallback,
                skill_name=skill.name,
                skill_message=None,
                error=None,
            )
        message = build_skill_message(
            SkillInfo(
                name=skill.name,
                description=skill.description,
                path=skill.path,
                scope=skill.scope,
            ),
            content,
        )
        return SkillOutput(
            success=True,
            tool_result=message,
            skill_name=skill.name,
            skill_message=message,
            error=None,
        )
