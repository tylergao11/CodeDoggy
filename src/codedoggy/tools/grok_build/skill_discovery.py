"""Skill discovery — SKILL.md frontmatter + path scopes (Grok spirit).

Ported from:
  implementations/skills/discovery.rs
    normalize_skill_name, is_valid_skill_name, parse_skill_frontmatter
    find_skill_md_paths, walk_for_skill_md, parse_skill_files
    discover_skills_for_paths (simplified roots; no CompatConfig)
  SkillScope: local / repo / user / bundled / server / plugin

Paths (CodeDoggy product homes, Grok uses .grok):
  local:  {cwd}/.codedoggy/skills and {cwd}/.grok/skills
  repo:   {git_root}/.codedoggy/skills and .grok/skills
  user:   ~/.codedoggy/skills and ~/.grok/skills
  host:   extra['skills_registry'] or extra['skill_paths']
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from codedoggy.tools.grok_build.skill_logic import (
    SCOPE_LOCAL,
    SCOPE_REPO,
    SCOPE_USER,
    SkillInfo,
    extract_skill_body,
)

# Grok discovery.rs constants
MAX_DESCRIPTION_LEN = 1024
MAX_NAME_LEN = 64
MAX_FRONTMATTER_BYTES = 4096
MAX_BODY_PEEK_BYTES = 2048
MAX_SKILL_WALK_DEPTH = 5

# Grok RECOVERABLE_KEYS — line recovery must not mangle list/map fields
_RECOVERABLE_KEYS = frozenset({"name", "description", "when-to-use", "when_to_use"})

# Grok vendor-default denylist (path-gated)
CURSOR_DEFAULT_SKILLS = frozenset(
    {
        "babysit",
        "canvas",
        "create-hook",
        "create-rule",
        "create-skill",
        "create-subagent",
        "loop",
        "migrate-to-skills",
        "sdk",
        "shell",
        "split-to-prs",
        "statusline",
        "update-cli-config",
        "update-cursor-settings",
    }
)
CLAUDE_DEFAULT_SKILLS = frozenset({"pdf", "docx", "xlsx", "pptx", "skill-creator"})


@dataclass
class ParsedFrontmatter:
    """Grok ``ParsedFrontmatter`` — result of ``parse_skill_frontmatter``."""

    name: str
    description: str = ""
    license: str | None = None
    compatibility: str | None = None
    short_description: str | None = None
    author: str | None = None
    metadata: dict[str, str] | None = None
    argument_hint: str | None = None
    allowed_tools: list[str] | None = None
    model: str | None = None
    effort: str | None = None
    user_invocable: bool = True
    disable_model_invocation: bool = False
    when_to_use: str | None = None
    has_user_specified_description: bool = False
    paths: list[str] | None = None


class SkillParseError(Exception):
    """Grok ``SkillParseError`` family."""

    def __init__(self, kind: str, detail: str = "") -> None:
        self.kind = kind
        self.detail = detail
        super().__init__(detail or kind)


def normalize_skill_name(name: str) -> str:
    """Grok ``normalize_skill_name`` — non [a-z0-9] → hyphen, collapse, trim."""
    result: list[str] = []
    for c in (name or "").strip():
        ch = c.lower()
        if "a" <= ch <= "z" or "0" <= ch <= "9":
            result.append(ch)
        else:
            if result and result[-1] == "-":
                continue
            result.append("-")
    return "".join(result).strip("-")


def is_valid_skill_name(name: str) -> bool:
    """Grok ``is_valid_skill_name`` — lowercase, digits, single hyphens, ≤64."""
    if not name or len(name) > MAX_NAME_LEN:
        return False
    if name.startswith("-") or name.endswith("-") or "--" in name:
        return False
    return all(c.islower() or c.isdigit() or c == "-" for c in name)


def is_vendor_default_skill(path: str, name: str) -> bool:
    """Grok ``is_vendor_default_skill`` — path-gated denylist."""
    p = path.replace("\\", "/")
    in_cursor = "/.cursor/" in p
    in_claude = "/.claude/" in p
    return (in_cursor and name in CURSOR_DEFAULT_SKILLS) or (
        in_claude and name in CLAUDE_DEFAULT_SKILLS
    )


def parse_skill_frontmatter(
    content: str,
    fallback_name: str | None = None,
) -> ParsedFrontmatter:
    """Grok ``parse_skill_frontmatter`` — PyYAML (= serde_yaml spirit).

    Pipeline (discovery.rs)::
      1. yaml.safe_load(frontmatter)
      2. on fail → quote_problematic_values + safe_load retry
      3. on fail → recover_scalar_fields (name/description/when-to-use only)
    """
    text = (content or "").lstrip()
    if not text.startswith("---"):
        raise SkillParseError("NoFrontmatter")
    after = text[3:]
    idx = after.find("\n---")
    if idx < 0:
        raise SkillParseError("NoFrontmatter")
    yaml_content = after[:idx].strip()

    frontmatter = _load_frontmatter_map(yaml_content)

    fm_name = _coerce_to_string(frontmatter.get("name"))
    if fm_name is None and not fallback_name:
        raise SkillParseError("YamlError", "missing 'name' and no directory fallback")
    name: str | None = None
    for cand in (fm_name, fallback_name):
        if not cand:
            continue
        n = normalize_skill_name(cand)
        if is_valid_skill_name(n):
            name = n
            break
    if name is None:
        raise SkillParseError(
            "InvalidName",
            normalize_skill_name(fm_name or "") if fm_name else "",
        )

    desc_val = frontmatter.get("description")
    coerced_description = _coerce_to_string(desc_val)
    has_desc = coerced_description is not None
    description = _cap_string(coerced_description or "", MAX_DESCRIPTION_LEN)

    when_raw = frontmatter.get("when-to-use")
    if when_raw is None:
        when_raw = frontmatter.get("when_to_use")
    when = _coerce_to_string(when_raw)
    if when is not None:
        when = _cap_string(when, MAX_DESCRIPTION_LEN)

    paths = _parse_skill_paths(frontmatter.get("paths"))
    short_desc, author, metadata = _parse_metadata(frontmatter.get("metadata"))

    # Absent user-invocable → true; only explicit true / "true" is true for disable
    if "user-invocable" not in frontmatter and "user_invocable" not in frontmatter:
        user_invocable = True
    else:
        user_invocable = _parse_boolean_frontmatter(
            frontmatter.get("user-invocable", frontmatter.get("user_invocable"))
        )
    disable_model = _parse_boolean_frontmatter(
        frontmatter.get(
            "disable-model-invocation",
            frontmatter.get("disable_model_invocation"),
        )
    )

    return ParsedFrontmatter(
        name=name,
        description=description,
        license=_coerce_to_string(frontmatter.get("license")),
        compatibility=_coerce_to_string(frontmatter.get("compatibility")),
        short_description=short_desc,
        author=author,
        metadata=metadata,
        argument_hint=_coerce_to_string(
            frontmatter.get("argument-hint", frontmatter.get("argument_hint"))
        ),
        allowed_tools=_coerce_tool_list(
            frontmatter.get("allowed-tools", frontmatter.get("allowed_tools"))
        ),
        model=_coerce_to_string(frontmatter.get("model")),
        effort=_coerce_to_string(frontmatter.get("effort")),
        user_invocable=user_invocable,
        disable_model_invocation=disable_model,
        when_to_use=when,
        has_user_specified_description=has_desc,
        paths=paths,
    )


def parse_skill_md(path: Path, *, scope: str = SCOPE_USER) -> SkillInfo | None:
    """Parse a SKILL.md file into SkillInfo (Grok parse_skill_files item path)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    fallback = path.parent.name if path.name == "SKILL.md" else path.stem
    try:
        parsed = parse_skill_frontmatter(text, fallback)
    except SkillParseError as e:
        if e.kind in {"NoFrontmatter", "YamlError"}:
            name = normalize_skill_name(fallback)
            if not is_valid_skill_name(name):
                return None
            parsed = ParsedFrontmatter(name=name)
        else:
            return None

    body = extract_skill_body(text).lstrip("\n")
    description = parsed.description
    if not description:
        peek = body[:MAX_BODY_PEEK_BYTES]
        description = (_first_paragraph(peek) or parsed.name)[:MAX_DESCRIPTION_LEN]

    path_str = str(path.resolve())
    if is_vendor_default_skill(path_str, parsed.name):
        return None

    return SkillInfo(
        name=parsed.name,
        description=description,
        path=path_str,
        scope=scope,
        when_to_use=parsed.when_to_use,
        short_description=parsed.short_description,
        author=parsed.author,
        argument_hint=parsed.argument_hint,
        license=parsed.license,
        compatibility=parsed.compatibility,
        metadata=parsed.metadata,
        allowed_tools=parsed.allowed_tools,
        model=parsed.model,
        effort=parsed.effort,
        user_invocable=parsed.user_invocable,
        disable_model_invocation=parsed.disable_model_invocation,
        body=body or None,
        has_user_specified_description=parsed.has_user_specified_description,
        paths=parsed.paths,
    )


def _load_frontmatter_map(yaml_content: str) -> dict[str, Any]:
    """Grok: safe_load → quote retry → recover_scalar_fields."""
    for candidate in (yaml_content, _quote_problematic_values(yaml_content)):
        try:
            loaded = yaml.safe_load(candidate)
        except yaml.YAMLError:
            continue
        if loaded is None:
            return {}
        if isinstance(loaded, dict):
            # normalize keys to str
            return {str(k): v for k, v in loaded.items()}
        # non-mapping root → empty (coerce per-field will miss all)
        return {}
    return _recover_scalar_fields(yaml_content)


def _quote_problematic_values(frontmatter: str) -> str:
    """Grok ``quote_problematic_values`` — retry path for colon / indicator chars."""
    needs_chars = set("{}[]*&#!|>%@`")

    def needs_quoting(v: str) -> bool:
        return any(c in needs_chars for c in v) or (": " in v)

    lines: list[str] = []
    for line in frontmatter.splitlines():
        colon = line.find(":")
        if colon < 0:
            lines.append(line)
            continue
        key = line[:colon]
        if not key or not all(c.isalpha() or c in "_-" for c in key):
            lines.append(line)
            continue
        after = line[colon + 1 :]
        value = after.lstrip()
        # require whitespace after colon and non-empty value
        if not value or len(value) == len(after):
            lines.append(line)
            continue
        value = value.rstrip()
        already = (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        )
        if already or not needs_quoting(value):
            lines.append(line)
            continue
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{key}: "{escaped}"')
    return "\n".join(lines)


def _recover_scalar_fields(yaml_text: str) -> dict[str, Any]:
    """Grok ``recover_scalar_fields`` — listing scalars only (first-wins)."""
    out: dict[str, Any] = {}
    for line in yaml_text.splitlines():
        if not line or line[0] in " \t":
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        if key not in _RECOVERABLE_KEYS or key in out:
            continue
        raw = val.strip()
        if (raw.startswith('"') and raw.endswith('"')) or (
            raw.startswith("'") and raw.endswith("'")
        ):
            raw = raw[1:-1]
        elif " #" in raw:
            raw = raw.split(" #", 1)[0].rstrip()
        # bare block-scalar indicator → skip (body fallback supplies description)
        if not raw:
            continue
        if raw[0] in "|>" and all(c in "+-0123456789" for c in raw[1:]):
            continue
        out[key] = raw
    return out


def _coerce_to_string(value: Any) -> str | None:
    """Grok ``coerce_to_string`` — scalar only; null/blank/non-scalar → None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # YAML may give int/float
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, str):
        t = value.strip()
        return t or None
    return None


def _parse_boolean_frontmatter(value: Any) -> bool:
    """Grok ``parse_boolean_frontmatter`` — only true / \"true\" are true."""
    if value is True:
        return True
    if isinstance(value, str) and value == "true":
        return True
    return False


def _coerce_tool_list(value: Any) -> list[str] | None:
    """Grok ``coerce_tool_list`` — string (top-level split) or YAML sequence."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = _split_top_level(value, "(", ")", split_ws=True)
        return parts or None
    if isinstance(value, list):
        out = [str(x) for x in value if isinstance(x, str) and x]
        return out or None
    return None


def _coerce_path_list(value: Any) -> list[str] | None:
    """Grok ``coerce_path_list``."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = _split_top_level(value, "{", "}", split_ws=False)
        return parts or None
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.extend(_split_top_level(item, "{", "}", split_ws=False))
        return out or None
    return None


def _normalize_skill_paths(patterns: list[str] | None) -> list[str] | None:
    """Grok ``normalize_skill_paths`` — strip ``/**``; all-``**`` → None."""
    if not patterns:
        return None
    cleaned: list[str] = []
    for p in patterns:
        if p.endswith("/**"):
            p = p[:-3]
        if p:
            cleaned.append(p)
    if not cleaned or all(p == "**" for p in cleaned):
        return None
    return cleaned


def _parse_skill_paths(value: Any) -> list[str] | None:
    return _normalize_skill_paths(_coerce_path_list(value))


def _parse_metadata(
    value: Any,
) -> tuple[str | None, str | None, dict[str, str] | None]:
    """Grok ``parse_metadata`` — promote short-description / author."""
    if not isinstance(value, dict):
        return None, None, None
    short_description: str | None = None
    author: str | None = None
    rest: dict[str, str] = {}
    for k, v in value.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        if k == "short-description":
            short_description = v
        elif k == "author":
            author = v
        else:
            rest[k] = v
    return short_description, author, (rest or None)


def _cap_string(s: str, max_len: int) -> str:
    if len(s) > max_len:
        return "".join(list(s)[:max_len])
    return s


def _split_top_level(
    input_s: str, open_c: str, close_c: str, *, split_ws: bool
) -> list[str]:
    """Grok ``split_top_level`` — keep open/close groups intact."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for c in input_s:
        if c == open_c:
            depth += 1
            current.append(c)
        elif c == close_c:
            depth -= 1
            current.append(c)
        elif depth <= 0 and (c == "," or (split_ws and c.isspace())):
            t = "".join(current).strip()
            if t:
                parts.append(t)
            current = []
        else:
            current.append(c)
    t = "".join(current).strip()
    if t:
        parts.append(t)
    return parts


def _first_paragraph(body: str) -> str:
    lines: list[str] = []
    for line in body.splitlines():
        s = line.strip()
        if not s:
            if lines:
                break
            continue
        if s.startswith("#"):
            continue
        lines.append(s)
        if len(" ".join(lines)) > 200:
            break
    return " ".join(lines)[:240]


def discover_skill_dirs(cwd: Path | str | None = None) -> list[tuple[Path, str]]:
    """Return (dir, scope) candidates in Grok priority order (local first)."""
    cwd_p = Path(cwd or Path.cwd()).resolve()
    homes: list[tuple[Path, str]] = []
    for name, scope in (
        (cwd_p / ".codedoggy" / "skills", SCOPE_LOCAL),
        (cwd_p / ".grok" / "skills", SCOPE_LOCAL),
    ):
        homes.append((name, scope))
    root = _find_git_root(cwd_p)
    if root is not None and root != cwd_p:
        homes.append((root / ".codedoggy" / "skills", SCOPE_REPO))
        homes.append((root / ".grok" / "skills", SCOPE_REPO))
    home = Path.home()
    homes.append((home / ".codedoggy" / "skills", SCOPE_USER))
    homes.append((home / ".grok" / "skills", SCOPE_USER))
    return homes


def discover_skills(
    cwd: Path | str | None = None,
    *,
    extra_dirs: Iterable[Path | str] | None = None,
) -> list[SkillInfo]:
    """Scan skill directories; local overrides repo overrides user (by name).

    Priority: extra_dirs / local > repo > user (Grok SkillScope order).
    """
    dirs = list(discover_skill_dirs(cwd))
    if extra_dirs:
        for d in extra_dirs:
            dirs.insert(0, (Path(d), SCOPE_LOCAL))
    # Process low priority first so higher priority overwrites
    found: dict[str, SkillInfo] = {}
    for directory, scope in reversed(dirs):
        if not directory.is_dir():
            continue
        for skill_file in _iter_skill_files(directory):
            info = parse_skill_md(skill_file, scope=scope)
            if info is None:
                continue
            found[info.name.lower()] = info
    return list(found.values())


def _iter_skill_files(directory: Path) -> list[Path]:
    """Grok ``find_skill_md_paths`` over a skills root directory."""
    return find_skill_md_paths(directory)


def walk_for_skill_md(
    directory: Path, paths: list[Path] | None = None, depth: int = 0
) -> list[Path]:
    """Grok ``walk_for_skill_md`` — recursive SKILL.md, max depth 5."""
    out = paths if paths is not None else []
    if depth > MAX_SKILL_WALK_DEPTH:
        return out
    try:
        dirs = sorted(
            [p for p in directory.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )
    except OSError:
        return out
    for path in dirs:
        skill_md = path / "SKILL.md"
        if skill_md.is_file():
            out.append(skill_md)
        walk_for_skill_md(path, out, depth + 1)
    return out


def _find_git_root(start: Path) -> Path | None:
    cur = start
    for _ in range(32):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def resolve_skill(
    name: str,
    skills: list[SkillInfo],
) -> SkillInfo | None:
    """Resolve bare name or scope:name / plugin:name."""
    raw = (name or "").strip()
    if not raw:
        return None
    low = raw.lower()
    # qualified
    if ":" in raw:
        for s in skills:
            if format_skill_name_local(s).lower() == low:
                return s
            if s.name.lower() == low.split(":", 1)[-1] and (
                s.scope.lower() == low.split(":", 1)[0]
                or (s.plugin_name or "").lower() == low.split(":", 1)[0]
            ):
                return s
    for s in skills:
        if s.name.lower() == low:
            return s
    return None


def format_skill_name_local(skill: SkillInfo) -> str:
    if skill.plugin_name:
        return f"{skill.plugin_name}:{skill.name}"
    return f"{skill.scope}:{skill.name}"


def load_skills_from_extra(extra: dict[str, Any] | None, cwd: Path | str | None) -> list[SkillInfo]:
    """Host inject: skills_registry list / skill_paths / filesystem discovery.

    An **explicit** ``skills_registry`` list (including empty) is host-owned and
    skips disk discovery — matches Grok AvailableSkills resource injection.
    """
    bag = extra or {}
    if "skills_registry" in bag and isinstance(bag.get("skills_registry"), list):
        reg = bag["skills_registry"]
        out: list[SkillInfo] = []
        for item in reg:
            if isinstance(item, SkillInfo):
                out.append(item)
            elif isinstance(item, dict):
                body_raw = item.get("body") or item.get("content")
                out.append(
                    SkillInfo(
                        name=str(item.get("name") or ""),
                        description=str(item.get("description") or ""),
                        path=str(item.get("path") or ""),
                        scope=str(item.get("scope") or SCOPE_USER),
                        body=str(body_raw) if body_raw is not None else None,
                        plugin_name=item.get("plugin_name"),
                        enabled=bool(item.get("enabled", True)),
                        user_invocable=bool(item.get("user_invocable", True)),
                        disable_model_invocation=bool(
                            item.get("disable_model_invocation", False)
                        ),
                    )
                )
        return [
            s
            for s in out
            if s.name and s.enabled and not s.disable_model_invocation
        ]
    extra_dirs = bag.get("skill_paths") or bag.get("skills_dirs")
    dirs: list[Path | str] = []
    if isinstance(extra_dirs, (list, tuple)):
        dirs = list(extra_dirs)
    return [
        s
        for s in discover_skills(cwd, extra_dirs=dirs or None)
        if s.enabled and not s.disable_model_invocation
    ]


# Grok discovery.rs APIs
def find_skill_md_paths(directory: Path | str) -> list[Path]:
    """Grok ``find_skill_md_paths`` — root SKILL.md + recursive walk."""
    d = Path(directory)
    paths: list[Path] = []
    self_skill = d / "SKILL.md"
    if self_skill.is_file():
        paths.append(self_skill)
    walk_for_skill_md(d, paths, 0)
    return paths


def find_skill_paths(directory: Path | str) -> list[Path]:
    """Grok ``find_skill_paths`` — ``{dir}/skills/**/SKILL.md``."""
    skills_dir = Path(directory) / "skills"
    if not skills_dir.is_dir():
        return []
    paths: list[Path] = []
    walk_for_skill_md(skills_dir, paths, 0)
    return paths


def parse_skill_files(
    skill_files: list[tuple[Path, str]],
) -> list[SkillInfo]:
    """Grok ``parse_skill_files`` — (path, scope) list + vendor denylist."""
    out: list[SkillInfo] = []
    for path, scope in skill_files:
        info = parse_skill_md(path, scope=scope)
        if info is not None:
            out.append(info)
    return out


def discover_skills_for_paths(
    roots: list[tuple[Path | str, str]],
) -> list[SkillInfo]:
    """Discover skills under explicit (dir, scope) roots (Doggy host helper).

    Note: Grok's ``discover_skills_for_paths`` walks upward from accessed files;
    this variant is the static multi-root scan used by host config paths.
    """
    pairs: list[tuple[Path, str]] = []
    for directory, scope in roots:
        d = Path(directory)
        if not d.is_dir():
            continue
        for p in find_skill_md_paths(d):
            pairs.append((p, scope))
    return parse_skill_files(pairs)
