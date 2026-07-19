"""Grok-aligned MCP configuration loading.

Source alignment:
  - xai-grok-config-types/src/mcp.rs
  - xai-grok-shell/src/util/config/mcp.rs

The merge order intentionally matches Grok: TOML entries win over Claude,
Cursor, and ``.mcp.json`` imports; nearer project files replace farther ones
as complete server entries rather than deep-merging individual fields.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_STARTUP_TIMEOUT_SECS = 30
DEFAULT_TOOL_TIMEOUT_SECS = 6000
logger = logging.getLogger(__name__)


class McpConfigError(ValueError):
    """A discovered MCP configuration file is malformed."""


class McpTransport(str, Enum):
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"
    SSE = "sse"


@dataclass(slots=True)
class McpOAuthConfig:
    """Resolved BYO OAuth settings passed beside the HTTP transport."""

    client_id: str | None = None
    client_secret: str | None = field(default=None, repr=False)
    scopes: tuple[str, ...] = ()
    callback_port: int | None = None


@dataclass(slots=True)
class McpServerConfig:
    """Resolved equivalent of Grok ``McpServerConfig`` + ACP transport."""

    name: str
    transport: McpTransport
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    startup_timeout_sec: int = DEFAULT_STARTUP_TIMEOUT_SECS
    tool_timeout_sec: int = DEFAULT_TOOL_TIMEOUT_SECS
    tool_timeouts: dict[str, int] = field(default_factory=dict)
    expose_image_base64: bool = False
    oauth: McpOAuthConfig | None = None
    setup_required: bool = False
    disabled_tools: frozenset[str] = frozenset()
    source: str = "unknown"

    def tool_timeout_for(self, tool_name: str) -> int:
        return max(1, int(self.tool_timeouts.get(tool_name, self.tool_timeout_sec)))

    def fingerprint(self) -> tuple[Any, ...]:
        """Transport identity used by Grok-style diff updates.

        ``source`` is diagnostic metadata and does not tear down a healthy
        connection when only provenance changes.
        """

        return (
            self.name,
            self.transport.value,
            self.command,
            self.args,
            tuple(sorted(self.env.items())),
            self.cwd,
            self.url,
            tuple(sorted(self.headers.items())),
            self.enabled,
            self.startup_timeout_sec,
            self.tool_timeout_sec,
            tuple(sorted(self.tool_timeouts.items())),
            self.expose_image_base64,
            (
                (
                    self.oauth.client_id,
                    self.oauth.client_secret,
                    self.oauth.scopes,
                    self.oauth.callback_port,
                )
                if self.oauth
                else None
            ),
            self.setup_required,
            tuple(sorted(self.disabled_tools)),
        )

    def connection_fingerprint(self) -> tuple[Any, ...]:
        """Fields that require replacing the underlying client/transport."""

        return self.fingerprint()[:-1]


@dataclass(slots=True)
class McpConfigSnapshot:
    servers: list[McpServerConfig]
    paths: tuple[Path, ...]


_ENV_PATTERN = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
)


def expand_env_string(value: str) -> str:
    """Expand Grok's ``${VAR}`` and ``${VAR:-default}`` syntax."""

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        current = os.environ.get(name)
        if current not in (None, ""):
            return current
        default = match.group("default")
        return default if default is not None else ""

    return _ENV_PATTERN.sub(replace, value)


def grok_home() -> Path:
    raw = os.environ.get("GROK_HOME", "").strip()
    return Path(raw).expanduser().resolve() if raw else Path.home() / ".grok"


def _git_root(cwd: Path) -> Path:
    current = cwd.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def _project_chain(cwd: Path) -> list[Path]:
    cwd = cwd.resolve()
    root = _git_root(cwd)
    if root == cwd:
        return [cwd]
    relative = cwd.relative_to(root)
    out = [root]
    current = root
    for part in relative.parts:
        current = current / part
        out.append(current)
    return out


def discover_mcp_config_paths(cwd: str | Path) -> tuple[Path, ...]:
    """Return all Grok-compatible paths, including files not created yet."""

    root = Path(cwd).resolve()
    chain = _project_chain(root)
    paths: list[Path] = [
        grok_home() / "config.toml",
        grok_home() / "mcp_preferences.json",
        Path.home() / ".claude.json",
        Path.home() / ".cursor" / "mcp.json",
    ]
    for directory in chain:
        paths.extend(
            (
                directory / ".grok" / "config.toml",
                directory / ".mcp.json",
                directory / ".cursor" / "mcp.json",
            )
        )
    # Stable order and no duplicate watch targets on case-insensitive systems.
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = os.path.normcase(str(path.resolve(strict=False)))
        if key not in seen:
            seen.add(key)
            out.append(path)
    return tuple(out)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise McpConfigError(f"failed to read MCP JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise McpConfigError(f"MCP JSON root must be an object: {path}")
    return value


def _read_toml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise McpConfigError(f"failed to read MCP TOML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise McpConfigError(f"MCP TOML root must be a table: {path}")
    return value


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    try:
        return _read_json(path)
    except McpConfigError as exc:
        logger.warning("ignored malformed MCP JSON: %s", exc)
        return None


def _read_toml_optional(path: Path) -> dict[str, Any] | None:
    try:
        return _read_toml(path)
    except McpConfigError as exc:
        logger.warning("ignored malformed MCP TOML: %s", exc)
        return None


def _server_map(
    root: Mapping[str, Any] | None, *, strict: bool = False
) -> dict[str, dict[str, Any]]:
    if not root:
        return {}
    raw = root.get("mcpServers")
    if raw is None:
        raw = root.get("mcp_servers")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        if strict:
            raise McpConfigError("MCP servers must be an object/table")
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, config in raw.items():
        if isinstance(name, str) and name.strip() and isinstance(config, Mapping):
            out[name.strip()] = dict(config)
        elif strict:
            raise McpConfigError(f"invalid MCP server entry {name!r}")
    return out


def _lookup_project_entry(
    projects: Mapping[str, Any], cwd: Path
) -> Mapping[str, Any] | None:
    wanted = os.path.normcase(str(cwd.resolve()))
    for key, value in projects.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            continue
        try:
            normalized = os.path.normcase(str(Path(key).expanduser().resolve()))
        except OSError:
            normalized = os.path.normcase(key)
        if normalized == wanted:
            return value
    return None


def _load_preferences(*, strict: bool = False) -> dict[str, Any]:
    reader = _read_json if strict else _read_json_optional
    value = reader(grok_home() / "mcp_preferences.json")
    if not value:
        return {}
    servers = value.get("servers")
    if servers is None:
        return {}
    if not isinstance(servers, Mapping):
        if strict:
            raise McpConfigError("MCP preferences servers must be an object")
        return {}
    return dict(servers)


def _render_template(value: str, variables: Mapping[str, str]) -> str:
    pattern = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")

    def replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        if key not in variables:
            raise McpConfigError(f"unresolved MCP setup variable {key!r}")
        return variables[key]

    return pattern.sub(replace, value)


def _resolve_setup(
    name: str, raw: dict[str, Any], preferences: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    setup = raw.get("setup")
    if not isinstance(setup, Mapping):
        return raw, False
    fields = setup.get("fields")
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)) or len(fields) != 1:
        raise McpConfigError(
            f"MCP server {name!r} setup must declare exactly one select field"
        )
    field_spec = fields[0]
    if not isinstance(field_spec, Mapping) or str(field_spec.get("type", "")).lower() != "select":
        raise McpConfigError(f"MCP server {name!r} setup field must be a select")
    field_id = str(field_spec.get("id") or "").strip()
    options = field_spec.get("options")
    if (
        not field_id
        or not isinstance(options, Sequence)
        or isinstance(options, (str, bytes))
        or not options
    ):
        raise McpConfigError(
            f"MCP server {name!r} setup field must have an id and non-empty options"
        )
    allowed: set[str] = set()
    for option in options:
        if not isinstance(option, Mapping) or option.get("value") is None:
            raise McpConfigError(f"MCP server {name!r} has an invalid setup option")
        allowed.add(str(option["value"]))

    pref = preferences.get(name)
    values = pref.get("values") if isinstance(pref, Mapping) else None
    selected = values.get(field_id) if isinstance(values, Mapping) else None
    # The schema's display default is not consent. Grok requires a persisted
    # preference and re-prompts when a former option is no longer valid.
    if selected is None or str(selected) not in allowed:
        return raw, True

    selected_value = str(selected)
    variables: dict[str, str] = {}
    derived = setup.get("variables")
    if not isinstance(derived, Mapping):
        derived = setup.get("values")
    if derived is None:
        derived = {}
    if not isinstance(derived, Mapping):
        raise McpConfigError(f"MCP server {name!r} setup variables must be a table")
    for var_name, spec in derived.items():
        if not isinstance(spec, Mapping):
            raise McpConfigError(
                f"MCP server {name!r} setup variable {var_name!r} is invalid"
            )
        source = str(spec.get("from") or "")
        if source != field_id:
            raise McpConfigError(
                f"MCP server {name!r} setup variable {var_name!r} "
                f"references unknown field {source!r}"
            )
        mapping = spec.get("map")
        if not isinstance(mapping, Mapping):
            raise McpConfigError(
                f"MCP server {name!r} setup variable {var_name!r} needs a map"
            )
        resolved = mapping.get(selected_value)
        if resolved is None:
            return raw, True
        variables[str(var_name)] = str(resolved)

    rendered = dict(raw)
    for key in ("command", "url", "urlTemplate", "url_template", "cwd"):
        if isinstance(rendered.get(key), str):
            rendered[key] = _render_template(rendered[key], variables)
    for key in ("args",):
        if isinstance(rendered.get(key), list):
            rendered[key] = [
                _render_template(item, variables) if isinstance(item, str) else item
                for item in rendered[key]
            ]
    for key in ("env", "headers"):
        if isinstance(rendered.get(key), Mapping):
            rendered[key] = {
                str(k): _render_template(str(v), variables)
                for k, v in rendered[key].items()
            }
    rendered.pop("setup", None)
    return rendered, False


def _int_field(raw: Mapping[str, Any], *names: str, default: int) -> int:
    for name in names:
        value = raw.get(name)
        if value is not None:
            try:
                return max(1, int(value))
            except (TypeError, ValueError) as exc:
                raise McpConfigError(f"{name} must be a positive integer") from exc
    return default


def _parse_oauth_config(raw: Mapping[str, Any]) -> McpOAuthConfig | None:
    block = raw.get("oauth")
    block = block if isinstance(block, Mapping) else {}

    # Grok gives transport-level oauth_* fields precedence over the JSON
    # OAuth block when a BYO client id is present.
    transport_client_id = raw.get("oauth_client_id") or raw.get("oauthClientId")
    if transport_client_id:
        client_id = str(transport_client_id)
        secret_env = raw.get("oauth_client_secret_env_var") or raw.get(
            "oauthClientSecretEnvVar"
        )
        raw_scopes = raw.get("oauth_scopes") or raw.get("oauthScopes")
        callback_port = None
    else:
        client_id_value = block.get("client_id") or block.get("clientId")
        if not client_id_value:
            return None
        client_id = str(client_id_value)
        secret_env = block.get("client_secret_env_var") or block.get(
            "clientSecretEnvVar"
        )
        raw_scopes = block.get("scopes")
        callback_port = block.get("callback_port") or block.get("callbackPort")

    if raw_scopes is None:
        scopes: tuple[str, ...] = ()
    elif isinstance(raw_scopes, Sequence) and not isinstance(raw_scopes, (str, bytes)):
        scopes = tuple(str(scope) for scope in raw_scopes)
    else:
        raise McpConfigError("MCP oauth scopes must be an array")

    resolved_port: int | None = None
    if callback_port is not None:
        try:
            resolved_port = int(callback_port)
        except (TypeError, ValueError) as exc:
            raise McpConfigError("MCP oauth callback_port must be an integer") from exc
        if not 1 <= resolved_port <= 65535:
            raise McpConfigError("MCP oauth callback_port must be between 1 and 65535")

    secret = None
    if isinstance(secret_env, str) and secret_env:
        secret = os.environ.get(secret_env)
    return McpOAuthConfig(
        client_id=client_id,
        client_secret=secret,
        scopes=scopes,
        callback_port=resolved_port,
    )


def parse_mcp_server_config(
    name: str,
    raw_config: Mapping[str, Any],
    *,
    source: str = "unknown",
    preferences: Mapping[str, Any] | None = None,
    disabled_tools: Iterable[str] = (),
) -> McpServerConfig:
    """Parse JSON/TOML server shapes into one resolved transport config."""

    raw, setup_required = _resolve_setup(name, dict(raw_config), preferences or {})
    enabled = bool(raw.get("enabled", True))
    command = raw.get("command")
    url = raw.get("url") or raw.get("urlTemplate") or raw.get("url_template")

    if isinstance(command, str) and command.strip():
        transport = McpTransport.STDIO
        command = expand_env_string(command.strip())
        url = None
    elif isinstance(url, str) and url.strip():
        transport_type = str(raw.get("type") or "").lower()
        expanded_url = expand_env_string(url.strip())
        transport = (
            McpTransport.SSE
            if transport_type == "sse" or expanded_url.rstrip("/").endswith("/sse")
            else McpTransport.STREAMABLE_HTTP
        )
        url = expanded_url
        command = None
    elif setup_required:
        # Preserve a setup-required entry in status snapshots without spawning.
        transport = McpTransport.STREAMABLE_HTTP
        command = None
        url = None
    else:
        raise McpConfigError(
            f"MCP server {name!r} from {source} needs either command or url"
        )

    raw_args = raw.get("args") or []
    if not isinstance(raw_args, Sequence) or isinstance(raw_args, (str, bytes)):
        raise McpConfigError(f"MCP server {name!r} args must be an array")
    args = tuple(expand_env_string(str(item)) for item in raw_args)

    raw_env = raw.get("env") or {}
    if not isinstance(raw_env, Mapping):
        raise McpConfigError(f"MCP server {name!r} env must be an object/table")
    env = {str(k): expand_env_string(str(v)) for k, v in raw_env.items()}

    raw_headers = raw.get("headers") or {}
    if not isinstance(raw_headers, Mapping):
        raise McpConfigError(f"MCP server {name!r} headers must be an object/table")
    headers = {str(k): expand_env_string(str(v)) for k, v in raw_headers.items()}
    bearer_var = raw.get("bearer_token_env_var") or raw.get("bearerTokenEnvVar")
    if isinstance(bearer_var, str) and bearer_var:
        token = os.environ.get(bearer_var)
        if token:
            headers["Authorization"] = f"Bearer {token}"

    raw_timeouts = raw.get("tool_timeouts") or raw.get("toolTimeouts") or {}
    if not isinstance(raw_timeouts, Mapping):
        raise McpConfigError(f"MCP server {name!r} tool_timeouts must be a table")
    tool_timeouts: dict[str, int] = {}
    for tool, timeout in raw_timeouts.items():
        try:
            tool_timeouts[str(tool)] = max(1, int(timeout))
        except (TypeError, ValueError) as exc:
            raise McpConfigError(
                f"MCP server {name!r} timeout for {tool!r} must be an integer"
            ) from exc

    cwd_value = raw.get("cwd")
    oauth = _parse_oauth_config(raw)
    return McpServerConfig(
        name=name,
        transport=transport,
        command=command,
        args=args,
        env=env,
        cwd=expand_env_string(str(cwd_value)) if cwd_value else None,
        url=url if isinstance(url, str) else None,
        headers=headers,
        enabled=enabled,
        startup_timeout_sec=_int_field(
            raw,
            "startup_timeout_sec",
            "startupTimeoutSec",
            default=DEFAULT_STARTUP_TIMEOUT_SECS,
        ),
        tool_timeout_sec=_int_field(
            raw,
            "tool_timeout_sec",
            "toolTimeoutSec",
            default=DEFAULT_TOOL_TIMEOUT_SECS,
        ),
        tool_timeouts=tool_timeouts,
        expose_image_base64=bool(
            raw.get("expose_image_base64", raw.get("exposeImageBase64", False))
        ),
        oauth=oauth,
        setup_required=setup_required,
        disabled_tools=frozenset(str(t) for t in disabled_tools),
        source=source,
    )


def _merge_imports(
    cwd: Path, *, strict: bool = False
) -> tuple[dict[str, tuple[dict[str, Any], str]], set[str], dict[str, set[str]]]:
    """Build the raw server map from low to high priority."""

    merged: dict[str, tuple[dict[str, Any], str]] = {}
    disabled_servers: set[str] = set()
    disabled_tools: dict[str, set[str]] = {}
    chain = _project_chain(cwd)
    read_json = _read_json if strict else _read_json_optional
    read_toml = _read_toml if strict else _read_toml_optional

    # Lowest priority: .mcp.json, repo root -> cwd (closest wins).
    for directory in chain:
        path = directory / ".mcp.json"
        for name, raw in _server_map(read_json(path), strict=strict).items():
            merged[name] = (raw, str(path))

    # Cursor: global first, then project files root -> cwd.
    cursor_paths = [Path.home() / ".cursor" / "mcp.json"]
    cursor_paths.extend(directory / ".cursor" / "mcp.json" for directory in chain)
    for path in cursor_paths:
        for name, raw in _server_map(read_json(path), strict=strict).items():
            merged[name] = (raw, str(path))

    # Claude: user then exact project entry.
    claude_path = Path.home() / ".claude.json"
    claude = read_json(claude_path)
    if claude:
        for name, raw in _server_map(claude, strict=strict).items():
            merged[name] = (raw, str(claude_path))
        projects = claude.get("projects")
        if isinstance(projects, Mapping):
            project = _lookup_project_entry(projects, cwd)
            for name, raw in _server_map(project, strict=strict).items():
                merged[name] = (raw, f"{claude_path}#projects[{cwd}]")

    # Highest priority: global Grok TOML, then project TOML root -> cwd.
    toml_paths = [grok_home() / "config.toml"]
    toml_paths.extend(directory / ".grok" / "config.toml" for directory in chain)
    for path in toml_paths:
        root = read_toml(path)
        if not root:
            continue
        for name, raw in _server_map(root, strict=strict).items():
            merged[name] = (raw, str(path))
        raw_disabled = root.get("disabled_mcp_servers")
        if isinstance(raw_disabled, Sequence) and not isinstance(raw_disabled, (str, bytes)):
            disabled_servers.update(str(name) for name in raw_disabled)
        raw_disabled_tools = root.get("disabled_mcp_tools")
        if isinstance(raw_disabled_tools, Mapping):
            for name, tools in raw_disabled_tools.items():
                if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes)):
                    disabled_tools[str(name)] = {str(tool) for tool in tools}

    return merged, disabled_servers, disabled_tools


def load_mcp_config_snapshot(
    cwd: str | Path, *, strict: bool = False
) -> McpConfigSnapshot:
    """Load the effective MCP config.

    Initial discovery keeps Grok's tolerant import behavior. Hot reload uses
    ``strict=True`` so a partially-written file or invalid high-priority entry
    rejects the whole candidate snapshot and preserves last-known-good state.
    """

    cwd_path = Path(cwd).resolve()
    merged, disabled_servers, disabled_tools = _merge_imports(cwd_path, strict=strict)
    preferences = _load_preferences(strict=strict)
    servers: list[McpServerConfig] = []
    for name, (raw, source) in merged.items():
        if name in disabled_servers:
            raw = dict(raw)
            raw["enabled"] = False
        try:
            servers.append(
                parse_mcp_server_config(
                    name,
                    raw,
                    source=source,
                    preferences=preferences,
                    disabled_tools=disabled_tools.get(name, ()),
                )
            )
        except McpConfigError as exc:
            if strict:
                raise
            # Grok deserializes each TOML entry independently and drops only
            # the invalid server; malformed JSON files are already ignored as
            # a whole by _read_json_optional.
            logger.warning("ignored invalid MCP server %s: %s", name, exc)
    return McpConfigSnapshot(
        servers=servers,
        paths=discover_mcp_config_paths(cwd_path),
    )


def load_mcp_server_configs(cwd: str | Path) -> list[McpServerConfig]:
    return load_mcp_config_snapshot(cwd).servers


def coerce_mcp_server_configs(
    configs: Mapping[str, Mapping[str, Any]] | Sequence[McpServerConfig],
    *,
    source: str = "injected",
) -> list[McpServerConfig]:
    """Normalize explicit configs used by embedding hosts and focused tests."""

    if isinstance(configs, Mapping):
        return [
            parse_mcp_server_config(str(name), raw, source=source)
            for name, raw in configs.items()
        ]
    out = list(configs)
    if not all(isinstance(item, McpServerConfig) for item in out):
        raise TypeError("MCP config sequence must contain McpServerConfig objects")
    return out


__all__ = [
    "DEFAULT_STARTUP_TIMEOUT_SECS",
    "DEFAULT_TOOL_TIMEOUT_SECS",
    "McpConfigError",
    "McpConfigSnapshot",
    "McpOAuthConfig",
    "McpServerConfig",
    "McpTransport",
    "coerce_mcp_server_configs",
    "discover_mcp_config_paths",
    "expand_env_string",
    "grok_home",
    "load_mcp_config_snapshot",
    "load_mcp_server_configs",
    "parse_mcp_server_config",
]
