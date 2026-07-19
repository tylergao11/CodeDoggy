"""Secret redaction for tool output and history replay (Hermes ``agent/redact``).

Applied when preparing wire messages so tool_use inputs and tool results
do not re-send live credentials to the model (Hermes #19798 / #35519).

Ported patterns from Hermes agent/redact.py (prefixes, env/config, JSON/YAML,
auth headers, private keys, DB URLs, JWTs, bare-token userinfo, form bodies).
"""

from __future__ import annotations

import json
import os
import re
import shlex
from typing import Any

# ---------------------------------------------------------------------------
# Enable / disable (snapshot at import; force=True always redacts)
# ---------------------------------------------------------------------------

_REDACT_ENABLED = os.getenv("CODEDOGGY_REDACT_SECRETS", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# ---------------------------------------------------------------------------
# Vendor prefixes (Hermes _PREFIX_PATTERNS)
# ---------------------------------------------------------------------------

_PREFIX_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{10,}",
    r"ghp_[A-Za-z0-9]{10,}",
    r"github_pat_[A-Za-z0-9_]{10,}",
    r"gho_[A-Za-z0-9]{10,}",
    r"ghu_[A-Za-z0-9]{10,}",
    r"ghs_[A-Za-z0-9]{10,}",
    r"ghr_[A-Za-z0-9]{10,}",
    r"xapp-\d+-[A-Za-z0-9-]{10,}",
    r"xox[baprs]-[A-Za-z0-9-]{10,}",
    r"AIza[A-Za-z0-9_-]{30,}",
    r"pplx-[A-Za-z0-9]{10,}",
    r"fal_[A-Za-z0-9_-]{10,}",
    r"fc-[A-Za-z0-9]{10,}",
    r"bb_live_[A-Za-z0-9_-]{10,}",
    r"gAAAA[A-Za-z0-9_=-]{20,}",
    r"AKIA[A-Z0-9]{16}",
    r"sk_live_[A-Za-z0-9]{10,}",
    r"sk_test_[A-Za-z0-9]{10,}",
    r"rk_live_[A-Za-z0-9]{10,}",
    r"SG\.[A-Za-z0-9_-]{10,}",
    r"hf_[A-Za-z0-9]{10,}",
    r"r8_[A-Za-z0-9]{10,}",
    r"npm_[A-Za-z0-9]{10,}",
    r"pypi-[A-Za-z0-9_-]{10,}",
    r"dop_v1_[A-Za-z0-9]{10,}",
    r"doo_v1_[A-Za-z0-9]{10,}",
    r"am_[A-Za-z0-9_-]{10,}",
    r"sk_[A-Za-z0-9_]{10,}",
    r"tvly-[A-Za-z0-9]{10,}",
    r"exa_[A-Za-z0-9]{10,}",
    r"gsk_[A-Za-z0-9]{10,}",
    r"syt_[A-Za-z0-9]{10,}",
    r"retaindb_[A-Za-z0-9]{10,}",
    r"hsk-[A-Za-z0-9]{10,}",
    r"mem0_[A-Za-z0-9]{10,}",
    r"brv_[A-Za-z0-9]{10,}",
    r"xai-[A-Za-z0-9]{30,}",
    r"ntn_[A-Za-z0-9]{10,}",
    r"fw-[A-Za-z0-9]{30,}",
    r"fw_[A-Za-z0-9]{30,}",
    r"fpk_[A-Za-z0-9]{30,}",
]

_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(" + "|".join(_PREFIX_PATTERNS) + r")(?![A-Za-z0-9_-])"
)


def _extract_literal_prefix(pattern: str) -> str:
    meta = r"[(\\.?*+|{^$"
    for i, ch in enumerate(pattern):
        if ch in meta:
            return pattern[:i]
    return pattern


_PREFIX_SUBSTRINGS = tuple(_extract_literal_prefix(p) for p in _PREFIX_PATTERNS)

_SENSITIVE_QUERY_PARAMS = frozenset(
    {
        "access_token",
        "refresh_token",
        "id_token",
        "token",
        "api_key",
        "apikey",
        "client_secret",
        "password",
        "auth",
        "jwt",
        "session",
        "secret",
        "key",
        "code",
        "signature",
        "x-amz-signature",
    }
)

_SECRET_ENV_NAMES = r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)"
_ENV_ASSIGN_RE = re.compile(
    rf"([A-Z0-9_]{{0,50}}{_SECRET_ENV_NAMES}[A-Z0-9_]{{0,50}})\s*=\s*(['\"]?)(\S+)\2",
)

_SECRET_CFG_NAMES = r"(?:api[ _.\-]?key|token|secret|passwd|password|credential|auth)"
_CFG_VALUE = r"(['\"]?)([^\s&]+?)\2(?=[\s&]|$)"
_ENV_LOOKUP_VALUE_RE = re.compile(
    r"^(?:os\.(?:getenv|environ)|process\.env|\$ENV\{)"
)
_CFG_DOTTED_RE = re.compile(
    rf"((?:[A-Za-z0-9_\-]+\.)+[A-Za-z0-9_.\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_.\-]*"
    rf"|[A-Za-z0-9_.\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_.\-]*\.[A-Za-z0-9_.\-]+)"
    rf"={_CFG_VALUE}",
    re.IGNORECASE,
)
_CFG_ANCHORED_RE = re.compile(
    rf"(^[ \t]*(?:export[ \t]+)?[A-Za-z0-9_\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_\-]*)={_CFG_VALUE}",
    re.IGNORECASE | re.MULTILINE,
)

_YAML_CFG_NAMES = r"(?:api[ _.\-]?key|token|secret|passwd|password|credential)"
_YAML_ASSIGN_RE = re.compile(
    rf"(^[ \t]*[A-Za-z0-9_.\-]*{_YAML_CFG_NAMES}[A-Za-z0-9_.\-]*)(:[ \t]*)(?!['\"])([^\s&]+)",
    re.IGNORECASE | re.MULTILINE,
)

_JSON_KEY_NAMES = (
    r"(?:api_?[Kk]ey|token|secret|password|access_token|refresh_token|"
    r"auth_token|bearer|secret_value|raw_secret|secret_input|key_material)"
)
_JSON_FIELD_RE = re.compile(
    rf'("{_JSON_KEY_NAMES}")\s*:\s*"([^"]+)"',
    re.IGNORECASE,
)

_AUTH_HEADER_RE = re.compile(
    r"((?:Proxy-)?Authorization:\s*)([A-Za-z][\w.+-]*\s+)?([^\s\"']+)",
    re.IGNORECASE,
)

_SECRET_HEADER_NAMES = (
    r"(?:x-api-key|x-goog-api-key|api-key|apikey|x-api-token|x-auth-token|x-access-token)"
)
_SECRET_HEADER_RE = re.compile(
    rf"({_SECRET_HEADER_NAMES}\s*:\s*)(\S+)",
    re.IGNORECASE,
)

_TELEGRAM_RE = re.compile(r"(bot)?(\d{8,}):([-A-Za-z0-9_]{30,})")

_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"
)

_DB_CONNSTR_RE = re.compile(
    r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:\s]+:)([^@\s]+)(@)",
    re.IGNORECASE,
)

_URL_BARE_TOKEN_RE = re.compile(
    r"((?:https?|wss?|git|ssh|ftp|ftps|sftp)://)"
    r"([^\s:@/]{8,})"
    r"(@[^\s]+)",
    re.IGNORECASE,
)

_JWT_RE = re.compile(
    r"eyJ[A-Za-z0-9_-]{10,}" r"(?:\.[A-Za-z0-9_=-]{4,}){0,2}"
)

_SIGNAL_PHONE_RE = re.compile(r"(\+[1-9]\d{6,14})(?![A-Za-z0-9])")

_FORM_BODY_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*(?:&[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*)+$"
)

_ENV_DUMP_COMMANDS = frozenset({"env", "printenv", "set", "export", "declare"})


def mask_secret(
    value: str,
    *,
    head: int = 4,
    tail: int = 4,
    floor: int = 12,
    placeholder: str = "***",
    empty: str = "",
) -> str:
    """Mask a secret for display, preserving head/tail when long enough."""
    if not value:
        return empty
    if len(value) < floor:
        return placeholder
    return f"{value[:head]}...{value[-tail:]}"


def _mask_token(token: str) -> str:
    if not token:
        return "***"
    return mask_secret(token, head=6, tail=4, floor=18)


def _mask_token_nonreusable(token: str) -> str:
    """Non-reusable sentinel so agents cannot write truncated keys back (#35519).

    Fully opaque — no vendor prefix in the marker — so tool output and API
    replay never re-emit ``ghp_`` / ``sk-`` substrings that regexes or
    agents might treat as live credentials.
    """
    return "«redacted-secret»"


def _has_known_prefix_substring(text: str) -> bool:
    return any(p in text for p in _PREFIX_SUBSTRINGS)


def _redact_query_string(query: str) -> str:
    if not query:
        return query
    parts = []
    for pair in query.split("&"):
        if "=" not in pair:
            parts.append(pair)
            continue
        key, _, value = pair.partition("=")
        if key.lower() in _SENSITIVE_QUERY_PARAMS:
            parts.append(f"{key}=***")
        else:
            parts.append(pair)
    return "&".join(parts)


def _redact_form_body(text: str) -> str:
    if not text or "\n" in text or "&" not in text:
        return text
    if not _FORM_BODY_RE.match(text.strip()):
        return text
    return _redact_query_string(text.strip())


def is_env_dump_command(command: str | None) -> bool:
    if not command or not isinstance(command, str):
        return False
    segments = re.split(r"[|;&]+", command)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        try:
            tokens = shlex.split(seg)
        except ValueError:
            tokens = seg.split()
        if tokens and tokens[0] in _ENV_DUMP_COMMANDS:
            return True
    return False


def redact_sensitive_text(
    text: str | None,
    *,
    force: bool = True,
    code_file: bool = False,
    file_read: bool = False,
) -> str | None:
    """Apply Hermes-grade redaction patterns to free text.

    ``force=True`` (default for API/tool paths) always redacts regardless of
    ``CODEDOGGY_REDACT_SECRETS``. Prefix matches use a non-reusable sentinel so
    truncated keys cannot be written back into config (#35519).
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    if not (force or _REDACT_ENABLED):
        return text

    if file_read:
        code_file = True

    # Known vendor prefixes
    if _has_known_prefix_substring(text):
        _prefix_sub = _mask_token_nonreusable if (file_read or force) else _mask_token
        text = _PREFIX_RE.sub(lambda m: _prefix_sub(m.group(1)), text)

    if not code_file:
        if "=" in text:

            def _redact_env(m: re.Match[str]) -> str:
                name, quote, value = m.group(1), m.group(2), m.group(3)
                if _ENV_LOOKUP_VALUE_RE.match(value):
                    return m.group(0)
                masked = (
                    _mask_token_nonreusable(value)
                    if force
                    else _mask_token(value)
                )
                return f"{name}={quote}{masked}{quote}"

            text = _ENV_ASSIGN_RE.sub(_redact_env, text)
            if "://" not in text:
                text = _CFG_DOTTED_RE.sub(_redact_env, text)
                text = _CFG_ANCHORED_RE.sub(_redact_env, text)

        if ":" in text and '"' in text:

            def _redact_json(m: re.Match[str]) -> str:
                key, value = m.group(1), m.group(2)
                if _ENV_LOOKUP_VALUE_RE.match(value):
                    return m.group(0)
                masked = (
                    _mask_token_nonreusable(value)
                    if force
                    else _mask_token(value)
                )
                return f'{key}: "{masked}"'

            text = _JSON_FIELD_RE.sub(_redact_json, text)

        if ":" in text and "://" not in text:

            def _redact_yaml(m: re.Match[str]) -> str:
                key, sep, value = m.group(1), m.group(2), m.group(3)
                if _ENV_LOOKUP_VALUE_RE.match(value):
                    return m.group(0)
                masked = (
                    _mask_token_nonreusable(value)
                    if force
                    else _mask_token(value)
                )
                return f"{key}{sep}{masked}"

            text = _YAML_ASSIGN_RE.sub(_redact_yaml, text)

    if "uthorization" in text or "UTHORIZATION" in text:
        text = _AUTH_HEADER_RE.sub(
            lambda m: m.group(1)
            + (m.group(2) or "")
            + (
                _mask_token_nonreusable(m.group(3))
                if force
                else _mask_token(m.group(3))
            ),
            text,
        )

    if ":" in text:
        text = _SECRET_HEADER_RE.sub(
            lambda m: m.group(1)
            + (
                _mask_token_nonreusable(m.group(2))
                if force
                else _mask_token(m.group(2))
            ),
            text,
        )

        def _redact_telegram(m: re.Match[str]) -> str:
            prefix = m.group(1) or ""
            digits = m.group(2)
            return f"{prefix}{digits}:***"

        text = _TELEGRAM_RE.sub(_redact_telegram, text)

    if "BEGIN" in text and "-----" in text:
        text = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)

    if "://" in text:
        if code_file:

            def _redact_db(m: re.Match[str]) -> str:
                pw = m.group(2)
                if pw.startswith("{") and pw.endswith("}"):
                    return m.group(0)
                return f"{m.group(1)}***{m.group(3)}"

            text = _DB_CONNSTR_RE.sub(_redact_db, text)
        else:
            text = _DB_CONNSTR_RE.sub(
                lambda m: f"{m.group(1)}***{m.group(3)}", text
            )

        text = _URL_BARE_TOKEN_RE.sub(
            lambda m: f"{m.group(1)}"
            + (
                _mask_token_nonreusable(m.group(2))
                if force
                else _mask_token(m.group(2))
            )
            + m.group(3),
            text,
        )

    if "eyJ" in text:
        text = _JWT_RE.sub(
            lambda m: (
                _mask_token_nonreusable(m.group(0))
                if force
                else _mask_token(m.group(0))
            ),
            text,
        )

    # Form bodies only (web URL query pass-through by design — Hermes note)
    if "&" in text and "=" in text:
        text = _redact_form_body(text)

    if "+" in text:

        def _redact_phone(m: re.Match[str]) -> str:
            phone = m.group(1)
            if len(phone) <= 8:
                return phone[:2] + "****" + phone[-2:]
            return phone[:4] + "****" + phone[-4:]

        text = _SIGNAL_PHONE_RE.sub(_redact_phone, text)

    return text


def redact_terminal_output(
    output: str, command: str | None = None, *, force: bool = False
) -> str:
    """Redact secrets from terminal/process stdout (Hermes policy)."""
    if not output:
        return output
    code_file = not is_env_dump_command(command or "")
    return redact_sensitive_text(output, force=force, code_file=code_file) or ""


def redact_tool_arguments(arguments: Any) -> Any:
    """Redact secrets inside tool call arguments (dict or JSON string)."""
    if isinstance(arguments, dict):
        return {k: _redact_value(v) for k, v in arguments.items()}
    if isinstance(arguments, list):
        return [_redact_value(v) for v in arguments]
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return redact_sensitive_text(arguments) or ""
        if isinstance(parsed, (dict, list)):
            return json.dumps(redact_tool_arguments(parsed), ensure_ascii=False)
        return redact_sensitive_text(arguments) or ""
    return arguments


def _redact_value(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _redact_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_redact_value(x) for x in v]
    if isinstance(v, str):
        return redact_sensitive_text(v) or ""
    return v


def redact_messages_for_api(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deep-copy-ish message list with tool args/results redacted for replay."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        m = dict(msg)
        tcs = m.get("tool_calls")
        if isinstance(tcs, list):
            new_tcs = []
            for tc in tcs:
                if not isinstance(tc, dict):
                    new_tcs.append(tc)
                    continue
                tc2 = dict(tc)
                fn = dict(tc2.get("function") or {})
                if "arguments" in fn:
                    fn["arguments"] = redact_tool_arguments(fn["arguments"])
                tc2["function"] = fn
                new_tcs.append(tc2)
            m["tool_calls"] = new_tcs
        if m.get("role") == "tool":
            c = m.get("content")
            if isinstance(c, str):
                m["content"] = redact_sensitive_text(c)
            elif isinstance(c, list):
                m["content"] = [_redact_value(p) for p in c]
        # Free-text content on any role may embed secrets (logs, curl dumps)
        elif isinstance(m.get("content"), str) and m.get("role") in {
            "user",
            "assistant",
            "system",
        }:
            # Skip system by default? Hermes redacts tool paths primarily;
            # still redact assistant/user free text lightly for safety.
            if m.get("role") != "system":
                m["content"] = redact_sensitive_text(m["content"])
        pdata = m.get("provider_data")
        if isinstance(pdata, dict) and "anthropic_content_blocks" in pdata:
            blocks = []
            for b in pdata["anthropic_content_blocks"] or []:
                if not isinstance(b, dict):
                    blocks.append(b)
                    continue
                b2 = dict(b)
                if b2.get("type") == "tool_use" and "input" in b2:
                    b2["input"] = redact_tool_arguments(b2["input"])
                elif b2.get("type") == "tool_result" and "content" in b2:
                    c = b2["content"]
                    if isinstance(c, str):
                        b2["content"] = redact_sensitive_text(c)
                    else:
                        b2["content"] = _redact_value(c)
                blocks.append(b2)
            m["provider_data"] = {**pdata, "anthropic_content_blocks": blocks}
        out.append(m)
    return out
