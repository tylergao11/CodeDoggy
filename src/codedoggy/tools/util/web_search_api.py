"""Responses API web_search HTTP client (tool-layer).

Ported from grok-build:
  implementations/web_search/client.rs  (WebSearchClient)
  implementations/web_search/types.rs   (WebSearchConfig)
  xai-grok-workspace session/tool_config.rs  (default model / env)

Grok calls:
  POST {base_url}/responses
  with tools=[{type: web_search, filters?: {allowed_domains}}], input=query

Config (env, first wins for key):
  CODEDOGGY_WEB_SEARCH_API_KEY | XAI_API_KEY | CODEDOGGY_API_KEY | OPENAI_API_KEY
  CODEDOGGY_WEB_SEARCH_BASE_URL | XAI_BASE_URL  (default https://api.x.ai/v1)
  CODEDOGGY_WEB_SEARCH_MODEL | GROK_WEB_SEARCH_MODEL  (default grok-4.20-multi-agent)
  CODEDOGGY_WEB_SEARCH_ENABLED  (0/false to force disable)

When no API key is configured the client is Disabled → not_supported at tool layer.
Optional CODEDOGGY_WEB_SEARCH_URL: GET custom backend (honest override; not Grok).
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from codedoggy.tools.grok_build.web_search import (
    DEFAULT_BASE_URL,
    DEFAULT_WEB_SEARCH_MODEL,
    REDACTED_API_KEY,
    build_search_request,
    extract_citations,
    extract_output_text,
    format_prompt_output,
)

DEFAULT_TIMEOUT_S = 120.0


class WebSearchNotSupported(Exception):
    """API missing, disabled, or endpoint does not support web search."""

    def __init__(self, message: str, *, code: str = "not_supported") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class WebSearchError(Exception):
    """Execution / HTTP failure from the Responses web_search path."""

    def __init__(self, message: str, *, code: str = "search_failed") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class WebSearchConfig:
    """Mirrors types.rs WebSearchConfig Enabled / Disabled."""

    enabled: bool
    base_url: str
    api_key: str | None
    model: str
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = DEFAULT_TIMEOUT_S
    reason_disabled: str = ""
    # Honest CodeDoggy-only override (not Grok): GET backend returning text/JSON
    custom_search_url: str | None = None

    def is_enabled(self) -> bool:
        return self.enabled

    def redacted(self) -> WebSearchConfig:
        """types.rs WebSearchConfig::redacted — api_key → ***REDACTED***."""
        if not self.enabled:
            return WebSearchConfig(
                enabled=False,
                base_url=self.base_url,
                api_key=None,
                model=self.model,
                extra_headers=dict(self.extra_headers),
                timeout_s=self.timeout_s,
                reason_disabled=self.reason_disabled,
                custom_search_url=self.custom_search_url,
            )
        return WebSearchConfig(
            enabled=True,
            base_url=self.base_url,
            api_key=REDACTED_API_KEY,
            model=self.model,
            extra_headers=dict(self.extra_headers),
            timeout_s=self.timeout_s,
            reason_disabled="",
            custom_search_url=self.custom_search_url,
        )

    @classmethod
    def from_env(cls) -> WebSearchConfig:
        custom = os.environ.get("CODEDOGGY_WEB_SEARCH_URL", "").strip() or None
        flag = os.environ.get("CODEDOGGY_WEB_SEARCH_ENABLED", "1").strip().lower()
        if flag in {"0", "false", "off", "no"}:
            return cls(
                enabled=False,
                base_url=DEFAULT_BASE_URL,
                api_key=None,
                model=DEFAULT_WEB_SEARCH_MODEL,
                reason_disabled="CODEDOGGY_WEB_SEARCH_ENABLED is off",
                custom_search_url=custom,
            )
        key = (
            os.environ.get("CODEDOGGY_WEB_SEARCH_API_KEY")
            or os.environ.get("XAI_API_KEY")
            or os.environ.get("CODEDOGGY_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip() or None
        base = (
            os.environ.get("CODEDOGGY_WEB_SEARCH_BASE_URL")
            or os.environ.get("XAI_BASE_URL")
            or DEFAULT_BASE_URL
        ).strip().rstrip("/")
        model = (
            os.environ.get("CODEDOGGY_WEB_SEARCH_MODEL")
            or os.environ.get("GROK_WEB_SEARCH_MODEL")
            or DEFAULT_WEB_SEARCH_MODEL
        ).strip() or DEFAULT_WEB_SEARCH_MODEL
        timeout = DEFAULT_TIMEOUT_S
        raw_t = os.environ.get("CODEDOGGY_WEB_SEARCH_TIMEOUT_S", "").strip()
        if raw_t:
            try:
                timeout = float(raw_t)
            except ValueError:
                pass

        # Custom URL is a real HTTP backend; allow without Responses key.
        if custom and not key:
            return cls(
                enabled=True,
                base_url=base,
                api_key=None,
                model=model,
                timeout_s=timeout,
                custom_search_url=custom,
            )

        if not key:
            return cls(
                enabled=False,
                base_url=base,
                api_key=None,
                model=model,
                timeout_s=timeout,
                custom_search_url=custom,
                reason_disabled=(
                    "Web search is not supported: no API key configured. "
                    "Set CODEDOGGY_WEB_SEARCH_API_KEY (or XAI_API_KEY / "
                    "CODEDOGGY_API_KEY / OPENAI_API_KEY) for the Responses API "
                    f"web_search tool (default base {DEFAULT_BASE_URL}), "
                    "or set CODEDOGGY_WEB_SEARCH_URL for a custom GET backend."
                ),
            )
        return cls(
            enabled=True,
            base_url=base,
            api_key=key,
            model=model,
            timeout_s=timeout,
            custom_search_url=custom,
        )


@dataclass
class WebSearchResult:
    """Structured result matching Grok WebSearchOutput fields (sans pre_formatted)."""

    query: str
    content: str
    citations: list[str]
    allowed_domains: list[str] | None = None


def search(
    query: str,
    *,
    allowed_domains: list[str] | None = None,
    config: WebSearchConfig | None = None,
) -> WebSearchResult:
    """Perform web search via Responses API (or custom URL).

    Raises WebSearchNotSupported when disabled / no key.
    Raises WebSearchError with Grok-aligned messages on HTTP/parse failures.
    """
    cfg = config or WebSearchConfig.from_env()
    if not cfg.enabled:
        # Grok: Cannot create WebSearchClient from disabled config
        raise WebSearchNotSupported(
            cfg.reason_disabled
            or "Cannot create WebSearchClient from disabled config",
            code="not_supported",
        )

    if cfg.custom_search_url and not cfg.api_key:
        return _search_custom(cfg.custom_search_url, query, allowed_domains)

    if not cfg.api_key:
        raise WebSearchNotSupported(
            "Cannot create WebSearchClient from disabled config",
            code="not_supported",
        )

    # Prefer Responses API when key present (Grok path). Custom URL only if no key.
    return _search_responses(cfg, query, allowed_domains)


def _search_responses(
    cfg: WebSearchConfig,
    query: str,
    allowed_domains: list[str] | None,
) -> WebSearchResult:
    payload = build_search_request(query, cfg.model, allowed_domains)
    url = f"{cfg.base_url.rstrip('/')}/responses"
    try:
        body = _post_json(
            url,
            payload,
            api_key=cfg.api_key or "",
            extra_headers=cfg.extra_headers,
            timeout_s=cfg.timeout_s,
        )
    except WebSearchError:
        raise
    except Exception as e:  # noqa: BLE001
        # Grok: "HTTP request failed: {e}"
        raise WebSearchError(
            f"HTTP request failed: {e}",
            code="search_failed",
        ) from e

    content = extract_output_text(body)
    citations = extract_citations(body)
    return WebSearchResult(
        query=query,
        content=content,
        citations=citations,
        allowed_domains=list(allowed_domains) if allowed_domains is not None else None,
    )


def _search_custom(
    base: str,
    query: str,
    allowed_domains: list[str] | None,
) -> WebSearchResult:
    """Honest non-Grok GET override: CODEDOGGY_WEB_SEARCH_URL?q=…"""
    sep = "&" if "?" in base else "?"
    qs = f"q={urllib.parse.quote(query)}"
    if allowed_domains:
        qs += "&allowed_domains=" + urllib.parse.quote(",".join(allowed_domains))
    url = f"{base}{sep}{qs}"
    try:
        raw = _http_get(url, timeout_s=DEFAULT_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001
        raise WebSearchError(
            f"HTTP request failed: {e}",
            code="search_failed",
        ) from e
    # Treat body as content; no citations from opaque backends.
    content = raw.strip() or "No search results found."
    return WebSearchResult(
        query=query,
        content=content,
        citations=[],
        allowed_domains=list(allowed_domains) if allowed_domains is not None else None,
    )


def format_result(result: WebSearchResult, *, pre_formatted: str | None = None) -> str:
    """Apply Grok prompt format to a structured result."""
    return format_prompt_output(
        result.query,
        result.content,
        pre_formatted=pre_formatted,
    )


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    extra_headers: dict[str, str] | None,
    timeout_s: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "CodeDoggy-web_search/0.1",
        "Accept": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            err_body = "Failed to read error body"
        if not err_body:
            err_body = "Failed to read error body"
        if e.code == 401:
            # Grok exact: Responses API returned 401 Unauthorized: {body}
            raise WebSearchError(
                f"Responses API returned 401 Unauthorized: {err_body}",
                code="auth_failed",
            ) from e
        if e.code in {404, 405, 501}:
            raise WebSearchNotSupported(
                f"Web search is not supported by this API endpoint "
                f"({url}, HTTP {e.code}). {err_body[:400]}".strip(),
                code="not_supported",
            ) from e
        # Grok: Responses API returned {status}: {body}
        raise WebSearchError(
            f"Responses API returned {e.code}: {err_body}",
            code="search_failed",
        ) from e
    except urllib.error.URLError as e:
        raise WebSearchError(
            f"HTTP request failed: {e.reason}",
            code="network_error",
        ) from e
    except TimeoutError as e:
        raise WebSearchError(
            f"HTTP request failed: {e}",
            code="timeout",
        ) from e

    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        # Grok: Failed to parse response: {e}
        raise WebSearchError(
            f"Failed to parse response: {e}",
            code="bad_response",
        ) from e


def _http_get(url: str, *, timeout_s: float) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CodeDoggy-web_search/0.1"},
        method="GET",
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
            return resp.read(1_000_000).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:400]
        except Exception:  # noqa: BLE001
            pass
        raise WebSearchError(
            f"Responses API returned {e.code}: {err_body or e.reason}",
            code="search_failed",
        ) from e
    except urllib.error.URLError as e:
        raise WebSearchError(
            f"HTTP request failed: {e.reason}",
            code="network_error",
        ) from e
