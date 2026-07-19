"""Vertex AI OpenAI-compatible endpoint (Hermes vertex_adapter).

Optional: ``google-auth``. Mints a short-lived access token and routes through
:class:`OpenAICompatClient` at the Vertex OpenAI base URL.

Env (optional):
  GOOGLE_APPLICATION_CREDENTIALS / VERTEX_CREDENTIALS_PATH — SA JSON path
  VERTEX_PROJECT_ID — override project
  VERTEX_REGION — default ``global``
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from codedoggy.model.openai_compat import ModelError, OpenAICompatClient
from codedoggy.model.profile import ProviderProfile
from codedoggy.model.profile_registry import get_profile
from codedoggy.model.stream_cancel import run_cancellable_request
from codedoggy.model.types import ChatMessage, CompletionResult, ModelConfig

logger = logging.getLogger(__name__)

DEFAULT_REGION = "global"
_creds_cache: dict[str, Any] = {}


def clear_vertex_credentials_cache() -> None:
    """Drop cached Credentials objects (tests / profile switch)."""
    _creds_cache.clear()


def get_vertex_token(
    credentials_path: str | None = None,
) -> tuple[str | None, str | None]:
    """Return (access_token, project_id) or (None, None)."""
    try:
        import google.auth
        import google.auth.transport.requests
        from google.oauth2 import service_account
    except ImportError:
        logger.warning("google-auth not installed — cannot use Vertex")
        return None, None

    path = credentials_path
    if not path or not os.path.exists(path):
        path = (
            os.environ.get("VERTEX_CREDENTIALS_PATH")
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            or ""
        ).strip() or None
    cache_key = path or "__adc__"

    try:
        cached = _creds_cache.get(cache_key)
        if cached is None:
            if path and os.path.exists(path):
                creds = service_account.Credentials.from_service_account_file(
                    path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                project_id = creds.project_id
            else:
                creds, project_id = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            _creds_cache[cache_key] = (creds, project_id)
        else:
            creds, project_id = cached

        needs_refresh = (
            not getattr(creds, "token", None)
            or getattr(creds, "expired", False)
            or (
                getattr(creds, "expiry", None) is not None
                and (creds.expiry.timestamp() - time.time()) < 300
            )
        )
        if needs_refresh:
            creds.refresh(google.auth.transport.requests.Request())

        override = (os.environ.get("VERTEX_PROJECT_ID") or "").strip()
        if override:
            project_id = override
        return creds.token, project_id
    except Exception as exc:
        logger.error("Vertex credentials failed: %s", exc)
        _creds_cache.pop(cache_key, None)
        # ADC failed — retry once with SA path if env provides one later
        if cache_key == "__adc__":
            sa_path = (
                os.environ.get("VERTEX_CREDENTIALS_PATH")
                or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                or ""
            ).strip()
            if sa_path and os.path.exists(sa_path) and sa_path != path:
                return get_vertex_token(sa_path)
        return None, None


def build_vertex_base_url(project_id: str, region: str = DEFAULT_REGION) -> str:
    """OpenAI-compatible Vertex base URL.

    ``global`` uses bare ``aiplatform.googleapis.com``; regional uses
    ``{region}-aiplatform.googleapis.com``.
    """
    host = (
        "aiplatform.googleapis.com"
        if region == "global"
        else f"{region}-aiplatform.googleapis.com"
    )
    return (
        f"https://{host}/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi"
    )


def resolve_vertex_region(
    explicit: str | None = None,
    config_extra: dict[str, Any] | None = None,
) -> str:
    if explicit:
        return explicit
    if config_extra and config_extra.get("region"):
        return str(config_extra["region"])
    return (os.environ.get("VERTEX_REGION") or DEFAULT_REGION).strip() or DEFAULT_REGION


class VertexClient:
    """Thin wrapper: refresh ADC token → OpenAI-compat complete."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        profile: ProviderProfile | None = None,
    ) -> None:
        self._config = config
        self._profile = profile or get_profile(config.provider)
        self._region = resolve_vertex_region(
            config.extra.get("region") if isinstance(config.extra, dict) else None,
            config.extra if isinstance(config.extra, dict) else None,
        )

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def profile(self) -> ProviderProfile | None:
        return self._profile

    @property
    def region(self) -> str:
        return self._region

    def complete(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        cancel_event: Any | None = None,
    ) -> CompletionResult:
        try:
            # ADC/service-account refresh is network I/O too.  Acquire the
            # short-lived client inside an abandonable owner before handing
            # the actual request to OpenAICompat's own cancellable owner.
            client = run_cancellable_request(self._client, cancel_event)
            return client.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                cancel_event=cancel_event,
            )
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"Vertex complete failed: {exc}") from exc

    def complete_stream(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_delta: Any | None = None,
        cancel_event: Any | None = None,
    ) -> CompletionResult:
        try:
            client = run_cancellable_request(self._client, cancel_event)
            stream_fn = getattr(client, "complete_stream", None)
            if callable(stream_fn):
                return stream_fn(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    on_delta=on_delta,
                    cancel_event=cancel_event,
                )
            return self.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                cancel_event=cancel_event,
            )
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"Vertex stream failed: {exc}") from exc

    def _client(self) -> OpenAICompatClient:
        token, project_id = get_vertex_token()
        if not token or not project_id:
            raise ModelError(
                "Vertex credentials unavailable. Set GOOGLE_APPLICATION_CREDENTIALS "
                "or VERTEX_CREDENTIALS_PATH and install google-auth "
                "(pip install google-auth)."
            )
        base = self._config.base_url
        if not base or "aiplatform.googleapis.com" not in base:
            base = build_vertex_base_url(project_id, self._region)
        cfg = ModelConfig(
            provider="vertex",
            model=self._config.model,
            base_url=base,
            api_key=token,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            timeout_s=self._config.timeout_s,
            context_window=self._config.context_window,
            extra_headers=dict(self._config.extra_headers),
            extra=dict(self._config.extra),
        )
        # Vertex OpenAI endpoint speaks chat.completions
        return OpenAICompatClient(cfg, profile=get_profile("openai_compat") or get_profile("openai"))
