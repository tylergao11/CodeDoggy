"""Active connection — single source of truth for provider/model/auth readiness.

Env / ProviderProfile are *import sources* only. Runtime display, sample path,
and TUI gates read :class:`ActiveConnection` via :class:`ConnectionService`.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any, Literal

from codedoggy.model.auth import auth_kind_for_provider, auth_status, is_imperial
from codedoggy.model.chat_sampler import ChatSampler
from codedoggy.model.context_limits import ensure_model_context_window
from codedoggy.model.profile_registry import get_profile
from codedoggy.model.provider_switch import rewrite_system_model_identity
from codedoggy.model.registry import create_client, model_config_from_env
from codedoggy.model.types import ModelConfig

ConnectionSource = Literal["bootstrap", "panel", "reload", "refresh"]


@dataclass(frozen=True, slots=True)
class ActiveConnection:
    """Immutable snapshot of the live main-model connection.

    Secrets never live here — only readiness and routing identity.
    """

    provider: str
    model: str
    base_url: str
    api_mode: str
    auth_kind: str
    auth_source: str
    logged_in: bool
    auth_detail: str
    temperature: float | None
    max_tokens: int | None
    context_window: int | None
    timeout_s: float
    # From ModelConfig.extra["reasoning"] — product default is high when unset.
    reasoning_enabled: bool
    reasoning_effort: str
    aux_provider: str
    aux_model: str
    aux_base_url: str
    generation: int
    source: str
    last_error: str | None
    updated_at: float

    @property
    def ready_to_sample(self) -> bool:
        """Whether the product should allow starting a turn."""
        if self.provider in {"ollama", "custom"} and not is_imperial(self.provider):
            # Local / custom: ollama always; custom needs a prior successful resolve.
            if self.provider == "ollama":
                return True
            return bool(self.logged_in)
        if not is_imperial(self.provider):
            return bool(self.logged_in)
        return bool(self.logged_in)

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def reasoning_label(self) -> str:
        """Short UI label for reasoning effort (e.g. ``推理:high`` / ``推理:off``)."""
        if not self.reasoning_enabled:
            return "推理:off"
        effort = (self.reasoning_effort or "high").strip().lower() or "high"
        return f"推理:{effort}"

    @property
    def model_mode_caption(self) -> str:
        """Bottom-bar style ``model · …`` left half (mode filled by surface)."""
        return self.model

    def to_model_config(self) -> ModelConfig:
        """Routing config without secrets (auth layer fills keys on create)."""
        reasoning: dict[str, Any]
        if not self.reasoning_enabled:
            reasoning = {"enabled": False}
        else:
            reasoning = {
                "enabled": True,
                "effort": (self.reasoning_effort or "high").strip().lower() or "high",
            }
        return ModelConfig(
            provider=self.provider,
            model=self.model,
            base_url=self.base_url,
            api_key=None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_s=self.timeout_s,
            context_window=self.context_window,
            extra={"reasoning": reasoning},
        )


def reasoning_from_extra(extra: dict[str, Any] | None) -> tuple[bool, str]:
    """Parse ``ModelConfig.extra['reasoning']`` → ``(enabled, effort)``.

    Product default when unset: enabled + ``high`` (matches registry env defaults).
    """
    if not isinstance(extra, dict):
        return True, "high"
    rc = extra.get("reasoning")
    if not isinstance(rc, dict):
        return True, "high"
    effort = str(rc.get("effort") or "").strip().lower()
    if rc.get("enabled") is False or effort == "off":
        return False, "off"
    if not effort:
        effort = "high"
    if effort in {"max", "ultra"}:
        effort = "xhigh"
    elif effort == "minimal":
        effort = "low"
    if effort not in {"low", "medium", "high", "xhigh"}:
        effort = "high"
    return True, effort


def connection_from_config(
    main: ModelConfig,
    *,
    aux: ModelConfig | None = None,
    source: ConnectionSource = "bootstrap",
    generation: int = 0,
    last_error: str | None = None,
) -> ActiveConnection:
    """Build a snapshot from configs + live auth probe (no client swap)."""
    provider = (main.provider or "ollama").strip().lower()
    profile = get_profile(provider)
    api_mode = str(getattr(profile, "api_mode", "") or "chat_completions")
    st = auth_status(provider)
    aux_cfg = aux or main
    reasoning_on, reasoning_effort = reasoning_from_extra(getattr(main, "extra", None))
    return ActiveConnection(
        provider=provider,
        model=str(main.model or "").strip() or (profile.default_model if profile else "model"),
        base_url=str(main.base_url or "").strip(),
        api_mode=api_mode,
        auth_kind=str(auth_kind_for_provider(provider) or st.kind or ""),
        auth_source=str(st.source or ""),
        logged_in=bool(st.logged_in) or provider == "ollama",
        auth_detail=str(st.detail or ""),
        temperature=main.temperature,
        max_tokens=main.max_tokens,
        context_window=main.context_window,
        timeout_s=float(main.timeout_s or 120.0),
        reasoning_enabled=reasoning_on,
        reasoning_effort=reasoning_effort,
        aux_provider=str(aux_cfg.provider or provider).strip().lower(),
        aux_model=str(aux_cfg.model or main.model or "").strip(),
        aux_base_url=str(aux_cfg.base_url or main.base_url or "").strip(),
        generation=generation,
        source=source,
        last_error=last_error,
        updated_at=time.time(),
    )


class ConnectionService:
    """Session-scoped owner of :class:`ActiveConnection` and MAIN client push.

    Write paths:
      - :meth:`bootstrap` / constructor — cold start from env-derived config
      - :meth:`apply` — panel / login reload (provider and/or model)
      - :meth:`refresh_auth` — re-probe credentials without changing model

    Read path: :meth:`snapshot` only.
    """

    def __init__(
        self,
        state: ActiveConnection,
        *,
        client: Any | None = None,
        runner: Any | None = None,
        kernel: Any | None = None,
        context: Any | None = None,
    ) -> None:
        self._lock = RLock()
        self._state = state
        self._client = client
        self._runner = runner
        self._kernel = kernel
        self._context = context

    @classmethod
    def bootstrap(
        cls,
        main: ModelConfig,
        *,
        aux: ModelConfig | None = None,
        client: Any | None = None,
        runner: Any | None = None,
    ) -> ConnectionService:
        state = connection_from_config(main, aux=aux, source="bootstrap", generation=0)
        return cls(state, client=client, runner=runner)

    def bind_runner(self, runner: Any) -> None:
        with self._lock:
            self._runner = runner

    def bind_runtime(
        self,
        *,
        runner: Any | None = None,
        kernel: Any | None = None,
        context: Any | None = None,
    ) -> None:
        """Bind every runtime consumer updated by a connection switch.

        MAIN, context budget, and child-agent sampler creation must observe one
        generation.  Keeping this binding here prevents the TUI from reaching
        into each subsystem independently.
        """
        with self._lock:
            if runner is not None:
                self._runner = runner
            if kernel is not None:
                self._kernel = kernel
            if context is not None:
                self._context = context

    def bind_client(self, client: Any) -> None:
        with self._lock:
            self._client = client

    def snapshot(self) -> ActiveConnection:
        with self._lock:
            return deepcopy(self._state)

    def client(self) -> Any | None:
        with self._lock:
            return self._client

    def new_sampler(self) -> ChatSampler:
        """Return an isolated sampler over the current client generation.

        Parallel child agents must not share ChatSampler's mutable tool-call id
        sequence or retain the bootstrap client after a TUI provider switch.
        """
        with self._lock:
            client = self._client
        if client is None:
            raise RuntimeError("active connection has no model client")
        return ChatSampler(client)

    def refresh_auth(self) -> ActiveConnection:
        """Re-probe auth for the current provider; does not rebuild the client."""
        with self._lock:
            cur = self._state
            st = auth_status(cur.provider)
            logged_in = bool(st.logged_in) or cur.provider == "ollama"
            self._state = replace(
                cur,
                auth_kind=str(auth_kind_for_provider(cur.provider) or st.kind or cur.auth_kind),
                auth_source=str(st.source or ""),
                logged_in=logged_in,
                auth_detail=str(st.detail or ""),
                source="refresh",
                last_error=None if logged_in else (st.detail or cur.last_error),
                updated_at=time.time(),
            )
            return deepcopy(self._state)

    def apply(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        reasoning_enabled: bool | None = None,
        require_auth: bool = True,
        source: ConnectionSource = "panel",
    ) -> ActiveConnection:
        """Resolve config, rebuild MAIN client/sampler, publish new snapshot.

        Raises on hard failure; previous connection remains active.
        """
        with self._lock:
            cur = self._state
            prov = (provider if provider is not None else cur.provider).strip().lower()
            if not prov:
                prov = cur.provider

            if model is not None and str(model).strip():
                mod = str(model).strip()
            elif provider is not None and prov != cur.provider:
                profile = get_profile(prov)
                mod = (
                    (profile.default_model if profile else None)
                    or cur.model
                    or "model"
                )
            else:
                mod = cur.model

            cfg = model_config_from_env(provider=prov, model=mod)
            if prov != cur.provider:
                cfg = _clean_cross_provider_config(cfg, provider=prov)
            # After cross-provider base_url clean, re-derive window once.
            cfg = ensure_model_context_window(cfg)
            # Preserve session sampling knobs when env does not override.
            if cfg.temperature is None:
                cfg = replace_config(cfg, temperature=cur.temperature)
            if cfg.max_tokens is None and cur.max_tokens is not None:
                cfg = replace_config(cfg, max_tokens=cur.max_tokens)
            # Reasoning effort: explicit panel choice wins; else keep connection.
            # Env is only published after create_client succeeds (no failed-swap leak).
            cfg = _apply_reasoning_to_config(
                cfg,
                effort=reasoning_effort,
                enabled=reasoning_enabled,
                fallback_effort=cur.reasoning_effort,
                fallback_enabled=cur.reasoning_enabled,
                publish_env=False,
            )

            try:
                # Ollama is deliberately credential-free.  The TUI hard auth
                # gate must not turn a local model change into "API key needed".
                client = create_client(
                    cfg,
                    require_auth=bool(require_auth and prov != "ollama"),
                )
            except Exception as exc:
                self._state = replace(
                    cur,
                    last_error=str(exc),
                    updated_at=time.time(),
                )
                raise

            _publish_reasoning_env(cfg)
            self._push_runtime(client, cfg)
            st = auth_status(prov)
            profile = get_profile(prov)
            api_mode = str(getattr(profile, "api_mode", "") or cur.api_mode or "chat_completions")
            # AUX is independently configured and the compactor keeps that
            # client.  Changing these labels without rebuilding AUX made the
            # connection snapshot claim a runtime that did not exist.
            aux_provider = cur.aux_provider
            aux_model = cur.aux_model
            aux_base = cur.aux_base_url

            self._client = client
            reasoning_on, reasoning_effort = reasoning_from_extra(getattr(cfg, "extra", None))
            self._state = ActiveConnection(
                provider=prov,
                model=str(cfg.model or mod).strip(),
                base_url=str(cfg.base_url or "").strip(),
                api_mode=api_mode,
                auth_kind=str(auth_kind_for_provider(prov) or st.kind or ""),
                auth_source=str(st.source or ""),
                logged_in=bool(st.logged_in) or prov == "ollama",
                auth_detail=str(st.detail or ""),
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                context_window=cfg.context_window,
                timeout_s=float(cfg.timeout_s or cur.timeout_s or 120.0),
                reasoning_enabled=reasoning_on,
                reasoning_effort=reasoning_effort,
                aux_provider=str(aux_provider).strip().lower(),
                aux_model=str(aux_model or mod).strip(),
                aux_base_url=str(aux_base or cfg.base_url or "").strip(),
                generation=cur.generation + 1,
                source=source,
                last_error=None,
                updated_at=time.time(),
            )
            return deepcopy(self._state)

    def _push_runtime(self, client: Any, cfg: ModelConfig) -> None:
        runner = self._runner
        if runner is not None:
            old = getattr(runner, "sampler", None)
            stream = bool(getattr(old, "stream", False))
            on_delta = getattr(old, "on_delta", None)
            runner.sampler = ChatSampler(client, stream=stream, on_delta=on_delta)
            sp = getattr(runner, "system_prompt", None)
            if isinstance(sp, str) and sp:
                runner.system_prompt = rewrite_system_model_identity(
                    sp, model=cfg.model, provider=cfg.provider
                )

        kernel = self._kernel
        if kernel is not None:
            sp = getattr(kernel, "base_system_prompt", None)
            if isinstance(sp, str) and sp:
                kernel.base_system_prompt = rewrite_system_model_identity(
                    sp, model=cfg.model, provider=cfg.provider
                )
            # Keep tool_extra['connection'] current so image/video/search follow login.
            refresh = getattr(kernel, "refresh_tool_extra", None)
            if callable(refresh):
                try:
                    refresh()
                except Exception:  # noqa: BLE001
                    pass

        context = self._context
        if context is None and runner is not None:
            context = getattr(runner, "context_compactor", None)
        bind = getattr(context, "bind_model_window", None)
        if callable(bind):
            bind(
                context_window=cfg.context_window,
                max_completion_tokens=cfg.max_tokens,
            )
        budget = getattr(context, "budget", None)
        if budget is not None:
            # Usage belongs to the previous provider generation.  Preserve the
            # transcript, but show an unknown live count until the new model's
            # first response supplies real usage.
            budget.last_prompt_tokens = None
            budget.last_completion_tokens = None
            budget.awaiting_usage_ensures = 0


def _normalize_reasoning_choice(
    *,
    effort: str | None,
    enabled: bool | None,
    fallback_effort: str = "high",
    fallback_enabled: bool = True,
) -> tuple[bool, str]:
    """Resolve panel/connection reasoning into ``(enabled, effort)``."""
    if enabled is None and effort is None:
        # Connection snapshot is truth — do not re-open env defaults here.
        on = bool(fallback_enabled)
        eff = (fallback_effort or "high").strip().lower() or "high"
    else:
        on = fallback_enabled if enabled is None else bool(enabled)
        eff = (effort or fallback_effort or "high").strip().lower() or "high"
    if not on or eff == "off":
        return False, "off"
    if eff in {"max", "ultra"}:
        eff = "xhigh"
    elif eff == "minimal":
        eff = "low"
    elif eff not in {"low", "medium", "high", "xhigh"}:
        eff = "high"
    return True, eff


def _apply_reasoning_to_config(
    cfg: ModelConfig,
    *,
    effort: str | None,
    enabled: bool | None,
    fallback_effort: str = "high",
    fallback_enabled: bool = True,
    publish_env: bool = False,
) -> ModelConfig:
    """Merge reasoning into ``ModelConfig.extra`` (optionally publish env)."""
    on, eff = _normalize_reasoning_choice(
        effort=effort,
        enabled=enabled,
        fallback_effort=fallback_effort,
        fallback_enabled=fallback_enabled,
    )
    extra = dict(cfg.extra)
    if on:
        extra["reasoning"] = {"enabled": True, "effort": eff}
    else:
        extra["reasoning"] = {"enabled": False}
    out = replace_config(cfg, extra=extra)
    if publish_env:
        _publish_reasoning_env(out)
    return out


def _publish_reasoning_env(cfg: ModelConfig) -> None:
    """Mirror successful connection reasoning into process env for later reloads."""
    import os

    on, eff = reasoning_from_extra(getattr(cfg, "extra", None))
    if on:
        os.environ["CODEDOGGY_REASONING_ENABLED"] = "1"
        os.environ["CODEDOGGY_REASONING_EFFORT"] = eff
    else:
        os.environ["CODEDOGGY_REASONING_ENABLED"] = "0"
        os.environ.pop("CODEDOGGY_REASONING_EFFORT", None)


def replace_config(cfg: ModelConfig, **kwargs: Any) -> ModelConfig:
    """ModelConfig is slots dataclass — small helper for partial updates."""
    data = {
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "api_key": cfg.api_key,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "timeout_s": cfg.timeout_s,
        "context_window": cfg.context_window,
        "extra_headers": dict(cfg.extra_headers),
        "extra": dict(cfg.extra),
    }
    data.update(kwargs)
    return ModelConfig(**data)


def _clean_cross_provider_config(cfg: ModelConfig, *, provider: str) -> ModelConfig:
    """Drop credentials/endpoints imported from the previous ecosystem.

    ``CODEDOGGY_BASE_URL`` and ``CODEDOGGY_API_KEY`` describe the bootstrap
    connection.  A runtime panel switch must resolve the target profile's own
    endpoint and auth source instead of sending an Anthropic request to xAI (or
    reusing an OpenAI key as a Claude token).
    """
    profile = get_profile(provider)
    canonical = str(getattr(profile, "name", None) or provider).strip().lower()
    if canonical in {"custom", "openai_compat"}:
        return cfg
    base_url = profile.resolve_base_url(None) if profile is not None else cfg.base_url
    extra = dict(cfg.extra)
    extra.pop("auth_kind", None)
    extra.pop("auth_source", None)
    return replace_config(
        cfg,
        base_url=str(base_url or "").strip(),
        api_key=None,
        extra_headers={},
        extra=extra,
    )


def connection_of(session: Any) -> ConnectionService | None:
    """Resolve ConnectionService from a Session (extensions or kernel)."""
    ext = getattr(session, "extensions", None)
    conn = getattr(ext, "connection", None) if ext is not None else None
    if conn is not None:
        return conn
    kernel = getattr(ext, "kernel", None) if ext is not None else None
    return getattr(kernel, "connection", None) if kernel is not None else None
