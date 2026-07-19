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
    def model_mode_caption(self) -> str:
        """Bottom-bar style ``model · …`` left half (mode filled by surface)."""
        return self.model

    def to_model_config(self) -> ModelConfig:
        """Routing config without secrets (auth layer fills keys on create)."""
        return ModelConfig(
            provider=self.provider,
            model=self.model,
            base_url=self.base_url,
            api_key=None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_s=self.timeout_s,
            context_window=self.context_window,
        )


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
    ) -> None:
        self._lock = RLock()
        self._state = state
        self._client = client
        self._runner = runner

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

    def bind_client(self, client: Any) -> None:
        with self._lock:
            self._client = client

    def snapshot(self) -> ActiveConnection:
        with self._lock:
            return deepcopy(self._state)

    def client(self) -> Any | None:
        with self._lock:
            return self._client

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
            # Preserve session sampling knobs when env does not override.
            if cfg.temperature is None:
                cfg = replace_config(cfg, temperature=cur.temperature)
            if cfg.max_tokens is None and cur.max_tokens is not None:
                cfg = replace_config(cfg, max_tokens=cur.max_tokens)
            if cfg.context_window is None and cur.context_window is not None:
                cfg = replace_config(cfg, context_window=cur.context_window)

            try:
                client = create_client(cfg, require_auth=require_auth)
            except Exception as exc:
                self._state = replace(
                    cur,
                    last_error=str(exc),
                    updated_at=time.time(),
                )
                raise

            self._push_runtime(client, cfg)
            st = auth_status(prov)
            profile = get_profile(prov)
            api_mode = str(getattr(profile, "api_mode", "") or cur.api_mode or "chat_completions")
            aux_provider = cur.aux_provider if prov == cur.provider else prov
            aux_model = cur.aux_model
            if prov != cur.provider and profile is not None:
                aux_model = profile.default_aux_model or profile.default_model or mod
            aux_base = cfg.base_url if prov != cur.provider else cur.aux_base_url

            self._client = client
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
        if runner is None:
            return
        old = getattr(runner, "sampler", None)
        stream = bool(getattr(old, "stream", False))
        on_delta = getattr(old, "on_delta", None)
        runner.sampler = ChatSampler(client, stream=stream, on_delta=on_delta)
        sp = getattr(runner, "system_prompt", None)
        if isinstance(sp, str) and sp:
            runner.system_prompt = rewrite_system_model_identity(
                sp, model=cfg.model, provider=cfg.provider
            )


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


def connection_of(session: Any) -> ConnectionService | None:
    """Resolve ConnectionService from a Session (extensions or kernel)."""
    ext = getattr(session, "extensions", None)
    conn = getattr(ext, "connection", None) if ext is not None else None
    if conn is not None:
        return conn
    kernel = getattr(ext, "kernel", None) if ext is not None else None
    return getattr(kernel, "connection", None) if kernel is not None else None
