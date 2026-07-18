"""Main + optional aux model profiles (context fold summarizer, etc.).

Aux is *not* a quality auditor — only a cheap secondary model when configured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from codedoggy.model.registry import create_client, model_config_from_env
from codedoggy.model.types import ModelConfig
from codedoggy.model.provider import ChatClient


@dataclass(slots=True)
class ModelProfiles:
    """Two model configs: coding agent + optional aux (summarizer)."""

    main: ModelConfig
    aux: ModelConfig

    def main_client(self) -> ChatClient:
        return create_client(self.main)

    def aux_client(self) -> ChatClient:
        return create_client(self.aux)


def model_profiles_from_env() -> ModelProfiles:
    """Resolve dual profiles from environment.

    Main:
      CODEDOGGY_PROVIDER / CODEDOGGY_MODEL / CODEDOGGY_BASE_URL / CODEDOGGY_API_KEY

    Aux (falls back to main when unset):
      CODEDOGGY_AUX_PROVIDER / CODEDOGGY_AUX_MODEL /
      CODEDOGGY_AUX_BASE_URL / CODEDOGGY_AUX_API_KEY /
      CODEDOGGY_AUX_TEMPERATURE (default 0.1)
    """
    main = model_config_from_env()

    aux_provider = os.environ.get("CODEDOGGY_AUX_PROVIDER") or main.provider
    aux_model = os.environ.get("CODEDOGGY_AUX_MODEL") or main.model
    aux_base = os.environ.get("CODEDOGGY_AUX_BASE_URL") or main.base_url
    aux_key = os.environ.get("CODEDOGGY_AUX_API_KEY") or main.api_key
    aux_temp = _env_float("CODEDOGGY_AUX_TEMPERATURE", 0.1)
    aux_max = _env_int("CODEDOGGY_AUX_MAX_TOKENS", 800)
    aux_timeout = _env_float("CODEDOGGY_AUX_TIMEOUT_S", main.timeout_s)

    aux = ModelConfig(
        provider=str(aux_provider).strip().lower(),
        model=str(aux_model).strip(),
        base_url=str(aux_base).strip(),
        api_key=aux_key,
        temperature=aux_temp,
        max_tokens=aux_max,
        timeout_s=aux_timeout or 120.0,
        extra_headers=dict(main.extra_headers),
    )
    return ModelProfiles(main=main, aux=aux)


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default
