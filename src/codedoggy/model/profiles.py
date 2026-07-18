"""Main vs auxiliary (audit) model profiles — Hermes-style dual brain.

Main agent and resident auditor can share one Ollama model or use different
ones (e.g. main=qwen3:8b, audit=qwen3:8b same, or a smaller aux later).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from codedoggy.model.registry import create_client, model_config_from_env
from codedoggy.model.types import ModelConfig
from codedoggy.model.provider import ChatClient


@dataclass(slots=True)
class ModelProfiles:
    """Two model configs: coding agent + audit brain."""

    main: ModelConfig
    audit: ModelConfig

    def main_client(self) -> ChatClient:
        return create_client(self.main)

    def audit_client(self) -> ChatClient:
        return create_client(self.audit)


def model_profiles_from_env() -> ModelProfiles:
    """Resolve dual profiles from environment.

    Main:
      CODEDOGGY_PROVIDER / CODEDOGGY_MODEL / CODEDOGGY_BASE_URL / CODEDOGGY_API_KEY

    Audit (falls back to main when unset):
      CODEDOGGY_AUDIT_PROVIDER or CODEDOGGY_AUX_PROVIDER
      CODEDOGGY_AUDIT_MODEL or CODEDOGGY_AUX_MODEL
      CODEDOGGY_AUDIT_BASE_URL or CODEDOGGY_AUX_BASE_URL
      CODEDOGGY_AUDIT_API_KEY or CODEDOGGY_AUX_API_KEY
      CODEDOGGY_AUDIT_TEMPERATURE (default 0.1)
    """
    main = model_config_from_env()

    audit_provider = (
        os.environ.get("CODEDOGGY_AUDIT_PROVIDER")
        or os.environ.get("CODEDOGGY_AUX_PROVIDER")
        or main.provider
    )
    audit_model = (
        os.environ.get("CODEDOGGY_AUDIT_MODEL")
        or os.environ.get("CODEDOGGY_AUX_MODEL")
        or main.model
    )
    audit_base = (
        os.environ.get("CODEDOGGY_AUDIT_BASE_URL")
        or os.environ.get("CODEDOGGY_AUX_BASE_URL")
        or main.base_url
    )
    audit_key = (
        os.environ.get("CODEDOGGY_AUDIT_API_KEY")
        or os.environ.get("CODEDOGGY_AUX_API_KEY")
        or main.api_key
    )
    audit_temp = _env_float(
        "CODEDOGGY_AUDIT_TEMPERATURE",
        _env_float("CODEDOGGY_AUX_TEMPERATURE", 0.1),
    )
    audit_max = _env_int(
        "CODEDOGGY_AUDIT_MAX_TOKENS",
        _env_int("CODEDOGGY_AUX_MAX_TOKENS", 800),
    )
    audit_timeout = _env_float(
        "CODEDOGGY_AUDIT_TIMEOUT_S",
        main.timeout_s,
    )

    audit = ModelConfig(
        provider=str(audit_provider).strip().lower(),
        model=str(audit_model).strip(),
        base_url=str(audit_base).strip(),
        api_key=audit_key,
        temperature=audit_temp,
        max_tokens=audit_max,
        timeout_s=audit_timeout or 120.0,
        extra_headers=dict(main.extra_headers),
    )
    return ModelProfiles(main=main, audit=audit)


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
