"""Suggested model ids per provider — product catalog for the connection panel.

Not a remote list API. Profiles supply default_model; this list is the UX
shortlist for real model switching until a live catalog exists.
"""

from __future__ import annotations

from codedoggy.model.profile_registry import get_profile

# Curated shortlists. Order = display order; first is usually the default.
_PROVIDER_MODELS: dict[str, tuple[str, ...]] = {
    "grok": (
        "grok-4.5",
        "grok-4",
        "grok-4-0709",
        "grok-3",
        "grok-3-mini",
        "grok-2",
        "grok-2-vision-1212",
    ),
    "claude": (
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-sonnet-4-0",
        "claude-3-5-haiku-latest",
    ),
    "codex": (
        "gpt-5.6-sol",
        "gpt-5.6",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.1-codex",
        "gpt-5.1",
        "o4-mini",
    ),
    "openai": (
        "gpt-5.6-sol",
        "gpt-5.6",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.4",
        "gpt-4o",
        "gpt-4o-mini",
        "o4-mini",
    ),
    "deepseek": (
        "deepseek-reasoner",
        "deepseek-chat",
    ),
    "ollama": (
        "qwen3:8b",
        "qwen2.5-coder:7b",
        "llama3.2",
        "deepseek-r1:8b",
        "mistral",
    ),
    "custom": (
        "gpt-5.6-sol",
        "gpt-5.6-luna",
        "gpt-4o-mini",
    ),
    "openai_compat": (
        "gpt-5.6-sol",
        "gpt-5.6-luna",
        "gpt-4o-mini",
    ),
    "bedrock": (
        "anthropic.claude-opus-4-5-20251101-v1:0",
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "anthropic.claude-haiku-4-5-20251001-v1:0",
    ),
    "vertex": (
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
    ),
    "codex_app_server": (
        "gpt-5.6-sol",
        "gpt-5.6",
        "gpt-5.6-luna",
        "gpt-5.1-codex",
    ),
}


def suggested_models(provider: str | None) -> list[str]:
    """Return ordered model ids for the connection panel."""
    pid = (provider or "").strip().lower()
    if not pid:
        return []
    prof = get_profile(pid)
    canon = (prof.name if prof is not None else pid).strip().lower()
    raw = list(_PROVIDER_MODELS.get(canon) or _PROVIDER_MODELS.get(pid) or ())
    default = (prof.default_model if prof is not None else "") or ""
    default = default.strip()
    out: list[str] = []
    if default:
        out.append(default)
    for mid in raw:
        if mid and mid not in out:
            out.append(mid)
    return out


def resolve_model_choice(provider: str | None, model: str | None) -> str:
    """Pick model id: explicit → provider default → first suggested → 'model'."""
    if model and str(model).strip():
        return str(model).strip()
    pid = (provider or "").strip().lower()
    prof = get_profile(pid)
    if prof is not None and prof.default_model:
        return prof.default_model.strip()
    suggested = suggested_models(pid)
    if suggested:
        return suggested[0]
    return "model"
