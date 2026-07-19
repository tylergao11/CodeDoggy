"""Built-in ProviderProfile instances.

Layer 1 (auth): oauth (Grok/Claude/Codex) vs api_key others.
Layer 2 (api):  chat_completions (OpenAI 系) vs anthropic_messages.
"""

from __future__ import annotations

from typing import Any

from codedoggy.model.profile import (
    API_ANTHROPIC_MESSAGES,
    API_CHAT_COMPLETIONS,
    API_CODEX_RESPONSES,
    AUTH_MODE_API_KEY,
    AUTH_MODE_OAUTH,
    REASONING_REQUIRE,
    REASONING_STRIP,
    ProviderProfile,
)
from codedoggy.model.profile_registry import register_profile


def _deepseek_thinking_model(model: str | None) -> bool:
    m = (model or "").strip().lower()
    if not m:
        return False
    if m.startswith("deepseek-v") and not m.startswith("deepseek-v3"):
        return True
    if m == "deepseek-reasoner":
        return True
    if "reasoner" in m or "r1" in m:
        return True
    if "deepseek-v4" in m:
        return True
    return False


class DeepSeekProfile(ProviderProfile):
    def reasoning_policy_for_model(self, model: str | None) -> str:
        if _deepseek_thinking_model(model):
            return REASONING_REQUIRE
        return REASONING_STRIP

    def build_api_kwargs_extras(
        self,
        *,
        model: str | None = None,
        reasoning_config: dict[str, Any] | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        if not _deepseek_thinking_model(model):
            return extra_body, top_level
        enabled = True
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            enabled = False
        extra_body["thinking"] = {"type": "enabled" if enabled else "disabled"}
        if not enabled:
            return extra_body, top_level
        if isinstance(reasoning_config, dict):
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort in {"xhigh", "max", "ultra"}:
                top_level["reasoning_effort"] = "max"
            elif effort in {"low", "medium", "high"}:
                top_level["reasoning_effort"] = effort
        return extra_body, top_level


def build_builtin_profiles() -> list[ProviderProfile]:
    return [
        # ── OAuth session providers ─────────────────────────────
        ProviderProfile(
            name="grok",
            aliases=("xai", "x-ai", "x.ai"),
            display_name="Grok (xAI)",
            description="Grok OAuth / API — Hermes xai uses Responses API",
            env_vars=("XAI_API_KEY",),
            base_url="https://api.x.ai/v1",
            base_url_env_var="XAI_BASE_URL",
            default_model="grok-3",
            default_aux_model="grok-3-mini",
            auth_mode=AUTH_MODE_OAUTH,
            # Hermes: transport=codex_responses for xai
            api_mode=API_CODEX_RESPONSES,
            reasoning_policy=REASONING_STRIP,
        ),
        ProviderProfile(
            name="claude",
            aliases=("anthropic",),
            display_name="Claude",
            description="Claude via OAuth (Claude Code login) or ANTHROPIC_API_KEY",
            env_vars=("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
            base_url="https://api.anthropic.com",
            base_url_env_var="ANTHROPIC_BASE_URL",
            default_model="claude-sonnet-4-5",
            default_aux_model="claude-haiku-4-5",
            auth_mode=AUTH_MODE_OAUTH,
            api_mode=API_ANTHROPIC_MESSAGES,  # Anthropic 系
            reasoning_policy=REASONING_STRIP,
            # Hermes system_and_3 — without markers Claude re-bills full prefix
            prompt_cache=True,
            prompt_cache_ttl="5m",
        ),
        ProviderProfile(
            name="codex",
            aliases=("openai-codex",),
            display_name="Codex (OpenAI)",
            description="Codex / ChatGPT OAuth — Responses API (Hermes codex_responses)",
            env_vars=("OPENAI_API_KEY",),
            # ChatGPT backend-api codex path is OAuth-specific; public key
            # users still hit api.openai.com/v1/responses.
            base_url="https://api.openai.com/v1",
            base_url_env_var="OPENAI_BASE_URL",
            default_model="gpt-5.1-codex",
            default_aux_model="gpt-4.1-mini",
            auth_mode=AUTH_MODE_OAUTH,
            api_mode=API_CODEX_RESPONSES,
            reasoning_policy=REASONING_STRIP,
        ),
        # ── API Key / OpenAI 系 ─────────────────────────────────
        ProviderProfile(
            name="openai",
            display_name="OpenAI API",
            description="OpenAI official API key (not Codex OAuth)",
            env_vars=("OPENAI_API_KEY",),
            base_url="https://api.openai.com/v1",
            base_url_env_var="OPENAI_BASE_URL",
            default_model="gpt-4o-mini",
            default_aux_model="gpt-4o-mini",
            auth_mode=AUTH_MODE_API_KEY,
            api_mode=API_CHAT_COMPLETIONS,
            reasoning_policy=REASONING_STRIP,
        ),
        ProviderProfile(
            name="openai_compat",
            aliases=("custom",),
            display_name="OpenAI-compatible",
            description="Generic OpenAI chat/completions endpoint",
            env_vars=("OPENAI_API_KEY", "CODEDOGGY_API_KEY"),
            base_url="https://api.openai.com/v1",
            base_url_env_var="OPENAI_BASE_URL",
            default_model="gpt-4o-mini",
            auth_mode=AUTH_MODE_API_KEY,
            api_mode=API_CHAT_COMPLETIONS,
            reasoning_policy=REASONING_STRIP,
        ),
        DeepSeekProfile(
            name="deepseek",
            aliases=("deepseek-chat",),
            display_name="DeepSeek",
            description="DeepSeek API key (OpenAI-compatible)",
            env_vars=("DEEPSEEK_API_KEY",),
            base_url="https://api.deepseek.com/v1",
            base_url_env_var="DEEPSEEK_BASE_URL",
            default_model="deepseek-chat",
            default_aux_model="deepseek-chat",
            auth_mode=AUTH_MODE_API_KEY,
            api_mode=API_CHAT_COMPLETIONS,
            reasoning_policy=REASONING_STRIP,
        ),
        ProviderProfile(
            name="ollama",
            display_name="Ollama",
            description="Local Ollama OpenAI-compatible server",
            env_vars=("OLLAMA_API_KEY",),
            base_url="http://127.0.0.1:11434/v1",
            base_url_env_var="OLLAMA_HOST",
            default_model="qwen3:8b",
            default_aux_model="qwen3:8b",
            auth_mode=AUTH_MODE_API_KEY,
            api_mode=API_CHAT_COMPLETIONS,
            reasoning_policy=REASONING_STRIP,
        ),
        ProviderProfile(
            name="custom",
            display_name="Custom endpoint",
            description="User-defined OpenAI-compatible base_url + API key",
            env_vars=("CODEDOGGY_API_KEY", "OPENAI_API_KEY"),
            base_url="",
            base_url_env_var="CODEDOGGY_BASE_URL",
            default_model="gpt-4o-mini",
            auth_mode=AUTH_MODE_API_KEY,
            api_mode=API_CHAT_COMPLETIONS,
            reasoning_policy=REASONING_STRIP,
        ),
        # ── Cloud / special runtimes (Hermes) ───────────────────
        ProviderProfile(
            name="bedrock",
            aliases=("aws-bedrock", "amazon-bedrock"),
            display_name="AWS Bedrock",
            description="Bedrock Converse API (boto3 credential chain)",
            env_vars=("AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_BEARER_TOKEN_BEDROCK"),
            base_url="",  # not HTTP OpenAI
            default_model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            auth_mode=AUTH_MODE_API_KEY,
            api_mode="bedrock_converse",
            reasoning_policy=REASONING_STRIP,
        ),
        ProviderProfile(
            name="vertex",
            aliases=("vertex-ai", "google-vertex"),
            display_name="Vertex AI",
            description="Gemini via Vertex OpenAI-compatible endpoint (google-auth)",
            env_vars=("GOOGLE_APPLICATION_CREDENTIALS", "VERTEX_CREDENTIALS_PATH"),
            base_url="",
            default_model="google/gemini-2.5-pro",
            auth_mode=AUTH_MODE_API_KEY,
            api_mode=API_CHAT_COMPLETIONS,  # OpenAI-compat surface after token mint
            reasoning_policy=REASONING_STRIP,
        ),
        ProviderProfile(
            name="codex_app_server",
            aliases=("codex-app-server", "codex-runtime"),
            display_name="Codex App Server",
            description="Local `codex app-server` JSON-RPC (optional; needs codex CLI)",
            env_vars=("OPENAI_API_KEY",),
            base_url="",
            default_model="gpt-5.1-codex",
            auth_mode=AUTH_MODE_OAUTH,
            api_mode="codex_app_server",
            reasoning_policy=REASONING_STRIP,
        ),
    ]


def register_builtin_profiles() -> None:
    for p in build_builtin_profiles():
        register_profile(p, replace=True)
