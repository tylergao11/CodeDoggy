"""Model stack: auth (oauth Grok/Claude/Codex | api_key) × transport (openai | anthropic)."""

from codedoggy.model.anthropic_messages import AnthropicMessagesClient
from codedoggy.model.codex_responses import CodexResponsesClient
from codedoggy.model.prompt_caching import apply_anthropic_cache_control
from codedoggy.model.protocol_context import prepare_wire_messages
from codedoggy.model.provider_switch import (
    reprepare_messages_for_provider,
    rewrite_system_model_identity,
)
from codedoggy.model.profile import API_CODEX_RESPONSES
from codedoggy.model.auth import (
    AUTH_API_KEY,
    AUTH_OAUTH,
    IMPERIAL_OAUTH,
    AuthCredential,
    AuthStatus,
    LoginRequired,
    apply_auth_to_config,
    auth_kind_for_provider,
    auth_status,
    begin_login,
    is_imperial,
    resolve_credential,
)
from codedoggy.model.chat_sampler import ChatSampler
from codedoggy.model.openai_compat import ModelError, OpenAICompatClient
from codedoggy.model.profile import (
    API_ANTHROPIC_MESSAGES,
    API_CHAT_COMPLETIONS,
    AUTH_MODE_API_KEY,
    AUTH_MODE_OAUTH,
    ProviderProfile,
)
from codedoggy.model.profile_registry import get_profile, list_profiles
from codedoggy.model.profiles import ModelProfiles, model_profiles_from_env
from codedoggy.model.provider import ChatClient
from codedoggy.model.registry import (
    create_client,
    list_providers,
    model_config_from_env,
    register_builtin_providers,
    register_provider,
)
from codedoggy.model.types import ChatMessage, CompletionResult, ModelConfig

__all__ = [
    "API_ANTHROPIC_MESSAGES",
    "API_CHAT_COMPLETIONS",
    "API_CODEX_RESPONSES",
    "AUTH_API_KEY",
    "AUTH_MODE_API_KEY",
    "AUTH_MODE_OAUTH",
    "AUTH_OAUTH",
    "AnthropicMessagesClient",
    "AuthCredential",
    "AuthStatus",
    "ChatClient",
    "ChatMessage",
    "ChatSampler",
    "CodexResponsesClient",
    "CompletionResult",
    "IMPERIAL_OAUTH",
    "LoginRequired",
    "ModelConfig",
    "ModelError",
    "ModelProfiles",
    "OpenAICompatClient",
    "ProviderProfile",
    "apply_anthropic_cache_control",
    "apply_auth_to_config",
    "auth_kind_for_provider",
    "auth_status",
    "begin_login",
    "create_client",
    "get_profile",
    "is_imperial",
    "list_profiles",
    "list_providers",
    "model_config_from_env",
    "model_profiles_from_env",
    "prepare_wire_messages",
    "register_builtin_providers",
    "register_provider",
    "reprepare_messages_for_provider",
    "resolve_credential",
    "rewrite_system_model_identity",
]
