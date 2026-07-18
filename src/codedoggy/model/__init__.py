"""Model clients: config, provider registry, Ollama / OpenAI-compat transport."""

from codedoggy.model.chat_sampler import ChatSampler
from codedoggy.model.openai_compat import ModelError, OpenAICompatClient
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
    "ChatClient",
    "ChatMessage",
    "ChatSampler",
    "CompletionResult",
    "ModelConfig",
    "ModelError",
    "ModelProfiles",
    "OpenAICompatClient",
    "create_client",
    "list_providers",
    "model_config_from_env",
    "model_profiles_from_env",
    "register_builtin_providers",
    "register_provider",
]
