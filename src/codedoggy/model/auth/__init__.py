"""Auth layer — *who you are* (OAuth session vs API key)."""

from codedoggy.model.auth.base import (
    AUTH_API_KEY,
    AUTH_OAUTH,
    AuthCredential,
    AuthKind,
    AuthStatus,
    LoginRequired,
)
from codedoggy.model.auth.registry import register_auth_provider
from codedoggy.model.auth.resolve import (
    IMPERIAL_OAUTH,
    apply_auth_to_config,
    auth_kind_for_provider,
    auth_status,
    begin_login,
    get_auth_provider,
    is_imperial,
    list_auth_providers,
    resolve_credential,
)

__all__ = [
    "AUTH_API_KEY",
    "AUTH_OAUTH",
    "IMPERIAL_OAUTH",
    "AuthCredential",
    "AuthKind",
    "AuthStatus",
    "LoginRequired",
    "apply_auth_to_config",
    "auth_kind_for_provider",
    "auth_status",
    "begin_login",
    "get_auth_provider",
    "is_imperial",
    "list_auth_providers",
    "register_auth_provider",
    "resolve_credential",
]
