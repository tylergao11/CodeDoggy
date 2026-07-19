"""MCP liveness and status types aligned with Grok's dispatcher surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class McpClientEventKind(str, Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    TRANSPORT_CLOSED = "transport_closed"
    HANDSHAKE_FAILED = "handshake_failed"
    TOOLS_LIST_CHANGED = "tools_list_changed"
    RESOURCES_LIST_CHANGED = "resources_list_changed"
    CONFIG_ADDED = "config_added"
    CONFIG_REMOVED = "config_removed"
    CONFIG_CHANGED = "config_changed"


class McpServerStatus(str, Enum):
    READY = "ready"
    INITIALIZING = "initializing"
    UNAVAILABLE = "unavailable"
    NEEDS_AUTH = "needs_auth"


class McpServerStatusReason(str, Enum):
    TRANSPORT_CLOSED = "transport_closed"
    HANDSHAKE_FAILED = "handshake_failed"
    CONFIG_ADDED = "config_added"
    CONFIG_REMOVED = "config_removed"
    CONFIG_CHANGED = "config_changed"
    DISABLED = "disabled"
    AUTH_EXPIRED = "auth_expired"
    INITIALIZED = "initialized"
    RESTART_SUCCEEDED = "restart_succeeded"
    RESTART_FAILED = "restart_failed"
    SETUP_REQUIRED = "setup_required"
    TOOLS_CHANGED = "tools_changed"


@dataclass(slots=True)
class McpClientEvent:
    server_name: str
    kind: McpClientEventKind
    client_id: int | None = None
    connection_generation: int | None = None
    detail: str | None = None
    reason: McpServerStatusReason | None = None
    tools: list[dict[str, Any]] | None = None

    @property
    def coalesce_key(self) -> tuple[str, McpClientEventKind]:
        return (self.server_name, self.kind)


@dataclass(slots=True)
class McpServerStatusPayload:
    session_id: str
    name: str
    source: str
    status: McpServerStatus
    reason: McpServerStatusReason
    detail: str | None = None
    tools: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "source": self.source,
            "status": self.status.value,
            "reason": self.reason.value,
            "detail": self.detail,
            "tools": list(self.tools),
        }


def status_for_event(event: McpClientEvent) -> tuple[McpServerStatus, McpServerStatusReason]:
    reason = event.reason
    if event.kind is McpClientEventKind.READY:
        return McpServerStatus.READY, reason or McpServerStatusReason.INITIALIZED
    if event.kind is McpClientEventKind.INITIALIZING:
        return McpServerStatus.INITIALIZING, reason or McpServerStatusReason.CONFIG_ADDED
    if event.kind is McpClientEventKind.TRANSPORT_CLOSED:
        return McpServerStatus.UNAVAILABLE, reason or McpServerStatusReason.TRANSPORT_CLOSED
    if event.kind is McpClientEventKind.HANDSHAKE_FAILED:
        lowered = (event.detail or "").lower()
        if "401" in lowered or "403" in lowered or "oauth" in lowered or "auth" in lowered:
            return McpServerStatus.NEEDS_AUTH, reason or McpServerStatusReason.AUTH_EXPIRED
        return McpServerStatus.UNAVAILABLE, reason or McpServerStatusReason.HANDSHAKE_FAILED
    if event.kind is McpClientEventKind.CONFIG_ADDED:
        return McpServerStatus.INITIALIZING, reason or McpServerStatusReason.CONFIG_ADDED
    if event.kind is McpClientEventKind.CONFIG_CHANGED:
        return McpServerStatus.INITIALIZING, reason or McpServerStatusReason.CONFIG_CHANGED
    if event.kind is McpClientEventKind.CONFIG_REMOVED:
        return McpServerStatus.UNAVAILABLE, reason or McpServerStatusReason.CONFIG_REMOVED
    return McpServerStatus.READY, reason or McpServerStatusReason.TOOLS_CHANGED


__all__ = [
    "McpClientEvent",
    "McpClientEventKind",
    "McpServerStatus",
    "McpServerStatusPayload",
    "McpServerStatusReason",
    "status_for_event",
]
