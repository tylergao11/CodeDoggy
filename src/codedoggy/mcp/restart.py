"""Bounded MCP recovery aligned with Grok ``mcp_restart.rs``."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from codedoggy.mcp.config import McpTransport
from codedoggy.mcp.events import (
    McpClientEvent,
    McpClientEventKind,
    McpServerStatus,
    McpServerStatusPayload,
    McpServerStatusReason,
)

if TYPE_CHECKING:
    from codedoggy.mcp.servers import McpClient, McpState

logger = logging.getLogger(__name__)

# Grok wall-clock targets: t=1s, t=5s, t=21s.
STDIO_BACKOFF_SECONDS = (1.0, 4.0, 16.0)
# First HTTP attempt is immediate, followed by the Grok recovery window.
HTTP_RECOVERY_BACKOFF_SECONDS = (1.0, 4.0, 16.0, 30.0, 30.0, 30.0, 30.0)


class McpRestartController:
    """Deduplicate and fence reconnect work for one Session."""

    def __init__(
        self,
        *,
        session_id: str,
        state: "McpState",
        status_sink: Callable[[McpServerStatusPayload], None],
    ) -> None:
        self.session_id = session_id
        self.state = state
        self.status_sink = status_sink
        self._in_flight: set[str] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    def maybe_schedule(self, event: McpClientEvent) -> bool:
        if self._closed or event.kind not in {
            McpClientEventKind.TRANSPORT_CLOSED,
            McpClientEventKind.HANDSHAKE_FAILED,
        }:
            return False
        name = event.server_name
        config = self.state.get_config(name)
        token = self.state.config_token(name)
        if (
            config is None
            or token is None
            or not config.enabled
            or config.setup_required
            or name in self.state.shutting_down
            or name in self._in_flight
        ):
            return False
        self._in_flight.add(name)
        if config.transport is McpTransport.STDIO:
            coro = self._restart_stdio(name, token)
        else:
            client = self.state.get_client(name)
            if client is None or event.kind is not McpClientEventKind.TRANSPORT_CLOSED:
                self._in_flight.discard(name)
                return False
            coro = self._recover_http(name, config.transport, token, client)
        task = asyncio.create_task(coro, name=f"mcp-restart:{name}")
        self._tasks.add(task)

        def done(completed: asyncio.Task[None]) -> None:
            self._tasks.discard(completed)
            self._in_flight.discard(name)
            if not completed.cancelled():
                exc = completed.exception()
                if exc is not None:
                    logger.warning("MCP recovery task failed for %s: %s", name, exc)

        task.add_done_callback(done)
        return True

    def _still_configured(
        self,
        name: str,
        transport: McpTransport,
        token: tuple[int, tuple[Any, ...]],
    ) -> bool:
        config = self.state.get_config(name)
        return bool(
            not self._closed
            and config is not None
            and config.enabled
            and not config.setup_required
            and config.transport is transport
            and self.state.config_token(name) == token
            and name not in self.state.shutting_down
        )

    def _push(
        self,
        name: str,
        status: McpServerStatus,
        reason: McpServerStatusReason,
        detail: str | None = None,
    ) -> None:
        config = self.state.get_config(name)
        self.status_sink(
            McpServerStatusPayload(
                session_id=self.session_id,
                name=name,
                source=config.source if config is not None else "unknown",
                status=status,
                reason=reason,
                detail=detail,
                tools=self.state.tools_for_server(name),
            )
        )

    async def _attempt_stdio(
        self,
        name: str,
        token: tuple[int, tuple[Any, ...]],
        attempt: int,
        total: int,
    ) -> bool:
        try:
            await self.state.respawn_stdio(name, expected_token=token)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._push(
                name,
                McpServerStatus.UNAVAILABLE,
                McpServerStatusReason.RESTART_FAILED,
                f"attempt {attempt} of {total}: {exc}",
            )
            return False
        self._push(
            name,
            McpServerStatus.READY,
            McpServerStatusReason.RESTART_SUCCEEDED,
        )
        return True

    async def _restart_stdio(
        self, name: str, token: tuple[int, tuple[Any, ...]]
    ) -> None:
        for index, delay in enumerate(STDIO_BACKOFF_SECONDS, 1):
            await asyncio.sleep(delay)
            if not self._still_configured(name, McpTransport.STDIO, token):
                return
            if self.state.client_is_ready(name):
                return
            if await self._attempt_stdio(
                name, token, index, len(STDIO_BACKOFF_SECONDS)
            ):
                return
        self.state.unregister_server_tools(name)
        self._push(
            name,
            McpServerStatus.UNAVAILABLE,
            McpServerStatusReason.RESTART_FAILED,
            f"exhausted after {len(STDIO_BACKOFF_SECONDS)} attempts",
        )

    async def _recover_http(
        self,
        name: str,
        transport: McpTransport,
        token: tuple[int, tuple[Any, ...]],
        client: "McpClient",
    ) -> None:
        delays = (0.0, *HTTP_RECOVERY_BACKOFF_SECONDS)
        for delay in delays:
            if delay:
                await asyncio.sleep(delay)
            if not self._still_configured(name, transport, token):
                return
            if self.state.get_client(name) is not client:
                return
            if self.state.client_is_ready(name, expected=client):
                return
            try:
                await self.state.reset_http_client(
                    name,
                    expected_token=token,
                    expected_client=client,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("HTTP MCP recovery attempt failed for %s: %s", name, exc)
            else:
                return
        # Grok parks an exhausted HTTP client in-place. Its catalog remains
        # registered and a later tool call can still lazily reset/reconnect.
        logger.warning("HTTP MCP recovery exhausted for %s after %s attempts", name, len(delays))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._in_flight.clear()


__all__ = [
    "HTTP_RECOVERY_BACKOFF_SECONDS",
    "McpRestartController",
    "STDIO_BACKOFF_SECONDS",
]
