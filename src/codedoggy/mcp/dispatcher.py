"""50 ms MCP liveness dispatcher aligned with Grok ``mcp_dispatcher.rs``."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

from codedoggy.mcp.events import (
    McpClientEvent,
    McpClientEventKind,
    McpServerStatusPayload,
    status_for_event,
)

if TYPE_CHECKING:
    from codedoggy.mcp.restart import McpRestartController
    from codedoggy.mcp.servers import McpState

logger = logging.getLogger(__name__)

COALESCE_WINDOW_SECONDS = 0.050
SERVER_STATUS_METHOD = "x.ai/mcp/server_status"
StatusSink = Callable[[McpServerStatusPayload], None]


class McpDispatcher:
    """Coalesce liveness/config events and fan them into state + status.

    The latest event wins within each ``(server, kind)`` key.  Dead-client
    eviction is identity-gated before restart scheduling, preventing a stale
    close from deleting a replacement client.
    """

    def __init__(
        self,
        *,
        session_id: str,
        queue: asyncio.Queue[McpClientEvent | None],
        state: "McpState",
        status_sink: StatusSink,
    ) -> None:
        self.session_id = session_id
        self.queue = queue
        self.state = state
        self.status_sink = status_sink
        self.restart: McpRestartController | None = None
        self._closed = False

    def bind_restart_controller(self, controller: "McpRestartController") -> None:
        self.restart = controller

    async def run(self) -> None:
        while not self._closed:
            first = await self.queue.get()
            if first is None:
                break
            batch: dict[tuple[str, McpClientEventKind], McpClientEvent] = {
                first.coalesce_key: first
            }
            # Grok keeps a second accumulator for every TransportClosed
            # identity observed in the window.  Last-write-wins is correct for
            # status pushes, but not for lifecycle: a late close from an old
            # client must not overwrite the close belonging to the currently
            # registered client.
            closed: dict[str, list[McpClientEvent]] = defaultdict(list)
            if first.kind is McpClientEventKind.TRANSPORT_CLOSED:
                closed[first.server_name].append(first)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + COALESCE_WINDOW_SECONDS
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    event = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                except TimeoutError:
                    break
                if event is None:
                    self._closed = True
                    break
                batch[event.coalesce_key] = event
                if event.kind is McpClientEventKind.TRANSPORT_CLOSED:
                    closed[event.server_name].append(event)

            # Restore a current close if LWW happened to leave a stale close
            # in the wire buffer.  If every observed identity is stale, strip
            # the close completely: it must not evict, publish unavailable, or
            # schedule recovery for a healthy replacement.
            for server_name, events in closed.items():
                current = [event for event in events if self.state.event_is_current(event)]
                key = (server_name, McpClientEventKind.TRANSPORT_CLOSED)
                if current:
                    batch[key] = current[-1]
                else:
                    batch.pop(key, None)
            for event in batch.values():
                try:
                    await self._handle(event)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "MCP dispatcher failed handling %s for %s",
                        event.kind.value,
                        event.server_name,
                    )

    async def _handle(self, event: McpClientEvent) -> None:
        client_event = event.client_id is not None
        if client_event and not self.state.event_is_current(event):
            logger.debug(
                "ignored stale MCP event server=%s client=%s generation=%s",
                event.server_name,
                event.client_id,
                event.connection_generation,
            )
            return

        config = self.state.get_config(event.server_name)
        is_http = config is not None and config.transport.value in {
            "streamable_http",
            "sse",
        }

        if event.kind is McpClientEventKind.TOOLS_LIST_CHANGED:
            await self.state.refresh_server_tools(event.server_name)
        elif event.kind is McpClientEventKind.TRANSPORT_CLOSED and not is_http:
            # Grok evicts dead stdio clients. HTTP/SSE clients remain
            # registered and recover their transport in place so the tool
            # catalog stays usable during a rolling remote restart.
            await self.state.evict_dead_client(event)
            self.state.unregister_server_tools(event.server_name)

        status, reason = status_for_event(event)
        payload = McpServerStatusPayload(
            session_id=self.session_id,
            name=event.server_name,
            source=config.source if config is not None else "unknown",
            status=status,
            reason=reason,
            detail=event.detail,
            tools=self.state.tools_for_server(event.server_name),
        )
        self.status_sink(payload)

        if self.restart is not None and (
            (event.kind is McpClientEventKind.TRANSPORT_CLOSED)
            or (event.kind is McpClientEventKind.HANDSHAKE_FAILED and not is_http)
        ):
            self.restart.maybe_schedule(event)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.queue.put_nowait(None)


__all__ = [
    "COALESCE_WINDOW_SECONDS",
    "McpDispatcher",
    "SERVER_STATUS_METHOD",
]
