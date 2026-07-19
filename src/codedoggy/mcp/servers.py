"""Grok-aligned MCP clients and consolidated session state.

Source alignment:
  - xai-grok-mcp/src/servers.rs
  - xai-grok-shell/src/session/mcp_servers.rs

The Python MCP SDK supplies JSON-RPC framing and transport primitives.  This
module owns the Grok semantics around them: one session-owned state object,
single-flight initialization, per-server timeouts, client identity guards,
incremental config diffs, paginated ``tools/list``, and one reconnect retry.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import re
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Iterable, Mapping

from codedoggy.mcp.config import McpServerConfig, McpTransport
from codedoggy.mcp.events import (
    McpClientEvent,
    McpClientEventKind,
    McpServerStatusReason,
)

logger = logging.getLogger(__name__)

MCP_TOOL_NAME_DELIMITER = "__"
_CLIENT_IDS = itertools.count(1)


class McpError(RuntimeError):
    def __init__(self, message: str, *, code: str = "mcp_error") -> None:
        super().__init__(message)
        self.code = code


class ClientStateKind(str, Enum):
    EMPTY = "empty"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


class InitProgressKind(str, Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    FINISHED = "finished"


@dataclass(slots=True)
class McpConfigDiff:
    added: list[str]
    removed: list[str]
    retained: list[str]


def validate_tool_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise McpError("MCP tool name is required", code="mcp_invalid_tool_name")
    if MCP_TOOL_NAME_DELIMITER not in name:
        raise McpError(
            f"MCP tool name {name!r} must be qualified as server__tool",
            code="mcp_invalid_tool_name",
        )
    server, tool = name.split(MCP_TOOL_NAME_DELIMITER, 1)
    if not server or not tool:
        raise McpError(
            f"invalid qualified MCP tool name: {name!r}", code="mcp_invalid_tool_name"
        )


def parse_mcp_tool_name(name: str) -> tuple[str, str] | None:
    if MCP_TOOL_NAME_DELIMITER not in name:
        return None
    server, tool = name.split(MCP_TOOL_NAME_DELIMITER, 1)
    return (server, tool) if server and tool else None


def _safe_log_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "mcp"


def _mcp_log_path(server_name: str) -> Path:
    home = os.environ.get("CODEDOGGY_HOME", "").strip()
    base = Path(home).expanduser() if home else Path.home() / ".codedoggy"
    path = base / "logs" / "mcp" / f"{_safe_log_segment(server_name)}.stderr.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _is_retriable_transport_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    tokens = (
        "connection closed",
        "closedresource",
        "brokenresource",
        "broken pipe",
        "connection reset",
        "connection aborted",
        "transport closed",
        "server disconnected",
        "end of stream",
        "eof",
        "502",
        "503",
        "504",
    )
    return any(token in text for token in tokens)


def _model_dump(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(by_alias=True, mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return value


def _content_to_dict(item: Any, *, expose_image_base64: bool) -> dict[str, Any]:
    raw = _model_dump(item)
    if not isinstance(raw, dict):
        return {"type": "text", "text": str(raw)}
    if raw.get("type") == "image" and not expose_image_base64 and "data" in raw:
        data = raw.pop("data", None)
        if isinstance(data, str):
            return {
                "type": "text",
                "text": (
                    f"[MCP image: {raw.get('mimeType') or raw.get('mime_type') or 'image'}, "
                    f"{len(data)} base64 characters]"
                ),
            }
    return raw


def _call_result_to_envelope(result: Any, *, expose_image_base64: bool) -> dict[str, Any]:
    content = getattr(result, "content", None)
    if content is None and isinstance(result, Mapping):
        content = result.get("content")
    content = content if isinstance(content, list) else []
    is_error = bool(
        getattr(result, "isError", False)
        or getattr(result, "is_error", False)
        or (result.get("isError") if isinstance(result, Mapping) else False)
        or (result.get("is_error") if isinstance(result, Mapping) else False)
    )
    envelope: dict[str, Any] = {
        "content": [
            _content_to_dict(item, expose_image_base64=expose_image_base64)
            for item in content
        ],
        "isError": is_error,
    }
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        envelope["structuredContent"] = _model_dump(structured)
    return envelope


class _LivenessClientSessionMixin:
    """Set an event when the SDK receive loop actually terminates."""

    transport_closed: asyncio.Event

    async def _receive_loop(self) -> None:  # pragma: no cover - exercised end-to-end
        try:
            await super()._receive_loop()  # type: ignore[misc]
        finally:
            self.transport_closed.set()


def _liveness_session_type() -> type[Any]:
    # Kept lazy so importing CodeDoggy without enabling MCP gives a precise
    # runtime error if the optional environment was not installed correctly.
    try:
        from mcp import ClientSession
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise McpError(
            "Python package 'mcp' is required for MCP transports",
            code="mcp_dependency_missing",
        ) from exc
    if not callable(getattr(ClientSession, "_receive_loop", None)):
        raise McpError(
            "installed MCP SDK no longer exposes the receive-loop lifecycle hook",
            code="mcp_sdk_incompatible",
        )

    class LivenessClientSession(_LivenessClientSessionMixin, ClientSession):
        def __init__(self, *args: Any, transport_closed: asyncio.Event, **kwargs: Any) -> None:
            self.transport_closed = transport_closed
            super().__init__(*args, **kwargs)

    return LivenessClientSession


EventSink = Callable[[McpClientEvent], None]


class McpClient:
    """One Grok-style MCP client with a single owner task per connection.

    AnyIO requires the task entering its cancel scope to exit it.  The owner
    task therefore keeps the SDK transport/session contexts alive and receives
    stop requests; callers never close those contexts from a foreign task.
    """

    def __init__(self, config: McpServerConfig, event_sink: EventSink) -> None:
        self.config = config
        self.name = config.name
        self.client_id = next(_CLIENT_IDS)
        self.state = ClientStateKind.EMPTY
        self._event_sink = event_sink
        self._init_lock = asyncio.Lock()
        self._owner_task: asyncio.Task[None] | None = None
        self._ready_future: asyncio.Future[Any] | None = None
        self._stop_event: asyncio.Event | None = None
        self._session: Any = None
        self._connection_generation = 0
        self._intentional_generations: set[int] = set()
        self._oauth_interactive = False
        self._closed = False

    @property
    def connection_generation(self) -> int:
        return self._connection_generation

    def _emit(
        self,
        kind: McpClientEventKind,
        *,
        detail: str | None = None,
        reason: McpServerStatusReason | None = None,
    ) -> None:
        self._event_sink(
            McpClientEvent(
                server_name=self.name,
                kind=kind,
                client_id=self.client_id,
                connection_generation=self._connection_generation,
                detail=detail,
                reason=reason,
            )
        )

    async def _message_handler(self, message: Any) -> None:
        root = getattr(message, "root", message)
        cls = type(root).__name__.lower()
        if "toollistchanged" in cls or "toolslistchanged" in cls:
            self._emit(McpClientEventKind.TOOLS_LIST_CHANGED)
        elif "resourcelistchanged" in cls or "resourceslistchanged" in cls:
            self._emit(McpClientEventKind.RESOURCES_LIST_CHANGED)
        # Requests from MCP servers (sampling/elicitation) are intentionally
        # left to the SDK default handler until CodeDoggy exposes those Grok
        # gateway capabilities.

    async def ensure_initialized(self) -> Any:
        if self._closed:
            raise McpError(f"MCP client {self.name!r} is closed", code="mcp_closed")

        async with self._init_lock:
            if (
                self.state is ClientStateKind.READY
                and self._session is not None
                and self._owner_task is not None
                and not self._owner_task.done()
            ):
                return self._session

            if self._owner_task is None or self._owner_task.done():
                self._connection_generation += 1
                generation = self._connection_generation
                loop = asyncio.get_running_loop()
                self._ready_future = loop.create_future()
                self._stop_event = asyncio.Event()
                self.state = ClientStateKind.INITIALIZING
                self._emit(McpClientEventKind.INITIALIZING)
                self._owner_task = asyncio.create_task(
                    self._connection_owner(
                        generation,
                        self._ready_future,
                        self._stop_event,
                    ),
                    name=f"mcp-owner:{self.name}:{generation}",
                )
            ready = self._ready_future

        if ready is None:  # defensive; creation above always assigns it
            raise McpError(f"MCP client {self.name!r} has no init future")
        timeout = self.config.startup_timeout_sec + 1
        try:
            return await asyncio.wait_for(asyncio.shield(ready), timeout=timeout)
        except TimeoutError as exc:
            await self.reset_transport(intentional=False)
            raise McpError(
                f"MCP server {self.name!r} initialization timed out after "
                f"{self.config.startup_timeout_sec}s",
                code="mcp_startup_timeout",
            ) from exc
        except McpError:
            raise
        except Exception as exc:
            raise McpError(
                f"MCP server {self.name!r} initialization failed: {exc}",
                code="mcp_handshake_failed",
            ) from exc

    async def _connection_owner(
        self,
        generation: int,
        ready: asyncio.Future[Any],
        stop_event: asyncio.Event,
    ) -> None:
        transport_closed = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                read_stream: Any
                write_stream: Any
                if self.config.transport is McpTransport.STDIO:
                    from mcp import StdioServerParameters
                    from mcp.client.stdio import stdio_client

                    if not self.config.command:
                        raise McpError(f"stdio MCP server {self.name!r} has no command")
                    child_env = os.environ.copy()
                    child_env.update(self.config.env)
                    params = StdioServerParameters(
                        command=self.config.command,
                        args=list(self.config.args),
                        env=child_env,
                        cwd=self.config.cwd,
                    )
                    try:
                        errlog = _mcp_log_path(self.name).open(
                            "a", encoding="utf-8", errors="replace"
                        )
                    except OSError:
                        logger.warning(
                            "unable to open MCP stderr log for %s; using stderr",
                            self.name,
                            exc_info=True,
                        )
                        errlog = sys.stderr
                    else:
                        stack.callback(errlog.close)
                    streams = await stack.enter_async_context(
                        stdio_client(params, errlog=errlog)
                    )
                    read_stream, write_stream = streams[0], streams[1]
                elif self.config.transport is McpTransport.SSE:
                    from mcp.client.sse import sse_client
                    from codedoggy.mcp.oauth import build_oauth_httpx_auth

                    if not self.config.url:
                        raise McpError(f"SSE MCP server {self.name!r} has no URL")
                    auth = await build_oauth_httpx_auth(
                        self.config,
                        interactive=self._oauth_interactive,
                        stack=stack,
                    )
                    streams = await stack.enter_async_context(
                        sse_client(
                            self.config.url,
                            headers=dict(self.config.headers),
                            timeout=float(self.config.startup_timeout_sec),
                            auth=auth,
                        )
                    )
                    read_stream, write_stream = streams[0], streams[1]
                else:
                    import httpx
                    from mcp.client.streamable_http import streamable_http_client
                    from codedoggy.mcp.oauth import build_oauth_httpx_auth

                    if not self.config.url:
                        raise McpError(f"HTTP MCP server {self.name!r} has no URL")
                    timeout = httpx.Timeout(
                        connect=float(self.config.startup_timeout_sec),
                        read=float(
                            max(self.config.startup_timeout_sec, self.config.tool_timeout_sec)
                        ),
                        write=float(self.config.tool_timeout_sec),
                        pool=float(self.config.startup_timeout_sec),
                    )
                    auth = await build_oauth_httpx_auth(
                        self.config,
                        interactive=self._oauth_interactive,
                        stack=stack,
                    )
                    http_client = await stack.enter_async_context(
                        httpx.AsyncClient(
                            headers=dict(self.config.headers),
                            timeout=timeout,
                            follow_redirects=True,
                            auth=auth,
                        )
                    )
                    streams = await stack.enter_async_context(
                        streamable_http_client(
                            self.config.url,
                            http_client=http_client,
                        )
                    )
                    read_stream, write_stream = streams[0], streams[1]

                session_cls = _liveness_session_type()
                session = session_cls(
                    read_stream,
                    write_stream,
                    message_handler=self._message_handler,
                    transport_closed=transport_closed,
                )
                session = await stack.enter_async_context(session)
                await asyncio.wait_for(
                    session.initialize(), timeout=self.config.startup_timeout_sec
                )
                if generation != self._connection_generation or self._closed:
                    raise asyncio.CancelledError
                self._session = session
                self.state = ClientStateKind.READY
                if not ready.done():
                    ready.set_result(session)
                self._emit(McpClientEventKind.READY)

                stop_waiter = asyncio.create_task(stop_event.wait())
                closed_waiter = asyncio.create_task(transport_closed.wait())
                try:
                    done, pending = await asyncio.wait(
                        {stop_waiter, closed_waiter},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    if (
                        closed_waiter in done
                        and generation not in self._intentional_generations
                        and not self._closed
                    ):
                        self.state = ClientStateKind.FAILED
                        self._emit(
                            McpClientEventKind.TRANSPORT_CLOSED,
                            detail="MCP transport receive loop closed",
                        )
                finally:
                    for task in (stop_waiter, closed_waiter):
                        if not task.done():
                            task.cancel()
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            raise
        except Exception as exc:  # noqa: BLE001
            self.state = ClientStateKind.FAILED
            error = (
                exc
                if isinstance(exc, McpError)
                else McpError(
                    f"MCP server {self.name!r} handshake failed: {exc}",
                    code="mcp_handshake_failed",
                )
            )
            if not ready.done():
                ready.set_exception(error)
            if generation not in self._intentional_generations and not self._closed:
                self._emit(McpClientEventKind.HANDSHAKE_FAILED, detail=str(error))
        finally:
            self._intentional_generations.discard(generation)
            if generation == self._connection_generation:
                self._session = None
                if self._closed:
                    self.state = ClientStateKind.CLOSED
                elif self.state is ClientStateKind.READY:
                    self.state = ClientStateKind.EMPTY

    async def reset_transport(self, *, intentional: bool = True) -> None:
        owner = self._owner_task
        generation = self._connection_generation
        if owner is None:
            if not self._closed:
                self.state = ClientStateKind.EMPTY
            return
        if intentional:
            self._intentional_generations.add(generation)
        stop = self._stop_event
        if stop is not None:
            stop.set()
        if owner is asyncio.current_task():
            return
        try:
            await asyncio.wait_for(asyncio.shield(owner), timeout=5.0)
        except TimeoutError:
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
        except Exception:  # noqa: BLE001
            await asyncio.gather(owner, return_exceptions=True)
        if generation == self._connection_generation:
            self._owner_task = None
            self._ready_future = None
            self._stop_event = None
            self._session = None
            if not self._closed:
                self.state = ClientStateKind.EMPTY

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.reset_transport(intentional=True)
        self.state = ClientStateKind.CLOSED

    async def list_tools(self) -> list[Any]:
        session = await self.ensure_initialized()
        tools: list[Any] = []
        cursor: str | None = None
        while True:
            try:
                result = await asyncio.wait_for(
                    session.list_tools(cursor), timeout=self.config.startup_timeout_sec
                )
            except Exception as exc:  # noqa: BLE001
                raise McpError(
                    f"MCP server {self.name!r} tools/list failed: {exc}",
                    code="mcp_list_tools_failed",
                ) from exc
            tools.extend(list(getattr(result, "tools", None) or []))
            cursor = getattr(result, "nextCursor", None) or getattr(
                result, "next_cursor", None
            )
            if not cursor:
                return tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        timeout = self.config.tool_timeout_for(tool_name)
        session = await self.ensure_initialized()
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments), timeout=timeout
            )
        except TimeoutError as exc:
            # Grok resets a wedged transport on the outer timeout and surfaces
            # the timeout without replaying a possibly mutating call.
            if self.config.transport is not McpTransport.STDIO:
                await self.reset_transport(intentional=True)
            raise McpError(
                f"MCP tool {self.name}{MCP_TOOL_NAME_DELIMITER}{tool_name} "
                f"timed out after {timeout}s",
                code="mcp_tool_timeout",
            ) from exc
        except Exception as first:  # noqa: BLE001
            if (
                self.config.transport is McpTransport.STDIO
                or not _is_retriable_transport_error(first)
            ):
                raise McpError(
                    f"MCP tool {self.name}{MCP_TOOL_NAME_DELIMITER}{tool_name} failed: {first}",
                    code="mcp_tool_failed",
                ) from first
            # Same one reconnect + one retry policy as Grok's try_call_tool.
            await self.reset_transport(intentional=True)
            session = await self.ensure_initialized()
            try:
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments), timeout=timeout
                )
            except Exception as second:  # noqa: BLE001
                raise McpError(
                    f"MCP tool {self.name}{MCP_TOOL_NAME_DELIMITER}{tool_name} "
                    f"failed after reconnect: {second}",
                    code="mcp_tool_failed",
                ) from second
        return _call_result_to_envelope(
            result, expose_image_base64=self.config.expose_image_base64
        )

    async def read_resource(self, uri: str) -> Any:
        session = await self.ensure_initialized()
        result = await asyncio.wait_for(
            session.read_resource(uri), timeout=self.config.tool_timeout_sec
        )
        return _model_dump(result)


SnapshotSink = Callable[[list[dict[str, Any]], list[dict[str, Any]], bool], None]


class McpState:
    """Consolidated MCP state behind the runtime event loop."""

    def __init__(
        self,
        event_sink: EventSink,
        snapshot_sink: SnapshotSink,
    ) -> None:
        self.configs: dict[str, McpServerConfig] = {}
        self.owned_clients: dict[str, McpClient] = {}
        self.shared_clients: dict[str, McpClient] = {}
        self.generation = 0
        # Per-name lifecycle epochs fence restart tasks without penalizing an
        # unchanged server when an unrelated server is reconfigured.
        self.config_versions: dict[str, int] = {}
        self.mcp_tool_meta: dict[str, Any] = {}
        self.auth_required: set[str] = set()
        self.init_failed: dict[str, str] = {}
        self.disabled_tool_registrations: dict[str, dict[str, Any]] = {}
        self.init_progress = InitProgressKind.NOT_STARTED
        self.handshaking: set[str] = set()
        self.shutting_down: set[str] = set()
        self._catalog_by_server: dict[str, list[dict[str, Any]]] = {}
        self._event_sink = event_sink
        self._snapshot_sink = snapshot_sink
        self._initialized_event = asyncio.Event()
        self._background: set[asyncio.Task[Any]] = set()
        self._apply_lock = asyncio.Lock()
        self._closed = False

    @property
    def is_initialized(self) -> bool:
        return self.init_progress is InitProgressKind.FINISHED and not self.handshaking

    @property
    def is_initializing(self) -> bool:
        return self.init_progress is InitProgressKind.INITIALIZING or bool(self.handshaking)

    def _spawn(
        self, awaitable: Coroutine[Any, Any, Any], *, name: str
    ) -> asyncio.Task[Any]:
        task = asyncio.create_task(awaitable, name=name)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    def _emit_config(
        self,
        name: str,
        kind: McpClientEventKind,
        *,
        detail: str | None = None,
        reason: McpServerStatusReason | None = None,
    ) -> None:
        self._event_sink(
            McpClientEvent(
                server_name=name,
                kind=kind,
                detail=detail,
                reason=reason,
            )
        )

    def _server_snapshot(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, config in self.configs.items():
            tools = self._catalog_by_server.get(name, [])
            client = self.owned_clients.get(name) or self.shared_clients.get(name)
            out.append(
                {
                    "name": name,
                    "source": config.source,
                    "transport": config.transport.value,
                    "enabled": config.enabled,
                    "setup_required": config.setup_required,
                    "status": (
                        "needs_auth"
                        if name in self.auth_required
                        else (
                            "unavailable"
                            if name in self.init_failed
                            else (
                                "initializing"
                                if name in self.handshaking
                                else (
                                    client.state.value
                                    if client is not None
                                    else "unavailable"
                                )
                            )
                        )
                    ),
                    "tool_count": len(tools),
                    "tool_names": [str(tool.get("name") or "") for tool in tools],
                }
            )
        return out

    def _publish_snapshot(self) -> None:
        catalog: list[dict[str, Any]] = []
        for name in self.configs:
            catalog.extend(dict(tool) for tool in self._catalog_by_server.get(name, []))
        self._snapshot_sink(catalog, self._server_snapshot(), self.is_initialized)
        if self.is_initialized:
            self._initialized_event.set()
        else:
            self._initialized_event.clear()

    async def wait_initialized(self, timeout: float | None = None) -> bool:
        if self.is_initialized:
            return True
        try:
            await asyncio.wait_for(self._initialized_event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    def get_config(self, name: str) -> McpServerConfig | None:
        return self.configs.get(name)

    def get_client(self, name: str) -> McpClient | None:
        return self.owned_clients.get(name) or self.shared_clients.get(name)

    def config_token(self, name: str) -> tuple[int, tuple[Any, ...]] | None:
        config = self.configs.get(name)
        if config is None:
            return None
        return (self.config_versions.get(name, 0), config.connection_fingerprint())

    def client_is_ready(self, name: str, *, expected: McpClient | None = None) -> bool:
        client = self.get_client(name)
        return bool(
            client is not None
            and (expected is None or client is expected)
            and client.state is ClientStateKind.READY
            and client._owner_task is not None
            and not client._owner_task.done()
        )

    def event_is_current(self, event: McpClientEvent) -> bool:
        if event.client_id is None:
            return True
        client = self.get_client(event.server_name)
        if client is None or client.client_id != event.client_id:
            return False
        return (
            event.connection_generation is None
            or client.connection_generation == event.connection_generation
        )

    async def evict_dead_client(self, event: McpClientEvent) -> bool:
        if not self.event_is_current(event):
            return False
        client = self.owned_clients.pop(event.server_name, None)
        if client is None:
            return False
        await client.close()
        return True

    async def apply_configs(self, new_configs: Iterable[McpServerConfig]) -> McpConfigDiff | None:
        """Apply one non-interleavable config transaction.

        Client teardown contains awaits, so event-loop affinity alone is not
        enough to make reloads atomic.  The lock spans the whole diff and all
        ownership transfers; callers may time out waiting, but the runtime
        deliberately does not cancel a transaction once submitted.
        """

        async with self._apply_lock:
            return await self._apply_configs_locked(new_configs)

    def _apply_disabled_tools_local(self, name: str, disabled: frozenset[str]) -> None:
        prefix = f"{name}{MCP_TOOL_NAME_DELIMITER}"
        visible: list[dict[str, Any]] = []
        for entry in self._catalog_by_server.get(name, []):
            bare = str(entry.get("tool_name") or "")
            qualified = str(entry.get("name") or f"{prefix}{bare}")
            if bare in disabled:
                self.disabled_tool_registrations[qualified] = entry
            else:
                visible.append(entry)

        for qualified, entry in list(self.disabled_tool_registrations.items()):
            if not qualified.startswith(prefix):
                continue
            bare = str(entry.get("tool_name") or qualified[len(prefix) :])
            if bare not in disabled:
                visible.append(entry)
                self.disabled_tool_registrations.pop(qualified, None)
        self._catalog_by_server[name] = visible

    async def _apply_configs_locked(
        self, new_configs: Iterable[McpServerConfig]
    ) -> McpConfigDiff | None:
        new_map = {config.name: config for config in new_configs}
        old_map = self.configs
        if old_map.keys() == new_map.keys() and all(
            old_map[name].connection_fingerprint()
            == new_map[name].connection_fingerprint()
            for name in old_map
        ):
            disabled_changed = [
                name
                for name in old_map
                if old_map[name].disabled_tools != new_map[name].disabled_tools
            ]
            self.configs = new_map
            for name in disabled_changed:
                # Grok stashes disabled registrations specifically so a
                # toggle is a local catalog operation, not another network
                # tools/list request that can fail halfway through policy.
                self._apply_disabled_tools_local(name, new_map[name].disabled_tools)
            if self.init_progress is InitProgressKind.NOT_STARTED:
                self.init_progress = InitProgressKind.FINISHED
                self._publish_snapshot()
            elif disabled_changed:
                self._publish_snapshot()
            return None

        removed: list[str] = []
        added: list[str] = []
        retained: list[str] = []
        for name, old in old_map.items():
            new = new_map.get(name)
            if new is None or old.connection_fingerprint() != new.connection_fingerprint():
                removed.append(name)
            else:
                retained.append(name)
        for name, new in new_map.items():
            old = old_map.get(name)
            if old is None or old.connection_fingerprint() != new.connection_fingerprint():
                added.append(name)

        self.generation = (self.generation + 1) & ((1 << 64) - 1)
        generation = self.generation
        self.init_progress = InitProgressKind.INITIALIZING
        self._initialized_event.clear()

        changed_names = set(removed) | set(added)
        for name in changed_names:
            self.config_versions[name] = self.config_versions.get(name, 0) + 1

        clients_to_close: list[McpClient] = []
        for name in removed:
            self.shutting_down.add(name)
            self.handshaking.discard(name)
            client = self.owned_clients.pop(name, None)
            if client is not None:
                clients_to_close.append(client)
            self._catalog_by_server.pop(name, None)
            self.init_failed.pop(name, None)
            self.auth_required.discard(name)
            prefix = f"{name}{MCP_TOOL_NAME_DELIMITER}"
            self.mcp_tool_meta = {
                key: value
                for key, value in self.mcp_tool_meta.items()
                if not key.startswith(prefix)
            }
            self.disabled_tool_registrations = {
                key: value
                for key, value in self.disabled_tool_registrations.items()
                if not key.startswith(prefix)
            }
            reason = (
                McpServerStatusReason.CONFIG_CHANGED
                if name in new_map
                else McpServerStatusReason.CONFIG_REMOVED
            )
            self._emit_config(
                name,
                McpClientEventKind.CONFIG_REMOVED,
                reason=reason,
            )

        self.configs = new_map
        start_names: list[str] = []
        for name in added:
            config = new_map[name]
            self.shutting_down.discard(name)
            if not config.enabled:
                self.shutting_down.add(name)
                self._emit_config(
                    name,
                    McpClientEventKind.CONFIG_REMOVED,
                    reason=McpServerStatusReason.DISABLED,
                )
                continue
            if config.setup_required:
                self._emit_config(
                    name,
                    McpClientEventKind.CONFIG_REMOVED,
                    detail="MCP server setup is required",
                    reason=McpServerStatusReason.SETUP_REQUIRED,
                )
                continue
            start_names.append(name)
            self.handshaking.add(name)
            self._emit_config(
                name,
                (
                    McpClientEventKind.CONFIG_CHANGED
                    if name in old_map
                    else McpClientEventKind.CONFIG_ADDED
                ),
            )

        # A global generation change invalidates in-flight initialization.
        # Healthy retained clients stay alive; retained clients that were
        # mid-handshake are explicitly restarted in the new generation so an
        # old completion cannot clear or poison the new lifecycle state.
        for name in retained:
            if old_map[name].disabled_tools != new_map[name].disabled_tools:
                self._apply_disabled_tools_local(name, new_map[name].disabled_tools)
            client = self.get_client(name)
            if client is not None and client.state is ClientStateKind.INITIALIZING:
                owned = self.owned_clients.pop(name, None)
                if owned is not None:
                    clients_to_close.append(owned)
                start_names.append(name)
                self.handshaking.add(name)

        if not self.handshaking and not start_names:
            self.init_progress = InitProgressKind.FINISHED
        self._publish_snapshot()

        if clients_to_close:
            await asyncio.gather(
                *(client.close() for client in clients_to_close),
                return_exceptions=True,
            )
        for name in start_names:
            self._spawn(
                self.initialize_server(
                    name,
                    expected_generation=generation,
                    expected_config_version=self.config_versions.get(name, 0),
                ),
                name=f"mcp-init:{name}:{generation}",
            )
        return McpConfigDiff(added=added, removed=removed, retained=retained)

    async def initialize_server(
        self,
        name: str,
        *,
        expected_generation: int,
        expected_config_version: int,
    ) -> None:
        config = self.configs.get(name)
        if (
            config is None
            or not config.enabled
            or config.setup_required
            or self._closed
            or self.generation != expected_generation
            or self.config_versions.get(name, 0) != expected_config_version
        ):
            return
        client = self.owned_clients.get(name)
        if client is None:
            client = McpClient(config, self._event_sink)
            self.owned_clients[name] = client
        self.init_failed.pop(name, None)
        try:
            await client.ensure_initialized()
            await self.refresh_server_tools(
                name,
                expected_client=client,
                expected_generation=expected_generation,
                expected_config_version=expected_config_version,
            )
        except Exception as exc:  # noqa: BLE001
            if (
                self.generation == expected_generation
                and self.config_versions.get(name, 0) == expected_config_version
                and self.get_client(name) is client
            ):
                lowered = str(exc).lower()
                if "401" in lowered or "403" in lowered or "oauth" in lowered:
                    self.auth_required.add(name)
                    self.init_failed.pop(name, None)
                else:
                    self.init_failed[name] = str(exc)
                logger.warning("MCP initialization failed for %s: %s", name, exc)
        finally:
            if (
                self.generation == expected_generation
                and self.config_versions.get(name, 0) == expected_config_version
            ):
                self.handshaking.discard(name)
                self._finish_init_if_settled()

    def _finish_init_if_settled(self) -> None:
        if not self.handshaking:
            self.init_progress = InitProgressKind.FINISHED
        self._publish_snapshot()

    async def refresh_server_tools(
        self,
        name: str,
        *,
        expected_client: McpClient | None = None,
        expected_generation: int | None = None,
        expected_config_version: int | None = None,
    ) -> bool:
        client = self.get_client(name)
        if client is None or (expected_client is not None and client is not expected_client):
            return False
        connection_generation = client.connection_generation
        raw_tools = await client.list_tools()
        config = self.configs.get(name)
        if config is None:
            return False
        catalog, tool_meta, disabled_registrations = self._build_tool_catalog(
            name, raw_tools, config
        )
        if (
            (expected_generation is not None and self.generation != expected_generation)
            or (
                expected_config_version is not None
                and self.config_versions.get(name, 0) != expected_config_version
            )
            or self.get_client(name) is not client
            or client.connection_generation != connection_generation
        ):
            return False
        prefix = f"{name}{MCP_TOOL_NAME_DELIMITER}"
        self.mcp_tool_meta = {
            key: value
            for key, value in self.mcp_tool_meta.items()
            if not key.startswith(prefix)
        }
        self.disabled_tool_registrations = {
            key: value
            for key, value in self.disabled_tool_registrations.items()
            if not key.startswith(prefix)
        }
        self.mcp_tool_meta.update(tool_meta)
        self.disabled_tool_registrations.update(disabled_registrations)
        self._catalog_by_server[name] = catalog
        self._publish_snapshot()
        return True

    def _build_tool_catalog(
        self,
        name: str,
        raw_tools: Iterable[Any],
        config: McpServerConfig,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
        disabled = config.disabled_tools
        catalog: list[dict[str, Any]] = []
        tool_meta: dict[str, Any] = {}
        disabled_registrations: dict[str, dict[str, Any]] = {}
        for tool in raw_tools:
            bare = str(getattr(tool, "name", "") or "").strip()
            if not bare:
                continue
            schema = getattr(tool, "inputSchema", None) or getattr(
                tool, "input_schema", None
            )
            if not isinstance(schema, Mapping):
                schema = {}
            meta = getattr(tool, "meta", None) or getattr(tool, "_meta", None)
            qualified = f"{name}{MCP_TOOL_NAME_DELIMITER}{bare}"
            entry = {
                "name": qualified,
                "server": name,
                "server_name": name,
                "tool_name": bare,
                "description": str(getattr(tool, "description", "") or ""),
                "parameters": dict(schema),
                "input_schema": dict(schema),
            }
            if meta is not None:
                entry["_meta"] = _model_dump(meta)
                tool_meta[qualified] = entry["_meta"]
            if bare in disabled:
                disabled_registrations[qualified] = entry
                continue
            catalog.append(entry)
        return catalog, tool_meta, disabled_registrations

    def tools_for_server(self, name: str) -> list[str]:
        return [str(tool.get("name") or "") for tool in self._catalog_by_server.get(name, [])]

    def unregister_server_tools(self, name: str) -> None:
        self._catalog_by_server.pop(name, None)
        prefix = f"{name}{MCP_TOOL_NAME_DELIMITER}"
        self.mcp_tool_meta = {
            key: value for key, value in self.mcp_tool_meta.items() if not key.startswith(prefix)
        }
        self.disabled_tool_registrations = {
            key: value
            for key, value in self.disabled_tool_registrations.items()
            if not key.startswith(prefix)
        }
        self._publish_snapshot()

    def _commit_tool_catalog(
        self,
        name: str,
        catalog: list[dict[str, Any]],
        tool_meta: dict[str, Any],
        disabled_registrations: dict[str, dict[str, Any]],
    ) -> None:
        prefix = f"{name}{MCP_TOOL_NAME_DELIMITER}"
        self.mcp_tool_meta = {
            key: value
            for key, value in self.mcp_tool_meta.items()
            if not key.startswith(prefix)
        }
        self.disabled_tool_registrations = {
            key: value
            for key, value in self.disabled_tool_registrations.items()
            if not key.startswith(prefix)
        }
        self.mcp_tool_meta.update(tool_meta)
        self.disabled_tool_registrations.update(disabled_registrations)
        self._catalog_by_server[name] = catalog

    async def respawn_stdio(
        self,
        name: str,
        *,
        expected_token: tuple[int, tuple[Any, ...]],
    ) -> None:
        config = self.configs.get(name)
        if (
            config is None
            or config.transport is not McpTransport.STDIO
            or not config.enabled
            or config.setup_required
            or self.config_token(name) != expected_token
        ):
            raise McpError(f"MCP server {name!r} is no longer configured")

        current = self.owned_clients.get(name)
        if current is not None and self.client_is_ready(name, expected=current):
            return

        # Handshake and enumerate before publishing the replacement. This is
        # Grok's atomic respawn seam: a config change during a multi-second
        # handshake drops the candidate instead of letting it kill/replace the
        # new config's client.
        # Suppress first-time Ready while the candidate is private; restart
        # success is owned by the restart controller. Liveness is wired only
        # when the candidate becomes the registered client.
        candidate = McpClient(config, lambda _event: None)
        published = False
        try:
            await candidate.ensure_initialized()
            raw_tools = await candidate.list_tools()
            catalog, tool_meta, disabled = self._build_tool_catalog(
                name, raw_tools, config
            )
            if (
                candidate.state is not ClientStateKind.READY
                or candidate._owner_task is None
                or candidate._owner_task.done()
            ):
                raise McpError(
                    f"MCP server {name!r} respawn candidate closed before commit",
                    code="mcp_transport_closed",
                )
            if self._closed or self.config_token(name) != expected_token:
                raise McpError(
                    f"MCP server {name!r} respawn raced with config change",
                    code="mcp_config_changed",
                )
            current = self.owned_clients.get(name)
            if current is not None and self.client_is_ready(name, expected=current):
                return
            candidate._event_sink = self._event_sink
            self.owned_clients[name] = candidate
            self._commit_tool_catalog(name, catalog, tool_meta, disabled)
            self.init_failed.pop(name, None)
            self.auth_required.discard(name)
            self.shutting_down.discard(name)
            self._publish_snapshot()
            published = True
        finally:
            if not published:
                await candidate.close()

        if current is not None:
            await current.close()

    async def reset_http_client(
        self,
        name: str,
        *,
        expected_token: tuple[int, tuple[Any, ...]],
        expected_client: McpClient,
    ) -> None:
        config = self.configs.get(name)
        if (
            config is None
            or config.transport is McpTransport.STDIO
            or not config.enabled
            or config.setup_required
            or self.config_token(name) != expected_token
            or self.get_client(name) is not expected_client
        ):
            raise McpError(f"HTTP MCP server {name!r} is no longer current")
        if self.client_is_ready(name, expected=expected_client):
            return
        if expected_client.state is ClientStateKind.INITIALIZING:
            # A lazy tool call may already be rebuilding this same HTTP
            # client. Join its single-flight handshake instead of resetting a
            # connection underneath the caller.
            await expected_client.ensure_initialized()
        else:
            await expected_client.reset_transport(intentional=True)
            await expected_client.ensure_initialized()
        if self.config_token(name) != expected_token or self.get_client(name) is not expected_client:
            raise McpError(
                f"HTTP MCP server {name!r} recovery raced with config change",
                code="mcp_config_changed",
            )
        refreshed = await self.refresh_server_tools(
            name,
            expected_client=expected_client,
            expected_config_version=expected_token[0],
        )
        if not refreshed:
            raise McpError(
                f"HTTP MCP server {name!r} recovery became stale",
                code="mcp_config_changed",
            )
        self.init_failed.pop(name, None)
        self.auth_required.discard(name)
        self.shutting_down.discard(name)

    async def authenticate_server(self, name: str) -> None:
        """Run an explicit interactive OAuth recovery for one HTTP server."""

        config = self.configs.get(name)
        token = self.config_token(name)
        if (
            config is None
            or token is None
            or config.transport is McpTransport.STDIO
            or not config.enabled
            or config.setup_required
        ):
            raise McpError(
                f"MCP server {name!r} is not an enabled HTTP server",
                code="mcp_auth_unsupported",
            )
        client = self.owned_clients.get(name)
        if client is None:
            client = McpClient(config, self._event_sink)
            self.owned_clients[name] = client
        client._oauth_interactive = True
        try:
            await client.reset_transport(intentional=True)
            await client.ensure_initialized()
            if self.config_token(name) != token or self.get_client(name) is not client:
                raise McpError(
                    f"MCP OAuth for {name!r} raced with config change",
                    code="mcp_config_changed",
                )
            refreshed = await self.refresh_server_tools(
                name,
                expected_client=client,
                expected_config_version=token[0],
            )
            if not refreshed:
                raise McpError(
                    f"MCP OAuth for {name!r} became stale",
                    code="mcp_config_changed",
                )
        except Exception:
            if self.config_token(name) == token and self.get_client(name) is client:
                self.auth_required.add(name)
            raise
        finally:
            client._oauth_interactive = False
        self.auth_required.discard(name)
        self.init_failed.pop(name, None)

    async def call_tool(self, qualified_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        validate_tool_name(qualified_name)
        parsed = parse_mcp_tool_name(qualified_name)
        if parsed is None:
            raise McpError(f"invalid MCP tool name: {qualified_name}")
        server_name, tool_name = parsed
        config = self.configs.get(server_name)
        if config is None or not config.enabled:
            raise McpError(
                f"MCP server {server_name!r} is not configured",
                code="mcp_server_not_found",
            )
        if config.setup_required:
            raise McpError(
                f"MCP server {server_name!r} requires setup",
                code="mcp_setup_required",
            )
        if tool_name in config.disabled_tools:
            raise McpError(
                f"MCP tool {qualified_name!r} is disabled", code="mcp_tool_disabled"
            )
        client = self.get_client(server_name)
        if client is None:
            client = McpClient(config, self._event_sink)
            self.owned_clients[server_name] = client
        return await client.call_tool(tool_name, arguments)

    async def read_resource(self, server_name: str, uri: str) -> Any:
        client = self.get_client(server_name)
        if client is None:
            config = self.configs.get(server_name)
            if config is None or not config.enabled:
                raise McpError(f"MCP server {server_name!r} is not configured")
            client = McpClient(config, self._event_sink)
            self.owned_clients[server_name] = client
        return await client.read_resource(uri)

    async def close(self) -> None:
        async with self._apply_lock:
            if self._closed:
                return
            self._closed = True
            self.shutting_down.update(self.configs)
            for task in list(self._background):
                task.cancel()
            if self._background:
                await asyncio.gather(*self._background, return_exceptions=True)
            clients = list(self.owned_clients.values())
            self.owned_clients.clear()
            await asyncio.gather(
                *(client.close() for client in clients), return_exceptions=True
            )
            self.handshaking.clear()


__all__ = [
    "ClientStateKind",
    "InitProgressKind",
    "MCP_TOOL_NAME_DELIMITER",
    "McpClient",
    "McpConfigDiff",
    "McpError",
    "McpState",
    "parse_mcp_tool_name",
    "validate_tool_name",
]
