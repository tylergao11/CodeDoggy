"""Session-owned MCP runtime wired to CodeDoggy's synchronous tool surface."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codedoggy.mcp.config import (
    McpConfigError,
    McpServerConfig,
    coerce_mcp_server_configs,
    load_mcp_config_snapshot,
)
from codedoggy.mcp.dispatcher import McpDispatcher
from codedoggy.mcp.events import McpClientEvent, McpServerStatusPayload
from codedoggy.mcp.restart import McpRestartController
from codedoggy.mcp.servers import McpError, McpState, parse_mcp_tool_name
from codedoggy.mcp.watcher import McpConfigWatcher
from codedoggy.tools.mcp.tool_index import (
    Bm25ToolSearchIndex,
    ServerMetadata,
    tools_from_mcp_catalog,
)
from codedoggy.tools.mcp.types import ToolIndex

if TYPE_CHECKING:
    from codedoggy.session.kernel import RuntimeKernel
    from codedoggy.tools.runtime import ToolCallContext

logger = logging.getLogger(__name__)


class McpRuntime:
    """Own the MCP event loop, clients, dispatcher, restart, and live index.

    CodeDoggy's agent loop is synchronous.  A dedicated asyncio thread is the
    language seam corresponding to Grok's session-local Tokio ``LocalSet``;
    MCP lifecycle and state remain Session-owned rather than global.
    """

    def __init__(
        self,
        cwd: str | Path,
        *,
        session_id: str,
        configs: Mapping[str, Mapping[str, Any]] | Sequence[McpServerConfig] | None = None,
        watch: bool = True,
        auto_restart: bool = True,
    ) -> None:
        self.cwd = Path(cwd).resolve()
        self.session_id = str(session_id)
        self._explicit_configs = (
            coerce_mcp_server_configs(configs) if configs is not None else None
        )
        self.watch_enabled = bool(watch and configs is None)
        self.auto_restart = auto_restart
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread_id: int | None = None
        self._ready = threading.Event()
        self._reload_lock = threading.Lock()
        self._closed = False
        self._start_error: BaseException | None = None
        self._state: McpState | None = None
        self._dispatcher: McpDispatcher | None = None
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._restart: McpRestartController | None = None
        self._watcher: McpConfigWatcher | None = None
        self._kernel_ref: weakref.ReferenceType[RuntimeKernel] | None = None

        self._snapshot_lock = threading.RLock()
        self._tools: list[dict[str, Any]] = []
        self._servers: list[dict[str, Any]] = []
        self._statuses: list[dict[str, Any]] = []
        self._status_by_server: dict[str, dict[str, Any]] = {}
        self._configs_by_name: dict[str, McpServerConfig] = {}
        self._initialized = False
        self._index = Bm25ToolSearchIndex([], mcp_initialized=False)
        self._tool_index = ToolIndex(index=self._index)

    @property
    def tools(self) -> list[dict[str, Any]]:
        return self._tools

    @property
    def servers(self) -> list[dict[str, Any]]:
        return self._servers

    @property
    def statuses(self) -> list[dict[str, Any]]:
        return self._statuses

    @property
    def tool_index(self) -> ToolIndex:
        return self._tool_index

    @property
    def initialized(self) -> bool:
        with self._snapshot_lock:
            return self._initialized

    @property
    def connecting_servers(self) -> list[str]:
        with self._snapshot_lock:
            return sorted(
                str(server.get("name"))
                for server in self._servers
                if server.get("name") and server.get("status") == "initializing"
            )

    def attach_kernel(self, kernel: "RuntimeKernel") -> None:
        self._kernel_ref = weakref.ref(kernel)
        kernel.mcp_runtime = self
        kernel.refresh_tool_extra()

    def populate_tool_extra(self, extra: dict[str, Any]) -> None:
        """Install stable live resources into a freshly rebuilt tool bag."""

        extra["mcp_runtime"] = self
        extra["mcp_inner_dispatch"] = self
        extra["mcp_dispatch"] = self.dispatch_legacy
        extra["mcp_tools"] = self._tools
        extra["mcp_servers"] = self._servers
        extra["mcp_status"] = self._statuses
        extra["mcp_tool_index"] = self._tool_index
        extra["mcp_initialized"] = self.initialized
        extra["mcp_authenticate"] = self.authenticate

    def start(self, *, timeout: float = 10.0) -> "McpRuntime":
        if self._closed:
            raise McpError("cannot start a closed MCP runtime", code="mcp_closed")
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"codedoggy-mcp-{self.session_id[:8]}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            self.close()
            raise McpError("MCP runtime event loop did not start", code="mcp_startup_timeout")
        if self._start_error is not None:
            raise McpError(f"MCP runtime failed to start: {self._start_error}") from self._start_error
        if self.watch_enabled:
            self._start_watcher()
        return self

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._loop_thread_id = threading.get_ident()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_start())
        except BaseException as exc:  # noqa: BLE001
            self._start_error = exc
            self._ready.set()
            try:
                loop.run_until_complete(self._async_cleanup_after_start_error())
            except Exception:  # noqa: BLE001
                pass
            loop.close()
            return
        self._ready.set()
        if self._closed:
            try:
                loop.run_until_complete(self._async_close())
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
            return
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _async_start(self) -> None:
        queue: asyncio.Queue[McpClientEvent | None] = asyncio.Queue()

        def emit(event: McpClientEvent) -> None:
            queue.put_nowait(event)

        state = McpState(emit, self._publish_snapshot)
        dispatcher = McpDispatcher(
            session_id=self.session_id,
            queue=queue,
            state=state,
            status_sink=self._publish_status,
        )
        restart = McpRestartController(
            session_id=self.session_id,
            state=state,
            status_sink=self._publish_status,
        )
        if self.auto_restart:
            dispatcher.bind_restart_controller(restart)
        self._state = state
        self._dispatcher = dispatcher
        self._restart = restart
        self._dispatcher_task = asyncio.create_task(
            dispatcher.run(), name=f"mcp-dispatcher:{self.session_id}"
        )

        if self._explicit_configs is not None:
            configs = list(self._explicit_configs)
        else:
            snapshot = load_mcp_config_snapshot(self.cwd)
            configs = snapshot.servers
        await state.apply_configs(configs)
        self._set_config_snapshot(configs)

    async def _async_cleanup_after_start_error(self) -> None:
        if self._state is not None:
            await self._state.close()
        if self._dispatcher is not None:
            await self._dispatcher.close()
        if self._dispatcher_task is not None:
            await asyncio.gather(self._dispatcher_task, return_exceptions=True)

    def _set_config_snapshot(self, configs: Sequence[McpServerConfig]) -> None:
        with self._snapshot_lock:
            self._configs_by_name = {config.name: config for config in configs}

    def _publish_snapshot(
        self,
        tools: list[dict[str, Any]],
        servers: list[dict[str, Any]],
        initialized: bool,
    ) -> None:
        with self._snapshot_lock:
            self._tools[:] = [dict(tool) for tool in tools]
            self._servers[:] = [dict(server) for server in servers]
            self._initialized = bool(initialized)
            server_meta: list[ServerMetadata | tuple[str, str | None]] = [
                ServerMetadata(
                    name=str(server.get("name") or ""),
                    description=None,
                )
                for server in servers
                if server.get("name")
                and server.get("enabled", True)
                and not server.get("setup_required", False)
                and server.get("status") == "ready"
            ]
            self._index.update(
                tools_from_mcp_catalog(self._tools),
                servers=server_meta,
                mcp_initialized=self._initialized,
            )
        kernel = self._kernel_ref() if self._kernel_ref is not None else None
        if kernel is not None and isinstance(getattr(kernel, "tool_extra", None), dict):
            kernel.tool_extra["mcp_initialized"] = self._initialized

    def _publish_status(self, payload: McpServerStatusPayload) -> None:
        raw = payload.as_dict()
        with self._snapshot_lock:
            self._status_by_server[payload.name] = raw
            self._statuses[:] = list(self._status_by_server.values())

    def _start_watcher(self, paths: Sequence[Path] | None = None) -> None:
        try:
            watch_paths = tuple(paths) if paths is not None else load_mcp_config_snapshot(self.cwd).paths
        except McpConfigError:
            logger.warning("MCP config watcher initial discovery failed", exc_info=True)
            return
        def reload_callback() -> None:
            self.reload_from_disk()

        watcher = McpConfigWatcher(watch_paths, reload_callback)
        if watcher.start():
            self._watcher = watcher

    def _refresh_watcher(self, paths: Sequence[Path]) -> None:
        if not self.watch_enabled or self._closed:
            return
        previous = self._watcher
        self._watcher = None
        if previous is not None:
            previous.close()
        self._start_watcher(paths)

    def reload_from_disk(self) -> bool:
        # Watchdog timers run on ordinary threads and may overlap. Loading and
        # applying as one serialized operation prevents an older snapshot from
        # committing after a newer file event.
        with self._reload_lock:
            if self._closed:
                return False
            try:
                snapshot = load_mcp_config_snapshot(self.cwd, strict=True)
                self.apply_configs(snapshot.servers, timeout=None)
            except (McpConfigError, McpError) as exc:
                # A half-written file is not a request to tear down the last
                # healthy runtime. The next filesystem event retries it.
                logger.warning(
                    "MCP config reload rejected; retaining current clients: %s", exc
                )
                return False
            # A reload may have been caused by creation of a previously missing
            # .grok/.cursor directory. Rebuild watches so its files are now direct
            # targets without recursively watching the whole workspace/home.
            self._refresh_watcher(snapshot.paths)
            return True

    def apply_configs(
        self,
        configs: Mapping[str, Mapping[str, Any]] | Sequence[McpServerConfig],
        *,
        timeout: float | None = None,
    ) -> Any:
        normalized = coerce_mcp_server_configs(configs)
        state = self._require_state()
        return self._submit(
            self._apply_configs_transaction(state, normalized),
            timeout=timeout,
            cancel_on_timeout=False,
        )

    async def _apply_configs_transaction(
        self,
        state: McpState,
        normalized: Sequence[McpServerConfig],
    ) -> Any:
        result = await state.apply_configs(normalized)
        self._set_config_snapshot(normalized)
        return result

    def _require_state(self) -> McpState:
        if self._state is None or self._loop is None or self._closed:
            raise McpError("MCP runtime is not running", code="mcp_not_running")
        return self._state

    def _submit(
        self,
        awaitable: Any,
        *,
        timeout: float | None,
        cancel_on_timeout: bool = True,
    ) -> Any:
        loop = self._loop
        if loop is None or self._closed:
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            raise McpError("MCP runtime is not running", code="mcp_not_running")
        if threading.get_ident() == self._loop_thread_id:
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            raise McpError(
                "synchronous MCP dispatch cannot run on the MCP event-loop thread",
                code="mcp_reentrant_dispatch",
            )
        future = asyncio.run_coroutine_threadsafe(awaitable, loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            if cancel_on_timeout:
                future.cancel()
            raise McpError("MCP runtime operation timed out", code="mcp_runtime_timeout") from exc

    def __call__(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        ctx: "ToolCallContext | None" = None,
    ) -> Any:
        return self.call(tool_name, tool_input, ctx)

    def call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        ctx: "ToolCallContext | None" = None,
    ) -> Any:
        del ctx  # policy/mutation preparation remains in use_tool before dispatch
        state = self._require_state()
        parsed = parse_mcp_tool_name(tool_name)
        with self._snapshot_lock:
            config = self._configs_by_name.get(parsed[0]) if parsed else None
        timeout = (
            config.tool_timeout_for(parsed[1]) + config.startup_timeout_sec + 2
            if config is not None and parsed is not None
            else 60.0
        )
        return self._submit(state.call_tool(tool_name, dict(tool_input)), timeout=timeout)

    def dispatch_legacy(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        return self.call(tool_name, tool_input, None)

    def read_resource(self, server_name: str, uri: str) -> Any:
        state = self._require_state()
        with self._snapshot_lock:
            config = self._configs_by_name.get(server_name)
        timeout = (config.tool_timeout_sec + config.startup_timeout_sec + 2) if config else 60
        return self._submit(state.read_resource(server_name, uri), timeout=timeout)

    def authenticate(self, server_name: str, *, timeout: float = 330.0) -> None:
        """Open the browser and complete OAuth for one HTTP MCP server."""

        state = self._require_state()
        self._submit(state.authenticate_server(server_name), timeout=timeout)

    def wait_initialized(self, timeout: float = 30.0) -> bool:
        state = self._require_state()
        return bool(
            self._submit(
                state.wait_initialized(timeout),
                timeout=timeout + 1,
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        # Fence watcher reloads and new synchronous dispatch before beginning
        # teardown; the async close path below does not consult this flag.
        self._closed = True
        watcher = self._watcher
        self._watcher = None
        if watcher is not None:
            watcher.close()
        loop = self._loop
        thread = self._thread
        if loop is not None and thread is not None and thread.is_alive():
            future = asyncio.run_coroutine_threadsafe(self._async_close(), loop)
            try:
                future.result(timeout=10.0)
            except Exception:  # noqa: BLE001
                logger.warning("MCP runtime shutdown did not finish cleanly", exc_info=True)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=10.0)
        with self._snapshot_lock:
            self._initialized = False

    async def _async_close(self) -> None:
        if self._restart is not None:
            await self._restart.close()
        if self._state is not None:
            await self._state.close()
        if self._dispatcher is not None:
            await self._dispatcher.close()
        if self._dispatcher_task is not None:
            await asyncio.gather(self._dispatcher_task, return_exceptions=True)


__all__ = ["McpRuntime"]
