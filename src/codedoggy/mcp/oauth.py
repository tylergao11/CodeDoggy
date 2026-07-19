"""OAuth transport glue backed by Grok's MCP credential location.

The Python MCP SDK owns RFC discovery, PKCE, token refresh, and HTTP auth.
This module supplies the host-owned pieces Grok also owns: BYO client config,
loopback browser callback, and an atomic ``$GROK_HOME/mcp_credentials.json``
store keyed by ``server:url``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import threading
import time
import webbrowser
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import parse_qs, urlsplit

from codedoggy.mcp.config import McpServerConfig, grok_home

logger = logging.getLogger(__name__)

_STORE_LOCK = threading.Lock()


def _credential_path() -> Path:
    return grok_home() / "mcp_credentials.json"


@contextmanager
def _cross_process_lock(path: Path) -> Iterator[None]:
    """Best-effort cross-process lock matching Grok's merge-before-save rule."""

    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:  # pragma: no cover - Windows is the primary workspace
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _read_store(path: Path, *, strict: bool) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        if strict:
            raise
        logger.warning("unable to read MCP OAuth credentials from %s", path, exc_info=True)
        return {}
    if not isinstance(value, dict):
        if strict:
            raise ValueError("MCP credential root must be an object")
        return {}
    return value


def _write_store(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.codedoggy.tmp"
    )
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if os.name != "nt":  # pragma: no cover
        os.chmod(temp, 0o600)
    os.replace(temp, path)
    if os.name != "nt":  # pragma: no cover
        os.chmod(path, 0o600)


class GrokOAuthTokenStorage:
    """MCP SDK ``TokenStorage`` compatible with Grok's credential file."""

    def __init__(
        self,
        config: McpServerConfig,
        client_metadata: Any,
        *,
        interactive: bool,
    ) -> None:
        self.config = config
        self.client_metadata = client_metadata
        self.interactive = interactive
        self.path = _credential_path()
        self.key = f"{config.name}:{config.url}"

    def _entry(self) -> dict[str, Any]:
        raw = _read_store(self.path, strict=False).get(self.key)
        return dict(raw) if isinstance(raw, Mapping) else {}

    def _mutate(self, mutate: Any) -> None:
        with _STORE_LOCK, _cross_process_lock(self.path):
            store = _read_store(self.path, strict=True)
            raw = store.get(self.key)
            entry = dict(raw) if isinstance(raw, Mapping) else {}
            mutate(entry)
            store[self.key] = entry
            _write_store(self.path, store)

    async def get_tokens(self) -> Any | None:
        from mcp.shared.auth import OAuthToken

        entry = await asyncio.to_thread(self._entry)
        raw = entry.get("token_response")
        if not isinstance(raw, Mapping) or not raw.get("access_token"):
            return None
        access_token = str(raw["access_token"])
        expires_in = raw.get("expires_in")
        received_at = entry.get("token_received_at")
        try:
            if (
                expires_in is not None
                and received_at is not None
                and int(received_at) + int(expires_in) <= int(time.time())
                and raw.get("refresh_token")
            ):
                # An empty access token makes the SDK take its refresh path.
                access_token = ""
        except (TypeError, ValueError):
            pass
        scope = raw.get("scope")
        if scope is None and isinstance(entry.get("granted_scopes"), list):
            scope = " ".join(str(item) for item in entry["granted_scopes"])
        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=int(expires_in) if expires_in is not None else None,
            scope=str(scope) if scope else None,
            refresh_token=(
                str(raw["refresh_token"]) if raw.get("refresh_token") else None
            ),
        )

    async def set_tokens(self, tokens: Any) -> None:
        raw = tokens.model_dump(mode="json", exclude_none=True)
        raw["token_type"] = str(raw.get("token_type") or "bearer").lower()

        def mutate(entry: dict[str, Any]) -> None:
            entry["token_response"] = raw
            entry["token_received_at"] = int(time.time())
            scopes = str(raw.get("scope") or "").split()
            entry["granted_scopes"] = scopes

        await asyncio.to_thread(self._mutate, mutate)

    async def get_client_info(self) -> Any | None:
        from mcp.shared.auth import OAuthClientInformationFull

        entry = await asyncio.to_thread(self._entry)
        stored = entry.get("_codedoggy_client_info")
        if isinstance(stored, Mapping):
            try:
                info = OAuthClientInformationFull.model_validate(stored)
                configured = self.config.oauth
                current_redirects = {
                    str(uri) for uri in (self.client_metadata.redirect_uris or [])
                }
                stored_redirects = {str(uri) for uri in (info.redirect_uris or [])}
                # A dynamically registered client is bound to its callback
                # URI. Explicit interactive auth on a new ephemeral listener
                # must register a new client instead of reusing stale DCR
                # metadata. Non-interactive refresh can safely reuse it.
                if (
                    not self.interactive
                    or (configured and configured.client_id)
                    or current_redirects == stored_redirects
                ):
                    return info
            except Exception:  # noqa: BLE001
                logger.warning("ignored invalid stored MCP OAuth client info", exc_info=True)

        configured = self.config.oauth
        client_id = (
            configured.client_id if configured and configured.client_id else entry.get("client_id")
        )
        if not client_id:
            return None
        values = self.client_metadata.model_dump(mode="json", exclude_none=True)
        values.update(
            {
                "client_id": str(client_id),
                "client_secret": configured.client_secret if configured else None,
            }
        )
        return OAuthClientInformationFull.model_validate(values)

    async def set_client_info(self, client_info: Any) -> None:
        raw = client_info.model_dump(mode="json", exclude_none=True)

        def mutate(entry: dict[str, Any]) -> None:
            entry["client_id"] = raw.get("client_id")
            entry.setdefault("token_response", None)
            entry.setdefault("granted_scopes", [])
            entry["_codedoggy_client_info"] = raw

        await asyncio.to_thread(self._mutate, mutate)


class OAuthCallbackError(RuntimeError):
    pass


class OAuthLoopbackCallback:
    """One loopback listener that can service repeated OAuth grants."""

    def __init__(self) -> None:
        self.server: asyncio.AbstractServer | None = None
        self.port = 0
        self._results: asyncio.Queue[tuple[str, str | None] | BaseException] = (
            asyncio.Queue()
        )

    @classmethod
    async def create(cls, requested_port: int | None) -> "OAuthLoopbackCallback":
        self = cls()
        self.server = await asyncio.start_server(
            self._handle,
            "127.0.0.1",
            requested_port or 0,
        )
        sockets = self.server.sockets or []
        if not sockets:
            await self.close()
            raise OAuthCallbackError("OAuth callback server has no listening socket")
        self.port = int(sockets[0].getsockname()[1])
        return self

    async def redirect_handler(self, url: str) -> None:
        logger.info("opening browser for MCP OAuth consent: %s", url)
        opened = await asyncio.to_thread(webbrowser.open, url)
        if not opened:
            logger.warning("browser did not open; visit MCP OAuth URL manually: %s", url)

    async def callback_handler(self) -> tuple[str, str | None]:
        result = await asyncio.wait_for(self._results.get(), timeout=300.0)
        if isinstance(result, BaseException):
            raise result
        return result

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        status = "200 OK"
        body = "MCP authorization complete. You can close this window."
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5.0)
            request_line = head.split(b"\r\n", 1)[0].decode("ascii", errors="replace")
            _method, target, _version = request_line.split(" ", 2)
            parsed = urlsplit(target)
            query = parse_qs(parsed.query)
            if parsed.path != "/callback":
                raise OAuthCallbackError("unexpected OAuth callback path")
            if query.get("error"):
                raise OAuthCallbackError(str(query["error"][0]))
            code = str((query.get("code") or [""])[0])
            state = (query.get("state") or [None])[0]
            if not code:
                raise OAuthCallbackError("OAuth callback did not contain a code")
            await self._results.put((code, str(state) if state is not None else None))
        except Exception as exc:  # noqa: BLE001
            status = "400 Bad Request"
            body = "MCP authorization failed. Return to CodeDoggy and retry."
            await self._results.put(
                exc if isinstance(exc, OAuthCallbackError) else OAuthCallbackError(str(exc))
            )
        payload = body.encode("utf-8")
        writer.write(
            f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode(
                "ascii"
            )
            + payload
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def close(self) -> None:
        if self.server is None:
            return
        self.server.close()
        await self.server.wait_closed()
        self.server = None


def _reserve_ephemeral_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def build_oauth_httpx_auth(
    config: McpServerConfig,
    *,
    interactive: bool,
    stack: Any,
) -> Any | None:
    """Build SDK OAuth auth for an HTTP server lacking a static bearer header."""

    if not config.url or any(key.lower() == "authorization" for key in config.headers):
        return None

    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata
    from pydantic import AnyUrl

    oauth = config.oauth
    callback: OAuthLoopbackCallback | None = None
    if interactive:
        callback = await OAuthLoopbackCallback.create(
            oauth.callback_port if oauth else None
        )
        stack.push_async_callback(callback.close)
        port = callback.port
    else:
        port = oauth.callback_port if oauth and oauth.callback_port else _reserve_ephemeral_port()
    redirect_uri = f"http://127.0.0.1:{port}/callback"
    metadata = OAuthClientMetadata(
        redirect_uris=[AnyUrl(redirect_uri)],
        token_endpoint_auth_method=(
            "client_secret_post" if oauth and oauth.client_secret else "none"
        ),
        scope=" ".join(oauth.scopes) if oauth and oauth.scopes else None,
        client_name="CodeDoggy",
    )
    storage = GrokOAuthTokenStorage(config, metadata, interactive=interactive)
    if not interactive and await storage.get_tokens() is None:
        # Match Grok's non-interactive probe: do not perform DCR or begin a
        # browser grant during background session startup. A plain 401 is
        # surfaced as needs_auth; ``McpRuntime.authenticate`` is the explicit
        # interactive recovery path.
        return None
    return OAuthClientProvider(
        config.url,
        metadata,
        storage,
        redirect_handler=callback.redirect_handler if callback else None,
        callback_handler=callback.callback_handler if callback else None,
        timeout=300.0,
    )


__all__ = [
    "GrokOAuthTokenStorage",
    "OAuthLoopbackCallback",
    "build_oauth_httpx_auth",
]
