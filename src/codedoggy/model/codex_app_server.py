"""Codex app-server JSON-RPC + session (Hermes transports/codex_app_server*).

Optional opt-in runtime: spawn ``codex app-server`` over stdio, handshake,
``thread/start`` + ``turn/start``, project ``item/*`` notifications, handle
server-initiated approvals, interrupt / timeout / OAuth classification.

Default Codex path remains HTTP Responses API (``provider=codex``).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from codedoggy.model.codex_event_projector import CodexEventProjector
from codedoggy.model.openai_compat import ModelError
from codedoggy.model.profile import ProviderProfile
from codedoggy.model.profile_registry import get_profile
from codedoggy.model.redact import redact_sensitive_text
from codedoggy.model.types import ChatMessage, CompletionResult, ModelConfig

logger = logging.getLogger(__name__)

MIN_CODEX_VERSION = (0, 125, 0)
_STDERR_TAIL_LINES = 12
_TURN_ABORTED_MARKERS = ("<turn_aborted>", "<turn_aborted/>")

_OAUTH_REFRESH_FAILURE_HINTS = (
    "invalid_grant",
    "invalid grant",
    "refresh token",
    "refresh_token",
    "token refresh",
    "token_refresh",
    "token has expired",
    "expired_token",
    "expired token",
    "not authenticated",
    "unauthenticated",
    "unauthorized",
    "401 unauthorized",
    "re-authenticate",
    "reauthenticate",
    "please log in",
    "please login",
    "auth profile",
    "no auth profile",
    "oauth",
)


@dataclass
class CodexAppServerError(RuntimeError):
    code: int
    message: str
    data: Any = None

    def __str__(self) -> str:
        return f"codex app-server error {self.code}: {self.message}"


@dataclass
class _Pending:
    queue: queue.Queue
    method: str
    sent_at: float = field(default_factory=time.time)


@dataclass
class TurnResult:
    """Result of one user→assistant→tool turn through the codex app-server."""

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None
    turn_id: Optional[str] = None
    thread_id: Optional[str] = None
    token_usage_last: Optional[dict[str, Any]] = None
    token_usage_total: Optional[dict[str, Any]] = None
    model_context_window: Optional[int] = None
    compacted: bool = False
    should_retire: bool = False


@dataclass
class _ServerRequestRouting:
    auto_approve_exec: bool = False
    auto_approve_apply_patch: bool = False


def _classify_oauth_failure(*parts: str) -> Optional[str]:
    haystack = " ".join(p for p in parts if p).lower()
    if not haystack:
        return None
    for needle in _OAUTH_REFRESH_FAILURE_HINTS:
        if needle in haystack:
            return (
                "Codex authentication failed — your ChatGPT/Codex login "
                "looks expired or invalid. Run `codex login` to refresh, "
                "then retry. (Fall back to HTTP Responses with provider=codex "
                "if the issue persists.)"
            )
    return None


def _has_turn_aborted_marker(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _TURN_ABORTED_MARKERS)


def _coerce_turn_input_text(user_input: Any) -> str:
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list):
        parts: list[str] = []
        for item in user_input:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if not isinstance(item, dict):
                if item is not None:
                    parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item_type in {"image", "image_url", "input_image"}:
                parts.append("[image attached]")
        text = "\n\n".join(p for p in parts if p).strip()
        return text or "What do you see in this image?"
    return "" if user_input is None else str(user_input)


def _apply_token_usage_notification(result: TurnResult, note: dict) -> None:
    params = note.get("params") or {}
    usage = params.get("tokenUsage") or params.get("usage") or {}
    if not isinstance(usage, dict) or not usage:
        turn = params.get("turn") or {}
        usage = turn.get("tokenUsage") or turn.get("usage") or {}
    if not isinstance(usage, dict) or not usage:
        return
    result.token_usage_last = dict(usage)
    # Accumulate when possible
    total = result.token_usage_total or {}
    for k, v in usage.items():
        if isinstance(v, (int, float)):
            total[k] = int(total.get(k, 0) or 0) + int(v)
        else:
            total[k] = v
    result.token_usage_total = total
    cw = params.get("modelContextWindow") or params.get("contextWindow")
    if isinstance(cw, int):
        result.model_context_window = cw


def _apply_compaction_notification(result: TurnResult, note: dict) -> None:
    method = str(note.get("method") or "")
    if "compact" in method.lower():
        result.compacted = True
    params = note.get("params") or {}
    if params.get("compacted") or params.get("compaction"):
        result.compacted = True


def parse_codex_version(output: str) -> Optional[tuple[int, int, int]]:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output or "")
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def check_codex_binary(
    codex_bin: str = "codex",
    min_version: tuple[int, int, int] = MIN_CODEX_VERSION,
) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [codex_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False, (
            f"codex CLI not found at {codex_bin!r}. Install with: "
            f"npm i -g @openai/codex"
        )
    except subprocess.TimeoutExpired:
        return False, "codex --version timed out"
    if proc.returncode != 0:
        return False, f"codex --version exited {proc.returncode}: {proc.stderr.strip()}"
    version = parse_codex_version(proc.stdout)
    if version is None:
        return False, f"could not parse codex version from: {proc.stdout!r}"
    if version < min_version:
        return False, (
            f"codex {'.'.join(map(str, version))} is older than required "
            f"{'.'.join(map(str, min_version))}. Run: npm i -g @openai/codex"
        )
    return True, ".".join(map(str, version))


class CodexAppServerRpc:
    """Newline-delimited JSON-RPC 2.0 over stdio to ``codex app-server``."""

    def __init__(
        self,
        codex_bin: str = "codex",
        *,
        codex_home: str | None = None,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        spawn_env = os.environ.copy()
        if env:
            spawn_env.update(env)
        if codex_home:
            spawn_env["CODEX_HOME"] = codex_home
        spawn_env.setdefault("RUST_LOG", "warn")
        cmd = [codex_bin, "app-server"] + list(extra_args or [])
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=spawn_env,
        )
        self._next_id = 1
        self._pending: dict[int, _Pending] = {}
        self._pending_lock = threading.Lock()
        self._notifications: queue.Queue = queue.Queue()
        self._server_requests: queue.Queue = queue.Queue()
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self._closed = False
        self._initialized = False
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

    def initialize(
        self,
        client_name: str = "codedoggy",
        client_title: str = "CodeDoggy",
        client_version: str = "0.1",
        capabilities: dict | None = None,
        timeout: float = 15.0,
    ) -> dict:
        if self._initialized:
            raise RuntimeError("already initialized")
        result = self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": client_name,
                    "title": client_title,
                    "version": client_version,
                },
                "capabilities": capabilities or {},
            },
            timeout=timeout,
        )
        self.notify("initialized")
        self._initialized = True
        return result

    def request(
        self, method: str, params: dict | None = None, timeout: float = 60.0
    ) -> dict:
        rid = self._take_id()
        q: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[rid] = _Pending(queue=q, method=method)
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        try:
            msg = q.get(timeout=timeout)
        except queue.Empty:
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise TimeoutError(
                f"codex app-server {method!r} timed out after {timeout}s"
            )
        if "error" in msg:
            err = msg["error"] or {}
            raise CodexAppServerError(
                code=int(err.get("code", -1)),
                message=str(err.get("message") or ""),
                data=err.get("data"),
            )
        return msg.get("result") or {}

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def respond(self, request_id: Any, result: dict) -> None:
        self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    def respond_error(
        self,
        request_id: Any,
        code: int,
        message: str,
        data: Any | None = None,
    ) -> None:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        self._send({"jsonrpc": "2.0", "id": request_id, "error": err})

    def take_notification(self, timeout: float = 0.0) -> dict | None:
        try:
            if timeout <= 0:
                return self._notifications.get_nowait()
            return self._notifications.get(timeout=timeout)
        except queue.Empty:
            return None

    def take_server_request(self, timeout: float = 0.0) -> dict | None:
        try:
            if timeout <= 0:
                return self._server_requests.get_nowait()
            return self._server_requests.get(timeout=timeout)
        except queue.Empty:
            return None

    def stderr_tail(self, n: int = 20) -> list[str]:
        with self._stderr_lock:
            return list(self._stderr_lines[-n:])

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def close(self, timeout: float = 3.0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=timeout)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

    def __enter__(self) -> CodexAppServerRpc:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _take_id(self) -> int:
        rid = self._next_id
        self._next_id += 1
        return rid

    def _send(self, msg: dict) -> None:
        if self._closed:
            raise RuntimeError("codex app-server client is closed")
        if not self._proc.stdin:
            raise RuntimeError("codex app-server stdin closed")
        try:
            line = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise RuntimeError(
                f"codex app-server stdin closed unexpectedly: {exc}"
            ) from exc

    def _read_stdout(self) -> None:
        if self._proc.stdout is None:
            return
        try:
            for raw in iter(self._proc.stdout.readline, b""):
                if not raw:
                    break
                line = raw.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    with self._stderr_lock:
                        self._stderr_lines.append(
                            f"<non-json on stdout> {line[:200]!r}"
                        )
                    continue
                self._dispatch(msg)
        except Exception as exc:
            with self._stderr_lock:
                self._stderr_lines.append(f"<stdout reader error> {exc}")

    def _dispatch(self, msg: dict) -> None:
        if "id" in msg and ("result" in msg or "error" in msg):
            with self._pending_lock:
                try:
                    rid = int(msg["id"])
                except (TypeError, ValueError):
                    rid = msg["id"]
                pending = self._pending.pop(rid, None)
            if pending:
                try:
                    pending.queue.put_nowait(msg)
                except queue.Full:
                    pass
            return
        if "id" in msg and "method" in msg:
            self._server_requests.put(msg)
            return
        if "method" in msg:
            self._notifications.put(msg)

    def _read_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        try:
            for line in iter(self._proc.stderr.readline, b""):
                if not line:
                    break
                with self._stderr_lock:
                    self._stderr_lines.append(
                        line.decode("utf-8", "replace").rstrip()
                    )
                    if len(self._stderr_lines) > 500:
                        self._stderr_lines = self._stderr_lines[-500:]
        except Exception:
            pass


# Back-compat alias used by older imports / tests
CodexAppServerClient_RPC = CodexAppServerRpc


class CodexAppServerSession:
    """One Codex thread per session — lifecycle owned by the agent loop."""

    def __init__(
        self,
        *,
        cwd: str | None = None,
        codex_bin: str = "codex",
        codex_home: str | None = None,
        approval_callback: Callable[..., str] | None = None,
        on_event: Callable[[dict], None] | None = None,
        request_routing: _ServerRequestRouting | None = None,
        client_factory: Callable[..., CodexAppServerRpc] | None = None,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._codex_bin = codex_bin
        self._codex_home = codex_home
        self._approval_callback = approval_callback
        self._on_event = on_event
        self._routing = request_routing or _ServerRequestRouting()
        self._client_factory = client_factory or CodexAppServerRpc
        self._client: CodexAppServerRpc | None = None
        self._thread_id: str | None = None
        self._interrupt_event = threading.Event()
        self._pending_file_changes: dict[str, str] = {}
        self._closed = False

    def ensure_started(self) -> str:
        if self._thread_id is not None:
            return self._thread_id
        if self._client is None:
            self._client = self._client_factory(
                self._codex_bin, codex_home=self._codex_home
            )
        self._client.initialize()
        params: dict[str, Any] = {"cwd": self._cwd}
        result = self._client.request("thread/start", params, timeout=15)
        thread_obj = result.get("thread") or {}
        thread_id = (
            thread_obj.get("id")
            or thread_obj.get("sessionId")
            or result.get("sessionId")
            or result.get("threadId")
            or result.get("id")
        )
        if not thread_id:
            raise CodexAppServerError(
                code=-32603,
                message=(
                    "codex thread/start returned no thread id "
                    f"(payload keys: {sorted(result.keys())})"
                ),
            )
        self._thread_id = str(thread_id)
        logger.info(
            "codex app-server thread started: id=%s cwd=%s",
            self._thread_id[:8],
            self._cwd,
        )
        return self._thread_id

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._thread_id = None

    def __enter__(self) -> CodexAppServerSession:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def request_interrupt(self) -> None:
        self._interrupt_event.set()

    def _format_error_with_stderr(
        self,
        prefix: str,
        exc: Any = "",
        *,
        tail_lines: int = _STDERR_TAIL_LINES,
    ) -> str:
        exc_str = str(exc) if exc != "" and exc is not None else ""
        base = f"{prefix}: {exc_str}" if exc_str else prefix
        if self._client is None:
            return base
        try:
            tail = self._client.stderr_tail(tail_lines)
        except Exception:
            return base
        if not tail:
            return base
        joined = "\n".join(line.rstrip() for line in tail if line)
        if not joined.strip():
            return base
        redacted = redact_sensitive_text(joined, force=True) or joined
        return f"{base}\ncodex stderr (last {len(tail)} lines):\n{redacted}"

    def run_turn(
        self,
        user_input: Any,
        *,
        turn_timeout: float = 600.0,
        notification_poll_timeout: float = 0.25,
        post_tool_quiet_timeout: float = 90.0,
        model: str | None = None,
    ) -> TurnResult:
        result = TurnResult()
        try:
            self.ensure_started()
        except (CodexAppServerError, TimeoutError, OSError) as exc:
            result.error = self._format_error_with_stderr(
                "codex app-server startup failed", exc
            )
            result.should_retire = True
            return result
        assert self._client is not None and self._thread_id is not None
        result.thread_id = self._thread_id

        self._interrupt_event.clear()
        projector = CodexEventProjector()
        user_input_text = _coerce_turn_input_text(user_input)

        turn_params: dict[str, Any] = {
            "threadId": self._thread_id,
            "input": [{"type": "text", "text": user_input_text}],
        }
        if model:
            turn_params["model"] = model

        try:
            ts = self._client.request("turn/start", turn_params, timeout=10)
        except CodexAppServerError as exc:
            stderr_blob = "\n".join(self._client.stderr_tail(40))
            hint = _classify_oauth_failure(exc.message, stderr_blob)
            if hint is not None:
                result.error = hint
                result.should_retire = True
            else:
                result.error = self._format_error_with_stderr(
                    "turn/start failed", exc
                )
            return result
        except TimeoutError as exc:
            stderr_blob = "\n".join(self._client.stderr_tail(40))
            hint = _classify_oauth_failure(stderr_blob)
            result.error = hint or self._format_error_with_stderr(
                "turn/start timed out", exc
            )
            result.should_retire = True
            return result

        result.turn_id = (ts.get("turn") or {}).get("id") or ts.get("turnId")
        deadline = time.monotonic() + turn_timeout
        turn_complete = False
        last_tool_completion_at: float | None = None

        while time.monotonic() < deadline and not turn_complete:
            if self._interrupt_event.is_set():
                self._issue_interrupt(result.turn_id)
                result.interrupted = True
                break

            if not self._client.is_alive():
                stderr_blob = "\n".join(self._client.stderr_tail(60))
                hint = _classify_oauth_failure(stderr_blob)
                if hint is not None:
                    result.error = hint
                else:
                    result.error = self._format_error_with_stderr(
                        "codex app-server subprocess exited unexpectedly",
                        tail_lines=20,
                    )
                result.should_retire = True
                break

            if (
                last_tool_completion_at is not None
                and (time.monotonic() - last_tool_completion_at)
                > post_tool_quiet_timeout
            ):
                self._issue_interrupt(result.turn_id)
                result.interrupted = True
                result.error = (
                    f"codex went silent for "
                    f"{post_tool_quiet_timeout:.0f}s after a tool result; "
                    f"retiring app-server session."
                )
                result.should_retire = True
                break

            sreq = self._client.take_server_request(timeout=0)
            if sreq is not None:
                for _ in range(8):
                    pending = self._client.take_notification(timeout=0)
                    if pending is None:
                        break
                    _apply_token_usage_notification(result, pending)
                    _apply_compaction_notification(result, pending)
                    self._track_pending_file_change(pending)
                    proj = projector.project(pending)
                    if proj.messages:
                        result.projected_messages.extend(proj.messages)
                    if proj.is_tool_iteration:
                        result.tool_iterations += 1
                        last_tool_completion_at = time.monotonic()
                    if proj.final_text is not None:
                        result.final_text = proj.final_text
                        if _has_turn_aborted_marker(proj.final_text):
                            turn_complete = True
                            result.interrupted = True
                            result.error = (
                                result.error or "codex reported turn_aborted"
                            )
                self._handle_server_request(sreq)
                last_tool_completion_at = None
                continue

            note = self._client.take_notification(
                timeout=notification_poll_timeout
            )
            if note is None:
                continue

            method = note.get("method", "")
            if self._on_event is not None:
                try:
                    self._on_event(note)
                except Exception:
                    logger.debug("on_event callback raised", exc_info=True)

            _apply_token_usage_notification(result, note)
            _apply_compaction_notification(result, note)
            self._track_pending_file_change(note)

            projection = projector.project(note)
            if projection.messages:
                result.projected_messages.extend(projection.messages)
            if projection.is_tool_iteration:
                result.tool_iterations += 1
                last_tool_completion_at = time.monotonic()
            else:
                if projection.messages or projection.final_text is not None:
                    last_tool_completion_at = None
            if projection.final_text is not None:
                result.final_text = projection.final_text
                if _has_turn_aborted_marker(projection.final_text):
                    turn_complete = True
                    result.interrupted = True
                    result.error = (
                        result.error or "codex reported turn_aborted"
                    )

            if method == "turn/completed":
                turn_complete = True
                turn_obj = (note.get("params") or {}).get("turn") or {}
                turn_status = turn_obj.get("status")
                if turn_status and turn_status not in {
                    "completed",
                    "interrupted",
                }:
                    err_obj = turn_obj.get("error")
                    err_msg = (
                        str(err_obj)
                        if err_obj is not None
                        else str(turn_status)
                    )
                    stderr_blob = "\n".join(self._client.stderr_tail(40))
                    hint = _classify_oauth_failure(err_msg, stderr_blob)
                    if hint is not None:
                        result.error = hint
                        result.should_retire = True
                    else:
                        result.error = self._format_error_with_stderr(
                            f"turn ended status={turn_status}", err_msg
                        )
                if turn_status == "interrupted":
                    result.interrupted = True

        if (
            not turn_complete
            and not result.interrupted
            and result.final_text
            and result.error is None
        ):
            logger.warning(
                "codex app-server turn reached deadline after assistant "
                "text but before turn/completed; accepting final text"
            )
            turn_complete = True

        if not turn_complete and not result.interrupted:
            self._issue_interrupt(result.turn_id)
            result.interrupted = True
            if not result.error:
                result.error = self._format_error_with_stderr(
                    f"turn timed out after {turn_timeout}s"
                )
            result.should_retire = True

        return result

    def _issue_interrupt(self, turn_id: str | None) -> None:
        if self._client is None or self._thread_id is None or turn_id is None:
            return
        try:
            self._client.request(
                "turn/interrupt",
                {"threadId": self._thread_id, "turnId": turn_id},
                timeout=5,
            )
        except (CodexAppServerError, TimeoutError) as exc:
            logger.debug("turn/interrupt non-fatal: %s", exc)

    def _track_pending_file_change(self, note: dict) -> None:
        method = str(note.get("method") or "")
        params = note.get("params") or {}
        item = params.get("item") or {}
        if "fileChange" not in method and item.get("type") != "fileChange":
            return
        item_id = str(item.get("id") or params.get("itemId") or "")
        if not item_id:
            return
        changes = item.get("changes") or []
        summary_parts = []
        for ch in changes:
            if isinstance(ch, dict):
                summary_parts.append(
                    f"{(ch.get('kind') or {}).get('type', 'update')}: "
                    f"{ch.get('path') or '?'}"
                )
        self._pending_file_changes[item_id] = "; ".join(summary_parts) or "file change"

    def _handle_server_request(self, req: dict) -> None:
        if self._client is None:
            return
        method = req.get("method", "")
        rid = req.get("id")
        params = req.get("params") or {}

        if method == "item/commandExecution/requestApproval":
            decision = self._decide_exec_approval(params)
            self._client.respond(rid, {"decision": decision})
        elif method == "item/fileChange/requestApproval":
            decision = self._decide_apply_patch_approval(params)
            self._client.respond(rid, {"decision": decision})
        elif method == "item/permissions/requestApproval":
            self._client.respond(rid, {"decision": "declined"})
        else:
            logger.debug("unknown codex server request: %s", method)
            try:
                self._client.respond_error(
                    rid, -32601, f"method not supported: {method}"
                )
            except Exception:
                pass

    def _decide_exec_approval(self, params: dict) -> str:
        if self._approval_callback is not None:
            try:
                return str(
                    self._approval_callback("exec", params) or "declined"
                )
            except Exception:
                return "declined"
        if self._routing.auto_approve_exec:
            return "approved"
        # Non-interactive default: decline (safe). Interactive TUI can wire callback.
        return "declined"

    def _decide_apply_patch_approval(self, params: dict) -> str:
        item_id = str(params.get("itemId") or params.get("id") or "")
        summary = self._pending_file_changes.get(item_id, "")
        if self._approval_callback is not None:
            try:
                return str(
                    self._approval_callback(
                        "apply_patch", {**params, "summary": summary}
                    )
                    or "declined"
                )
            except Exception:
                return "declined"
        if self._routing.auto_approve_apply_patch:
            return "approved"
        return "declined"


class CodexAppServerClient:
    """ChatClient façade: one-shot turn via CodexAppServerSession."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        profile: ProviderProfile | None = None,
        codex_bin: str | None = None,
    ) -> None:
        self._config = config
        self._profile = profile or get_profile(config.provider)
        self._codex_bin = codex_bin or os.environ.get("CODEX_BIN") or "codex"

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def profile(self) -> ProviderProfile | None:
        return self._profile

    def complete(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResult:
        if not shutil.which(self._codex_bin):
            raise ModelError(
                f"codex binary not found ({self._codex_bin!r}). "
                "Install Codex CLI or use provider=codex (HTTP Responses)."
            )
        ok, ver_msg = check_codex_binary(self._codex_bin)
        if not ok:
            logger.warning("codex version check: %s", ver_msg)

        session = CodexAppServerSession(codex_bin=self._codex_bin)
        try:
            user_text = _last_user_text(messages)
            turn = session.run_turn(
                user_text,
                turn_timeout=float(self._config.timeout_s or 120),
                model=self._config.model,
            )
            if turn.error and not turn.final_text:
                raise ModelError(f"codex app-server failed: {turn.error}")
            usage = {}
            if turn.token_usage_last:
                u = turn.token_usage_last
                usage = {
                    k: v
                    for k, v in {
                        "prompt_tokens": u.get("inputTokens")
                        or u.get("prompt_tokens"),
                        "completion_tokens": u.get("outputTokens")
                        or u.get("completion_tokens"),
                        "total_tokens": u.get("totalTokens")
                        or u.get("total_tokens"),
                    }.items()
                    if v is not None
                }
            return CompletionResult(
                content=turn.final_text or None,
                model=self._config.model,
                finish_reason="stop" if not turn.interrupted else "interrupted",
                tool_calls=[],
                raw={
                    "codex_app_server": True,
                    "turn_id": turn.turn_id,
                    "thread_id": turn.thread_id,
                    "projected_messages": turn.projected_messages,
                    "tool_iterations": turn.tool_iterations,
                    "error": turn.error,
                    "should_retire": turn.should_retire,
                    "compacted": turn.compacted,
                },
                usage=usage,
            )
        except (CodexAppServerError, TimeoutError, OSError) as exc:
            raise ModelError(f"codex app-server failed: {exc}") from exc
        finally:
            session.close()


def _last_user_text(messages: list[Any]) -> str:
    for m in reversed(messages):
        if isinstance(m, ChatMessage):
            if m.role == "user" and m.content:
                return str(m.content)
        elif isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return _coerce_turn_input_text(c)
    return ""
