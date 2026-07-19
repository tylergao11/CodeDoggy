"""Cancellation helpers for blocking stdlib/SDK response streams."""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, TypeVar

from codedoggy.model.errors import ModelStreamCancelled

_T = TypeVar("_T")
# Main + the Grok-aligned eight-child coordinator may sample concurrently.
# Keep a second generation of capacity for temporarily abandoned requests.
_REQUEST_SLOTS = threading.BoundedSemaphore(18)


class HTTPErrorSnapshot(Exception):
    """Detached HTTP failure safe to pass across the request-owner boundary."""

    def __init__(self, status: int, body: str = "", reason: str = "") -> None:
        super().__init__(f"HTTP {status}: {body or reason}")
        self.status = int(status)
        self.body = str(body)
        self.reason = str(reason)


def cancellation_requested(signal: Any | None) -> bool:
    """Return whether an Event-like or callable cancellation signal fired."""
    if signal is None:
        return False
    check = getattr(signal, "is_set", None)
    if callable(check):
        try:
            return bool(check())
        except Exception:  # noqa: BLE001
            return False
    if callable(signal):
        try:
            return bool(signal())
        except Exception:  # noqa: BLE001
            return False
    return False


def raise_if_cancelled(signal: Any | None) -> None:
    if cancellation_requested(signal):
        raise ModelStreamCancelled()


def cancellable_readline(response: Any, signal: Any | None) -> Any:
    """Read one stream line and translate close-on-cancel into cancellation."""
    raise_if_cancelled(signal)
    try:
        line = response.readline()
    except Exception as exc:  # noqa: BLE001
        if cancellation_requested(signal):
            raise ModelStreamCancelled() from exc
        raise
    raise_if_cancelled(signal)
    return line


def cancellable_read(response: Any, signal: Any | None) -> Any:
    """Read a non-stream response with the same cancellation semantics."""
    raise_if_cancelled(signal)
    try:
        body = response.read()
    except Exception as exc:  # noqa: BLE001
        if cancellation_requested(signal):
            raise ModelStreamCancelled() from exc
        raise
    raise_if_cancelled(signal)
    return body


def snapshot_http_error(
    error: Any,
    signal: Any | None,
    *,
    read_body: bool = True,
    max_body_bytes: int = 600,
) -> HTTPErrorSnapshot:
    """Consume and close an HTTPError inside its request-owner thread.

    ``urllib.error.HTTPError`` owns a live response.  Letting it cross back to
    the turn thread and reading ``error.fp`` there recreates an uncancellable
    blocking read.  This helper detaches only inert status/body/reason data;
    cancellation may abandon the owner while a broken server stalls its body.
    """
    status = int(getattr(error, "code", 0) or 0)
    reason = str(getattr(error, "reason", "") or "")
    body = ""
    try:
        if read_body and getattr(error, "fp", None) is not None:
            raise_if_cancelled(signal)
            try:
                raw = error.read(max(0, int(max_body_bytes)))
            except Exception:  # noqa: BLE001
                if cancellation_requested(signal):
                    raise ModelStreamCancelled()
            else:
                raise_if_cancelled(signal)
                if isinstance(raw, bytes):
                    body = raw.decode("utf-8", errors="replace")
                else:
                    body = str(raw or "")
    finally:
        close = getattr(error, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
    return HTTPErrorSnapshot(status, body, reason)


def run_cancellable_request(
    operation: Callable[[], _T],
    cancel_event: Any | None,
) -> _T:
    """Run the complete blocking request on one abandonable owner thread.

    CPython/Windows cannot reliably interrupt a ``BufferedReader.readline``
    from another thread, even after closing or shutting down its socket.  The
    safe ownership model is therefore the same one Grok applies to its request
    future: the turn waits on a result *or* cancellation, and cancellation
    drops the whole request task.  The daemon owner retains/then closes every
    urllib object itself.  Eighteen global slots (main + 8 children, twice)
    bound abandoned workers until
    their configured transport timeout expires.
    """
    if cancel_event is None:
        return operation()
    raise_if_cancelled(cancel_event)
    while not _REQUEST_SLOTS.acquire(timeout=0.05):
        raise_if_cancelled(cancel_event)
    # Cancellation may race the successful acquire after the loop's last
    # timeout check.  Do not start an already-cancelled network operation or
    # strand one of the globally bounded owner slots.
    try:
        raise_if_cancelled(cancel_event)
    except BaseException:
        _REQUEST_SLOTS.release()
        raise

    result: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
    abandoned = threading.Event()

    def _run() -> None:
        try:
            try:
                value = operation()
            except BaseException as exc:
                payload = (False, exc)
            else:
                payload = (True, value)
            if not abandoned.is_set():
                try:
                    result.put_nowait(payload)
                except queue.Full:
                    pass
        finally:
            _REQUEST_SLOTS.release()

    owner = threading.Thread(
        target=_run,
        name="model-request",
        daemon=True,
    )
    try:
        owner.start()
    except BaseException:
        _REQUEST_SLOTS.release()
        raise
    while True:
        if cancellation_requested(cancel_event):
            abandoned.set()
            raise ModelStreamCancelled()
        try:
            ok, value = result.get(timeout=0.05)
        except queue.Empty:
            continue
        if cancellation_requested(cancel_event):
            abandoned.set()
            raise ModelStreamCancelled()
        if ok:
            return value
        raise value
