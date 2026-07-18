"""Async soft-prefire for compaction-adjacent work (Grok prefire spirit).

While the model is *not* waiting on the critical path for the next sample,
we can start expensive side work (memory flush LLM call) on a background
thread. The next ``ensure()`` joins the future before deciding flush/fold
so we never double-flush or race the live message list.

This is *not* a full async compact of the live window (that must stay
single-threaded on ``messages``). Prefire only covers:

  - pre-computing whether flush should run
  - optionally running memory_flush against a *snapshot* of messages

Join happens in ContextCompactor.ensure / join_prefire.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cd-prefire")
        return _executor


@dataclass
class PrefireController:
    """One controller per ContextCompactor / session."""

    enabled: bool = True
    _future: Future[Any] | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def submit(self, fn: Callable[[], Any]) -> bool:
        """Queue *fn* if idle. Returns False if disabled or already running."""
        if not self.enabled:
            return False
        with self._lock:
            if self._future is not None and not self._future.done():
                return False
            try:
                self._future = _pool().submit(fn)
                return True
            except Exception as e:  # noqa: BLE001
                logger.warning("prefire submit failed: %s", e)
                self._future = None
                return False

    def join(self, *, timeout_s: float = 30.0) -> Any | None:
        """Wait for outstanding prefire; clear only after wait finishes.

        Previously cleared ``_future`` before ``result()``, so ``is_running()``
        lied while the worker was still executing (session close race).
        """
        with self._lock:
            fut = self._future
        if fut is None:
            return None
        try:
            return fut.result(timeout=timeout_s)
        except Exception as e:  # noqa: BLE001
            logger.warning("prefire join failed: %s", e)
            return None
        finally:
            with self._lock:
                if self._future is fut:
                    self._future = None

    def try_join(self) -> Any | None:
        """Non-blocking: return result only if already done."""
        with self._lock:
            fut = self._future
            if fut is None or not fut.done():
                return None
        try:
            return fut.result(timeout=0)
        except Exception as e:  # noqa: BLE001
            logger.warning("prefire try_join failed: %s", e)
            return None
        finally:
            with self._lock:
                if self._future is fut:
                    self._future = None

    def is_running(self) -> bool:
        with self._lock:
            fut = self._future
            return fut is not None and not fut.done()

    def cancel_pending(self) -> None:
        with self._lock:
            fut = self._future
        if fut is not None:
            fut.cancel()
            # Wait briefly so worker exits before session teardown
            try:
                fut.result(timeout=5.0)
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            if self._future is fut:
                self._future = None

    def clear(self) -> None:
        """Cancel and wait — safe for session close."""
        self.cancel_pending()
