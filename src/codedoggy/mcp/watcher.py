"""Event-driven MCP config watcher corresponding to Grok config/reloader.rs."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class McpConfigWatcher:
    """Watch known Grok config parents and debounce reload notifications."""

    def __init__(
        self,
        paths: Iterable[Path],
        callback: Callable[[], None],
        *,
        debounce_seconds: float = 0.10,
    ) -> None:
        self.paths = tuple(Path(path).resolve(strict=False) for path in paths)
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self._observer: Any = None
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._closed = False

    def _is_relevant(self, value: str | os.PathLike[str] | None) -> bool:
        if not value:
            return False
        candidate = Path(value).resolve(strict=False)
        key = os.path.normcase(str(candidate))
        known = {os.path.normcase(str(path)) for path in self.paths}
        if key in known:
            return True
        # A file can be created after startup; only accept Grok MCP names.
        return candidate.name in {
            ".mcp.json",
            ".claude.json",
            "mcp.json",
            ".grok",
            ".cursor",
        } or (
            candidate.name == "config.toml" and candidate.parent.name == ".grok"
        )

    def _schedule(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_seconds, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._timer = None
        try:
            self.callback()
        except Exception:  # noqa: BLE001
            logger.exception("MCP config reload callback failed")

    def start(self) -> bool:
        if self._observer is not None or self._closed:
            return False
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.warning("watchdog unavailable; MCP config hot reload disabled")
            return False

        owner = self

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event: Any) -> None:
                if owner._is_relevant(getattr(event, "src_path", None)) or owner._is_relevant(
                    getattr(event, "dest_path", None)
                ):
                    owner._schedule()

        observer = Observer()
        watched: set[str] = set()
        handler = Handler()
        for path in self.paths:
            parent = path.parent
            # Watch the nearest existing ancestor so creation of .grok/.cursor
            # directories is observed as well.
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            key = os.path.normcase(str(parent.resolve(strict=False)))
            if key in watched:
                continue
            try:
                observer.schedule(handler, str(parent), recursive=False)
            except OSError:
                logger.debug("unable to watch MCP config parent %s", parent, exc_info=True)
                continue
            watched.add(key)
        if not watched:
            return False
        observer.daemon = True
        observer.start()
        self._observer = observer
        return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            timer = self._timer
            self._timer = None
        if timer is not None:
            timer.cancel()
        observer = self._observer
        self._observer = None
        if observer is not None:
            observer.stop()
            observer.join(timeout=5.0)


__all__ = ["McpConfigWatcher"]
