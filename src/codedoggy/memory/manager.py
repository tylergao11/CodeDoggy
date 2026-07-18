"""MemoryManager — orchestrate curated + session FTS + one external provider.

Hermes ``agent/memory_manager.py`` spirit, sized for CodeDoggy:
  - builtin curated + builtin session FTS always first
  - at most ONE external provider (reject second)
  - prefetch_all / queue_prefetch_all / build_system_prompt / sync_all
  - failures never block the turn
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from codedoggy.memory.provider import (
    BaseMemoryProvider,
    CuratedMemoryProvider,
    MemoryProvider,
    SessionFtsProvider,
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """Single integration point for memory (session + audit + tools)."""

    def __init__(self) -> None:
        self._providers: list[Any] = []
        self._has_external = False
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        # Convenience handles (same objects as providers wrap)
        self.curated_store: Any | None = None
        self.session_store: Any | None = None

    @classmethod
    def create_default(
        cls,
        *,
        curated: Any | None = None,
        session_store: Any | None = None,
    ) -> MemoryManager:
        mm = cls()
        mm.curated_store = curated
        mm.session_store = session_store
        if curated is not None:
            mm.add_provider(CuratedMemoryProvider(curated))
        if session_store is not None:
            mm.add_provider(SessionFtsProvider(session_store))
        return mm

    def add_provider(self, provider: Any) -> bool:
        """Register provider. Returns False if rejected (second external)."""
        name = getattr(provider, "name", "") or "unnamed"
        is_builtin = name.startswith("builtin")
        if not is_builtin:
            if self._has_external:
                existing = next(
                    (p.name for p in self._providers if not p.name.startswith("builtin")),
                    "?",
                )
                logger.warning(
                    "Rejected memory provider %r — external %r already registered",
                    name,
                    existing,
                )
                return False
            self._has_external = True
        self._providers.append(provider)
        logger.info("Memory provider registered: %s", name)
        return True

    @property
    def providers(self) -> list[Any]:
        return list(self._providers)

    def build_system_prompt(self) -> str:
        blocks: list[str] = []
        for p in self._providers:
            try:
                block = p.system_prompt_block()
                if block and str(block).strip():
                    blocks.append(str(block).strip())
            except Exception as e:  # noqa: BLE001
                logger.warning("provider %s system_prompt_block failed: %s", p.name, e)
        return "\n\n".join(blocks)

    def prefetch_all(self, query: str, *, session_id: str = "") -> str:
        """Merge prefetch from all providers (fail-soft)."""
        q = (query or "").strip()
        if not q:
            return ""
        parts: list[str] = []
        for p in self._providers:
            try:
                result = p.prefetch(q, session_id=session_id)
                if result and str(result).strip():
                    parts.append(str(result).strip())
            except Exception as e:  # noqa: BLE001
                logger.debug("provider %s prefetch failed: %s", p.name, e)
        return "\n\n".join(parts)

    def queue_prefetch_all(self, query: str, *, session_id: str = "") -> None:
        providers = list(self._providers)
        if not providers:
            return
        q = (query or "").strip()
        if not q:
            return

        def _run() -> None:
            for p in providers:
                try:
                    p.queue_prefetch(q, session_id=session_id)
                except Exception as e:  # noqa: BLE001
                    logger.debug("provider %s queue_prefetch failed: %s", p.name, e)

        self._submit(_run)

    def sync_all(
        self,
        user_text: str,
        assistant_text: str,
        *,
        session_id: str = "",
    ) -> None:
        providers = list(self._providers)

        def _run() -> None:
            for p in providers:
                try:
                    p.sync_turn(
                        user_text or "",
                        assistant_text or "",
                        session_id=session_id,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.debug("provider %s sync_turn failed: %s", p.name, e)

        self._submit(_run)

    def as_audit_selector(self) -> Any:
        """HermesMemorySelector bound to curated + session for resident audit."""
        from codedoggy.memory.hermes_select import HermesMemorySelector

        return HermesMemorySelector(
            curated_store=self.curated_store,
            session_store=self.session_store,
        )

    def notify_memory_write(self, target: str = "memory") -> None:
        """Context flush / tool memory write — keep freeze in sync.

        Single spine: anything that mutates curated store should call this
        so system snapshot and manager-owned curated provider stay aligned.
        """
        store = self.curated_store
        if store is None:
            return
        refresh = getattr(store, "refresh_system_prompt_snapshot", None)
        if callable(refresh):
            try:
                refresh()
            except Exception as e:  # noqa: BLE001
                logger.warning("notify_memory_write refresh failed: %s", e)

    def shutdown(self, *, timeout_s: float = 5.0) -> None:
        with self._lock:
            ex = self._executor
            self._executor = None
        if ex is not None:
            ex.shutdown(wait=True, cancel_futures=True)

    def _submit(self, fn: Any) -> None:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="cd-memory"
                )
            ex = self._executor
        try:
            ex.submit(fn)
        except Exception as e:  # noqa: BLE001
            logger.warning("memory background submit failed: %s", e)
