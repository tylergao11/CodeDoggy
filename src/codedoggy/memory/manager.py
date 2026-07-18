"""MemoryManager — hermes-agent/agent/memory_manager.py orchestration surface.

Source: C:\\Ai\\hermes-agent\\agent\\memory_manager.py

  - builtin curated + session FTS first; at most ONE external provider
  - build_system_prompt / prefetch_all / queue_prefetch_all / sync_all
  - get_all_tool_schemas / handle_tool_call routing
  - on_turn_start / on_session_end / flush_pending / shutdown_all
  - background single-worker; failures never block the turn
  - drain timeout (Hermes _SYNC_DRAIN_TIMEOUT_S = 5.0)

Prefetch raw text is fenced by ``context_fence.build_memory_context_block``
and injected into the current user message at sample time only — not SYSTEM.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from codedoggy.memory.provider import (
    CuratedMemoryProvider,
    SessionFtsProvider,
)

logger = logging.getLogger(__name__)

# Hermes memory_manager._SYNC_DRAIN_TIMEOUT_S
_SYNC_DRAIN_TIMEOUT_S = 5.0

# Core tool names memory providers must not shadow (Hermes toolsets spirit)
_CORE_TOOL_NAMES = frozenset(
    {
        "read_file",
        "search_replace",
        "list_dir",
        "grep",
        "run_terminal_cmd",
        "memory",
        "session_search",
        "code_nav",
        "spawn_subagent",
        "parallel_tasks",
        "get_subagent_output",
    }
)


class MemoryManager:
    """Single integration point for memory (session + audit + tools)."""

    def __init__(self) -> None:
        self._providers: list[Any] = []
        self._has_external = False
        self._tool_to_provider: dict[str, Any] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._shutdown = False  # after shutdown_all, never recreate executor
        self.curated_store: Any | None = None
        self.session_store: Any | None = None
        self._session_id: str = ""

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

    # -- Registration (Hermes add_provider) -----------------------------------

    def add_provider(self, provider: Any) -> bool:
        """Register provider. Second external is rejected (Hermes one-external)."""
        name = getattr(provider, "name", "") or "unnamed"
        is_builtin = name.startswith("builtin") or name == "builtin"
        if not is_builtin:
            if self._has_external:
                existing = next(
                    (
                        p.name
                        for p in self._providers
                        if not str(getattr(p, "name", "")).startswith("builtin")
                        and getattr(p, "name", "") != "builtin"
                    ),
                    "?",
                )
                logger.warning(
                    "Rejected memory provider %r — external %r already registered. "
                    "Only one external memory provider is allowed.",
                    name,
                    existing,
                )
                return False
            self._has_external = True
        self._providers.append(provider)

        # Index tool schemas → provider (Hermes; skip core shadows)
        get_schemas = getattr(provider, "get_tool_schemas", None)
        if callable(get_schemas):
            try:
                for schema in get_schemas() or []:
                    if not isinstance(schema, dict):
                        continue
                    tool_name = schema.get("name") or ""
                    if (
                        isinstance(schema.get("function"), dict)
                        and schema.get("type") == "function"
                    ):
                        tool_name = schema["function"].get("name") or tool_name
                    if not tool_name or not isinstance(tool_name, str):
                        continue
                    if tool_name in _CORE_TOOL_NAMES:
                        logger.warning(
                            "Memory provider %r tool %r shadows a core tool; ignored",
                            name,
                            tool_name,
                        )
                        continue
                    if tool_name not in self._tool_to_provider:
                        self._tool_to_provider[tool_name] = provider
            except Exception as e:  # noqa: BLE001
                logger.warning("provider %s get_tool_schemas failed: %s", name, e)

        logger.info("Memory provider registered: %s", name)
        return True

    @property
    def providers(self) -> list[Any]:
        return list(self._providers)

    def initialize_all(self, session_id: str = "", **kwargs: Any) -> None:
        """Hermes initialize_all — warm providers for a session."""
        self._session_id = session_id or self._session_id
        for p in self._providers:
            init = getattr(p, "initialize", None)
            if not callable(init):
                continue
            try:
                init(session_id=self._session_id, **kwargs)
            except Exception as e:  # noqa: BLE001
                logger.warning("provider %s initialize failed: %s", p.name, e)

    # -- System prompt -------------------------------------------------------

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

    # -- Prefetch / recall ---------------------------------------------------

    def prefetch_all(
        self, query: str, *, session_id: str = "", cwd: str = ""
    ) -> str:
        """Merge prefetch from all providers (fail-soft). Hermes prefetch_all."""
        q = (query or "").strip()
        if not q:
            return ""
        parts: list[str] = []
        sid = session_id or self._session_id
        for p in self._providers:
            try:
                result = p.prefetch(q, session_id=sid, cwd=cwd)
                if result and str(result).strip():
                    parts.append(str(result).strip())
            except Exception as e:  # noqa: BLE001
                logger.debug("provider %s prefetch failed: %s", p.name, e)
        return "\n\n".join(parts)

    def queue_prefetch_all(
        self, query: str, *, session_id: str = "", cwd: str = ""
    ) -> None:
        """Background warm for next turn (Hermes queue_prefetch_all)."""
        providers = list(self._providers)
        if not providers:
            return
        q = (query or "").strip()
        if not q:
            return
        sid = session_id or self._session_id

        def _run() -> None:
            for p in providers:
                try:
                    p.queue_prefetch(q, session_id=sid, cwd=cwd)
                except Exception as e:  # noqa: BLE001
                    logger.debug("provider %s queue_prefetch failed: %s", p.name, e)

        self._submit_background(_run)

    def sync_all(
        self,
        user_text: str,
        assistant_text: str,
        *,
        session_id: str = "",
        cwd: str = "",
        messages: list[Any] | None = None,
    ) -> None:
        """Post-turn sync on background worker (Hermes sync_all)."""
        providers = list(self._providers)
        if not providers:
            return
        sid = session_id or self._session_id
        u = user_text or ""
        a = assistant_text or ""

        def _run() -> None:
            from codedoggy.memory.redact import redact_secrets

            safe_u = redact_secrets(u) or ""
            safe_a = redact_secrets(a) or ""
            safe_msgs = _redact_messages_for_provider(messages)
            for p in providers:
                try:
                    try:
                        p.sync_turn(
                            safe_u,
                            safe_a,
                            session_id=sid,
                            cwd=cwd,
                            messages=safe_msgs,
                        )
                    except TypeError:
                        p.sync_turn(safe_u, safe_a, session_id=sid, cwd=cwd)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Memory provider '%s' sync_turn failed: %s", p.name, e
                    )

        self._submit_background(_run)

    # -- Tools ---------------------------------------------------------------

    def get_all_tool_schemas(self) -> list[dict[str, Any]]:
        """Hermes get_all_tool_schemas — bare function schemas from providers."""
        schemas: list[dict[str, Any]] = []
        seen: set[str] = set()
        for p in self._providers:
            try:
                for raw in p.get_tool_schemas() or []:
                    if not isinstance(raw, dict):
                        continue
                    schema = raw
                    if raw.get("type") == "function" and isinstance(
                        raw.get("function"), dict
                    ):
                        schema = raw["function"]
                    name = schema.get("name")
                    if not name or name in seen or name in _CORE_TOOL_NAMES:
                        continue
                    schemas.append(schema)
                    seen.add(name)
            except Exception as e:  # noqa: BLE001
                logger.warning("provider %s get_tool_schemas failed: %s", p.name, e)
        return schemas

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_to_provider

    def handle_tool_call(
        self, tool_name: str, args: dict[str, Any], **kwargs: Any
    ) -> str:
        """Route external memory tool to provider (Hermes handle_tool_call)."""
        provider = self._tool_to_provider.get(tool_name)
        if provider is None:
            return json.dumps(
                {"success": False, "error": f"No memory provider handles tool {tool_name!r}"}
            )
        try:
            handle = getattr(provider, "handle_tool_call", None)
            if not callable(handle):
                return json.dumps(
                    {"success": False, "error": f"Provider has no handle_tool_call for {tool_name}"}
                )
            result = handle(tool_name, args, **kwargs)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Memory provider '%s' handle_tool_call(%s) failed: %s",
                provider.name,
                tool_name,
                e,
            )
            return json.dumps(
                {"success": False, "error": f"Memory tool '{tool_name}' failed: {e}"}
            )

    # -- Lifecycle hooks -----------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        for p in self._providers:
            hook = getattr(p, "on_turn_start", None)
            if not callable(hook):
                continue
            try:
                hook(turn_number, message, **kwargs)
            except Exception as e:  # noqa: BLE001
                logger.debug("provider %s on_turn_start failed: %s", p.name, e)

    def on_session_end(self, messages: list[Any] | None = None) -> None:
        for p in self._providers:
            hook = getattr(p, "on_session_end", None)
            if not callable(hook):
                continue
            try:
                hook(list(messages or []))
            except Exception as e:  # noqa: BLE001
                logger.warning("provider %s on_session_end failed: %s", p.name, e)

    def on_pre_compress(self, messages: list[Any] | None = None) -> str:
        """Hermes on_pre_compress — before context fold discards middle.

        Returns combined text from providers (may be empty). Callers may fold
        this into the compression summary prompt; side-effect-only providers
        just return "".
        """
        parts: list[str] = []
        snap = list(messages or [])
        for p in self._providers:
            hook = getattr(p, "on_pre_compress", None)
            if not callable(hook):
                continue
            try:
                result = hook(snap)
                if result and str(result).strip():
                    parts.append(str(result).strip())
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Memory provider '%s' on_pre_compress failed: %s", p.name, e
                )
        return "\n\n".join(parts)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        """Hermes on_session_switch — rebind provider session ids without teardown."""
        if not new_session_id:
            return
        self._session_id = new_session_id
        if rewound:
            kwargs["rewound"] = True
        for p in self._providers:
            hook = getattr(p, "on_session_switch", None)
            if not callable(hook):
                # Minimal: update _session_id attr if present
                if hasattr(p, "_session_id"):
                    try:
                        p._session_id = new_session_id  # noqa: SLF001
                    except Exception:  # noqa: BLE001
                        pass
                continue
            try:
                hook(
                    new_session_id,
                    parent_session_id=parent_session_id,
                    reset=reset,
                    **kwargs,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Memory provider '%s' on_session_switch failed: %s", p.name, e
                )

    def commit_session_boundary_async(
        self,
        messages: list[Any] | None,
        *,
        new_session_id: str,
        parent_session_id: str = "",
        reason: str = "new_session",
    ) -> None:
        """Hermes commit_session_boundary_async (#16454).

        Serialize on_session_end → on_session_switch on the single background
        worker so extraction never races rebind.
        """
        if not self._providers:
            self._session_id = new_session_id or self._session_id
            return
        snapshot = list(messages or [])

        def _run() -> None:
            try:
                self.on_session_end(snapshot)
            except Exception as e:  # noqa: BLE001
                logger.warning("Session-boundary extraction failed: %s", e)
            try:
                self.on_session_switch(
                    new_session_id,
                    parent_session_id=parent_session_id,
                    reset=True,
                    reason=reason,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Session-boundary switch failed: %s", e)

        self._submit_background(_run)

    def as_memory_selector(self) -> Any:
        """Hermes multi-source selector for turn prefetch."""
        from codedoggy.memory.hermes_select import HermesMemorySelector

        return HermesMemorySelector(
            curated_store=self.curated_store,
            session_store=self.session_store,
        )

    def notify_memory_write(
        self,
        target: str = "memory",
        *,
        action: str = "add",
        content: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Hermes on_memory_write — refresh freeze + mirror to external providers.

        Skips builtin_* sources of the write (Hermes skips name=='builtin').
        """
        store = self.curated_store
        if store is not None:
            refresh = getattr(store, "refresh_system_prompt_snapshot", None)
            if callable(refresh):
                try:
                    refresh()
                except Exception as e:  # noqa: BLE001
                    logger.warning("notify_memory_write refresh failed: %s", e)
        self.on_memory_write(action, target, content, metadata=metadata)

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Notify external providers when curated memory tool writes (Hermes)."""
        for p in self._providers:
            name = str(getattr(p, "name", "") or "")
            if name == "builtin" or name.startswith("builtin"):
                continue
            hook = getattr(p, "on_memory_write", None)
            if not callable(hook):
                continue
            try:
                try:
                    hook(action, target, content, metadata=dict(metadata or {}))
                except TypeError:
                    hook(action, target, content)
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Memory provider '%s' on_memory_write failed: %s", p.name, e
                )

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Hermes on_delegation — parent observes subagent task+result."""
        for p in self._providers:
            hook = getattr(p, "on_delegation", None)
            if not callable(hook):
                continue
            try:
                hook(
                    task or "",
                    result or "",
                    child_session_id=child_session_id,
                    **kwargs,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Memory provider '%s' on_delegation failed: %s", p.name, e
                )

    def flush_pending(self, timeout: float | None = None) -> bool:
        """Hermes flush_pending — barrier on single-worker queue."""
        with self._lock:
            executor = self._executor
        if executor is None:
            return True
        try:
            fut = executor.submit(lambda: None)
        except RuntimeError:
            return True
        try:
            fut.result(timeout=timeout)
            return True
        except Exception:  # noqa: BLE001
            return False

    def shutdown(self, *, timeout_s: float = _SYNC_DRAIN_TIMEOUT_S) -> None:
        """Hermes shutdown_all — drain then provider shutdown."""
        self.shutdown_all(timeout_s=timeout_s)

    def shutdown_all(self, *, timeout_s: float = _SYNC_DRAIN_TIMEOUT_S) -> None:
        # Drain queued work
        self.flush_pending(timeout=max(0.1, float(timeout_s)))
        with self._lock:
            self._shutdown = True
            ex = self._executor
            self._executor = None
            providers = list(self._providers)
        if ex is not None:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                ex.shutdown(wait=False)
            deadline = time.time() + max(0.1, float(timeout_s))
            while time.time() < deadline:
                alive = any(
                    t.is_alive() for t in (getattr(ex, "_threads", None) or [])
                )
                if not alive:
                    break
                time.sleep(0.05)
        for p in providers:
            close = getattr(p, "shutdown", None) or getattr(p, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    logger.debug("provider shutdown failed", exc_info=True)

    # -- Background dispatch (Hermes _submit_background) ---------------------

    def _submit_background(self, fn: Any) -> None:
        executor = self._get_executor()
        if executor is None:
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                logger.debug("Inline memory background task failed: %s", e)
            return
        try:
            executor.submit(fn)
        except RuntimeError:
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                logger.debug("Inline memory background task failed: %s", e)

    def _submit(self, fn: Any) -> None:
        self._submit_background(fn)

    def _get_executor(self) -> ThreadPoolExecutor | None:
        with self._lock:
            if self._shutdown:
                return None
            if self._executor is None:
                try:
                    self._executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="mem-sync"
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to create memory sync executor: %s", e)
                    return None
            return self._executor


def _redact_messages_for_provider(messages: list[Any] | None) -> list[Any] | None:
    """Best-effort redact of message content before external provider sync."""
    if not messages:
        return messages
    from codedoggy.memory.redact import redact_secrets

    out: list[Any] = []
    for m in messages:
        if isinstance(m, dict):
            d = dict(m)
            if "content" in d and isinstance(d["content"], str):
                d["content"] = redact_secrets(d["content"])
            out.append(d)
            continue
        content = getattr(m, "content", None)
        if isinstance(content, str):
            try:
                # Prefer copy with redacted content when dataclass-like
                if hasattr(m, "__dataclass_fields__"):
                    import dataclasses

                    out.append(
                        dataclasses.replace(m, content=redact_secrets(content))
                    )
                    continue
            except Exception:  # noqa: BLE001
                pass
        out.append(m)
    return out
