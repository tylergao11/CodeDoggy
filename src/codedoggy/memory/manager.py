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

import inspect
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

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

    @staticmethod
    def _supported_keyword_args(
        fn: Callable[..., Any], candidates: dict[str, Any]
    ) -> dict[str, Any]:
        """Select keyword arguments from a callable signature before invoking it.

        Compatibility is decided before the call so a runtime ``TypeError``
        cannot trigger a second execution of a provider side effect.
        """
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            # Some extension callables do not expose signatures. Prefer the
            # current contract and still invoke exactly once.
            return dict(candidates)
        params = signature.parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(candidates)
        return {
            key: value
            for key, value in candidates.items()
            if key in params
            and params[key].kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }

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
        if not u.strip():
            return

        def _run() -> None:
            from codedoggy.memory.redact import redact_secrets

            safe_u = redact_secrets(u) or ""
            safe_a = redact_secrets(a) or ""
            safe_msgs = _redact_messages_for_provider(messages)
            for p in providers:
                try:
                    candidates: dict[str, Any] = {
                        "session_id": sid,
                        "cwd": cwd,
                    }
                    if safe_msgs is not None:
                        candidates["messages"] = safe_msgs
                    kwargs = self._supported_keyword_args(
                        p.sync_turn,
                        candidates,
                    )
                    # Exactly one call. A TypeError raised by provider logic is
                    # a real provider failure, not evidence of an old signature.
                    p.sync_turn(safe_u, safe_a, **kwargs)
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

    def get_all_tool_names(self) -> set[str]:
        """Return provider-owned tool names for child/toolset isolation."""
        return set(self._tool_to_provider)

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
        self._refresh_curated_snapshot()
        self.on_memory_write(action, target, content, metadata=metadata)

    def _refresh_curated_snapshot(self) -> None:
        """Refresh the frozen curated prompt after a committed local write."""
        store = self.curated_store
        if store is not None:
            refresh = getattr(store, "refresh_system_prompt_snapshot", None)
            if callable(refresh):
                try:
                    refresh()
                except Exception as e:  # noqa: BLE001
                    logger.warning("notify_memory_write refresh failed: %s", e)

    @staticmethod
    def _provider_memory_write_metadata_mode(provider: Any) -> str:
        """Resolve metadata compatibility without retrying provider effects."""
        hook = getattr(provider, "on_memory_write", None)
        try:
            signature = inspect.signature(hook)
        except (TypeError, ValueError):
            return "keyword"
        params = signature.parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return "keyword"
        metadata_param = params.get("metadata")
        if metadata_param is not None and metadata_param.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            return "keyword"
        positional = [
            p
            for p in params.values()
            if p.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        return "positional" if len(positional) >= 4 else "legacy"

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
                metadata_mode = self._provider_memory_write_metadata_mode(p)
                if metadata_mode == "keyword":
                    hook(action, target, content, metadata=dict(metadata or {}))
                elif metadata_mode == "positional":
                    hook(action, target, content, dict(metadata or {}))
                else:
                    hook(action, target, content)
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Memory provider '%s' on_memory_write failed: %s", p.name, e
                )

    _MIRRORED_MEMORY_ACTIONS = frozenset({"add", "replace", "remove"})

    @staticmethod
    def _memory_tool_result_succeeded(result: Any) -> bool:
        """Fail closed unless the built-in tool committed a non-staged write."""
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:  # noqa: BLE001
                return False
        if not isinstance(result, dict):
            return False
        return result.get("success") is True and result.get("staged") is not True

    def notify_memory_tool_write(
        self,
        tool_result: Any,
        tool_args: dict[str, Any],
        *,
        build_metadata: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        """Refresh and mirror one successful built-in memory invocation.

        Batch expansion, result gating, per-operation metadata, and ``old_text``
        forwarding follow Hermes' built-in memory bridge contract.
        """
        if not self._memory_tool_result_succeeded(tool_result):
            return

        target = str(tool_args.get("target") or "memory")
        operations = tool_args.get("operations")
        if isinstance(operations, list) and operations:
            raw_operations = operations
        else:
            raw_operations = [
                {
                    "action": tool_args.get("action"),
                    "content": tool_args.get("content"),
                    "old_text": tool_args.get("old_text"),
                }
            ]

        mirrored = [
            op
            for op in raw_operations
            if isinstance(op, dict)
            and str(op.get("action") or "") in self._MIRRORED_MEMORY_ACTIONS
        ]
        if not mirrored:
            return

        # The local store has already committed. Refresh once for the complete
        # atomic batch, then fan out per-op notifications to external providers.
        self._refresh_curated_snapshot()
        for op in mirrored:
            action = str(op.get("action") or "")
            try:
                metadata = dict(build_metadata() if build_metadata else {})
                old_text = op.get("old_text")
                if old_text:
                    metadata["old_text"] = str(old_text)
                self.on_memory_write(
                    action,
                    target,
                    str(op.get("content") or ""),
                    metadata=metadata,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "notify_memory_tool_write failed for op %s: %s", action, e
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
            if self._shutdown:
                return True
            executor = self._executor
            if executor is None:
                return True
            try:
                # Submit under the lifecycle lock. Shutdown cannot move the
                # manager to CLOSED between observing the pool and enqueueing
                # the barrier.
                fut = executor.submit(lambda: None)
            except RuntimeError:
                return False
        try:
            fut.result(timeout=timeout)
            return True
        except Exception:  # noqa: BLE001
            return False

    def shutdown(self, *, timeout_s: float = _SYNC_DRAIN_TIMEOUT_S) -> None:
        """Hermes shutdown_all — drain then provider shutdown."""
        self.shutdown_all(timeout_s=timeout_s)

    def shutdown_all(self, *, timeout_s: float = _SYNC_DRAIN_TIMEOUT_S) -> None:
        """Atomically close submissions, then drain before provider teardown."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            ex = self._executor
            self._executor = None
            providers = list(reversed(self._providers))

        if ex is None:
            self._close_providers(providers)
            return

        # No caller can enqueue after CLOSED because submission takes the same
        # lock. Accepted work drains in order; providers close only afterwards.
        try:
            ex.shutdown(wait=False, cancel_futures=False)
        except Exception:  # noqa: BLE001
            logger.debug("memory executor stop-accepting failed", exc_info=True)

        drainer = threading.Thread(
            target=self._drain_executor_and_close,
            args=(ex, providers),
            daemon=True,
            name="mem-sync-drain",
        )
        drainer.start()
        drainer.join(timeout=max(0.0, float(timeout_s)))
        if drainer.is_alive():
            logger.warning(
                "Memory sync did not drain within %.2fs; provider teardown deferred",
                max(0.0, float(timeout_s)),
            )

    @staticmethod
    def _close_providers(providers: list[Any]) -> None:
        for p in providers:
            close = getattr(p, "shutdown", None) or getattr(p, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    logger.debug("provider shutdown failed", exc_info=True)

    @classmethod
    def _drain_executor_and_close(
        cls, executor: ThreadPoolExecutor, providers: list[Any]
    ) -> None:
        try:
            executor.shutdown(wait=True)
        except Exception:  # noqa: BLE001
            logger.debug("memory executor drain failed", exc_info=True)
        finally:
            cls._close_providers(providers)

    # -- Background dispatch (Hermes _submit_background) ---------------------

    def _submit_background(self, fn: Any) -> bool:
        """Enqueue once while OPEN; CLOSED managers drop late work."""
        with self._lock:
            if self._shutdown:
                return False
            executor = self._executor
            if executor is None:
                try:
                    from codedoggy.memory.daemon_pool import (
                        DaemonThreadPoolExecutor,
                    )

                    executor = DaemonThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="mem-sync"
                    )
                    self._executor = executor
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to create memory sync executor: %s", e)
                    return False
            try:
                # Submit while holding the lifecycle lock so shutdown cannot
                # interleave and force an inline replay of the side effect.
                executor.submit(fn)
                return True
            except RuntimeError:
                logger.debug("Memory background submission rejected", exc_info=True)
                return False

    def _submit(self, fn: Any) -> None:
        self._submit_background(fn)

    def _get_executor(self) -> ThreadPoolExecutor | None:
        with self._lock:
            if self._shutdown:
                return None
            if self._executor is None:
                try:
                    from codedoggy.memory.daemon_pool import (
                        DaemonThreadPoolExecutor,
                    )

                    self._executor = DaemonThreadPoolExecutor(
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
