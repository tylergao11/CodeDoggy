"""Host-side adapters that fill tool_extra for product sessions.

Tools stay pure: they only read ``ctx.extra``. This package wires optional
capabilities without inventing unrelated Grok backends (no graph-as-LSP and
no fake Job Object). The source-aligned MCP runtime is Session-owned and wired
by bootstrap rather than being a generic host adapter.

Main agent owns bootstrap/kernel injection. Parallel work lands adapters here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codedoggy.host.ask_user_cli import ask_user_cli, make_ask_user_fn
from codedoggy.host.memory_backend import build_memory_backend
from codedoggy.host.scheduler_runtime import start_scheduler_runtime
from codedoggy.host.scheduler_tick import fire_due, poll_due, run_tick_loop

if TYPE_CHECKING:
    from codedoggy.session.kernel import RuntimeKernel

__all__ = [
    "ask_user_cli",
    "build_memory_backend",
    "fire_due",
    "make_ask_user_fn",
    "poll_due",
    "run_tick_loop",
    "start_scheduler_runtime",
    "wire_default_host_extras",
]


def wire_default_host_extras(kernel: "RuntimeKernel", **opts: Any) -> dict[str, Any]:
    """Attach optional host adapters onto ``kernel.tool_extra``.

    Safe to call multiple times. Never overwrites keys already set by the host.
    Returns a dict of keys newly set (for tests/logging).

    Options:
      enable_memory_backend: bool = True when memory store present
      enable_ask_user_cli: bool = False (opt-in; avoid hanging tests)
      enable_scheduler_tick: bool = False (opt-in; does NOT auto-start a thread —
          only attaches a ``scheduler_tick`` handle with helpers for the host)
    """
    enable_memory_backend = opts.get("enable_memory_backend", True)
    enable_ask_user_cli = opts.get("enable_ask_user_cli", False)
    enable_scheduler_tick = opts.get("enable_scheduler_tick", False)
    start_scheduler_thread = opts.get("start_scheduler_thread", True)
    submit_prompt = opts.get("submit_prompt")

    if kernel.tool_extra is None:
        kernel.tool_extra = {}
    extra = kernel.tool_extra
    set_keys: list[str] = []

    if enable_memory_backend and "memory_backend" not in extra:
        store = getattr(kernel, "memory", None) or extra.get("memory_store")
        if store is not None:
            try:
                backend = build_memory_backend(store)
            except Exception:  # noqa: BLE001
                backend = None
            if backend is not None:
                extra["memory_backend"] = backend
                set_keys.append("memory_backend")

    if enable_ask_user_cli and "ask_user_fn" not in extra:
        try:
            fn = make_ask_user_fn()
        except Exception:  # noqa: BLE001
            fn = None
        if callable(fn):
            extra["ask_user_fn"] = fn
            set_keys.append("ask_user_fn")

    if enable_scheduler_tick and "scheduler_runtime" not in extra:
        try:
            handle = start_scheduler_runtime(
                kernel,
                start_thread=bool(start_scheduler_thread),
                submit_prompt=submit_prompt if callable(submit_prompt) else None,
            )
        except Exception:  # noqa: BLE001
            handle = None
        if handle is not None:
            set_keys.append("scheduler_runtime")
            if "scheduler_tick" in extra:
                set_keys.append("scheduler_tick")

    return {k: extra[k] for k in set_keys if k in extra}
