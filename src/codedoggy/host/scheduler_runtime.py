"""Product host: run scheduler tick and inject fired prompts into the agent.

NOT a Grok Tokio actor. Uses ``scheduler_tick.fire_due`` + interjection/prompt queue.

Main path:
  handle = start_scheduler_runtime(kernel)
  … session runs …
  handle.stop()  # kernel.close does this
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable

from codedoggy.host.scheduler_tick import FireResult, fire_due, run_tick_loop

logger = logging.getLogger(__name__)


@dataclass
class SchedulerRuntimeHandle:
    """Owns the optional background poll thread."""

    stop_event: threading.Event
    thread: threading.Thread | None
    scheduler: Any
    on_fire: Callable[[list[FireResult]], None] | None = None

    def stop(self, timeout: float = 2.0) -> None:
        self.stop_event.set()
        t = self.thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self.thread = None

    def poll_once(self) -> list[FireResult]:
        """Synchronous poll (tests / idle loop) — also injects via on_fire."""
        results = fire_due(self.scheduler)
        if results and self.on_fire is not None:
            try:
                self.on_fire(results)
            except Exception:  # noqa: BLE001
                logger.exception("scheduler poll_once on_fire failed")
        return results


def _default_on_fire(kernel: Any) -> Callable[[list[FireResult]], None]:
    def on_fire(results: list[FireResult]) -> None:
        if not results:
            return
        ib = getattr(kernel, "interjection_buffer", None)
        pq = getattr(kernel, "prompt_queue", None)
        for r in results:
            text = r.prompt if isinstance(r.prompt, str) else str(r.prompt)
            # Prefer mid-turn interjection when a turn can drain it
            if ib is not None and hasattr(ib, "push"):
                try:
                    ib.push(text, prompt_id=r.id)
                    continue
                except Exception:  # noqa: BLE001
                    logger.debug("interjection push failed", exc_info=True)
            if pq is not None and hasattr(pq, "push"):
                try:
                    from codedoggy.orchestration.prompt_queue import PromptQueueItem

                    pq.push(PromptQueueItem(text=text, prompt_id=r.id))
                    continue
                except Exception:  # noqa: BLE001
                    logger.debug("prompt_queue push failed", exc_info=True)
            logger.info("scheduler fired task_id=%s (no inject channel)", r.id)

    return on_fire


def start_scheduler_runtime(
    kernel: Any,
    *,
    interval_s: float = 1.0,
    start_thread: bool = True,
) -> SchedulerRuntimeHandle | None:
    """Start host tick that injects due scheduler prompts into the kernel.

    Returns None if kernel has no scheduler.
    """
    sched = getattr(kernel, "scheduler", None)
    if sched is None:
        extra = getattr(kernel, "tool_extra", None) or {}
        sched = extra.get("scheduler")
    if sched is None:
        return None

    stop = threading.Event()
    on_fire = _default_on_fire(kernel)
    thread: threading.Thread | None = None
    if start_thread:
        thread = threading.Thread(
            target=run_tick_loop,
            args=(sched, on_fire, stop),
            kwargs={"interval_s": max(0.2, float(interval_s))},
            daemon=True,
            name="codedoggy-scheduler-tick",
        )
        thread.start()

    handle = SchedulerRuntimeHandle(
        stop_event=stop, thread=thread, scheduler=sched, on_fire=on_fire
    )
    # Stash for tool_extra / close
    try:
        if kernel.tool_extra is None:
            kernel.tool_extra = {}
        kernel.tool_extra["scheduler_runtime"] = handle
        kernel.tool_extra["scheduler_tick"] = {
            "scheduler": sched,
            "fire_due": fire_due,
            "handle": handle,
        }
    except Exception:  # noqa: BLE001
        pass
    return handle
