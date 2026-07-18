"""Monitor token-bucket rate limiter — source port from Grok.

Ported from:
  grok-build/crates/codegen/xai-grok-tools/src/implementations/grok_build/monitor/rate_limiter.rs

Function map:
  TokenBucket / try_consume
  SuppressionTracker / process
  MonitorRateLimiter / process_event / is_killed
  RateLimitOutcome variants + notice strings
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from codedoggy.tools.grok_build.monitor_types import (
    AUTO_KILL_THRESHOLD_MS,
    DEFAULT_KILL_TOOL_NAME,
    RATE_LIMIT_REFILL_MS,
)


class RateLimitKind(Enum):
    Allowed = auto()
    Suppressed = auto()
    AutoKill = auto()


@dataclass
class RateLimitOutcome:
    kind: RateLimitKind
    catch_up_notice: Optional[str] = None
    message: Optional[str] = None


class TokenBucket:
    """Token bucket: starts full; 1 token per refill_interval_ms."""

    def __init__(self, capacity: int, refill_interval_ms: int) -> None:
        self.capacity = capacity
        self.tokens = capacity
        self.refill_interval_s = refill_interval_ms / 1000.0
        self.last_refill = time.monotonic()

    def try_consume(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        if self.refill_interval_s > 0:
            refills = int(elapsed / self.refill_interval_s)
            if refills > 0:
                self.tokens = min(self.capacity, self.tokens + refills)
                self.last_refill += self.refill_interval_s * refills
        if self.tokens > 0:
            self.tokens -= 1
            return True
        return False


@dataclass
class SuppressionTracker:
    suppressed_count: int = 0
    last_suppression: Optional[float] = None
    suppression_start: Optional[float] = None
    killed: bool = False
    kill_tool_name: str = ""

    def with_kill_tool_name(self, name: str) -> "SuppressionTracker":
        self.kill_tool_name = name
        return self

    def process(self, token_available: bool, _description: str) -> RateLimitOutcome:
        if self.killed:
            return RateLimitOutcome(kind=RateLimitKind.Suppressed)

        now = time.monotonic()
        if token_available:
            catch_up: Optional[str] = None
            if self.suppressed_count > 0:
                kill_name = self.kill_tool_name or DEFAULT_KILL_TOOL_NAME
                catch_up = (
                    f"[{self.suppressed_count} events suppressed -- output rate too high. "
                    f"Consider using {kill_name} to restart this monitor "
                    f"with a more selective filter.]"
                )
                self.suppressed_count = 0
                if (
                    self.last_suppression is not None
                    and (now - self.last_suppression)
                    > (RATE_LIMIT_REFILL_MS * 3) / 1000.0
                ):
                    self.suppression_start = None
            return RateLimitOutcome(
                kind=RateLimitKind.Allowed, catch_up_notice=catch_up
            )

        self.suppressed_count += 1
        self.last_suppression = now
        if self.suppression_start is None:
            self.suppression_start = now

        if self.suppression_start is not None:
            elapsed = now - self.suppression_start
            if elapsed > AUTO_KILL_THRESHOLD_MS / 1000.0:
                self.killed = True
                secs = int(elapsed)
                return RateLimitOutcome(
                    kind=RateLimitKind.AutoKill,
                    message=(
                        f"[Monitor stopped -- your script produced too much output "
                        f"({self.suppressed_count} events suppressed over {secs}s). "
                        f"Write a new monitor command that filters more aggressively -- "
                        f"pipe through grep --line-buffered, awk, or a wrapper script "
                        f"that only emits the specific events you need.]"
                    ),
                )
        return RateLimitOutcome(kind=RateLimitKind.Suppressed)


@dataclass
class MonitorRateLimiter:
    bucket: TokenBucket
    suppression: SuppressionTracker = field(default_factory=SuppressionTracker)

    @classmethod
    def new(cls, capacity: int, refill_interval_ms: int) -> "MonitorRateLimiter":
        return cls(bucket=TokenBucket(capacity, refill_interval_ms))

    def with_kill_tool_name(self, name: str) -> "MonitorRateLimiter":
        self.suppression = self.suppression.with_kill_tool_name(name)
        return self

    def process_event(self, description: str) -> RateLimitOutcome:
        available = self.bucket.try_consume()
        return self.suppression.process(available, description)

    def is_killed(self) -> bool:
        return self.suppression.killed
