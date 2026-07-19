"""Shared model transport failures.

Cancellation is a first-class outcome, not an API error.  Keeping a distinct
exception lets the synchronous Python transports mirror Grok's per-request
``CancellationToken`` without accidentally committing a partial response.
"""

from __future__ import annotations


class ModelError(Exception):
    """Transport or API failure talking to a model endpoint."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ModelStreamCancelled(ModelError):
    """The host cancelled an in-flight model response stream."""

    def __init__(self, message: str = "model stream cancelled") -> None:
        super().__init__(message)
