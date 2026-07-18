"""Session: workspace-bound conversation lifecycle."""

from codedoggy.session.config import SessionConfig
from codedoggy.session.extensions import SessionExtensions
from codedoggy.session.session import Session
from codedoggy.session.types import (
    SessionId,
    SessionPhase,
    TurnRequest,
    TurnResult,
    TurnStatus,
)

__all__ = [
    "Session",
    "SessionConfig",
    "SessionExtensions",
    "SessionId",
    "SessionPhase",
    "TurnRequest",
    "TurnResult",
    "TurnStatus",
]
