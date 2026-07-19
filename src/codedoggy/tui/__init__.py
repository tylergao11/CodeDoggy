"""Parallel task cockpit for CodeDoggy."""

from codedoggy.tui.app import run_tui
from codedoggy.tui.login_wizard import AuthWizard

__all__ = ["AuthWizard", "run_tui"]
