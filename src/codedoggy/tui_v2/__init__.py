"""Grok pager port (Python). See PORT.md.

Default product TUI for ``doggy``. Set ``CODEDOGGY_TUI=legacy`` for the old
task-card cockpit.
"""

from codedoggy.tui_v2.app import run_tui

__all__ = ["run_tui"]
