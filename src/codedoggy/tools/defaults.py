"""Hardcoded defaults for tool behavior.

Values mirror production tool defaults (timeouts, output caps, line limits).
"""

from __future__ import annotations

# ── shared output caps ───────────────────────────────────────────────
DEFAULT_TOOL_OUTPUT_BYTES: int = 40_000
DEFAULT_TOOL_OUTPUT_CHARS: int = 20_000  # shell / bash results

# ── read_file ────────────────────────────────────────────────────────
MAX_LINES_READ_DEFAULT: int = 1_000

# ── list_dir ─────────────────────────────────────────────────────────
LIST_DIR_MAX_OUTPUT_CHARS: int = 10_000
LIST_DIR_MAX_DEPTH: int = 3
LIST_DIR_MAX_GLOBAL_ITEMS: int = 100_000

# ── search_replace ───────────────────────────────────────────────────
PATH_COMPONENT_NAME_MAX: int = 255
SEARCH_REPLACE_INCLUDE_USER_EDIT_HINT: bool = True

# ── grep (mirror grok_build/grep/mod.rs) ─────────────────────────────
GREP_CONTENT_LINE_DEFAULT: int = 200
GREP_CONTENT_LINE_LIMIT: int = 2_000
GREP_FILE_COUNT_DEFAULT: int = 500
GREP_FILE_COUNT_LIMIT: int = 10_000
GREP_DEFAULT_MAX_CHARS_PER_LINE: int = 1_000
GREP_MAX_STDOUT_BYTES: int = 5_000_000
GREP_TIMEOUT_DEFAULT_SECS: int = 20
GREP_TIMEOUT_WSL_SECS: int = 60

# ── run_terminal_cmd (bash) ──────────────────────────────────────────
BASH_DEFAULT_TIMEOUT_MS: int = 120_000  # 2 minutes
BASH_DEFAULT_MAX_TIMEOUT_MS: int = 300_000  # 5 minutes schema max
BASH_DEFAULT_FOREGROUND_BLOCK_BUDGET_MS: int = 15_000
BASH_DEFAULT_TIMEOUT_SECS: float = 120.0
# Grok BashParams defaults (bash/mod.rs Default for BashParams):
# enabled_background=true, auto_background_on_timeout=false,
# allow_background_operator=true (reject & only when bg disabled or allow=false).
BASH_ENABLED_BACKGROUND: bool = True
BASH_AUTO_BACKGROUND_ON_TIMEOUT: bool = False
BASH_ALLOW_BACKGROUND_OPERATOR: bool = True
# Persist cwd (and env overlays) across run_terminal_cmd in tool_extra.shell_state.
BASH_PERSISTENT_SHELL_STATE: bool = True
# Background max lifetime (Grok BACKGROUND_MAX_RUNTIME = 10h)
BASH_BACKGROUND_MAX_RUNTIME_S: float = 36_000.0

# ── get_task_output (mirrors task_output_logic / Grok task_output/mod.rs) ─
TASK_OUTPUT_DEFAULT_WAIT_MS: int = 30_000
TASK_OUTPUT_MAX_WAIT_MS: int = 600_000  # 10 minutes
TASK_OUTPUT_MAX_MULTI_IDS: int = 20
