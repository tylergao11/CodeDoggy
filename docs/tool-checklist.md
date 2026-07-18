# Tool-class details checklist

Reminder list for model-facing tool contracts. Not orchestration (turn/session).

## Done (baseline)

- [x] Registry: register / pack / finalize / dispatch by client name
- [x] Namespace `Doggy:*` qualified ids
- [x] Core builtins: `read_file`, `search_replace`, `list_dir`, `grep`, `run_terminal_cmd`
- [x] Central defaults in `tools/defaults.py`
- [x] Static tool `description` + `parameters_schema` (param-level descriptions)
- [x] `name_override` / `description_override` on config
- [x] `tool_definitions()` → model ToolSpec list
- [x] Finite defaults: line window, grep head_limit, shell timeout/output caps
- [x] read_file: `N→` prefix rules, offset/limit clamp, binary reject, UTF-8 lossy, token-size guard
- [x] read_file description honest (text only; no false PDF/image claims)
- [x] search_replace: exact match, replace_all, empty→create, actionable errors, lossy UTF-8
- [x] list_dir: tree format, hide dots, char budget; description matches depth/budget behavior
- [x] grep: rg preferred, python fallback, head caps, timeout, workspace_result card + Found N
- [x] grep: without rg, reject context/`type`/multiline/complex glob (no silent drop)
- [x] search_replace: CRLF match (LF old_string) + preserve CRLF on write; bytes write
- [x] run_terminal_cmd: whole-command argv, shell detect, UTF-8 env; FG timeout 0→default 120s
- [x] run_terminal_cmd: `description` required; no is_background in schema; trailing `&` rejected
- [x] run_terminal_cmd: timeout kills process tree (taskkill /T / process group)
- [x] list_dir description honest: depth 3, char budget, no gitignore claim
- [x] read_file: no phantom empty line after trailing newline; offset=-1 = last content line
- [x] grep: rg exit 2 → error (not false “No matches”)
- [x] Highest privilege (no permission/sandbox) by product choice for now

## Description / listing (tool-class)

- [ ] Description **templates** with cross-tool name resolution (`read` name in `search_replace` text after override)
- [ ] Conditional description sections unified via renderer (unix-utils / semicolon already partial in shell desc)
- [ ] Schema property description re-render after param renames
- [ ] Per-turn `should_list` / dynamic description when mode changes
- [ ] Shared description test: no raw template placeholders leak to model
- [ ] Truncation / default numbers injected into descriptions from `defaults.py` consistently

## Observation / output contract

- [x] grep workspace_result + Found N / truncated footers
- [ ] Structured vs plain-text results (when to return machine JSON vs prose)
- [ ] Consistent error **codes** + model-facing messages matrix per tool
- [ ] Truncation markers + “how to get the rest” for all tools
- [ ] Soft vs hard output limits (bytes vs chars) documented per tool

## Edit / read ACI

- [x] search_replace: CRLF normalize match + preserve endings
- [ ] search_replace: unicode confusable / normalized fallback (optional flag)
- [ ] search_replace: edit snippet in success payload (line context after edit)
- [ ] read_file: PDF / image / pptx (optional future; not claimed in description)
- [ ] list_dir: gitignore respect, large-dir extension summary (BFS collapse)

## Shell / host

- [ ] Background tasks (`is_background`, task_id, kill/output tools)
- [ ] Auto-background on long FG wait
- [x] Trailing `&` rejection (foreground-only mode)
- [x] Timeout process-tree / process-group kill (honest description)
- [ ] Persistent shell state (cwd/env across commands) if desired
- [x] Exit-code formatting: `exit: N` / `exit: killed (timeout)`

## Grep / search

- [ ] Full output modes: content / files_with_matches / count
- [x] Without rg: fail closed on flags Python cannot honor (not silent ignore)
- [ ] Multiline + context parity on python fallback (optional future; currently reject)
- [x] Workspace result wrappers / “Found N matching lines”
- [ ] Deny globs / ignore wiring

## Cross-tool product surface (still tool-class, not turn)

- [ ] Tool kind filtering for capability modes (read-only session etc.)
- [ ] Per-tool params injection (BashParams / GrepParams style)
- [ ] Behavior versions / presets for stable contracts
- [ ] Streaming progress for long tools (shell/grep)
- [ ] MCP dynamic tools (same ToolSpec surface)

## Explicitly NOT tool-class (remind separately)

- Turn loop sample → tool_calls → writeback
- Parallel batching / same-file locks
- Permission / sandbox / hooks
- Context compression
- Session queue / cancel integration with tools

## When to revisit

Before claiming “tools are done”, walk this file top to bottom.
When adding a new builtin: tick observation + description + defaults rows for that tool.
