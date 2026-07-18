# Host `tool_extra` injection

Honest map of what mid-turn tools actually receive.
Source of truth: `RuntimeKernel.refresh_tool_extra` + each tool’s `ctx.extra` lookup.
Not a marketing list — missing keys fail soft or hard as coded.

## Flow

```
RuntimeKernel.refresh_tool_extra()
        │  rebuilds managed keys; preserves host keys
        ▼
kernel.tool_extra  ──►  AgentTurnRunner.run (refresh every turn)
        │
        ▼
run_agent_loop(..., tool_extra=…)  ──►  ToolCallContext.extra
```

- **Managed keys** are rewritten every refresh from kernel fields (when not `None`).
- **Host keys** (anything outside managed set) survive refresh if not overwritten.
- Bare unit tests may call tools with a hand-built `extra` bag; no kernel required.

Managed set (`_MANAGED_TOOL_EXTRA_KEYS`):

`kernel`, `memory_store`, `session_store`, `policy`, `memory_manager`, `graph`,
`audit`, `session_mode_state`, `interjection_buffer`, `subagent_coordinator`,
`subagent_run_fn`, `task_manager`, `scheduler`, `todo_state`.

---

## 1. Kernel-injected (managed)

| Key | Kernel field | Injected when | Used by (examples) |
|-----|--------------|---------------|--------------------|
| `kernel` | self | always | plan mode, update_goal, scheduler/tm resolve |
| `memory_store` | `memory` | bound | `memory_get`, `memory` (Doggy write) |
| `session_store` | `session_store` | bound | `session_search` |
| `policy` | `policy` | bound | gate, `use_tool` path checks, `code_nav` reindex |
| `memory_manager` | `memory_manager` | bound | Hermes provider tools |
| `graph` | `graph` | bound | `code_nav` only — **not** LSP |
| `audit` | `audit` | bound | audit hooks / P0 |
| `session_mode_state` | `session_mode_state` | bound | plan/goal gates |
| `interjection_buffer` | `interjection_buffer` | bound | mid-turn interject drain |
| `subagent_coordinator` | `subagent_coordinator` | bound | `spawn_subagent`, `get_*_output`, `kill_*`, `parallel_tasks` |
| `subagent_run_fn` | `subagent_run_fn` | bound | child turn runner |
| `task_manager` | `task_manager` | always after `__post_init__` (default `BackgroundTaskManager`) | bg shell, monitor, wait/kill/get task |
| `scheduler` | `scheduler` | always after `__post_init__` (default `Scheduler`) | `scheduler_*` |
| `todo_state` | `todo_state` | bound / lazy-filled by tool | `todo_write` |

`build_session` binds memory/policy/graph/subagent/audit and calls `refresh_tool_extra` after Hermes bind.

Runner defensive fill (if kernel missing a key): `memory_manager`, `memory_store`, `session_store`, `policy`, `graph`, plus optional `prefetch_user_block`.

**Lazy without kernel:** `ensure_task_manager` / `ensure_scheduler` create local instances into the bag.
Shell kill uses real Win32 Job Objects when assigned (`util/job_object.py`); scheduler tick is a light host poller, not a Grok Tokio actor.

---

## 2. Optional host keys (not managed)

Host or session UI injects these. Refresh **preserves** them.

| Key | Contract | Consumer |
|-----|----------|----------|
| `lsp_backend` | `.dispatch(args)` or `.run(args)` → str/dict | `lsp` |
| `memory_backend` | `.search(query, max_results=…, min_score=…)` → ranked hits | `memory_search` only |
| `mcp_dispatch` | `McpDispatch`：`callable(tool_name, tool_input)` | `use_tool` (**transport 仍 host**) |
| `mcp_tools` | `list[dict]` catalog (`name`, `description`, `parameters`/`input_schema`, …) | `search_tool`, schema prep；无 index 时自动建 BM25 |
| `mcp_tool_index` | Grok `ToolIndex` / `ToolSearchIndex`（`.search_snapshot` / `.list_server_summaries`；可选 `.get`） | `search_tool`；`ensure_mcp_tool_index` 从 catalog 生成 `ToolIndex(Bm25…)` |
| `mcp_servers` / `mcp_initialized` | server 列表 + 是否 ready | BM25 `is_ready` / system-reminder |
| `skills_registry` / `skill_paths` | `list[SkillInfo\|dict]` 或额外目录 | `skill` tool；否则扫 `.codedoggy/skills` / `.grok/skills` |
| `native_tool_correction` | Grok `UseToolParams` bool（默认 true） | `use_tool` 原生工具纠错 |
| `ask_user_fn` | `fn(list[question_dict])` → answers | `ask_user_question` |
| `plan_mode_consent_fn` | `fn() -> bool` | `enter_plan_mode` (decline → soft string) |
| `plan_mode_exit_fn` | host outcome hook | `exit_plan_mode` |
| `plan_file_path` / `plan_tool_hints` | plan path + tool name hints | enter/exit plan |
| `goal_ack_fn` | harness ack for `update_goal` | optional; else local ack |
| `shell_state` | cwd + env overlays across `run_terminal_command` | shell tools |
| `stream_sample` / `on_sample_delta` | streaming UI hooks | turn loop |
| `prefetch_user_block` | Hermes fence text | runner → loop |
| `writes_paused` | pause mutating tools | registry gate |

### MCP mutation envelope

If `mcp_dispatch` returns only a plain string, workspace side effects are not
recorded on the tool context. For write tools host **should** return structured
shapes (`mutations` / `mutation` / `mutated_paths` / `mutated_path`). See
`docs/grok-tool-surface.md`.

---

## 3. Missing behavior (no fake backends)

| Tool | Required extra | When missing |
|------|----------------|--------------|
| `lsp` | `lsp_backend` with `dispatch`/`run` | `ToolError` code `process_manager`: *“LSP tool is unavailable. Configure ~/.grok/lsp.json or \<cwd\>/.grok/lsp.json …”* |
| `memory_search` | `memory_backend.search` | Product `build_session` injects simple store backend when memory on. Soft Grok string only if backend missing. |
| `memory_get` | `memory_store` (`MemoryStore`) | Same soft experimental-memory string (not ToolError) |
| `memory` (Doggy) | `memory_store` | `ToolError` `memory_not_configured` |
| `session_search` | `session_store` | `ToolError` `session_store_not_configured` |
| `code_nav` | `graph` | `ToolError` `not_available` (graph — **not** LSP) |
| `search_tool` | `mcp_tool_index` or `mcp_tools` | Soft Grok note: *“No integration tools are configured. MCP servers are not connected.”* |
| `use_tool` | `mcp_dispatch` | `ToolError` `mcp_dispatch_missing` |
| `ask_user_question` | `ask_user_fn` | `MIGRATION_FALLBACK=True` → QuestionsSent soft text + `pending_user_questions` stash; no real UI wait |
| `spawn_subagent` / `parallel_tasks` | `subagent_coordinator` + `subagent_run_fn` | `ToolError` `missing_resource` |
| `enter_plan_mode` | `kernel` / `session_mode_state` optional | Still returns Entered prompt; mode gate may be absent |
| `exit_plan_mode` | plan path + optional exit_fn / kernel | Empty/ready plan strings; mode exit best-effort |
| `update_goal` | `goal_ack_fn` optional | Local ack path; no harness classifier |
| `task_manager` / `scheduler` | usually kernel-defaulted | Lazy create if bag empty |

**Soft** = model-visible string, tool “succeeds”.  
**Hard** = `ToolError` with code.  
Do not paper over missing host wiring with invented backends.

---

## 4. Forbidden inventions

Do **not** add or reintroduce:

| Invention | Why forbidden |
|-----------|----------------|
| **graph-as-LSP** | `code_nav` is graph; `lsp` needs real `lsp_backend`. No fallback between them. |
| **BM25 registry as Grok MCP index** | No in-tree `mcp_registry` BM25. Host owns `mcp_tool_index` / simple `mcp_tools` filter. |
| **Invented degrade / taskkill-as-Job** | Forbidden. Kill path is Grok protocol only: TerminateJobObject then child kill. Missing full actor → label **C**, do not invent a second mechanism. |
| Token-scored `MemoryStore` as `memory_search` | Search is only `memory_backend`; store is for get/write. |
| Mid-stream interjection interrupt as product | Deleted invention; interjections drain at safe points only. |

See also: `docs/grok-source-fidelity.md` “已删除的脑补”, `docs/release-boundaries.md` DEFERRED.

---

## 5. Host checklist (minimal)

```python
# Optional — only if you ship the capability
kernel.tool_extra["lsp_backend"] = my_lsp       # dispatch/run
kernel.tool_extra["memory_backend"] = my_mem    # search(...)
kernel.tool_extra["mcp_tools"] = catalog
kernel.tool_extra["mcp_tool_index"] = index     # optional BM25-like
kernel.tool_extra["mcp_dispatch"] = dispatch    # (name, input) -> str|dict
kernel.tool_extra["ask_user_fn"] = ask_ui
kernel.tool_extra["plan_mode_consent_fn"] = consent
kernel.tool_extra["plan_mode_exit_fn"] = on_exit
kernel.refresh_tool_extra()  # safe: host keys preserved
```

`task_manager` / `scheduler` / subagent / memory_store / graph are already filled when using `build_session` + bound extensions — inject only what the host actually implements.
