# Architecture

## 核心设计（母体生长）

> **以 Grok 为母体继续生长，用 Hermes 记忆、Graph、MAIN 强并行倾向（非自动编排）改造并增强整个系统。**

这是 CodeDoggy 的**唯一上位设计**，不是并列融合，也不是另起基座。

| 角色 | 是什么 | 不是什么 |
|------|--------|----------|
| **Grok** | **母体**：运行时协议、上下文、工具权限、采样、状态不变量、编排循环 | 可被替换的“其中一块” |
| **Hermes 记忆** | 在母体上**增强**召回 / 持久化 / provider 生命周期 | 改写 system/message/tool 协议的第二 runtime |
| **Parallel MAIN** | 主 agent **强并行倾向**（prompt + 可选工具）；MAIN 自己决定何时派工 | runtime 替 MAIN 自动拆任务/自动并行 |
| **Graph** | 挂在 Grok 读能力上的**代码导航** | 另起索引/权限/生命周期 |

**Shadow（写时软质检）已从产品路径移除** — 不再默认审计 mutation；`codedoggy.audit` 仅保留给遗留单测（`enable_audit=True`）。

**生长规则（必须同时满足）：**

1. 新能力先问：是否服从 Grok 的 context / tool / permission / transcript 边界？  
2. Hermes / Graph / Parallel 只**改造并增强**，不得削弱或分叉母体不变量。  
3. 冲突时 **Grok 胜出**；删除错误兼容层，不做双轨。  
4. `RuntimeKernel` 是薄适配层，不是第二母体。

```text
Grok Runtime（母体 · 唯一基座）
├─ Context / Sampling / Compaction
├─ Tool Runtime / Permission / Gate
├─ Turn loop / Orchestration（Agent 配置 · 两阶段工具 · 子会话）
├─ Hermes Memory Adapter     ← 增强记忆（USER 围栏召回，不升权 SYSTEM）
├─ MAIN parallel bias        ← prompt 倾向 + 可选工具；不替 MAIN 自动并行
└─ Graph as read capability  ← 增强导航（稳定 snapshot / 事件）
```

### Parallel MAIN (agent bias — not harness auto-parallel)

**Product rule:** MAIN has a **strong parallel tendency** (system prompt). The harness
**does not** auto-split work, auto-fan-out, or parallelize on MAIN's behalf. Parallelism
runs only when **MAIN chooses tools**.

- **Bias (prompt):** prefer dispatching independent slices to subagents rather than
  grinding everything serially yourself; you still own serial/critical path and final answer.
- **Tools (opt-in):** `parallel_tasks` / `spawn_subagent` / wait-get — invoked by MAIN only.
  - `wait=false` if MAIN wants to keep working after dispatch; `wait=true` if MAIN wants to join in-tool.
- **Types:** `explore` · `plan` · `general-purpose` (available workers, not auto-picked).
- **Coordinator:** `SubagentCoordinator` pool executes what MAIN spawned (`spawn_many` / `wait_all` helpers for the tool path).
- **Not in scope:** runtime forced fan-out / auto task-split as “product parallel”.
- **Separate (Grok-aligned):** batch **tool execute** uses path-lock parallel dispatch
  (`execute_approved_batch`) for tools MAIN already emitted — engine efficiency, not
  auto-decomposition.

### Release / audit boundaries

What is **DONE** (P0 / major P1 + attack tests), what is **GLUE**, and what is
**DEFERRED** lives only in:

→ **[`docs/release-boundaries.md`](docs/release-boundaries.md)**

Port map: [`docs/grok-source-map.md`](docs/grok-source-map.md) · Hermes seam:
[`docs/hermes-groke-seam.md`](docs/hermes-groke-seam.md). Do not claim features outside
those docs.

```
Session          outer lifecycle, cwd, extensions
  turn_runner    AgentTurnRunner → run_agent_loop
                 + live_messages + archive-at-create
                 + MemoryManager system/prefetch/sync
                 + rewind_context() session API
  tools          FinalizedToolset + WorkspacePolicy
                 (writes/shell gated; shell paths check_write)
                 + parallel_tasks / spawn_subagent
  context        ContextCompactor
                 (Grok pipeline + pre-tool async prefire + checkpoint rewind)
  memory         MemoryManager (curated + FTS warm cache + 1 external slot)
  graph          CodebaseGraph + code_nav (xai-codebase-graph API spirit)
  orchestration  Grok-aligned (codedoggy.orchestration)
                 Agent config · two-phase tools · subagent · plan mode · interject
                 · parallel fan-out (spawn_many / wait_all)
```

### Graph (GitHub-style navigation — from xai-codebase-graph)

Ported API surface (not a from-scratch design):

- `IndexBuilder.build(root)` → `ScopeGraphIndex`
- `Navigator.goto_definition` / `goto_references` (+ `_by_name`)
- `Location` / `NavigationResult` (1-indexed lines)
- `find_definitions_smart` / `find_references_smart` + same-language ranking
- aliases (import as), cache `.goto_index.json`
- tool: `code_nav` (definition | references | at_position | stats | reindex)

### GrokBuild default tool surface (ported)

Core: `read_file` · `search_replace` · `list_dir` · `grep` · `run_terminal_cmd`
Tasks: `get_task_output` · `kill_task` · `monitor` · `spawn_subagent` · `parallel_tasks` · `get_subagent_output`
Orchestration: `todo_write` · `update_goal` · `enter_plan_mode` · `exit_plan_mode` · `ask_user_question`
Web: `web_search` · `web_fetch` (SSRF)
Scheduler: `scheduler_create` · `scheduler_delete` · `scheduler_list`
Enhancements: `memory` · `session_search` · `code_nav`

Host wiring: `RuntimeKernel.task_manager` + `scheduler` injected via `tool_extra`.

Extract: tree-sitter queries (python / js / ts / rust / go), same capture
conventions as xai-codebase-graph `languages/*.rs`.

**Index pipeline (crate strengths):**

1. Collect files (git ls-files → walk)
2. **Phase 1 parallel** extract into `FileSymbols` (thread pool, chunked)
3. **Phase 2 sequential** merge into one `ScopeGraphIndex` (bounded batches)
4. `with_threads(N)` real (default CPU−1)
5. **IndexManager** FileEvent Created/Modified/Removed/Renamed → reindex_file
6. `query_version` + JSON cache; mutation → mark_dirty / FileEvent
7. **WorkspaceWatcher** (watchdog) → debounced FileEvents

**Hard deps:** tree-sitter + grammars + watchdog (no extract/watch fallback path).

| 项 | 状态 |
|----|------|
| SGIX `.goto_index.bin` 字节布局 | **不适合** Python 复刻；JSON + query_version 等价语义 |

## Turn loop (Grok SessionActor spirit)

```
user prompt
  → seed: SYSTEM(+MEMORY freeze) + prior live (non-system) + USER
  → Hermes seam: curated/provider static → SYSTEM; prefetch fence → sample-time USER only
  → loop:
       drain interjection_buffer → USER format_interjection (Grok xai-interjection-core)
       ContextCompactor.ensure
       sample(messages, tools)
       archive ASSISTANT
       if no tool_calls → done
       else Grok three-phase tools (tool_calls / tool_dispatch spirit):
         Phase1 prepare ALL (sequential): schema · pre_tool_use · plan gate · policy
           HookDeny → soft observation (turn continues)
           PermissionReject / PlanReject → hard-stop *remaining prepares* (earlier approved still run)
         Phase2 execute approved: path-lock parallel
           same file_path/path/target_file → mutex; else concurrent
           shell / apply_patch often no path key → no lock (GLUE)
         Phase3 writeback in *model emission order* + after_tool / after_mutation
           hooks optional (product default: no Shadow); abort stops *next sample*, not un-run phase2
       exit: completed | max_turns | cancelled | permission_reject | aborted | error
  → carry live_messages into next handle_prompt
```

- `max_turns`: sampling rounds per prompt; `None` = unlimited
- Tool errors / soft denies become observations (do not crash the loop)
- **Hard** prepare outcomes cancel remainder of *prepare*; already-approved still execute in phase 2
- **Shadow removed** from product path (`enable_audit=False`); `after_mutation` only if host wires legacy hooks
- **Live** window may be pruned; **archive** is create-time full fidelity

### Two “parallel” layers (do not confuse)

| Layer | Who decides | What runs parallel |
|-------|-------------|--------------------|
| **MAIN multi-agent bias** | **MAIN** (prompt + tools it chooses) | Subagent children when MAIN calls `spawn_subagent` / `parallel_tasks` |
| **Batch tool dispatch** | Engine (Grok-aligned) | Tool *executions* for tool_calls MAIN already emitted |

Harness **never** auto-splits user work into subagents. Batch parallel only speeds tools MAIN already requested.

## Orchestration (Grok port — `codedoggy.orchestration`)

Faithful map from grok-build (not a re-design):

| Grok | CodeDoggy |
|------|-----------|
| SessionActor | RuntimeKernel + turn loop |
| Agent (config package) | `AgentDefinition` / `Agent` — not the loop |
| ToolLoop + prepare/execute | `tool_pipeline.execute_tool_calls_two_phase` |
| path lock same-file | `path_lock.lock_path_for_args` |
| PlanModeTracker | `SessionModeState` + `plan_mode_edit_gate` |
| Interjection / prompt queue | `InterjectionBuffer` / `PromptQueue` |
| SubagentCoordinator | `SubagentCoordinator` + child `run_agent_loop` |
| CapabilityMode | `read-only` / `read-write` / `execute` / `all` |
| Built-ins explore / plan / general-purpose | `BUILTIN_AGENTS` + tools `spawn_subagent` / `parallel_tasks` / `get_subagent_output` |

- **Agent ≠ loop:** definition = tools whitelist + capability + prompt body
- **Subagent = full child session**, independent context; summary fold-back only
- **Plan mode hard gate** independent of yolo / auto-approve (only plan file edits)
- Session API: `session.interject`, `session.enter_plan_mode`, `session.exit_plan_mode`

## Context (Grok foundation pipeline)

```
before each sample → bind tools_reserve → ContextCompactor.ensure:
  0. suppress gate
  1. prune oversized tool results (P0 footers kept)
  2. under pressure: prune_retained
  3. memory_flush (soft)
  4. if over context_window * threshold_percent / 100 → fold middle
       commit gate: reject if no token savings (never replace with worse window)
  5. system / MEMORY never dropped
  on API context overflow → compact-and-resubmit (not silent fail)
  awaiting_real_usage: max 3 ensure cycles then fall back to estimates
```

- **Authority:** `context_window` (model / `CODEDOGGY_CONTEXT_WINDOW`), not a
  freestanding 30k char default
- Reserves: `completion_reserve` + per-sample `tools_reserve` (schema tokens)
- Env: `CODEDOGGY_CONTEXT_WINDOW`, `CODEDOGGY_COMPLETION_RESERVE`,
  `CODEDOGGY_CONTEXT_THRESHOLD_PERCENT` (default 85 Grok-like),
  `CODEDOGGY_CONTEXT_TARGET_PERCENT`, `CODEDOGGY_COMPACTION_MODE`, …
- Summarizer sees full middle sketch (large cap), not a tiny head
- Hermes (source: `C:\\Ai\\hermes-agent` — do not invent):
  - **Seam owner:** ``memory/hermes_seam.py`` — only lifecycle entry for
    bind / system block / prefetch fence / turn begin·end / pre_compress /
    rewound / session boundary / close (runner · kernel · compactor call seam)
  - curated MEMORY/USER frozen snapshot → system (tools/memory_tool.py)
  - load-time threat scan → ``[BLOCKED: …]`` in snapshot only; live keeps raw
  - external drift: round-trip OR entry > store limit → refuse + ``.bak`` (#26045)
  - consolidation fail cap → terminal ``done: true`` (#42405)
  - MemoryManager: one external provider; prefetch/sync background; tool routing;
    ``flush_pending`` / ``shutdown_all`` drain (memory_manager.py)
  - ``commit_session_boundary`` (seam) → ``commit_session_boundary_async``:
    on_session_end → on_session_switch (#16454)
  - plugin discovery: ``plugins/memory/`` + ``$CODEDOGGY_HOME/plugins/``
    (``CODEDOGGY_MEMORY_PROVIDER``); Hermes ``load_memory_provider`` surface
  - threat_patterns: Hermes scopes + NFKC + invisible unicode (threat_patterns.py)
  - prefetch → ``build_memory_context_block`` ``<memory-context>`` fence
  - inject into **current user message at sample time only**
    (conversation_loop.py / ``sample_messages_with_memory``) — not SYSTEM, not archived
  - store redacts secrets on write; FTS scoped by cwd/roles (user+assistant)
  - Session/Kernel ``new_session()`` rotates id + memory boundary via seam
  - ``on_pre_compress`` before fold (feeds summarizer; providers may extract)
  - ``load_on_disk_store()`` for CLI without live agent
  - bundled example external: ``CODEDOGGY_MEMORY_PROVIDER=notes``
  - docs: ``docs/hermes-groke-seam.md``
- Shell: scrub known API keys from child env; mutations recorded even on
  non-zero exit (Shadow partial-write path)
- Graph: incremental events mark dirty; real `respect_gitignore`; cache
  format version; reindex rebinds watcher manager
- Shadow P0: optional soft restore from mutation `before`
  (`CODEDOGGY_SHADOW_RESTORE`, default on) — best-effort, not full sandbox TX
- Busy `handle_prompt` → Grok interject queue (soft result, no crash)
- seed prior transcript: sanitize tool pairs before next sample
- **Port rule:** Grok from `C:\\Ai\\grok-build` only; wrong inventions **deleted**.
  Custom code only as **CodeDoggy glue** when Rust stack unportable.
  Map: ``docs/grok-source-map.md``
- Interjection: **ported** ``xai-interjection-core``; safe-point drain only
- Plan gate: plan file only; Goal: session flag + ``update_goal`` (no invent tool allowlist)
- Subagent: isolation/resume **contract** ``task.rs`` (completed-only resume)
- Worktree/merge: **glue** minimal git worktree (not full ``xai-fast-worktree``)
- Stream deltas: **glue** optional ``on_sample_delta`` (no mid-stream interject)
- Usage compaction: prefer API ``prompt_tokens`` when present

## Model layer

```
ModelConfig { provider, model, base_url, api_key, temperature, … }
  → register_provider(name, factory)   # Hermes-style registry
  → create_client(config)              # Grok-style config → client
  → ChatClient.complete(messages)
  → ChatSampler (turn.Sampler)
```

- Stock providers: `ollama` (default), `openai_compat` / `openai` / `custom`
- Bootstrap: `build_session()` wires `ChatSampler(main)` + tools + parallel coordinator
- Env main: `CODEDOGGY_PROVIDER`, `CODEDOGGY_MODEL`, `CODEDOGGY_BASE_URL`, `CODEDOGGY_API_KEY`
- Ollama default: `http://127.0.0.1:11434/v1` + model `qwen3:8b`
- Transport: OpenAI-compatible `chat/completions` (stdlib HTTP)
- Legacy: `CODEDOGGY_AUDIT_*` / dual profiles still exist for unused audit package tests

## Shadow 影子 (REMOVED from product path)

Package `codedoggy.audit` remains importable for legacy unit tests only.
Product sessions: `build_session(..., enable_audit=False)` (default). No
resident mutation auditor, no soft restore in the live agent path.

### Historical notes (legacy package only; not product)

<details><summary>Former Shadow design (do not re-enable by default)</summary>

## Shadow 影子 (write-time soft review — not a normal audit)

```
mutation (search_replace first-hand before/after)
  → MutationTrajectory (session write log)
  → MemorySelector (Hermes curated + FTS)
  → ShadowAuditor / ModelAuditor.review (model brain)
  → pass: silent | fail: soft observation footnote (rethink)
```

- **Name:** former product **Shadow / 影子**; code package `codedoggy.audit` (legacy)
- **Differs from normal audit:** in-loop, per mutation, soft only, no repo report
- Session **goal** = intent anchor (`session.goal` / `set_goal`)
- Shadow **must not write** workspace
- **P0 (`critical`)**: immediate red card on tool observation; prune/fold preserve
- **Non-P0**: buffered → turn end (`metadata.shadow_deferred`)
- Handle: `SessionExtensions.audit` (historical field name)
- Aliases: `ShadowAuditor`, `ShadowHooks`, `ShadowServices`

</details>

## Memory (Hermes-style: curated + big session store)

### Curated (always-on, small)

Bounded notes on disk under `{CODEDOGGY_HOME}/memories/` (default `~/.codedoggy/memories/`):

| File | Target | Role |
|------|--------|------|
| MEMORY.md | `memory` | agent notes (env, conventions, lessons) |
| USER.md | `user` | user profile (prefs, style) |

- Entries joined by `§`; char budgets (default 2200 / 1375)
- **Frozen snapshot** at `load_from_disk()` → system prompt; mid-session
  `memory` tool writes hit disk immediately but do **not** change the freeze
  until `refresh_system_prompt_snapshot()` (called after successful
  pre-compaction `memory_flush`) or next load
- Live path: `prefer_frozen=False` → `live_system_prompt_blocks()`
- Tool: `memory` (add / replace / remove / batch)
- Bind `MemoryStore` on `SessionExtensions.memory` so the turn loop injects it

### Big session store (on-demand, unlimited)

- SQLite `~/.codedoggy/state.db` with **FTS5** (LIKE fallback)
- Every turn transcript persisted via `AgentTurnRunner`
- Tool: `session_search` — discovery / scroll / read / browse
- Selector: `HermesMemorySelector` = curated blocks + FTS session hits

| Layer | Capacity | When loaded |
|-------|----------|-------------|
| MEMORY.md / USER.md | ~chars budget | every prompt (frozen) |
| Session FTS | unlimited | on demand (tool + optional select) |

## Tools

- Registry: `register` → `finalize` → `FinalizedToolset`
- Packs: `register_tool_pack` before first builder
- Qualified ids: `Doggy:short_id` (e.g. `Doggy:read_file`)
- Builtins: `read_file`, `search_replace`, `list_dir`, `grep`, `run_terminal_cmd`,
  `memory`, `session_search`, `code_nav`, `spawn_subagent`, `parallel_tasks`, `get_subagent_output`
- Defaults: `tools/defaults.py`
