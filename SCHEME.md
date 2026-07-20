# Architecture

CodeDoggy 的运行时结构说明。实现以本仓库源码为准。

## 上位规则

| 角色 | 职责 |
|------|------|
| **MAIN** | 主 agent：拆任务、派工、汇总；runtime 不替它自动并行 |
| **Turn loop** | 采样、两阶段工具、压缩、插话、plan/goal |
| **Memory** | curated 笔记 + 会话 FTS；prefetch 进当前 USER，不升权 SYSTEM |
| **Graph** | 代码导航（定义/引用），挂在读能力上 |
| **Connection** | 登录所选 provider 的凭证与 base_url；聊天与媒体工具共用 |

```text
RuntimeKernel
├─ Context / Sampling / Compaction
├─ Tool Runtime / Permission / Gate
├─ Turn loop / Orchestration（Agent · 两阶段工具 · 子会话）
├─ Memory Manager（curated + FTS + provider 插件）
├─ MAIN parallel bias（prompt + 可选工具；不自动 fan-out）
└─ Graph（code_nav）
```

### Parallel MAIN（agent 倾向，非 harness 自动并行）

- **Bias（prompt）：** 能拆的独立切片优先派子 agent；关键路径与最终答案仍由 MAIN 负责。
- **Tools（opt-in）：** `parallel_tasks` / `spawn_subagent` / wait-get — 仅 MAIN 调用。
- **Types：** `explore` · `plan` · `general-purpose`。
- **Coordinator：** `SubagentCoordinator` 只执行 MAIN 已经 spawn 的工作。
- **另一层并行：** 同一轮 `tool_calls` 的 batch 执行可按 path-lock 并行（引擎效率，不是自动拆任务）。

### Release / audit

完成度与边界见 [`docs/release-boundaries.md`](docs/release-boundaries.md)。

```
Session          outer lifecycle, cwd, extensions
  turn_runner    AgentTurnRunner → run_agent_loop
                 + live_messages + archive
                 + MemoryManager system/prefetch/sync
                 + rewind_context()
  tools          FinalizedToolset + WorkspacePolicy
                 + parallel_tasks / spawn_subagent
  context        ContextCompactor（预算 / fold / prefire / rewind）
  memory         MemoryManager（curated + FTS + external slot）
  graph          CodebaseGraph + code_nav
  orchestration  Agent · two-phase tools · subagent · plan · interject
  connection     ActiveConnection（model/provider 真源）
```

### Graph

- `IndexBuilder.build(root)` → `ScopeGraphIndex`
- `Navigator.goto_definition` / `goto_references`
- `Location` / `NavigationResult`（1-based lines）
- 工具：`code_nav`（definition | references | at_position | stats | reindex）
- 索引：tree-sitter（python / js / ts / rust / go）+ watchdog 增量

### Default tool surface

Core: `read_file` · `search_replace` · `list_dir` · `grep` · `run_terminal_cmd`  
Tasks: `get_task_output` · `kill_task` · `monitor` · `spawn_subagent` · `parallel_tasks` · `get_subagent_output`  
Orchestration: `todo_write` · `update_goal` · `record_plan` · `enter_plan_mode` · `exit_plan_mode` · `ask_user_question`  
Web: `web_search` · `web_fetch`  
Scheduler: `scheduler_create` · `scheduler_delete` · `scheduler_list`  
Extras: `memory` · `session_search` · `code_nav` · `image_gen` · `image_edit` · video tools  

Host wiring: `RuntimeKernel.task_manager` + `scheduler` via `tool_extra`。

## Turn loop

```
user prompt
  → seed: SYSTEM(+MEMORY freeze) + prior live + USER
  → memory: curated/provider → SYSTEM；prefetch fence → sample-time USER only
  → loop:
       drain interjection_buffer → USER
       ContextCompactor.ensure
       sample(messages, tools)
       archive ASSISTANT
       if no tool_calls → incomplete-work gate
         (todos / subagents / bg shell tasks / unmet plan-first)
         else → done
       else three-phase tools:
         Phase1 prepare ALL (schema · pre_tool_use · plan-first · plan gate · policy)
         Phase2 execute approved (path-lock parallel when safe)
         Phase3 writeback in emission order + after_tool / after_mutation
       exit: completed | max_turns | cancelled | permission_reject | aborted | error
  → carry live_messages into next handle_prompt
```

| Layer | Who decides | What runs parallel |
|-------|-------------|--------------------|
| MAIN multi-agent | MAIN（prompt + tools） | 子 agent |
| Batch tool dispatch | Engine | 已 emit 的 tool_calls |

## Orchestration (`codedoggy.orchestration`)

| Concept | Implementation |
|---------|----------------|
| Session spine | RuntimeKernel + turn loop |
| Agent config | `AgentDefinition` / `Agent` |
| Tool prepare/execute | `tool_pipeline.execute_tool_calls_two_phase` |
| Path lock | `path_lock.lock_path_for_args` |
| Plan mode | `SessionModeState` + `plan_mode_edit_gate` |
| Plan-first | go-steer `PlanFirstGate` + `record_plan` (mutate gated until recorded) |
| Interjection / queue | `InterjectionBuffer` / `PromptQueue` |
| Subagents | `SubagentCoordinator` + child `run_agent_loop` |
| Capability | `read-only` / `read-write` / `execute` / `all` |
| Built-ins | `BUILTIN_AGENTS` + spawn / parallel / get_output tools |

- Subagent = 完整子会话，独立 context；回传 summary
- Plan mode hard gate 独立于 auto-approve（仅 plan 文件可写）
- Session API: `session.interject`, `enter_plan_mode`, `exit_plan_mode`

## Context

```
before each sample → ContextCompactor.ensure:
  0. suppress gate
  1. under pressure: prune oversized + prune_retained
  2. memory_flush (soft; refreshes curated snapshot — does not append SYSTEM to live)
  3. if over usable_window * threshold → fold middle (commit only if token savings)
  4. system / MEMORY never dropped
  on API context overflow → compact-and-resubmit (threshold restore in finally)
```

Env: `CODEDOGGY_CONTEXT_WINDOW`, `CODEDOGGY_COMPLETION_RESERVE`,
`CODEDOGGY_CONTEXT_THRESHOLD_PERCENT`（默认 85）、`CODEDOGGY_CONTEXT_TARGET_PERCENT`、…

## Memory

### Curated（小、常驻）

`{CODEDOGGY_HOME}/memories/`（默认 `~/.codedoggy/memories/`）：

| File | Role |
|------|------|
| MEMORY.md | agent 笔记 |
| USER.md | 用户画像 |

- `§` 分隔；字符预算；load 时冻结进 SYSTEM
- 工具：`memory`（add / replace / remove / batch）

### Session store（大、按需）

- SQLite `~/.codedoggy/state.db` + FTS5
- 工具：`session_search`
- 生命周期入口：`memory/hermes_seam.py` + `memory/manager.py`（实现模块名，非外部产品依赖）
- Prefetch 围栏进 **当前 USER**（不进 SYSTEM、不进 archive）
- Provider 插件：`CODEDOGGY_MEMORY_PROVIDER`（例：`notes`）

## Model layer

```
ModelConfig { provider, model, base_url, api_key, … }
  → create_client(config)
  → ChatSampler
```

- Providers：`grok` / `claude` / `codex` / `openai` / `deepseek` / `ollama` / `custom` …
- Env：`CODEDOGGY_PROVIDER`, `CODEDOGGY_MODEL`, `CODEDOGGY_BASE_URL`, `CODEDOGGY_API_KEY`
- Aux（压缩摘要）：`CODEDOGGY_AUX_MODEL` / `CODEDOGGY_AUX_*`
- **ActiveConnection**：登录所选连接为聊天与 image/video/web_search 的凭证真源

## Tools

- Registry: `register` → `finalize` → `FinalizedToolset`
- Qualified ids: `Doggy:short_id`
- Defaults: `tools/defaults.py`
- Host bag: `docs/host-tool-extra.md`
- Tool contracts: `docs/tool-design.md` · `docs/tool-checklist.md`
