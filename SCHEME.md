# Architecture

CodeDoggy 的运行时结构说明。实现以本仓库源码为准。

**整合定位：** **GrokBuild** 基础工程 + **Hermes** 记忆，在 CodeDoggy 壳上融合。
TUI 默认产品面为 **Grok 会话壳**（`codedoggy.tui_v2`，对照 `D:\grok-build` pager）；
旧任务卡驾驶舱仅 `CODEDOGGY_TUI=legacy`。品牌 / Ctrl+L / 粘贴图片为 Doggy 例外。

## 融合硬规则（禁止混用）

| 域 | 唯一真源 | 默认产品面 | 禁止 |
|----|----------|------------|------|
| **记忆** | Hermes | `memory` + `session_search` + 冻结 MEMORY/USER 注入 SYSTEM；prefetch 围栏进当前 USER | 默认暴露 Grok `memory_search` / `memory_get`；默认注入 `memory_backend` |
| **基础工程** | GrokBuild | plan 四态、todo、两阶段工具、edit gate、subagent 契约 | 自创 plan 语义；用 Hermes 替代 plan/todo |

**Plan 文件（Grok）：** `{cwd}/.grok/sessions/<session_id>/plan.md`  
（工具无 session 时的 fallback 才是 `cwd/.grok/plan.md`。）

**Edit gate（Grok）：** 仅 `Active`；`ExitPending` 不挡写。

**Grok 读记忆工具：** 仅 `register_optional_grok_memory_tools()` + `CODEDOGGY_GROK_MEMORY_BACKEND=1` 实验路径。

## 上位规则

| 角色 | 职责 |
|------|------|
| **MAIN** | 主 agent：拆任务、派工、汇总；runtime 不替它自动并行 |
| **Turn loop** | 采样、两阶段工具、压缩、插话、plan/goal（GrokBuild） |
| **Memory** | Hermes：curated 笔记 + 会话 FTS；prefetch 进当前 USER，不升权 SYSTEM |
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
- **Tools（opt-in）：** `parallel_tasks` / `spawn_subagent`（wire `task`）/ wait-get — 仅 MAIN 调用。
- **Types：** `explore` · `plan` · `general-purpose`。
- **Coordinator：** `SubagentCoordinator` 只执行 MAIN 已经 spawn 的工作。
- **Isolation：** `none`（共享 cwd）或 `worktree`（`.codedoggy/worktrees/<id>`）；`cwd` 与 worktree 互斥。
  - 默认：`CODEDOGGY_SUBAGENT_ISOLATION=none|worktree|auto`（auto：explore=none，写角色=worktree）。
  - 合并：`merge_subagent_worktree`（MAIN 显式 land）。
- **Model pin：** `Task.model` 或 env `CODEDOGGY_SUBAGENT_MODELS` / `CODEDOGGY_SUBAGENT_MODEL_<TYPE>` → 子 sampler 真 pin。
- **Nesting：** `CODEDOGGY_MAX_SUBAGENT_DEPTH`（默认 1=Grok）；>1 时子 agent 可再 spawn。
- **Agent 发现：** `~/.codedoggy/agents`、`{cwd}/.codedoggy/agents`、`CODEDOGGY_AGENTS_PATHS` 下 `*.md`。
- **Cancel 不 auto-wake：** 取消后默认不 drain prompt_queue（`CODEDOGGY_DRAIN_AFTER_CANCEL=1` 可恢复）。
- **另一层并行：** 同一轮 `tool_calls` 的 batch 执行可按 path-lock 并行（引擎效率，不是自动拆任务）。
- **TUI 并行面：** 任务卡 ↳ roster（选中/进行中）；顶栏 `并行 n/m` badge → 底部 **全局** fleet 面板（跨任务 live 优先，↑↓/Enter/p 钉/`m` 合入）；详情页 Agent 芯片切换；worktree `wt`，完成后 **双击 m / 点合入** 确认 land（同 `merge_subagent_worktree`）。

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
Orchestration: `todo_write` · `update_goal` · `enter_plan_mode` · `exit_plan_mode` · `ask_user_question`  
Web: `web_search` · `web_fetch`  
Scheduler: `scheduler_create` · `scheduler_delete` · `scheduler_list`  
Extras: `memory` · `session_search` · `code_nav` · `image_gen` · `image_edit` · video tools  
（无默认 `memory_search` / `memory_get` — Hermes 记忆面）  

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
         (todos / subagents / bg shell tasks only)
         else → done
       else three-phase tools:
         Phase1 prepare ALL (schema · pre_tool_use · plan-mode gate · policy)
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
| Plan mode | `SessionModeState` + `plan_mode_edit_gate` + enter/exit tools (GrokBuild) |
| Interjection / queue | `InterjectionBuffer` / `PromptQueue` |
| Subagents | `SubagentCoordinator` + child `run_agent_loop` |
| Capability | `read-only` / `read-write` / `execute` / `all` |
| Built-ins | `BUILTIN_AGENTS` + spawn / parallel / get_output tools |

- Subagent = 完整子会话，独立 context；回传 summary
- Plan mode hard gate 独立于 auto-approve（仅 plan 文件可写）
- Product: 工程任务由模型软识别进 Plan → 正常对话/写 plan → `exit_plan_mode` 审批 → Auto 实现；闲聊直接交流；不强制问卷；无 S-Tab 手切 Plan/Auto
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
