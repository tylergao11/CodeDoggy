# Grok 源码级复刻对照表

**标准：能指到 Grok 文件+函数/常量，行为与错误文案一致 = 源码级；只能“形状像”= 近似；未接真后端 = stub。**

不再写「完美复刻」；每条工具只标等级。

## 等级

| 等级 | 含义 |
|------|------|
| **S** | Source-level：对照 Grok 源文件移植逻辑/常量/错误串，可逐段 diff |
| **C** | Contract：名字/schema/描述对齐，实现是简化版 |
| **A** | Approx：行为近似，算法未移植 |
| **X** | Stub：面在，真能力依赖 host 或未实现 |

## 对照（Grok → CodeDoggy）

| Grok 源 | 我们 | 等级 | 备注 |
|---------|------|------|------|
| `xai-grok-agent/src/config.rs` product renames | `tools/grok_surface.py` | **S** | 产品改名表 |
| `codex/apply_patch/parser.rs` | `tools/codex/apply_patch/parser.py` | **S** | 源码级移植 |
| `codex/apply_patch/seek_sequence.rs` | `tools/codex/apply_patch/seek_sequence.py` | **S** | |
| `codex/apply_patch/apply.rs` | `tools/codex/apply_patch/apply_logic.py` | **S** | |
| `codex/apply_patch/tool.rs` | `tools/builtins/apply_patch.py` | **C→S** | tool 壳接纯逻辑 |
| `grok_build/bash` format_default_prompt | `tools/grok_build/bash_format.py` | **S** | 输出卡片 |
| `computer/local/shell_state.rs` | `tools/util/shell_state.py` | **A** | cwd+env probe + `shell_env_overrides`；无 Unix FD dump |
| `bash` bg op / timeout helpers | `grok_build/bash_bg_op.py` + `bash_params.py` | **S** | `&` 检测 + should_reject；auto-bg wait=min(timeout,budget) |
| `bash` description template | `builtins/run_terminal_cmd.py` | **S** | Job Object 文案与 Grok 模板一致 |
| Win32 Job Object kill | `tools/util/job_object.py` + bash/task_manager | **C** | Grok ProcessGroup：Create/Assign/TerminateJobObject + child kill；**无 taskkill** |
| `grok_build/bash` terminal actor | `task_manager` + run_terminal | **A** | 非 Grok actor；杀树路径已接 Job Object |
| host memory_backend | `host/memory_backend.py` | **C/A** | MemoryStore 子串检索；非 Grok 全量 backend |
| host ask_user_cli | `host/ask_user_cli.py` | **C** | stdin 多选；TTY 自动注入 |
| host scheduler runtime | `host/scheduler_runtime.py` | **A** | fire→interjection；非 Tokio actor |
| `grok_build/read_file` extract + resolve offset | `tools/grok_build/read_file_extract.py` | **S** | harness 负 offset + trailing empty |
| `read_file/{pdf,pptx,image}` | `util/rich_files.py` + tool shell | **C** | 文本/元数据；无多模态 ImageContent |
| `grok_build/list_dir` DirNode/budget_expand | `tools/grok_build/list_dir.py` | **S** | seed+BFS 预算；walk 用 pathlib 非 ignore crate |
| `grok_build/grep` finalize/format | `tools/grok_build/grep_format.py` | **S** | Found/at least、count_matches、workspace_result |
| `grok_build/grep` rg spawn | `tools/builtins/grep.py` | **C** | `--heading` 对齐；无 stream/cgroup |
| `search_replace` helpers + messages | `search_replace_logic.py` + builtin | **S** | 归一化匹配 + Grok 错误/成功串 |
| `memory/search_tool.rs` | `memory_search.py` + `host/memory_backend.py` | **C** | schema+软文案 S；产品默认注入子串 backend（非 Grok 全量） |
| `memory/get_tool.rs` | `tools/builtins/memory_get.py` | **S** | `format_with_line_numbers` + `**File:**`/`**Lines:**`/软文案；slice=storage.read_file；host=`memory_store`（非 MemoryBackend） |
| `grok_build/lsp` | `tools/builtins/lsp.py` | **X** | schema S；**无** graph 顶替；需 `lsp_backend` |
| `types/tool_index.rs` DTOs + trait | `tools/mcp/types.py` | **S** | `ToolSearchResult`/`SearchSnapshot`/`ServerSummary`/`ToolSearchIndex`/`ToolIndex` 字段名对齐；`McpDispatch` host glue |
| `search_tool` / truncate / sanitize / reminder | `search_tool_logic.py` + `builtins/search_tool.py` | **S** | `SearchToolInput` + 文案+截断+FNV+reminder + grouped JSON ready/partial + 无连接 note |
| shell `Bm25ToolSearchIndex` | `tools/mcp/tool_index.py` | **S** (algo spirit) | split_identifier / normalize_query / exact-match / Okapi BM25 / list_server_summaries；无 Rust bm25 crate |
| auto index from `mcp_tools` | `ensure_mcp_tool_index` | **S** glue | 注入 `ToolIndex(Bm25…)` 对齐 shell `ToolIndex(Arc<dyn ToolSearchIndex>)` |
| `use_tool` | `use_tool_logic.py` + `builtins/use_tool.py` | **S** (types/strings) / **C** (host) | `UseToolInput`/`UseToolParams` **S**；native correction **S**；`mcp_dispatch` host；**transport X** |
| `skills/types.rs` + `skill.rs` | `skill_logic.py` | **S** | `SkillInfo`/`SkillInput`/`SkillOutput`/`SkillRef` + `<skill>` 信封 + substitutions |
| `skills/discovery.rs` | `skill_discovery.py` | **S** | `normalize_skill_name`/`ParsedFrontmatter`/`find_skill_md_paths`/`walk` depth5；**PyYAML** = serde_yaml 管线（safe_load → quote retry → recover_scalar） |
| `skill` tool + OpenCode DESCRIPTION | `builtins/skill.py` + `build_skill_tool_description` | **S/C** | 动态 `<available_skills>` **S**；`SkillInput`/invoke/discovery **S/C**；list_ctx 接线 **C** |
| `image_gen` / `image_edit` | `imagine_api.py` + `builtins/image_gen.py` | **S** (HTTP) / **X** (tier) | endpoint/payload/save `images/n.jpg`/MediaGen JSON；无 SuperGrok upsell |
| `video_gen` client + tools | `video_api.py` + `builtins/video_gen.py` | **S** (HTTP) / **X** (ZDR) | generations+poll+download；**无** ZDR/S3 |
| `web_search` Responses API | `grok_build/web_search.py` + `util/web_search_api.py` | **S** (HTTP path) | 无 key→`not_supported`；无 DDG mock |
| `todo` merge/replace | `grok_build/todo_logic.py` + `todo_write.py` | **S** | 状态标签/摘要串；存储在 `extra` 非 Resources |
| `update_goal` mod.rs | `grok_build/update_goal_logic.py` + `builtins/update_goal.py` | **S** (schema/summary/render_ack) / **A** (host) | `build_summary`+`render_ack_into_output`+RejectReason codes；本地无 classifier/oneshot，3×blocked 在 tool 侧；host=`goal_ack_fn` |
| `ask_user_question` format | `builtins/ask_user_question.py` | **S** (format) / **X** (TUI) | format A–D；host `ask_user_fn` |
| `enter_plan_mode` seed + prompt | `grok_build/plan_mode.py` + `builtins/enter_plan_mode.py` | **S** (seed/prompt/schema) / **X** (Resources) | empty `{}` schema；seed 不截断；to_prompt 6 步；host `session_mode_state`/`kernel`/`plan_mode_consent_fn`；无 NotificationHandle/TemplateRenderer |
| `exit_plan_mode` read + prompt | `grok_build/plan_mode.py` + `builtins/exit_plan_mode.py` | **S** (read/prompt/schema) / **X** (ACP) | empty `{}`；读盘非入参；PlanReady/EmptyPlan 串；host `plan_mode_exit_fn` outcome；无 PlanModeExited 总线 |
| `opencode/write` | `builtins/write.py` | **S** | create/overwrite 成功串；无 DisplayCwd |
| cgroup (Linux) | — | **X** | 未做 |
| Job Object (Windows kill tree) | `util/job_object.py` | **C** | Create/Assign/Terminate 真 API；非完整 terminal actor |
| `web_fetch/ssrf.rs` `is_blocked_ip`/`check_ssrf` | `tools/util/ssrf.py` | **S** | 精确 octet/ULA；错误串含 `private/internal IP` + gh hint |
| `web_fetch/error.rs` Display | `grok_build/web_fetch_error.py` | **S** | 错误文案一字不改 |
| `web_fetch/config.rs` caps/UA/allowlist | `grok_build/web_fetch_config.py` | **S** | MAX_URL/REDIRECTS/UA/10MB/100k/DEFAULT_ALLOWED_DOMAINS |
| `web_fetch/domain.rs` DomainMatcher | `grok_build/web_fetch_domain.py` | **S** | path-prefix 匹配；client 默认不 enforce（Grok 在 permission 层） |
| `web_fetch/cache.rs` FetchCache | `grok_build/web_fetch_cache.py` | **S** | TTL+evict；truncated 不入缓存 |
| `web_fetch/client.rs` validate/fetch/content | `grok_build/web_fetch_content.py` + `web_fetch_client.py` | **S/C** | validate/SSRF/redirects/binary/media **S**；html→md 无 htmd **A** |
| `web_fetch/overflow.rs` + artifact | `grok_build/web_fetch_overflow.py` | **S/A** | budget/footer/steer **S**；artifact 无 fs2 锁 **A** |
| `web_fetch/mod.rs` tool shell | `builtins/web_fetch.py` | **C** | 描述对齐；session_folder=cwd/extra |
| `scheduler/interval.rs` parse + human | `grok_build/scheduler_interval.py` | **S** | min clamp 60s；错误串 `invalid interval: …` |
| `scheduler/types.rs` task/expiry/missed | `grok_build/scheduler_types.py` | **S** | next_fire/expires/missed；id=uuid4 hex12（非 v7） |
| `scheduler/actor.rs` Create/Delete/List/fire | `tools/scheduler.py` | **S/C** | 命令+fire/missed **S**；**无** tokio timer/notifications **X** |
| host poll / tick (not actor timer) | `host/scheduler_tick.py` | **A/X** | `poll_due`/`fire_due`/`run_tick_loop` 调 store；**无** ToolNotificationHandle **X**；host 必须自注入 prompt |
| `scheduler/{create,delete,list}.rs` | `builtins/scheduler_tools.py` | **S** | schema/描述/成功失败串；durable 持久化 host **X** |
| `task` / `TaskToolInput` + formatters | `grok_build/task_format.py` + `builtins/spawn_subagent.py` | **S** (schema/strings) / **C** (dispatch) / **X** (cwd+model pin) | wire id `task`→product `spawn_subagent`; `build_task_description` product names + `by_kind.plan`→`todo_write` **S**；host coordinator **C** |
| Main `prompt.md` | `prompt/grok_system.py` `render_grok_base_prompt` | **S** (sections) / **C** (label CodeDoggy; no liquid) | action_safety / tool_calling / output_efficiency / formatting 源码级；identity 用产品 label |
| Subagent `subagent_prompt.md` | `render_grok_subagent_base` | **S** (structure) / **C** (no hashline branch) | Parallelize tool calls + AGENTS.md + user_info **S**；role-instructions 为 Doggy |
| Product appendix | `codedoggy_product_appendix` | **Doggy** | MAIN 并行倾向、code_nav、session_search — **非** Grok |
| `task_output/mod.rs` + `xai-tool-types` task.rs | `grok_build/task_output_logic.py` + `builtins/get_task_output.py` | **S/A** | `build_task_output_description` cli_default 锁死 **S**；schema/waits/cap/cards **S**；wait poll 非 Notify **A** |
| `task_output/wait_tasks.rs` | `builtins/wait_tasks.py` | **S/A** | `build_wait_tasks_description` 含 `background=true or background=true` **S**；wait_all/any **S**；join **A** |
| `kill_task/mod.rs` + `build_kill_task_description` | `builtins/kill_task.py` + `task_output_logic` | **S/C** | 描述 OS 动词 **S**；kill=TerminateJobObject+child kill（对齐 terminal.rs，无 taskkill）**C** |
| `use_tool` UseToolInput docs | `builtins/use_tool.py` + `use_tool_logic` | **S** (desc/schema) / **C** (dispatch) | description_template + schemars 字段文案 **S** |
| `monitor/{types,event,rate_limiter,tool}.rs` | `grok_build/monitor_*.py` + `builtins/monitor.py` | **S/A/X** | constants/validate/start 文案/line+rate pure **S**；spawn+PYTHONUNBUFFERED **S**；无 MonitorEvent 通知管线 **X** |

## 已删除的脑补（禁止再加）

| 脑补 | 处置 |
|------|------|
| graph-as-LSP fallback | 删除；`lsp` 无 backend 直接 unavailable |
| BM25 `mcp_registry.py` | 删除；搜索由 host `mcp_tool_index` / 简单 catalog 过滤 |
| 伪 token `memory_search` 扫 MemoryStore | 删除；仅 `memory_backend` |
| 声称「完美复刻」文档 | 改为本表等级 |

## Doggy 增强（非 Grok 产品面）

| 工具 | 说明 |
|------|------|
| `memory` | Hermes 写记忆 |
| `session_search` | 会话 FTS |
| `code_nav` | codebase graph 导航（**不是** LSP） |
| `parallel_tasks` | MAIN **主动调用**的多子 agent 派工工具；**不是** runtime 自动并行 |

- 纯 Grok：`finalize(grok_build_product_config())`
- CodeDoggy 默认：`finalize()` → `codedoggy_product_config()` = Grok + 上表增强

## 规则（强制）

1. 新移植模块顶部写：`Ported from grok-build/<path>` + 函数对应表。
2. 错误字符串与 Grok 相同则一字不改。
3. 不能 1:1 的标 **A/X**，禁止在文档写「完美」。
4. 测试尽量搬 Grok 同名用例语义。
5. **禁止**用 graph/BM25/假后端冒充 Grok runtime；缺 backend 就报 unavailable / soft text。
6. **禁止发明「兜底 / 降级」路径。** Grok 源码没有的第二套机制（如 taskkill 顶 Job Object）不准加。Grok 协议里的固定两步（如 hard-kill 先 Job 再 `start_kill`）是协议本身，不是 degrade ladder。未做完就标等级，别用替代实现糊墙。

## 当前冲刺顺序

1. ~~apply_patch parser/apply/seek~~
2. ~~bash_format~~
3. ~~删除脑补 LSP/MCP/memory；恢复 shell_state/rich_files 子集~~
4. ~~list_dir budget_expand~~
5. ~~read_file extract_file_content_lines~~
6. ~~grep finalize / format~~
7. ~~search_replace confusable + Grok strings~~
8. ~~web_fetch SSRF / cache~~
9. ~~todo_write / ask_user_question / write / web_search / imagine / video~~
10. ~~bash bg_op / timeout / desc / shell_env_overrides / Job Object 子集~~
11. ~~task 面 / scheduler / plan / goal / host 接线~~
12. 剩余：cgroup、full ShellState FD dump、真 LSP/MCP 传输、task kernel isolation
