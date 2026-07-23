# Release / audit boundaries

CodeDoggy **已交付 / 胶水层 / 延期** 的诚实分界。不要在文档外宣称未实现能力。

| Label | Meaning |
|-------|---------|
| **DONE** | 已实现，且有下列攻击/回归测试覆盖 |
| **GLUE** | 满足产品契约的精简实现，不是完整外部栈复刻 |
| **DEFERRED** | 未实现或 host stub；勿当产品完成项宣传 |
| **REMOVED** | 曾存在，已刻意移出产品路径 |

---

## 1. DONE — P0s, major P1s, product posture

### P0 (must not regress)

| Area | What is fixed | Attack / regression files |
|------|---------------|---------------------------|
| **apply_patch Move** | Path escape denied; source kept; `.env` deny before unlink; successful Move emits delete+create mutations | `tests/test_p0_attack.py` |
| **Context overflow** | Overflow resubmit bounded (`max_turns` + compact budget); no infinite sample spin | `tests/test_p0_attack.py` |
| **tool_call ids** | Fallback ids unique across samples; sanitize keeps multi-round results with colliding ids (FIFO) | `tests/test_p0_attack.py` |
| **Policy / gate** | Case-insensitive deny; shell protected-path detect; central gate blocks writes; missing tool results filled | `tests/test_p1_attack.py` |

### Major P1s (done)

| Area | What is fixed | Attack / regression files |
|------|---------------|---------------------------|
| **Busy prompt** | Concurrent `handle_prompt` → `QUEUED` / interject, not false `COMPLETED` | `tests/test_p1_attack.py`, `tests/test_p1_session_queue.py` |
| **SessionStore cwd** | `ensure_session` does not rehome cwd on re-open | `tests/test_p1_attack.py` |
| **session_search scope** | FTS results scoped by session cwd | `tests/test_p1_attack.py`, `tests/test_p1_tool_extra.py` |
| **Context budget** | Trigger uses usable window (window − reserves), not raw `context_window` | `tests/test_p1_attack.py` |
| **memory tool kind** | Curated `memory` is write-capable (`ToolKind.Edit`) for gate/capability | `tests/test_p1_attack.py` |
| **Secret redact** | AWS keys / DB URLs redacted on store path | `tests/test_p1_attack.py` |
| **Graph reindex** | Extract failure keeps prior defs; reindex/persist honor write policy | `tests/test_p1_graph.py`, `tests/test_p1_graph_internal.py` |
| **use_tool prepare** | Schema + policy before host `mcp_dispatch` (escape / protected path) | `tests/test_p1_use_tool.py` |
| **Provider tools** | Host-injected provider tools visible/callable; kernel `tool_extra` mid-turn path | `tests/test_p1_provider_tools.py`, `tests/test_p1_tool_extra.py` |

### Parallel MAIN (product — agent bias, not auto-orchestrate)

| Area | What ships | Tests |
|------|------------|-------|
| **MAIN parallel tendency** | System prompt biases MAIN to prefer multi-agent when work splits; **no runtime forced fan-out** | `tests/test_parallel_tasks.py` |
| **`parallel_tasks` tool** | Opt-in tool when MAIN calls it; `wait` true/false is MAIN's choice | `tests/test_parallel_tasks.py` |
| **Coordinator** | Executes what MAIN spawned (`spawn_many` / `wait_all`); not a policy engine | `tests/test_parallel_tasks.py` |
| **`general-purpose` agent** | Available type MAIN may name | `tests/test_parallel_tasks.py` |

### Fusion (GrokBuild base + Hermes memory)

| Area | What ships | Boundary |
|------|------------|----------|
| **Memory surface** | Hermes `memory` + `session_search` only on default product finalize | Grok `memory_search`/`memory_get` opt-in via `register_optional_grok_memory_tools` |
| **memory_backend** | Off by default; `CODEDOGGY_GROK_MEMORY_BACKEND=1` only | Not Hermes product path |
| **Plan file path** | `{cwd}/.grok/sessions/<id>/plan.md` (Grok session_dir) | Not shared workspace `.grok/plan.md` as authority |
| **Plan edit gate** | Active only (ExitPending allows non-plan edits again) | Matches Grok `is_active()` |

### Plan mode (GrokBuild)

Product policy (agent soft-decide, **not** harness force-every-turn):

- **Engineering work** → model should `enter_plan_mode`, talk normally, write plan file, `exit_plan_mode` → user **a/s/q** → on approve **Auto** implement.
- **Chat / Q&A** → direct dialogue; no plan mode, no questionnaire required.
- **`ask_user_question`** is optional multi-choice only — not required to "be in plan."
- TUI has **no** S-Tab Plan/Auto toggle; mode is tool-driven. Enter-plan host consent is not wired (avoids extra confirm). Exit approval remains.
- **Plan UX**: detail (计划 tab) is the main review surface; awaiting tasks show a card **批准** CTA (same path as `a`); open-detail defaults to 计划 when drafting/awaiting.

| Area | What ships | Tests |
|------|------------|-------|
| **`enter_plan_mode` / `exit_plan_mode`** | Session plan lifecycle (pending/active/exit_pending); exit approval (TUI a/s/q); no enter-consent strip | `tests/test_plan_mode.py` |
| **`plan_mode_edit_gate`** | **File edits only** while **Active** — only the plan file may be written; shell/MCP/spawn are **not** plan-gated (same spirit as Grok) | `tests/test_plan_mode.py`, `tests/test_orchestration.py` |
| **`plan.md` + `plan_mode.json`** | Plan body + lifecycle under `{cwd}/.grok/sessions/{id}/`; restart collapses pending/exit_pending | `tests/test_plan_mode.py` |
| **Approval re-park** | `awaiting_plan_approval` resume runs host approval before next turn (Grok re-park equivalent) | `tests/test_plan_mode.py` |
| **`todo_state.json` + badge** | TodoWrite persisted; TUI `完成/总数` badge (cancelled excluded) | `tests/test_todo_write.py`, `tests/test_tui.py` |
| **Subagent todo isolation** | Children get their own TodoState (never MAIN’s); disk under sanitized `parent__child` session dir; snapshot metadata checklist | `tests/test_todo_write.py`, `tests/test_spawn_subagent.py` |
| **Incomplete-work scope** | MAIN gates on MAIN todos + running children; child only on own todos (not siblings) | `tests/test_incomplete_work.py` |
| **Agent detail live todo** | Modal title badge + body checklist for open agent (MAIN or child) | `tests/test_tui.py` |
| **Task card todo chips** | Active card shows MAIN `n/m`; expanded agent rows show each child’s live badge; **click row opens that agent** | `tests/test_tui.py` |
| **Turn-status incomplete hint** | Running: Chinese chip for open MAIN todos / running kids (`未完·待办N·M子任务`) | `tests/test_tui.py` |
| **Incomplete-work model nudge** | Chinese primary steer + `[incomplete_work]` tag after prose-stop | `tests/test_incomplete_work.py` |
| **Badge → focus task** | Open todo pane focuses active/latest task card | `tests/test_tui.py` |
| **Badge Ctrl+left** | Opens MAIN agent **计划** tab + keeps checklist pane | `tests/test_tui.py` |
| **Task plan_state sync** | Session plan mode mirrors onto active task `planning` / `awaiting_approval` | `tests/test_tui.py` |
| **Goal ↔ Plan exclusive** | Goal clears plan approval chrome; caption prefers live plan_phase; plan enter is tool-driven (no S-Tab) | `tests/test_tui.py` |

### Incomplete-work / anti early-done

| Area | What ships | Tests |
|------|------------|-------|
| **No-tools gate** | Open todos / running subagents / bg shell tasks → nudge + continue (not COMPLETED). | `tests/test_incomplete_work.py` |
| **update_goal gate** | `completed=true` refused while incomplete-work reasons remain | `tests/test_incomplete_work.py` |

Related suites: `tests/test_orchestration.py`, `tests/test_spawn_subagent.py`, `tests/test_image_gen.py`.

---

## 2. GLUE — intentional thin implementations

| Area | What ships | Honest limit |
|------|------------|--------------|
| **Session spine** | `RuntimeKernel` + turn loop + prompt queue / interjection | 非完整宿主 session actor |
| **Resume transcript** | Serialize `Message` dicts for prior live | 非完整 ConversationItem 存储 |
| **Shell process kill** | Win32 TerminateJobObject + child kill；POSIX killpg | 非完整终端 actor + cgroup |
| **Subagent pool** | Thread-pool runs children MAIN spawned | 无自动 task-split |
| **Batch tool dispatch** | Phase-1 prepare + phase-2 path-lock parallel；writeback 按 emit 序 | 非 Tokio 全量 interruptible wait |
| **Media tools** | image/video/web_search 跟 ActiveConnection | 端点不支持则 `not_supported`，不跨 provider 偷密钥 |

---

## 3. DEFERRED

| Area | Notes |
|------|-------|
| **Interactive sandbox / approval UX** | Workspace policy/gate 存在；默认高权限是产品选择，非完整沙箱审批 UI |
| **Full LSP product** | 需 host `lsp_backend`；`code_nav` 不顶替 LSP |
| **Binary CLI installer** | Python 包 + `uv tool install`；无官方 install.ps1 二进制分发 |

Do **not** invent additional roadmap rows here without implementation.

---

## 4. Verification (smoke)

```bash
pytest tests/test_p0_attack.py tests/test_p1_attack.py -q
pytest tests/test_orchestration.py tests/test_parallel_tasks.py -q
pytest tests/test_image_gen.py tests/test_connection.py -q
```
