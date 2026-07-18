# Release / audit boundaries

Honest cut line for what CodeDoggy **ships**, what is **glue** (not a full Grok port), and what is **deferred**.  
Do not claim features outside this file. Port rules: [`docs/grok-source-map.md`](grok-source-map.md). Hermes seam: [`docs/hermes-groke-seam.md`](hermes-groke-seam.md).

| Label | Meaning |
|-------|---------|
| **DONE** | Implemented + covered by attack/regression tests listed below |
| **GLUE** | CodeDoggy-only subset that satisfies a *contract*; not a full Grok/Hermes/Codex stack |
| **DEFERRED** | Not implemented (or host-only stub); do not market as product-complete |
| **REMOVED** | Was present; deliberately out of product path |

---

## 1. DONE — P0s, major P1s, product posture

### P0 (must not regress)

| Area | What is fixed | Attack / regression files |
|------|---------------|---------------------------|
| **apply_patch Move** | Path escape denied; source kept; `.env` deny before unlink; successful Move emits delete+create mutations | `tests/test_p0_attack.py` |
| **Context overflow** | Overflow resubmit bounded (`max_turns` + compact budget); no infinite sample spin | `tests/test_p0_attack.py` |
| **tool_call ids** | Fallback ids unique across samples; sanitize keeps multi-round results with colliding ids (FIFO) | `tests/test_p0_attack.py` |
| **Policy / gate** | Case-insensitive deny; shell protected-path detect; central gate blocks writes; missing tool results filled | `tests/test_kernel_audit_fixes.py` |

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
| **Default no Shadow** | `build_session` `enable_audit=False`; no system-prompt Shadow; children get no audit hooks | `tests/test_bootstrap.py`, `tests/test_parallel_tasks.py` |
| **MAIN parallel tendency** | System prompt biases MAIN to prefer multi-agent when work splits; **no runtime forced fan-out** | `tests/test_parallel_tasks.py` |
| **`parallel_tasks` tool** | Opt-in tool when MAIN calls it; `wait` true/false is MAIN's choice | `tests/test_parallel_tasks.py` |
| **Coordinator** | Executes what MAIN spawned (`spawn_many` / `wait_all`); not a policy engine | `tests/test_parallel_tasks.py` |
| **`general-purpose` agent** | Available type MAIN may name | `tests/test_parallel_tasks.py` |

Related suites: `tests/test_orchestration.py`, `tests/test_hermes_seam.py`, `tests/test_grok_fidelity.py`, `tests/test_spawn_subagent.py`.

---

## 2. GLUE — contract satisfied, not full upstream stack

| Area | What ships | What it is *not* |
|------|------------|------------------|
| **Worktree isolation** | Minimal `git worktree` create / reattach / remove / merge (`orchestration/worktree.py`) | Full `xai-fast-worktree` + pool + shell session RPC |
| **MCP surface** | Wire tools `search_tool` / `use_tool`; host injects catalog + `mcp_dispatch` | Built-in MCP transport, BM25 registry, or production MCP client pack |
| **SessionActor spine** | `RuntimeKernel` + turn loop + prompt queue / interjection (spirit map in `SCHEME.md`) | Full Grok `SessionActor` (ACP session, all host channels, complete actor state machine) |
| **Stream deltas** | Optional `stream_sample` / `on_sample_delta` for host UI | Mid-stream interjection interrupt (explicitly deleted invention) |
| **Resume transcript** | Serialize `Message` dicts for prior live | Full ConversationItem / sampling-type store from Grok |
| **Shell process kill** | Win32 Job Object when available + taskkill fallback; POSIX process group | Full Grok terminal actor + Linux cgroup |
| **Subagent pool** | Thread-pool runs children MAIN spawned; `parallel_tasks` is opt-in | Auto task-split; full Grok multi-agent kernel / channel RPC |
| **Batch tool dispatch** | Phase-1 prepare all + phase-2 path-lock parallel execute (Grok `tool_calls` / `tool_dispatch` spirit); writeback in emission order | Full tokio FuturesUnordered + interruptible wait tools |

---

## 3. REMOVED — do not claim as product

| Area | Status | Notes |
|------|--------|-------|
| **Shadow / write-time audit** | **Removed from product path** | `enable_audit` default `False`. Package `codedoggy.audit` remains for unit tests only. |
| **Shadow soft restore** | Not product | Legacy tests may still exercise package code with `enable_audit=True`. |
| **Full TX Shadow** | Never shipped | Not re-planned as product. |

---

## 4. DEFERRED — do not claim

| Area | Status today | Notes |
|------|--------------|-------|
| **Codex / Grok sandbox + interactive permission** | Deferred | Workspace **policy/gate** exists; product still runs high privilege by choice (`docs/tool-checklist.md`). Not a Codex-style sandbox or approval UX. |
| **Full SessionActor** | Deferred | Kernel + loop is the host-facing subset only. |
| **MCP transport** | Deferred | Host-owned; tools fail soft / error without injection. No in-tree MCP wire protocol. |
| **LSP runtime** | Deferred / host | `lsp` tool needs `lsp_backend`; `code_nav` is graph, **not** LSP. |
| **Scheduler timer actor / notifications** | Deferred | Interval/types/tools **S**; no tokio-style timer bus. |
| **ShellState full FD dump** | Deferred | cwd + env probe only. |

Do **not** invent additional roadmap rows here. Port depth for tools: [`docs/grok-source-fidelity.md`](docs/grok-source-fidelity.md).

---

## 5. Release checklist — attack suites must be green

Before calling a build release-ready for audit boundaries, these **must** pass:

```bash
# Core P0 / P1 attack suites
pytest tests/test_p0_attack.py tests/test_p1_attack.py -q

# Kernel / gate regressions
pytest tests/test_kernel_audit_fixes.py -q

# Parallel MAIN product path + Grok batch dispatch
pytest tests/test_parallel_tasks.py tests/test_bootstrap.py tests/test_batch_parallel.py -q

# Major P1 attack expansions
pytest tests/test_p1_graph.py tests/test_p1_graph_internal.py \
       tests/test_p1_session_queue.py tests/test_p1_use_tool.py \
       tests/test_p1_provider_tools.py tests/test_p1_tool_extra.py -q

# Seam / port map locks (recommended same gate)
pytest tests/test_hermes_seam.py tests/test_grok_fidelity.py -q
```

Legacy Shadow package unit tests (optional; not product path):

```bash
pytest tests/test_shadow_restore.py tests/test_audit_p0.py tests/test_audit.py -q
```

One-shot product gate:

```bash
pytest \
  tests/test_p0_attack.py \
  tests/test_p1_attack.py \
  tests/test_kernel_audit_fixes.py \
  tests/test_parallel_tasks.py \
  tests/test_bootstrap.py \
  tests/test_p1_graph.py \
  tests/test_p1_graph_internal.py \
  tests/test_p1_session_queue.py \
  tests/test_p1_use_tool.py \
  tests/test_p1_provider_tools.py \
  tests/test_p1_tool_extra.py \
  tests/test_hermes_seam.py \
  tests/test_grok_fidelity.py \
  -q
```
