# Architecture

**Roadmap order:** thicken four pillars (context / memory / tools / **shadow**)
**before** external orchestration (multi-agent graph, gateway routing, etc.).

**Shadow (影子)** = write-time soft quality review *inside* the agent loop.
Not a normal offline “code audit” of the repo. Package path: `codedoggy.audit`
(import stability); product name: **Shadow**.

```
Session          outer lifecycle, cwd, extensions
  turn_runner    AgentTurnRunner → run_agent_loop
                 + live_messages + archive-at-create
                 + MemoryManager system/prefetch/sync
                 + rewind_context() session API
  tools          FinalizedToolset + WorkspacePolicy
                 (writes/shell gated; shell paths check_write)
  context        ContextCompactor
                 (Grok pipeline + pre-tool async prefire + checkpoint rewind)
  memory         MemoryManager (curated + FTS warm cache + 1 external slot)
  shadow         ModelAuditor / ShadowAuditor + Hermes select + policy
                 (extensions.audit handle; product name 影子)
  graph          CodebaseGraph + code_nav (xai-codebase-graph API spirit)
  orchestration  planned (after pillar thicken)
```

### Graph (GitHub-style navigation — from xai-codebase-graph)

Ported API surface (not a from-scratch design):

- `IndexBuilder.build(root)` → `ScopeGraphIndex`
- `Navigator.goto_definition` / `goto_references` (+ `_by_name`)
- `Location` / `NavigationResult` (1-indexed lines)
- `find_definitions_smart` / `find_references_smart` + same-language ranking
- aliases (import as), cache `.goto_index.json`
- tool: `code_nav` (definition | references | at_position | stats | reindex)

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

## Turn loop

```
user prompt
  → seed: SYSTEM(+MEMORY) + prior live (non-system) + USER
  → Hermes prefetch: session FTS hits → system (curated already injected)
  → sample(messages, tools)   # before each sample: ContextCompactor.ensure
  → archive ASSISTANT/TOOL as created (full body → SessionStore)
  → if tool_calls:
       for each call: execute → after_tool / after_mutation hooks → write TOOL observation
       (abort skips remaining calls; no orphan side-effects without transcript)
       → sample again
  → else: final text → done
  → stop on: no tools | max_turns | cancel | hook abort | sampler error
  → carry live_messages into next handle_prompt
```

- `max_turns`: sampling rounds per prompt; `None` = unlimited
- Tool errors become observations (do not crash the loop)
- Tool batch is sequential per-call (parallel is a future executor concern)
- `after_mutation`: resident quality audit (search_replace + detected shell writes)
- **Live** window may be pruned; **archive** is create-time full fidelity

## Context (Grok foundation pipeline)

```
before each sample → ContextCompactor.ensure:
  0. suppress gate (TURN / STICKY / UNTIL_SUCCESS)
  1. prune oversized tool results (P0 footers kept)
  2. under pressure: prune_retained (clear old tool bodies; P0 footers kept)
  3. memory_flush (soft threshold, once per compaction cycle incl. 0)
       → Hermes MEMORY.md via model → refresh_system_prompt_snapshot
       → inject SYSTEM "[MEMORY refreshed mid-turn…]"
  4. if over threshold_percent → fold middle (P0 stripped from sketch)
       mode: summary | transcript | segments (segment_*.md + INDEX)
       reinject open P0 as binding USER note if not still on a TOOL
  5. system / MEMORY never dropped
  SessionStore: create-time archive (full tool bodies); live window separate
```

- Env: `CODEDOGGY_CONTEXT_MAX_CHARS`, `CODEDOGGY_CONTEXT_THRESHOLD_PERCENT`,
  `CODEDOGGY_COMPACTION_MODE`, `CODEDOGGY_MEMORY_FLUSH`, `CODEDOGGY_RETAIN_RECENT_TOOLS`,
  `CODEDOGGY_COMPACTION_CHECKPOINT` (default on — pre-fold segment)
- Budget: token-first (optional **tiktoken** if installed; else CJK heuristic);
  `CODEDOGGY_CONTEXT_MAX_TOKENS` or `…_MAX_CHARS` (chars≈tokens×4); model
  `prompt_tokens` usage overrides when higher
- Soft prefire: compact again after each tool batch (not only loop-top)
- Hermes SUMMARY_PREFIX + END marker + Historical* headings; iterative
  previous_summary; protect_first_n (decays after first fold)
- Grok tool-pair safe split + hard_trim_safe / sanitize_tool_pairs
- update_from_response + awaiting_real_usage thrash guard; on_session_end
- Compaction prefix: Grok "REFERENCE ONLY"; **system-prompt** MEMORY/USER
  (incl. post-flush refresh) is authoritative over the summary — not raw disk alone
- Ported (partial): async prefire flush (thread), MemoryManager + 1 external slot,
  checkpoint rewind API, WorkspacePolicy
- Not ported: full rewind UI product, full OS sandbox, full hermes-agent tree

## Model layer

```
ModelConfig { provider, model, base_url, api_key, temperature, … }
  → register_provider(name, factory)   # Hermes-style registry
  → create_client(config)              # Grok-style config → client
  → ChatClient.complete(messages)
  → ChatSampler (turn.Sampler) | ModelAuditor (ResidentAuditor)
```

- Stock providers: `ollama` (default), `openai_compat` / `openai` / `custom`
- Dual profiles: `ModelProfiles` / `model_profiles_from_env()` — **main** + **audit**
- Bootstrap: `build_session()` wires `ChatSampler(main)` + `ModelAuditor(audit)` + tools
- Env main: `CODEDOGGY_PROVIDER`, `CODEDOGGY_MODEL`, `CODEDOGGY_BASE_URL`, `CODEDOGGY_API_KEY`
- Env audit: `CODEDOGGY_AUDIT_*` / `CODEDOGGY_AUX_*` (fallback to main)
- Ollama default: `http://127.0.0.1:11434/v1` + model `qwen3:8b`
- Transport: OpenAI-compatible `chat/completions` (stdlib HTTP)

## Shadow 影子 (write-time soft review — not a normal audit)

```
mutation (search_replace first-hand before/after)
  → MutationTrajectory (session write log)
  → MemorySelector (Hermes curated + FTS)
  → ShadowAuditor / ModelAuditor.review (model brain)
  → pass: silent | fail: soft observation footnote (rethink)
```

- **Name:** product **Shadow / 影子**; code package `codedoggy.audit` (stable imports)
- **Differs from normal audit:** in-loop, per mutation, soft only, no repo report
- Session **goal** = intent anchor (`session.goal` / `set_goal`)
- Shadow **must not write** workspace
- **P0 (`critical`)**: immediate red card on tool observation; prune/fold preserve
- **Non-P0**: buffered → turn end (`metadata.shadow_deferred` / `audit_deferred`)
- Handle: `SessionExtensions.audit` (historical field name)
- Aliases: `ShadowAuditor`, `ShadowHooks`, `ShadowServices`

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
- Live path for audit: `prefer_frozen=False` → `live_system_prompt_blocks()`
- Tool: `memory` (add / replace / remove / batch)
- Bind `MemoryStore` on `SessionExtensions.memory` so the turn loop injects it

### Big session store (on-demand, unlimited)

- SQLite `~/.codedoggy/state.db` with **FTS5** (LIKE fallback)
- Every turn transcript persisted via `AgentTurnRunner`
- Tool: `session_search` — discovery / scroll / read / browse
- Audit select: `HermesMemorySelector` = curated blocks + FTS session hits

| Layer | Capacity | When loaded |
|-------|----------|-------------|
| MEMORY.md / USER.md | ~chars budget | every prompt (frozen) |
| Session FTS | unlimited | on demand (tool + audit select) |

## Tools

- Registry: `register` → `finalize` → `FinalizedToolset`
- Packs: `register_tool_pack` before first builder
- Qualified ids: `Doggy:short_id` (e.g. `Doggy:read_file`)
- Builtins: `read_file`, `search_replace`, `list_dir`, `grep`, `run_terminal_cmd`, `memory`, `session_search`
- Defaults: `tools/defaults.py`
