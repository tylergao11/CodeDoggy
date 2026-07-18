# Grok tool surface fidelity

CodeDoggy tools aim to **replicate GrokBuild product contracts**, not invent a parallel API.

## Source of truth

| Layer | Location in grok-build |
|-------|------------------------|
| Wire tool ids | `xai-grok-tools` implementations (`ToolId::new(...)`) |
| Product renames | `xai-grok-agent/src/config.rs` (`bash_tool_config`, `task_tool_config`, …) |
| Default product set | `default_grok_build_toolset` + `workspace_grok_build_toolset` |
| CodeDoggy surface | `codedoggy/tools/grok_surface.py` |

## Product renames (must match Grok)

| Wire id | Client name | Param renames |
|---------|-------------|----------------|
| `run_terminal_cmd` | `run_terminal_command` | `is_background` → `background` |
| `task` | `spawn_subagent` | `run_in_background` → `background` |
| `get_task_output` | `get_command_or_subagent_output` | — |
| `wait_tasks` | `wait_commands_or_subagents` | — |
| `kill_task` | `kill_command_or_subagent` | — |

`ToolRegistryBuilder.finalize()` defaults to **CodeDoggy pack** (`codedoggy_product_config`).
Pure Grok list: `finalize(grok_build_product_config())`.

Wire-id aliases still resolve on `call()` so tests/legacy callers can use either name.

## Product tool list

### Grok (`grok_build_product_config`)

- Core: `run_terminal_command`, `read_file`, `search_replace`, `write`, `list_dir`, `grep`, `apply_patch`
- Tasks: `kill_command_or_subagent`, `get_command_or_subagent_output`, `wait_commands_or_subagents`, `spawn_subagent`, `monitor`
- Orchestration: `todo_write`, `update_goal`, `enter_plan_mode`, `exit_plan_mode`, `ask_user_question`
- Web/MCP: `web_search`, `web_fetch`, `search_tool`, `use_tool`
- Scheduler: `scheduler_create`, `scheduler_delete`, `scheduler_list`
- Memory (read): `memory_search`, `memory_get` — **backend host-injected**
- Media: `image_gen`, `image_edit`, `image_to_video`, `reference_to_video`
- `lsp` — **requires host `lsp_backend`**; no graph fake

### Doggy enhancements (`codedoggy_product_config` only)

| Tool | Status |
|------|--------|
| `memory` | Hermes write — **not in Grok** |
| `session_search` | session FTS — **not in Grok** |
| `code_nav` | graph nav — **not LSP** |

## Host injection (honest stubs)

| Tool | Required extra | Missing behavior |
|------|----------------|------------------|
| `lsp` | `lsp_backend` with `dispatch`/`run` | Grok unavailable string |
| `memory_search` | `memory_backend.search` | product injects simple `MemoryStore` backend; soft text if none |
| `memory_get` | `memory_store` | soft: experimental-memory |
| `search_tool` | `mcp_tool_index` or `mcp_tools` list | "No MCP tools registered." |
| `use_tool` | `mcp_dispatch` | ToolError mcp_dispatch_missing |

### `use_tool` host mutation contract (Shadow)

If `mcp_dispatch` returns only a plain string, **Shadow is blind** to MCP file
side effects. For write tools the host **MUST** return a structured envelope:

| Shape | Example |
|-------|---------|
| `mutations` list | `{"text": "...", "mutations": [{"path", "before"?, "after"?, "is_create"?, "is_delete"?}, ...]}` |
| single `mutation` | `{"output": "...", "mutation": {"path": "rel/a.py", ...}}` |
| minimal `mutated_paths` | `{"result": "...", "mutated_paths": ["rel/a.py", "rel/b.py"]}` |
| single `mutated_path` | `{"text": "...", "mutated_path": "rel/a.py", "before"?, "after"?}` |

- Relative paths preferred. Each entry with a non-empty `path` → `ctx.set_mutation`.
- Returned paths are **always recorded** (Shadow truth); policy attaches
  `args["_policy"]` via `set_mutation` (pre-dispatch tool_input paths still gated).
- Model text from `text` / `output` / `result` / `content` when present.
- Plain string return → no mutation (no false positives).

**Deleted inventions:** `mcp_registry` BM25, graph-as-LSP, token-scored MemoryStore as memory_search.

## Imagine / media

Uses **normal HTTP API** (xAI / OpenAI-compatible), not a mock-only path:

| Env | Meaning |
|-----|---------|
| `CODEDOGGY_IMAGINE_API_KEY` (or `XAI_API_KEY` / `CODEDOGGY_API_KEY` / `OPENAI_API_KEY`) | Bearer key |
| `CODEDOGGY_IMAGINE_BASE_URL` | default `https://api.x.ai/v1` |
| `CODEDOGGY_IMAGINE_MODEL` | default `grok-imagine-image-quality` |
| `CODEDOGGY_IMAGINE_ENABLED=0` | force disable |

| Tool | Behavior |
|------|----------|
| `image_gen` | `POST {base}/images/generations` → save under `images/n.jpg` |
| `image_edit` | `POST {base}/images/edits` |
| missing key / HTTP 404/501 | **report not supported** (`code=not_supported`) |
| `image_to_video` | `POST {base}/videos/generations` (model `grok-imagine-video-1.5-preview`) + poll → `videos/n.mp4` |
| `reference_to_video` | same API with 2–7 refs (model `grok-imagine-video`) + aspect_ratio |
| video ZDR / S3 upload | **X** — not ported; local download only |

## Honest gaps

| Grok | CodeDoggy status |
|------|------------------|
| Full language-server `lsp` | **X** — host backend only |
| Full ShellState dump (fd 3/4) | **A** — cwd probe + env overlays |
| Image multimodal presentation | metadata only without host vision |
| Job Object / cgroup | Win32 Job Object kill tree (**C**) + taskkill fallback; Linux cgroup **X** |
| MCP BM25 index | host-owned; tools only wire |
| Behavior versions / legacy-0.4.10 | not versioned |
| Monitor `MonitorEvent` chat inject | **X** — events on output_file; no pager notification bridge |
| Multi-id wait Notify/join_all | **A** — shared-deadline poll via `task_manager` |
