# Grok source map — port only; invent only as glue

> **Rule (non-negotiable):**  
> 1. Behavior that claims Grok must be a source-level port from `C:\Ai\grok-build`.  
> 2. **Wrong / invented “Grok” code is deleted**, not papered over.  
> 3. Custom code is allowed **only** where the Python glue layer cannot replicate
>    the Rust stack (label **CodeDoggy glue**).

Local rev: `C:\Ai\grok-build\SOURCE_REV`.

## Ported (source-backed)

| Area | Grok source | CodeDoggy |
|------|-------------|-----------|
| Interjection format | `common/xai-interjection-core/src/format.rs` | `orchestration/interjection.py` |
| Interjection drain | `…/buffer.rs` `drain_formatted` | `InterjectionBuffer.drain_formatted` |
| Drain timing | shell `turn.rs` safe points | loop head + post-tool / followup |
| Task isolation/resume schema | `common/xai-tool-types/src/task.rs` | `SubagentRequest` fields + resume checks |
| Resume completed-only / fail copy | shell `subagent/handle_request.rs` | `SubagentCoordinator.resume` |
| Plan edit gate | plan mode edit gate | `plan_mode_edit_gate` |
| Tool batch phase-2 path locks | shell `tool_calls.rs` + `tool_dispatch.rs` | `execute_approved_batch` + main `loop` phase-2 |
| Tool renames | grok-agent tool surface | `tools/grok_surface.py` |
| Task output / multi-wait / kill messages | `xai-tool-types/task.rs` + `grok_build/task_output` + `kill_task` + `types/output.rs` | `tools/grok_build/task_output_logic.py` + builtins |
| Monitor constants / line / rate-limit / start text | `grok_build/monitor/{types,event,rate_limiter,tool}.rs` | `tools/grok_build/monitor_*.py` + `builtins/monitor.py` |
| Hermes memory lifecycle | `C:\Ai\hermes-agent` | `memory/hermes_seam.py` |

## CodeDoggy glue only (cannot full-port)

| Area | Why not full port | What we keep |
|------|-------------------|--------------|
| Worktree engine | Full stack is `xai-fast-worktree` + pool + shell session | Minimal `git worktree` create/reattach/remove |
| Merge into parent | Host “explicitly merged” without workspace RPC | `merge_worktree_into_parent` (git merge/squash/ff) |
| Resume transcript store | No ConversationItem / sampling types | Serialize `Message` dicts for prior_messages |
| Progressive sample deltas | Host CLI/ACP display | Optional `stream_sample` / `on_sample_delta` |
| Monitor notification pipeline | `ToolNotificationHandle` + pager auto-wake | Output file + get_task_output poll |
| Multi-wait event-driven | `wait_*_event_driven` Notify/join_all | Shared-deadline poll on task_manager |
| Kill Job Object (Windows) | terminal actor Job Object | `util/job_object.py` + taskkill fallback |

## Deleted inventions (do not reintroduce)

| Deleted | Why |
|---------|-----|
| Mid-stream sample interrupt on interjection push | Grok drains at **safe points**, not mid-SSE |
| `goal_mode_tool_gate` allowlists | Not in Grok source; goal is state + `update_goal` tool |
| `[interjection]` USER prefix | Wrong wire shape vs `format_interjection` |

## Interjection wire (locked to source tests)

```
The user sent a message while you were working:
<user_query>
{user text}
</user_query>
```

## Checklist for every change

1. Open the Grok (or Hermes) file first.  
2. Port names, order, fail-closed rules.  
3. Cite path in module docstring + this map.  
4. No source → either **delete** or mark **CodeDoggy glue** (never “Grok”).
