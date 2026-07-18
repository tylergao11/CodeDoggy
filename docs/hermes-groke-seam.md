# Grok ↔ Hermes 接缝（记忆向）

> **Grok 是母体；Hermes 只增强记忆。**  
> 实现入口：`codedoggy.memory.hermes_seam`  
> 源码对照：`C:\Ai\hermes-agent\agent\memory_manager.py` 等

## 职责切分

| 层 | 负责 | 不负责 |
|----|------|--------|
| **Grok** | turn loop、sample、tool 执行、compaction 触发、transcript 协议 | 外置记忆后端语义 |
| **Hermes seam** | curated 冻结、FTS 归档召回、prefetch 围栏、provider 生命周期 | 改写 SYSTEM/tool 协议 |
| **CLI / tools** | 入口与工具实现 | 见其他 agent |

## 生命周期（必须走 seam）

```text
build_session
  → bind_session / initialize_all

handle_prompt / turn begin
  → build_system_memory_block  (SYSTEM: curated freeze + provider static)
  → prefetch_fenced            (raw → <memory-context>)
  → on_turn_begin              (on_turn_start + consolidation reset)

run_agent_loop sample
  → sample_messages_with_memory  (只改 sample 副本，不写 archive)

turn end
  → on_turn_end = sync_all + queue_prefetch_all

context fold
  → on_pre_compress (providers)
  → fold
  → on_transcript_rewound(rewound=True)  if folded

memory flush (mid-turn)
  → notify_curated_write → refresh freeze spine

rewind_context
  → on_transcript_rewound

new_session
  → commit_session_boundary (end → switch; async on manager)

session.close
  → on_session_close (end → flush → shutdown_all)
```

## 注入规则（Hermes conversation_loop）

1. **SYSTEM**：仅 curated MEMORY/USER 冻结块 + provider `system_prompt_block`  
2. **Prefetch**：`build_memory_context_block` → `<memory-context>`  
3. **时机**：仅 sample 前拼到**当前 user content**  
4. **禁止**：prefetch 写 SYSTEM、写 archive、单独 USER 轮次  

## seam API（唯一入口）

| 函数 | 何时 |
|------|------|
| `bind_session` | bootstrap / resume |
| `build_system_memory_block` | 每 turn 拼 SYSTEM |
| `prefetch_fenced` | 每 turn 取 fence 文本 |
| `on_turn_begin` / `on_turn_end` | turn 起止 |
| `sample_messages_with_memory` | 每次 sample 前 |
| `on_pre_compress` | fold 前 |
| `on_transcript_rewound` | fold/rewind 后 |
| `commit_session_boundary` | `new_session` |
| `on_session_close` | `close` |
| `notify_curated_write` | MEMORY/USER 写后 / flush |

## 关键文件

| 文件 | 角色 |
|------|------|
| `memory/hermes_seam.py` | **唯一**生命周期编排入口 |
| `memory/manager.py` | Hermes MemoryManager |
| `memory/context_fence.py` | fence / ephemeral inject 原语 |
| `memory/store.py` | MEMORY.md / USER.md |
| `memory/session_store.py` | FTS archive |
| `turn/runner.py` | turn 级 seam |
| `turn/loop.py` | sample-time inject via seam |
| `context/compactor.py` | pre_compress + rewound + flush notify |
| `session/kernel.py` | new_session / close |
| `bootstrap.py` | bind_session |
