# Hermes 记忆增强 — 源码级对照

> **Grok 是母体；Hermes 只增强记忆。**  
> 对照根：`C:\Ai\hermes-agent\agent\memory_manager.py` · `memory_provider.py`  
> CodeDoggy 入口：`memory/hermes_seam.py` + `memory/manager.py`

## 等级

| 等级 | 含义 |
|------|------|
| **S** | 源码级：生命周期/围栏/文案/钩子与 Hermes 一致 |
| **C** | 契约对齐：形状对，实现简化 |
| **A** | 近似 |
| **X** | 未做 / host 依赖 |
| **Doggy** | CodeDoggy 增强（非 Hermes） |

## 对照表

| Hermes 源 | CodeDoggy | 等级 | 备注 |
|-----------|-----------|------|------|
| `build_memory_context_block` + system note | `context_fence.build_memory_context_block` | **S** | 权威 reference note 文案一致 |
| `sanitize_context` | `context_fence.sanitize_context` | **S** | fence/note 剥离 |
| `StreamingContextScrubber` | `context_fence.StreamingContextScrubber` | **S** | 流式分块不泄漏 fence |
| Sample-time USER inject only | `sample_messages_with_memory` + loop | **S** | 不写 SYSTEM / 不写 archive |
| `MemoryManager.add_provider` 一外置 | `manager.add_provider` | **S** | 第二 external 拒绝 |
| Core tool shadow 拒绝 | `_CORE_TOOL_NAMES` | **C** | Doggy 核心工具表，非 Hermes `_HERMES_CORE_TOOLS` 全集 |
| `build_system_prompt` | `manager.build_system_prompt` | **S** | provider 静态块拼接 |
| `prefetch_all` / `queue_prefetch_all` | 同名 | **S/C** | 后台单 worker；无 skill-scaffold strip（无 /skill 展开）**C** |
| `sync_all` 后台 | `sync_all` + redact | **S** + **Doggy** | 后台 FIFO；密钥 redact 为增强 |
| `on_turn_start` / `on_session_end` | 同名 | **S** | fail-soft |
| `on_session_switch` + `rewound` 仅 True 传入 | `on_session_switch` | **S** | 不污染 kwargs |
| `commit_session_boundary_async` end→switch | 同名 | **S** | #16454 语义 |
| `on_pre_compress` | 同名 + seam + compactor | **S** | fold 前 |
| `on_memory_write` 镜像外置 | `on_memory_write` / `notify_memory_write` | **S** | 跳过 builtin* |
| `on_delegation` 父观察子 agent | `on_delegation` + subagent 完成时 seam | **S** | 子 agent 无 provider 会话 |
| Provider protocol 钩子 | `BaseMemoryProvider` | **S** | 可选钩子齐全 |
| Builtin curated MEMORY/USER freeze | `CuratedMemoryProvider` + `store` | **S** / **Doggy** | 冻结快照 + drift 规则 |
| Session FTS prefetch | `SessionFtsProvider` | **Doggy** | Hermes 无此 builtin；会话档案召回增强 |
| Plugin discovery | `memory/plugins` + env | **C** | 一外置；格式对齐 spirit |
| Skill-message strip in prefetch/sync | — | **X** | 无 Hermes /skill 展开路径 |
| DaemonThreadPoolExecutor | stdlib `ThreadPoolExecutor` | **C** | 单 worker 语义同；无 daemon atexit 特化 |

## 注入规则（增强记忆系统）

```text
SYSTEM  ← build_system_memory_block  (curated freeze + provider static only)
USER@sample ← <memory-context> fence  (prefetch only; ephemeral)
archive/live  ← 永不写入 fence
```

## 生命周期（seam 唯一入口）

见 [`hermes-groke-seam.md`](hermes-groke-seam.md)。对齐后新增：

| 钩子 | 时机 |
|------|------|
| `on_delegation` | 父 session 子 agent **completed** |
| `StreamingContextScrubber` | UI/stream 输出时剥 fence（host 可选） |
| FTS `rewound` | 清 warm cache |

## 测试

- `tests/test_hermes_seam.py` — 生命周期
- `tests/test_hermes_fence.py` — 围栏
- `tests/test_hermes_memory_align.py` — scrubber / rewound / delegation / memory_write
