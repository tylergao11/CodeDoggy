# CodeDoggy

Coding agent harness (Python).

## 核心设计

> **以 Grok 为母体继续生长，用 Hermes 记忆、Graph、MAIN 强并行倾向（非自动编排）改造并增强整个系统。**

- **Grok** = 母体（loop / context / tools / permission / orchestration 不变量）
- **Hermes 记忆** = 增强召回与持久化（不改 Grok 的 system/message/tool 协议）
- **Parallel MAIN** = **MAIN 自己的强并行倾向**（prompt + 可选工具）；**不是** runtime 替他自动并行
- **Graph** = 挂在读能力上的代码导航

**Shadow（写时软质检）已从产品路径移除** — 不再默认审计。

详见 [`SCHEME.md`](SCHEME.md)。

## Layout

- `session/` — workspace-bound session lifecycle and turn entry
- `turn/` — agentic loop (`sample → two-phase tools → writeback`, max_turns, hooks)
- `orchestration/` — Grok 编排：Agent 配置、两阶段工具、子会话、plan、插话
- `memory/` — Hermes 记忆适配（curated MEMORY/USER + session FTS）
- `model/` — provider registry + Ollama/OpenAI-compat + ChatSampler
- `context/` — Grok 窗口预算与压缩
- `tools/` — 注册、Gate 权限、builtins（含 MAIN 可选的 `parallel_tasks`）
- `graph/` — 代码图导航（`code_nav`）
- `audit/` — **legacy only**（单测可 `enable_audit=True`；产品默认关闭）

### Main brain + optional multi-agent tools

```python
from codedoggy import build_session

session = build_session(".", goal="只修登录相关代码", max_turns=24)
result = session.handle_prompt("fix the auth bug")
session.close()
```

产品姿态：

1. **MAIN 决定**是否拆任务、何时 `spawn_subagent` / `parallel_tasks`
2. 系统 **不**自动 fan-out、不替 MAIN 并行
3. 若 MAIN 选择并行：可边派子 agent 边做自己的串行工作，最后 **自己汇总**

可用子 agent 类型：`explore` · `plan` · `general-purpose`（均由 MAIN 点名）。

```bash
python -m codedoggy.cli --goal "only touch auth" "fix the login bug"
```

在交互式终端中，`codedoggy` 默认进入任务驾驶舱：老板看 MAIN 与并行子 Agent
的汇报；点击 Agent 或按 `Tab` 后回车，可打开近全屏输出窗口。

```bash
codedoggy                         # 打开交互式任务驾驶舱
codedoggy "检查登录链路"          # 打开驾驶舱并立即启动任务
codedoggy --plain "检查登录链路"  # 单次纯文本输出，适合脚本和 CI
codedoggy --smoke                 # 只验证 session wiring
```

| Role | Env (fallback chain) |
|------|----------------------|
| Main | `CODEDOGGY_PROVIDER` / `MODEL` / `BASE_URL` (default ollama + qwen3:8b) |

## Setup

```bash
# package: codedoggy  |  product: CodeDoggy  |  tool namespace: Doggy:*
cd C:\Ai\CodeDoggy
pip install -e ".[dev]"   # includes tree-sitter + grammars + watchdog (hard deps)
pytest
python -m codedoggy.cli
```
