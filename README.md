# CodeDoggy

Python coding-agent harness：会话循环、工具执行、记忆、代码导航与终端驾驶舱。

## 核心设计

- **MAIN 主脑** — 由主 agent 决定是否拆任务、何时派子 agent；runtime **不**自动并行
- **Turn loop** — `sample → 两阶段工具 → writeback`，含压缩、插话、plan/goal
- **记忆** — curated MEMORY/USER + 会话 FTS 检索
- **Graph** — 代码定义/引用导航（`code_nav`）
- **多模型** — Grok / Claude / Codex OAuth 与各类 API key，登录连接统一驱动聊天与媒体能力

详见 [`SCHEME.md`](SCHEME.md)。

## Layout

- `session/` — 工作区会话生命周期与入口
- `turn/` — agent 循环（采样、工具、max_turns、hooks）
- `orchestration/` — Agent 配置、两阶段工具、子会话、plan、插话
- `memory/` — curated 记忆 + session FTS
- `model/` — provider 注册、鉴权、ChatSampler
- `context/` — 上下文窗口预算与压缩
- `tools/` — 注册、Gate 权限、builtins（含可选 `parallel_tasks`）
- `graph/` — 代码图导航（`code_nav`）
- `tui/` — 交互式任务驾驶舱

### 快速调用

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
doggy --goal "only touch auth" "fix the login bug"
```

在交互式终端中，`doggy` 默认进入任务驾驶舱：查看 MAIN 与并行子 Agent
的汇报；点击 Agent 或按 `Tab` 后回车，可打开近全屏输出窗口。

```bash
doggy                         # 打开交互式任务驾驶舱
doggy "检查登录链路"          # 打开驾驶舱并立即启动任务
doggy --plain "检查登录链路"  # 单次纯文本输出，适合脚本和 CI
doggy --smoke                 # 只验证 session wiring
```

| Role | Env |
|------|-----|
| Main | `CODEDOGGY_PROVIDER` / `CODEDOGGY_MODEL` / `CODEDOGGY_BASE_URL`（默认 ollama + qwen3:8b） |

## Setup

```bash
# package: codedoggy  |  CLI: doggy  |  tool namespace: Doggy:*
cd /path/to/CodeDoggy
pip install -e ".[dev]"
pytest
doggy
```

也可用（装完直接敲 `doggy`）：

```bash
uv tool install -e .
# 或
uv tool install git+https://github.com/tylergao11/CodeDoggy.git
doggy
```


