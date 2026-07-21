# CodeDoggy

承载会话循环、工具执行、主脑驱动的多 Agent 编排、记忆与上下文压缩、代码结构导航；兼容 Grok / Claude / Codex / DeepSeek，提供终端任务驾驶舱，支持命令行与持续集成。

包名 `codedoggy`，命令 `doggy`。

## 安装

```bash
uv tool install git+https://github.com/tylergao11/CodeDoggy.git
doggy
```

打开后 **Ctrl+L** 登录模型（使用自己的账号）。

```bash
doggy              # 终端任务驾驶舱
doggy "修登录"      # 立刻开干
doggy --plain "…"  # 命令行 / CI 单次输出
```

## 说明

- Python 3.11+
- 本机 CLI，不是云端服务
- 仓库：https://github.com/tylergao11/CodeDoggy
