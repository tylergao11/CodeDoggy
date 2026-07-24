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
doggy              # Grok 风格会话壳（scrollback + prompt）
doggy "修登录"      # 立刻开干
doggy --plain "…"  # 命令行 / CI 单次输出
```

TUI 默认是 GrokBuild pager 外形（配色 GrokNight、底栏 prompt、scrollback）。
Doggy 保留：**两只狗品牌**、**Ctrl+L 登录**、**粘贴图片**；plan/auto 策略仍按 Doggy。
回退旧任务卡驾驶舱：`CODEDOGGY_TUI=legacy doggy`。

对照源码：`D:\grok-build`（`xai-org/grok-build`）。

## 说明

- Python 3.11+
- 本机 CLI，不是云端服务
- 仓库：https://github.com/tylergao11/CodeDoggy
