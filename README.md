# CodeDoggy

终端里的 coding agent 驾驶舱。

发任务、看 MAIN 与并行子 Agent、打开详情跟思考与工具。包名 `codedoggy`，命令 `doggy`。

## 安装

```bash
uv tool install git+https://github.com/tylergao11/CodeDoggy.git
doggy
```

打开后 **Ctrl+L** 登录模型（Grok / Claude / Codex 等，使用自己的账号）。

```bash
doggy              # 驾驶舱
doggy "修登录"      # 立刻开干
doggy --plain "…"  # 纯文本一次跑完
```

## 说明

- 需要 **Python 3.11+**
- 本机 CLI，不是云端服务
- 仓库：https://github.com/tylergao11/CodeDoggy
