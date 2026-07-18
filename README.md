# CodeDoggy

Coding agent harness (Python).

## Layout

- `session/` — workspace-bound session lifecycle and turn entry
- `turn/` — agentic loop (`sample → tools → writeback`, max_turns, hooks)
- `memory/` — curated persistent MEMORY.md / USER.md (frozen prompt snapshot)
- `audit/` — **Shadow (影子)** write-time soft review (package path kept; product name Shadow)
- `model/` — provider registry + Ollama/OpenAI-compat client + ChatSampler / ShadowAuditor
- `context/` — Grok-style in-session budget + compaction (Hermes MEMORY stays)
- `tools/` — tool registration, dispatch, builtins
- `graph/` — GitHub-style code graph (`code_nav`; xai-codebase-graph spirit)

**Principle:** foundation **Grok** (loop, tools, context window) · enhance **Hermes** (curated + FTS memory).

### Model brains (main + shadow)

Both the coding agent and the **Shadow (影子)** use model brains (Ollama by default).
Shadow ≠ normal offline code audit: it runs *inside* the agent on each write.

```python
from codedoggy import build_session

session = build_session(".", goal="只修登录相关代码", max_turns=24)
result = session.handle_prompt("fix the auth bug")
session.close()
```

```bash
python -m codedoggy.cli --goal "only touch auth" "fix the login bug"
```

| Role | Env (fallback chain) |
|------|----------------------|
| Main | `CODEDOGGY_PROVIDER` / `MODEL` / `BASE_URL` (default ollama + qwen3:8b) |
| Shadow | `CODEDOGGY_AUDIT_*` or `CODEDOGGY_AUX_*` → else same as main |

## Setup

```bash
# package: codedoggy  |  product: CodeDoggy  |  tool namespace: Doggy:*
cd C:\Ai\CodeDoggy
pip install -e ".[dev]"   # includes tree-sitter + grammars + watchdog (hard deps)
pytest
python -m codedoggy.cli
```
