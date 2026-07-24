# Port audit — product path + paint fidelity

**Date:** 2026-07-24  
**Source rev:** `95d84f443eddcbed6cbfd6eed22e2eafe6b3939d` (`D:\grok-build`)  
**Tests:** `pytest tests/test_tui_v2_port.py tests/test_tui_v2_submit.py -q` (**68 passed**)

---

## Wave (orchestration)

**Done**
- chrome: user/assistant accent `None`; thinking running bullet filled
- shortcuts: honest keys only (no S-Tab / j/k / Ctrl+Q)
- verb_group: `execute` not groupable; `VERB_GROUPABLE` synced
- app: multi-sample `draft_generation` reset on assistant live
- memory_search: Grok markdown parse
- tools: fixed present headers Read/List/Search/Edit/Fetch; read/execute full body when expanded
- Tool `DisplayMode` cycle: collapsed → truncated → expanded (`ScrollItem.cycle_fold` + ←/→)
- `bg_task.py` + project wiring; BTM lifecycle listeners (started/completed/failed/killed)
- tui_v2 app binds listeners + poll fallback → bg_task ScrollItems; failed exit code paint
- `session_event.py` + finish markers; richer kinds (model_unavailable, max_turns, …)
- system word-wrap; markdown HR `━━━`; `+` lists keep `+`
- text selection copy strips quote bars `│`
- tests: listeners, quote strip, upsert — **68 passed**

**Residual**
- OSC8, mermaid
- Full Grok `session_event` enum
- `bg_task` open viewer for stdout
- Truncated host wiring (`truncated` kwarg exists; project always full expand)

---

## Verdict (honest)

| Layer | State |
|-------|--------|
| Paint / block port | **High** on Doggy-critical paths; residual below |
| Critical product path (type → send → turn) | **Wired** — Enter submit, finish reconcile, seed, usage |
| Double user paint | **Fixed** — optimistic user row + `on_live` skip via `_should_skip_live_message` |
| Non-goals | Mermaid, credit_limit, btw, workflow, OSC8, table select |

---

## Product path (done)

| Item | Notes |
|------|--------|
| Enter / Ctrl+J | Multiline `TextArea`; bare Enter → submit when prompt focused; Ctrl+J hard newline (cap 12). No reliable `s-enter`. |
| Finish reconcile | `reconcile_turn_finish`: clear draft, project missing live rows, settle tools/assistant, final_text fallback, error system line |
| Seed history | Last `HISTORY_SEED_CAP = 80` via `seed_from_messages` |
| Usage bar | Sticky last-good from `session.extensions.context.budget` |
| Live slice | `_live_messages_since` — no re-project of seeded history when nothing new |
| Double user paint | Optimistic user paint; live mirror skipped with `_should_skip_live_message` |

### Tests

| File | Coverage |
|------|----------|
| `tests/test_tui_v2_port.py` | Paint/port unit |
| `tests/test_tui_v2_submit.py` | project/seed/reconcile/submit/live_since + skip-live |

---

## High fidelity (parallel audits)

| Area | Notes |
|------|--------|
| Glyphs | Match Grok pager-render set |
| Theme | GrokNight RGB |
| Layout | `A \| PL \| Content \| PR` |
| Edit | Gutters + snaps |
| Enter / Ctrl+J | Submit + hard newline |
| Usage seed | Budget tokens wired |
| Subagent headers | Present |
| Hook inline | Header / inline counts |
| Accent rail | User/assistant accent `None` (Grok) |
| Tool headers | Fixed present: Read/List/Search/Edit/Fetch |
| Tool DisplayMode | collapsed → truncated → expanded via `cycle_fold` + ←/→ |
| verb_group | `execute` not groupable; `VERB_GROUPABLE` synced |
| Shortcuts | Honest keys only |
| Multi-sample | `draft_generation` reset on assistant live |
| memory_search | Grok markdown parse |
| Tool expand | read/execute full body when expanded |
| bg_task | BTM listeners + poll fallback → ScrollItems; failed exit code |
| session_event | Finish markers + richer kinds (model_unavailable, max_turns, …) |
| Quote strip | Selection copy strips quote bars `│` |
| System / MD | Word-wrap; HR `━━━`; `+` lists keep `+` |

---

## Residual (must-have later / nits)

| Item | Notes |
|------|--------|
| OSC8 / mermaid | Terminal hyperlinks; diagram paint |
| Full Grok `session_event` enum | Richer kinds landed; not full enum |
| `bg_task` open viewer | stdout viewer not wired |
| Truncated host wiring | `truncated` kwarg exists; project always full expand |
| `PromptInfo` | Refreshes after turns; not continuous on every model/login event |
| status/context bars | Modules exist; app mostly composes status fragments itself |
| Interject | Works mid-turn; no separate queue UI |

---

## Non-goals

| Item | Notes |
|------|--------|
| Mermaid | Diagram paint / worker |
| OSC8 | Terminal hyperlinks |
| Table select | `table_geometry` multi-cell |
| credit_limit / btw / workflow | Full Grok product blocks |
| Legacy dual path | `CODEDOGGY_TUI=legacy` — not a fidelity target |
| Full Grok prompt editor | File-search chips, Apple Terminal S-Enter rescue, etc. |

---

## UX (keys that work)

| Input | Action |
|-------|--------|
| Enter (prompt focused) | Submit / queue interject |
| Ctrl+J | Hard newline |
| ↑ / ↓ | History (prompt) / block select (scrollback) |
| → / ← | Expand / re-fold verb group or tool body (DisplayMode cycle) |
| Tab | Toggle focus prompt ↔ scrollback |
| Drag / double-click | Text selection / expand group+tool |
| Ctrl+C / Ctrl+Y | Copy selection (else clear / cancel / exit) |
| Ctrl+L / Ctrl+V | Login / clipboard paste (Doggy) |
| Esc | Cancel turn / double-Esc clear prompt |

---

## Verify

```bash
pytest tests/test_tui_v2_port.py tests/test_tui_v2_submit.py -q
# Manual: type → Enter → single user row (no double paint); assistant remains after turn
# Manual: status bar tokens after turn; resume shows last 80
# Manual: tool ←/→ cycles collapsed → truncated → expanded
```
