# Faithful port of Grok pager → Python

**Source root:** `D:\grok-build`  
**SOURCE_REV:** `95d84f443eddcbed6cbfd6eed22e2eafe6b3939d`  
**Primary crate:** `crates/codegen/xai-grok-pager`  
**Theme/glyphs:** `crates/codegen/xai-grok-pager-render`

## Method (hard rules)

1. **Translate Grok Rust source** — do not invent Doggy chrome.
2. Map ratatui `Line`/`Span`/`Style` → `prompt_toolkit` `StyleAndTextTuples` only at the paint edge.
3. Keep Grok names for types/functions when practical.
4. **Doggy-only exceptions:** `tui/doggy_brand.py` logo; Ctrl+L login; image paste; plan policy.
5. Data from Doggy `Message` / coordinator projects **into** Grok-shaped block models, then paint.

## Ownership (parallel agents)

| Owner dir | Grok sources |
|-----------|----------------|
| `glyphs.py` `theme.py` | `pager-render/src/glyphs.rs`, `theme/groknight.rs`, `theme/mod.rs` Theme struct |
| `layout.py` | `pager/src/scrollback/layout.rs`, entry horizontal chrome |
| `blocks/user.py` `thinking.py` `markdown.py` | `blocks/user.rs`, `thinking.rs`, `markdown_content.rs`, `quote_bar.rs` |
| `blocks/tool/*` | `blocks/tool/*.rs` + `tool/snapshots/*.snap` |
| `blocks/subagent.py` `system.py` | `blocks/subagent.rs`, `system.rs`, `bg_task.rs` (minimal) |
| `verb_group.py` | `scrollback/state/verb_group.rs` |
| `prompt.py` `turn_status.py` `shortcuts.py` `status.py` | `views/prompt_widget`, `turn_status.rs`, `shortcuts_bar.rs`, `status_bar.rs`, `context_bar.rs` |
| `scrollback.py` `app.py` `project.py` | orchestration + Doggy Session wiring only |

## Output types (shared contract)

```python
# fragments: list[tuple[str, str]]  style class -> text (no newlines except end of row)
# Row = list of fragments ending with ("", "\n") when painted
# BlockPainter.paint(ctx) -> list[Row]
```

Style classes use `class:grok.*` prefix from theme.
