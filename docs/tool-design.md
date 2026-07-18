# Tool design (model-facing contract)

Tools are the model’s action/observation API. Small contract details change
success rate more than prompt wording. This document records what CodeDoggy hardens
in each common tool and why.

## Principles

1. **Prefer dedicated tools over shell** for read / search / edit.
2. **Observations must be short, localizable, and recoverable.**
3. **Errors are part of the API** — actionable, stable codes when useful.
4. **Defaults are finite** — omitted args never mean “unbounded dump.”
5. **Descriptions state hard constraints** (timeouts, missing unix utils, FG-only).
6. **Claim matches implement** — do not advertise Job Objects, PDF readers, or
   background tasks that are not wired.

## Per-tool details

### `read_file`

| Detail | Behavior | Why model-friendly |
|--------|----------|--------------------|
| Line prefix | `N→` on first visible line and every line number divisible by 10 | Cuts tokens vs numbering every line; still jumpable |
| Default window | 1000 lines | Caps context without model guessing |
| `offset` | 1-based; `0`→1; negatives from last content line (`-1` = last line) | Pagination without re-reading whole file |
| Sparse prefix | first visible line + every line_num % 10 | Token savings; still jumpable |
| Window then size guard | Page with offset/limit first; reject only if the window is huge | Offset/limit remain useful |
| Binary reject | extension list + null-byte / non-printable sample | Avoids garbage tokens |
| UTF-8 | `errors=replace` | Never crash mid-turn on mixed encodings |
| Empty file | empty string | Stable, no fake content |
| Formats | plain text only (no PDF/image claims) | Honest surface |

### `search_replace`

| Detail | Behavior | Why |
|--------|----------|-----|
| Exact match only | string equality on LF-normalized text | Deterministic edits |
| CRLF | Match ignores `\r`; write preserves original CRLF | Models send LF after `read_file` |
| Bytes write | `write_bytes` (not text mode) | Windows text mode would force CRLF on LF files |
| Ambiguous match | error unless `replace_all` | Prevents silent multi-edit |
| Empty `old_string` | create file | Explicit create path |
| Success messages | “updated successfully” / “created successfully” | Clear terminal state for next sample |
| No-match hint | nearest line snippet + re-read nudge | Recovery without shell `cat` |
| Path `NAME_MAX` | 255 | Fail early on bad names |
| Description warns | do not include `N→` in `old_string` | Stops the #1 edit failure mode |

### `list_dir`

| Detail | Behavior | Why |
|--------|----------|-----|
| Format | `- root/\n  - child` | Parseable hierarchy |
| Hide dot entries | yes (name starts with `.`) | Noise reduction |
| No gitignore | not applied | Explicit; use deeper path or grep |
| Char budget | 10_000 | No directory dumps |
| Depth cap | 3 | Bounded expansion |
| File-vs-dir errors | distinct messages | Model can correct path |

### `grep`

| Detail | Behavior | Why |
|--------|----------|-----|
| Prefer `rg` | subprocess when present | Fast, correct ignore rules |
| rg exit codes | 0/1 matches or miss; exit 2 → error (not “No matches”) | Bad regex must not look like a miss |
| No-rg fallback | pure Python; **rejects** context/`type`/multiline/complex glob | No silent flag drop |
| Output card | `<workspace_result>` + `Found N matching lines` | Stable parse surface |
| Default head | 200 content lines | Stops mega-repos early |
| Hard caps | 2000 content lines | Explicit limit still bounded |
| Line char cap | 1000 | Minified files don’t explode |
| Timeout | 20s (60s WSL) | Wall clock, not infinite |
| No matches | explicit message | Not empty silence |
| Mode | content only (path:line:text) | Schema matches behavior |

### `run_terminal_cmd`

| Detail | Behavior | Why |
|--------|----------|-----|
| Whole command as one arg | `-Command` / `-c` / `/C` | Pipes and quotes stay in the shell |
| Shell detect | pwsh → powershell → git bash → cmd | Matches host, not assumed bash |
| UTF-8 env | `PYTHONUTF8`, `PYTHONIOENCODING` | Child tools don’t mojibake |
| Default timeout | 120s, max 300s; `0`→default | Finite by default |
| Timeout kill | Windows: `taskkill /T`; POSIX: process group SIGTERM→SIGKILL | Descendants die with the shell |
| Output card | first line `exit: N` or `exit: killed (timeout)` | Always parseable |
| Output cap | 20_000 chars | Shell is noisy; hard stop |
| `description` required | yes | Forces intentional shell use |
| Background | not supported; trailing `&` rejected | FG-only until task subsystem exists |
| Description | no unix utils on PowerShell/cmd | Steers model to `grep`/`read_file` |
| Chain note | `;` when `&&` unsupported | Fewer shell syntax failures |

## What not to put in tools

- Parallel batching of multiple tool calls → turn executor
- Permission / sandbox → policy extension
- System-wide “how to code” rules → prompts / project rules
- Background task_id / kill_task → future tools pack

## Defaults module

All magic numbers live in `codedoggy/tools/defaults.py` so they stay
discoverable and testable.
