"""CLI host adapter for ``ask_user_question`` (stdin/stdout multi-choice).

Wired by the host via ``extra['ask_user_fn']`` / ``kernel.tool_extra['ask_user_fn']``.
Does not implement ACP/TUI chrome — plain numbered prompts only.

Host contract (see ``builtins/ask_user_question.py``):

* Input: list of question dicts
  ``{question, options[{label, description, preview?}], multi_select?}``
* Output: outcome dict accepted by ``coerce_host_result`` /
  ``ext_response_to_user_response``::

    {"outcome": "accepted", "answers": {question: [labels…]}}
    {"outcome": "cancelled"}
    {"outcome": "chat_about_this", "partial_answers": {question: str}}
    {"outcome": "skip_interview", "partial_answers": {question: str}}

Non-interactive / non-TTY: returns cancelled (or configurable skip) without
blocking on ``input()``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Any, TextIO

# Outcome names mirror AskUserQuestionExtResponse / coerce_host_result.
OUTCOME_ACCEPTED = "accepted"
OUTCOME_CANCELLED = "cancelled"
OUTCOME_CHAT = "chat_about_this"
OUTCOME_SKIP = "skip_interview"

_DEFAULT_NONINTERACTIVE = OUTCOME_CANCELLED

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]
IsattyFn = Callable[[], bool]


def is_interactive(
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> bool:
    """True when both stdin and stdout look like a real terminal."""
    sin = stdin if stdin is not None else sys.stdin
    sout = stdout if stdout is not None else sys.stdout
    try:
        in_tty = bool(sin.isatty())
    except Exception:  # noqa: BLE001 — broken stream objects
        in_tty = False
    try:
        out_tty = bool(sout.isatty())
    except Exception:  # noqa: BLE001
        out_tty = False
    return in_tty and out_tty


def make_ask_user_fn(
    *,
    input_fn: InputFn | None = None,
    output_fn: OutputFn | None = None,
    interactive: bool | None = None,
    noninteractive_outcome: str = _DEFAULT_NONINTERACTIVE,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Build an ``ask_user_fn`` suitable for ``extra['ask_user_fn']``.

    Parameters
    ----------
    input_fn / output_fn
        Inject for tests (defaults: ``input`` / print-to-stdout).
    interactive
        Force interactive or non-interactive mode. ``None`` → detect TTY.
    noninteractive_outcome
        ``"cancelled"`` (default) or ``"skip_interview"`` when not interactive.
    stdin / stdout
        Streams used for TTY detection and default I/O.
    """

    def _fn(questions: list[dict[str, Any]]) -> dict[str, Any]:
        return ask_user_cli(
            questions,
            input_fn=input_fn,
            output_fn=output_fn,
            interactive=interactive,
            noninteractive_outcome=noninteractive_outcome,
            stdin=stdin,
            stdout=stdout,
        )

    return _fn


def ask_user_cli(
    questions: list[dict[str, Any]],
    *,
    input_fn: InputFn | None = None,
    output_fn: OutputFn | None = None,
    interactive: bool | None = None,
    noninteractive_outcome: str = _DEFAULT_NONINTERACTIVE,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> dict[str, Any]:
    """Prompt the user for multi-choice answers on a CLI.

    Compatible with ``AskUserQuestionTool`` host payload and
    ``coerce_host_result`` outcome dicts.
    """
    if not isinstance(questions, list):
        return {"outcome": OUTCOME_CANCELLED}
    if not questions:
        return {
            "outcome": OUTCOME_ACCEPTED,
            "answers": {},
        }

    sin = stdin if stdin is not None else sys.stdin
    sout = stdout if stdout is not None else sys.stdout

    if interactive is None:
        interactive = is_interactive(stdin=sin, stdout=sout)

    if not interactive:
        return _noninteractive_result(questions, noninteractive_outcome)

    write = output_fn or (lambda s: print(s, file=sout, flush=True))
    read = input_fn or (lambda prompt: input(prompt))

    answers: dict[str, list[str]] = {}
    partial: dict[str, str] = {}

    n = len(questions)
    for idx, raw_q in enumerate(questions):
        q = _normalize_question(raw_q, index=idx)
        if q is None:
            continue
        write("")
        write(f"── Question {idx + 1}/{n} ──")
        result = _prompt_one(
            q,
            write=write,
            read=read,
        )
        if result["kind"] == "cancel":
            return {"outcome": OUTCOME_CANCELLED}
        if result["kind"] == "skip":
            return {
                "outcome": OUTCOME_SKIP,
                "partial_answers": dict(partial),
            }
        if result["kind"] == "chat":
            return {
                "outcome": OUTCOME_CHAT,
                "partial_answers": dict(partial),
            }
        labels: list[str] = result["labels"]
        answers[q["question"]] = labels
        partial[q["question"]] = ", ".join(labels)

    return {"outcome": OUTCOME_ACCEPTED, "answers": answers}


# ── internals ─────────────────────────────────────────────────────────────


def _noninteractive_result(
    questions: Sequence[dict[str, Any]],
    outcome: str,
) -> dict[str, Any]:
    if outcome == OUTCOME_SKIP:
        return {
            "outcome": OUTCOME_SKIP,
            "partial_answers": {},
        }
    # Default and unknown → cancelled (honest: no user present to answer).
    return {"outcome": OUTCOME_CANCELLED}


def _normalize_question(
    raw: Any, *, index: int
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("question") or "").strip()
    if not text:
        return None
    opts_raw = raw.get("options")
    options: list[dict[str, str | None]] = []
    if isinstance(opts_raw, list):
        for o in opts_raw:
            if not isinstance(o, dict):
                continue
            label = str(o.get("label") or "").strip()
            if not label:
                continue
            desc = o.get("description")
            preview = o.get("preview")
            options.append(
                {
                    "label": label,
                    "description": str(desc) if desc is not None else "",
                    "preview": str(preview) if preview is not None else None,
                }
            )
    multi = raw.get("multi_select")
    if multi is None and "multiSelect" in raw:
        multi = raw.get("multiSelect")
    multi_select = bool(multi) if multi is not None else False
    return {
        "question": text,
        "options": options,
        "multi_select": multi_select,
        "_index": index,
    }


def _prompt_one(
    q: dict[str, Any],
    *,
    write: OutputFn,
    read: InputFn,
) -> dict[str, Any]:
    """Prompt for a single question. Returns kind=cancel|skip|chat|answer."""
    options: list[dict[str, str | None]] = list(q["options"])
    multi = bool(q["multi_select"])
    question_text = str(q["question"])

    write(question_text)
    if multi:
        write("  (multi-select — enter numbers separated by commas, e.g. 1,3)")
    else:
        write("  (single-select — enter one number)")

    for i, opt in enumerate(options, start=1):
        write(f"  {i}) {opt['label']}")
        desc = (opt.get("description") or "").strip()
        if desc:
            write(f"      {desc}")
        preview = opt.get("preview")
        if preview:
            # Indent preview lines for readability; keep short dumps usable.
            for line in str(preview).splitlines() or [""]:
                write(f"      | {line}")

    other_n = len(options) + 1
    write(f"  {other_n}) Other (type your own answer)")
    write(
        "  Commands: c=cancel  s=skip interview  h=chat about this  ?=help"
    )

    while True:
        try:
            raw = read("> ")
        except EOFError:
            return {"kind": "cancel"}
        except KeyboardInterrupt:
            write("")
            return {"kind": "cancel"}

        line = (raw or "").strip()
        if not line:
            write("  Enter a number, or c / s / h.")
            continue

        lower = line.lower()
        if lower in {"c", "cancel", "q", "quit"}:
            return {"kind": "cancel"}
        if lower in {"s", "skip"}:
            return {"kind": "skip"}
        if lower in {"h", "chat", "talk"}:
            return {"kind": "chat"}
        if lower in {"?", "help"}:
            write(
                "  Pick option number(s). "
                "For free text: type the Other number, then your answer.\n"
                "  c cancel · s skip interview · h chat about this"
            )
            continue

        # Free-text "Other" via `o <text>` / `other <text>` shorthand.
        if lower.startswith("o ") or lower.startswith("other "):
            if lower.startswith("other "):
                text = line[6:].strip()
            else:
                text = line[2:].strip()
            if not text:
                write("  Other needs text after 'o '.")
                continue
            return {"kind": "answer", "labels": [text]}

        # Number selection (single or multi).
        if _looks_like_indices(line):
            picked = _parse_indices(line, n_options=len(options), other_n=other_n)
            if picked is None:
                write(
                    f"  Invalid selection. Use 1–{other_n}"
                    + (" (comma-separated ok)." if multi else ".")
                )
                continue
            if not multi and len(picked) > 1:
                write("  Single-select: pick one number only.")
                continue
            labels: list[str] = []
            need_other = False
            for num in picked:
                if num == other_n:
                    need_other = True
                else:
                    labels.append(str(options[num - 1]["label"]))
            if need_other:
                try:
                    other_text = read("  Other: ").strip()
                except (EOFError, KeyboardInterrupt):
                    return {"kind": "cancel"}
                if not other_text:
                    write("  Other answer cannot be empty.")
                    continue
                labels.append(other_text)
            if not labels:
                write("  No options selected.")
                continue
            # De-dupe preserving order
            seen: set[str] = set()
            uniq: list[str] = []
            for lab in labels:
                if lab not in seen:
                    seen.add(lab)
                    uniq.append(lab)
            return {"kind": "answer", "labels": uniq}

        # Unrecognized — do not silently treat free text as Other (too easy
        # to accept typos). Require explicit Other path.
        write(
            f"  Unrecognized input {line!r}. "
            f"Enter 1–{other_n}, or c / s / h (type ? for help)."
        )


def _looks_like_indices(line: str) -> bool:
    """True if line is only digits, commas, spaces, or ranges like 1-3."""
    cleaned = line.replace(",", " ").replace("-", " ")
    parts = cleaned.split()
    if not parts:
        return False
    return all(p.isdigit() for p in parts)


def _parse_indices(
    line: str, *, n_options: int, other_n: int
) -> list[int] | None:
    """Parse '1', '1,3', '1 2', '1-3' into 1-based indices within range."""
    tokens: list[str] = []
    for chunk in line.replace(",", " ").split():
        if "-" in chunk and chunk.count("-") == 1:
            a, b = chunk.split("-", 1)
            if a.isdigit() and b.isdigit():
                lo, hi = int(a), int(b)
                if lo > hi:
                    lo, hi = hi, lo
                tokens.extend(str(i) for i in range(lo, hi + 1))
                continue
        tokens.append(chunk)

    out: list[int] = []
    for t in tokens:
        if not t.isdigit():
            return None
        n = int(t)
        if n < 1 or n > other_n:
            return None
        out.append(n)
    if not out:
        return None
    return out
