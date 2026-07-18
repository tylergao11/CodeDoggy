"""CLI host adapter for ask_user_question — stdin multi-choice."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.host.ask_user_cli import (
    OUTCOME_ACCEPTED,
    OUTCOME_CANCELLED,
    OUTCOME_CHAT,
    OUTCOME_SKIP,
    ask_user_cli,
    is_interactive,
    make_ask_user_fn,
)
from codedoggy.tools.builtins.ask_user_question import (
    CANCEL_TEXT,
    coerce_host_result,
    parse_questions,
)
from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.runtime import ToolCallContext


def _q(
    text: str,
    labels: list[str],
    *,
    multi: bool | None = None,
    previews: dict[str, str] | None = None,
) -> dict:
    opts = []
    for lab in labels:
        o: dict = {"label": lab, "description": f"desc for {lab}"}
        if previews and lab in previews:
            o["preview"] = previews[lab]
        opts.append(o)
    d: dict = {"question": text, "options": opts}
    if multi is not None:
        d["multi_select"] = multi
    return d


class _Scripted:
    """Queue of input lines; records output lines."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = list(lines)
        self.out: list[str] = []
        self.prompts: list[str] = []

    def write(self, s: str) -> None:
        self.out.append(s)

    def read(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        if not self.lines:
            raise EOFError
        return self.lines.pop(0)


# ── Non-interactive ──────────────────────────────────────────────────────


def test_noninteractive_returns_cancelled() -> None:
    qs = [_q("Pick?", ["A", "B"])]
    out = ask_user_cli(qs, interactive=False)
    assert out == {"outcome": OUTCOME_CANCELLED}


def test_noninteractive_skip_outcome() -> None:
    qs = [_q("Pick?", ["A", "B"])]
    out = ask_user_cli(
        qs, interactive=False, noninteractive_outcome=OUTCOME_SKIP
    )
    assert out["outcome"] == OUTCOME_SKIP
    assert out["partial_answers"] == {}


def test_make_ask_user_fn_noninteractive() -> None:
    fn = make_ask_user_fn(interactive=False)
    out = fn([_q("Q?", ["X"])])
    assert out["outcome"] == OUTCOME_CANCELLED


def test_empty_questions_accepted() -> None:
    out = ask_user_cli([], interactive=True)
    assert out == {"outcome": OUTCOME_ACCEPTED, "answers": {}}


def test_bad_questions_type_cancelled() -> None:
    out = ask_user_cli("not-a-list")  # type: ignore[arg-type]
    assert out["outcome"] == OUTCOME_CANCELLED


# ── Interactive single-select ────────────────────────────────────────────


def test_single_select_by_number() -> None:
    script = _Scripted(["1"])
    qs = [_q("Which database?", ["Redis", "Postgres"])]
    out = ask_user_cli(
        qs,
        interactive=True,
        input_fn=script.read,
        output_fn=script.write,
    )
    assert out["outcome"] == OUTCOME_ACCEPTED
    assert out["answers"] == {"Which database?": ["Redis"]}
    joined = "\n".join(script.out)
    assert "Which database?" in joined
    assert "1) Redis" in joined
    assert "2) Postgres" in joined
    assert "Other" in joined


def test_single_select_second_option() -> None:
    script = _Scripted(["2"])
    out = ask_user_cli(
        [_q("Pick one?", ["A (Recommended)", "B"])],
        interactive=True,
        input_fn=script.read,
        output_fn=script.write,
    )
    assert out["answers"]["Pick one?"] == ["B"]


def test_other_via_number_then_text() -> None:
    script = _Scripted(["3", "Cassandra"])
    out = ask_user_cli(
        [_q("Which database?", ["Redis", "Postgres"])],
        interactive=True,
        input_fn=script.read,
        output_fn=script.write,
    )
    assert out["answers"]["Which database?"] == ["Cassandra"]


def test_other_shorthand() -> None:
    script = _Scripted(["o MySQL"])
    out = ask_user_cli(
        [_q("Which database?", ["Redis", "Postgres"])],
        interactive=True,
        input_fn=script.read,
        output_fn=script.write,
    )
    assert out["answers"]["Which database?"] == ["MySQL"]


def test_preview_is_printed() -> None:
    script = _Scripted(["1"])
    qs = [
        _q(
            "Pick DB?",
            ["Postgres", "SQLite"],
            previews={"SQLite": "```\nSELECT 1;\n```"},
        )
    ]
    ask_user_cli(
        qs,
        interactive=True,
        input_fn=script.read,
        output_fn=script.write,
    )
    joined = "\n".join(script.out)
    assert "SELECT 1;" in joined


# ── Multi-select ─────────────────────────────────────────────────────────


def test_multi_select_comma_separated() -> None:
    script = _Scripted(["1,3"])
    out = ask_user_cli(
        [
            _q(
                "Pick features?",
                ["Auth", "Billing", "Search"],
                multi=True,
            )
        ],
        interactive=True,
        input_fn=script.read,
        output_fn=script.write,
    )
    assert out["answers"]["Pick features?"] == ["Auth", "Search"]


def test_multi_select_rejects_in_single() -> None:
    """Single-select with multi numbers re-prompts, then accepts one."""
    script = _Scripted(["1,2", "1"])
    out = ask_user_cli(
        [_q("Pick one?", ["A", "B"])],
        interactive=True,
        input_fn=script.read,
        output_fn=script.write,
    )
    assert out["answers"]["Pick one?"] == ["A"]
    assert any("Single-select" in line for line in script.out)


# ── Global commands ──────────────────────────────────────────────────────


def test_cancel_command() -> None:
    script = _Scripted(["c"])
    out = ask_user_cli(
        [_q("Q?", ["A"])],
        interactive=True,
        input_fn=script.read,
        output_fn=script.write,
    )
    assert out == {"outcome": OUTCOME_CANCELLED}


def test_eof_is_cancel() -> None:
    script = _Scripted([])  # immediate EOF
    out = ask_user_cli(
        [_q("Q?", ["A"])],
        interactive=True,
        input_fn=script.read,
        output_fn=script.write,
    )
    assert out["outcome"] == OUTCOME_CANCELLED


def test_skip_with_partial() -> None:
    script = _Scripted(["1", "s"])
    out = ask_user_cli(
        [
            _q("Which database?", ["Redis", "Postgres"]),
            _q("Which framework?", ["React", "Vue"]),
        ],
        interactive=True,
        input_fn=script.read,
        output_fn=script.write,
    )
    assert out["outcome"] == OUTCOME_SKIP
    assert out["partial_answers"]["Which database?"] == "Redis"
    assert "Which framework?" not in out["partial_answers"]


def test_chat_about_this_with_partial() -> None:
    script = _Scripted(["2", "h"])
    out = ask_user_cli(
        [
            _q("Which database?", ["Redis", "Postgres"]),
            _q("Which framework?", ["React", "Vue"]),
        ],
        interactive=True,
        input_fn=script.read,
        output_fn=script.write,
    )
    assert out["outcome"] == OUTCOME_CHAT
    assert out["partial_answers"]["Which database?"] == "Postgres"


def test_two_questions_accepted() -> None:
    script = _Scripted(["1", "2"])
    out = ask_user_cli(
        [
            _q("Which database?", ["Redis", "Postgres"]),
            _q("Which framework?", ["React", "Vue"]),
        ],
        interactive=True,
        input_fn=script.read,
        output_fn=script.write,
    )
    assert out["outcome"] == OUTCOME_ACCEPTED
    assert out["answers"] == {
        "Which database?": ["Redis"],
        "Which framework?": ["Vue"],
    }


# ── coerce_host_result / tool integration ────────────────────────────────


def test_accepted_coerces_to_path_a() -> None:
    qs = parse_questions(
        [
            {
                "question": "Which database?",
                "options": [
                    {"label": "Redis", "description": "in-mem"},
                    {"label": "Postgres", "description": "sql"},
                ],
            }
        ]
    )
    host = {
        "outcome": OUTCOME_ACCEPTED,
        "answers": {"Which database?": ["Redis"]},
    }
    text = coerce_host_result(host, qs)
    assert text.startswith("User has answered your questions:")
    assert '"Which database?"="Redis"' in text


def test_cancelled_coerces_to_cancel_text() -> None:
    qs = parse_questions(
        [
            {
                "question": "Q?",
                "options": [{"label": "A", "description": "a"}],
            }
        ]
    )
    assert coerce_host_result({"outcome": OUTCOME_CANCELLED}, qs) == CANCEL_TEXT


def test_tool_with_cli_adapter(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    script = _Scripted(["1"])
    fn = make_ask_user_fn(
        interactive=True,
        input_fn=script.read,
        output_fn=script.write,
    )
    ctx = ToolCallContext(cwd=tmp_path, extra={"ask_user_fn": fn})
    out = tools.call(
        "ask_user_question",
        {
            "questions": [
                {
                    "question": "Which database?",
                    "options": [
                        {"label": "Redis", "description": "in-mem"},
                        {"label": "Postgres", "description": "sql"},
                    ],
                }
            ]
        },
        ctx,
    )
    assert out.startswith("User has answered your questions:")
    assert '"Which database?"="Redis"' in out


def test_tool_noninteractive_cancel(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    fn = make_ask_user_fn(interactive=False)
    ctx = ToolCallContext(cwd=tmp_path, extra={"ask_user_fn": fn})
    out = tools.call(
        "ask_user_question",
        {
            "questions": [
                {
                    "question": "Q?",
                    "options": [{"label": "A", "description": "a"}],
                }
            ]
        },
        ctx,
    )
    assert out == CANCEL_TEXT


def test_tool_skip_interview_path(tmp_path: Path) -> None:
    tools = ToolRegistryBuilder.new().finalize()
    script = _Scripted(["1", "s"])
    fn = make_ask_user_fn(
        interactive=True,
        input_fn=script.read,
        output_fn=script.write,
    )
    ctx = ToolCallContext(cwd=tmp_path, extra={"ask_user_fn": fn})
    out = tools.call(
        "ask_user_question",
        {
            "questions": [
                {
                    "question": "Which database?",
                    "options": [
                        {"label": "Redis", "description": "in-mem"},
                        {"label": "Postgres", "description": "sql"},
                    ],
                },
                {
                    "question": "Which framework?",
                    "options": [
                        {"label": "React", "description": "ui"},
                        {"label": "Vue", "description": "ui2"},
                    ],
                },
            ]
        },
        ctx,
    )
    assert "enough answers for the plan interview" in out
    assert "Answer: Redis" in out
    assert "(No answer provided)" in out


def test_is_interactive_false_on_non_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoTty:
        def isatty(self) -> bool:
            return False

    assert is_interactive(stdin=_NoTty(), stdout=_NoTty()) is False  # type: ignore[arg-type]
