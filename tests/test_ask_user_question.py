"""ask_user_question — Grok format.rs / mod.rs fidelity tests.

Ported test intent from:
  ask_user_question/format.rs
  ask_user_question/mod.rs
  ask_user_question/types.rs
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoggy.tools.builtins.ask_user_question import (
    CANCEL_TEXT,
    AskUserQuestionTool,
    Question,
    QuestionAnnotation,
    QuestionOption,
    UserQuestionResponseAccepted,
    UserQuestionResponseCancelled,
    UserQuestionResponseChatAboutThis,
    UserQuestionResponseSkipInterview,
    coerce_host_result,
    format_accepted_tool_result,
    format_chat_about_this,
    format_id_keyed_accepted_tool_result,
    format_skip_interview,
)
from codedoggy.tools.runtime import ToolCallContext, ToolError
from codedoggy.tools import ToolRegistryBuilder


def _tools():
    return ToolRegistryBuilder.new().finalize()


def _make_question(text: str, labels: list[str]) -> Question:
    return Question(
        question=text,
        options=[
            QuestionOption(label=lab, description=f"Description for {lab}")
            for lab in labels
        ],
    )


def _id_keyed_q(qid: str, prompt: str, opts: list[tuple[str, str]]) -> Question:
    return Question(
        question=prompt,
        options=[
            QuestionOption(label=label, description=label, id=oid)
            for oid, label in opts
        ],
        id=qid,
    )


# ── Path A: format_accepted_tool_result ──────────────────────────────────


def test_format_accepted_single_no_annotations() -> None:
    answers = {"Which database?": ["Redis (Recommended)"]}
    result = format_accepted_tool_result(answers, None)
    assert result == (
        'User has answered your questions: "Which database?"="Redis (Recommended)". '
        "You can now continue with the user's answers in mind."
    )


def test_format_accepted_multiple_with_annotations() -> None:
    answers = {
        "Which database?": ["Redis"],
        "Which framework?": ["React"],
    }
    anns = {
        "Which database?": QuestionAnnotation(
            preview="<div>redis preview</div>", notes=None
        ),
        "Which framework?": QuestionAnnotation(
            preview=None, notes="I prefer React hooks"
        ),
    }
    result = format_accepted_tool_result(answers, anns)
    assert result == (
        'User has answered your questions: "Which database?"="Redis" selected preview:\n'
        '<div>redis preview</div>, "Which framework?"="React" user notes: I prefer React hooks. '
        "You can now continue with the user's answers in mind."
    )


def test_format_accepted_multi_select() -> None:
    answers = {"Which features?": ["Auth", "Logging"]}
    result = format_accepted_tool_result(answers, None)
    assert result == (
        'User has answered your questions: "Which features?"="Auth, Logging". '
        "You can now continue with the user's answers in mind."
    )


def test_format_accepted_freeform_only() -> None:
    answers = {"Which database?": ["Other"]}
    anns = {
        "Which database?": QuestionAnnotation(
            preview=None, notes="I want to use DynamoDB"
        )
    }
    result = format_accepted_tool_result(answers, anns)
    assert result == (
        'User has answered your questions: "Which database?"="Other" '
        "user notes: I want to use DynamoDB. "
        "You can now continue with the user's answers in mind."
    )


def test_format_accepted_preview_and_notes() -> None:
    answers = {"Which layout?": ["Grid"]}
    anns = {
        "Which layout?": QuestionAnnotation(
            preview='<div class="grid">...</div>',
            notes="Use CSS Grid for the main layout",
        )
    }
    result = format_accepted_tool_result(answers, anns)
    assert result == (
        'User has answered your questions: "Which layout?"="Grid" selected preview:\n'
        '<div class="grid">...</div> user notes: Use CSS Grid for the main layout. '
        "You can now continue with the user's answers in mind."
    )


def test_format_accepted_empty() -> None:
    result = format_accepted_tool_result({}, None)
    assert result == (
        "User has answered your questions: . "
        "You can now continue with the user's answers in mind."
    )


def test_format_accepted_partial() -> None:
    answers = {"Which database?": ["Redis"]}
    result = format_accepted_tool_result(answers, None)
    assert result == (
        'User has answered your questions: "Which database?"="Redis". '
        "You can now continue with the user's answers in mind."
    )


def test_format_accepted_special_chars() -> None:
    answers = {'Which "option"?': ["Option with\nnewline"]}
    result = format_accepted_tool_result(answers, None)
    assert result == (
        'User has answered your questions: "Which "option"?"="Option with\nnewline". '
        "You can now continue with the user's answers in mind."
    )


# ── Alternate id-keyed formatter ─────────────────────────────────────────


def test_format_id_keyed_single_question_single_select_matches_capture() -> None:
    questions = [
        _id_keyed_q(
            "demo_pick",
            "This is a demo of the question tool. Which outcome should we pick?",
            [
                ("a", "Option A — fast path (minimal scope)"),
                ("b", "Option B — thorough path (extra validation)"),
                ("c", "Option C — I'll decide later"),
            ],
        )
    ]
    answers = {
        "This is a demo of the question tool. Which outcome should we pick?": [
            "Option A — fast path (minimal scope)"
        ]
    }
    result = format_id_keyed_accepted_tool_result(questions, answers, None)
    assert result == (
        "User questions responses:\nQuestion demo_pick: Selected option(s) a"
    )


def test_format_id_keyed_three_questions_matches_capture() -> None:
    questions = [
        _id_keyed_q(
            "q1",
            "Question 1 of 3: Morning drink?",
            [("coffee", "Coffee"), ("tea", "Tea"), ("water", "Water")],
        ),
        _id_keyed_q(
            "q2",
            "Question 2 of 3: How do you usually start a new task?",
            [
                ("read", "Read docs first"),
                ("code", "Jump into code"),
                ("plan", "Sketch a plan first"),
            ],
        ),
        _id_keyed_q(
            "q3",
            "Question 3 of 3: What do you lean on before a push? (pick any)",
            [("tests", "Tests"), ("types", "Types"), ("lint", "Lint/format")],
        ),
    ]
    answers = {
        "Question 1 of 3: Morning drink?": ["Tea"],
        "Question 2 of 3: How do you usually start a new task?": ["Jump into code"],
        "Question 3 of 3: What do you lean on before a push? (pick any)": ["Tests"],
    }
    result = format_id_keyed_accepted_tool_result(questions, answers, None)
    assert result == (
        "User questions responses:\n"
        "Question q1: Selected option(s) tea\n"
        "Question q2: Selected option(s) code\n"
        "Question q3: Selected option(s) tests"
    )


def test_format_id_keyed_multi_select_inferred_csv() -> None:
    questions = [
        _id_keyed_q(
            "q3",
            "What do you lean on before a push? (pick any)",
            [("tests", "Tests"), ("types", "Types"), ("lint", "Lint/format")],
        )
    ]
    answers = {
        "What do you lean on before a push? (pick any)": [
            "Tests",
            "Types",
            "Lint/format",
        ]
    }
    result = format_id_keyed_accepted_tool_result(questions, answers, None)
    assert result == (
        "User questions responses:\n"
        "Question q3: Selected option(s) tests, types, lint"
    )


def test_format_id_keyed_unanswered_questions_are_omitted() -> None:
    questions = [
        _id_keyed_q("q1", "First?", [("a", "Apple"), ("b", "Banana")]),
        _id_keyed_q("q2", "Second?", [("x", "Xenon"), ("y", "Yttrium")]),
    ]
    answers = {"First?": ["Apple"]}
    result = format_id_keyed_accepted_tool_result(questions, answers, None)
    assert result == "User questions responses:\nQuestion q1: Selected option(s) a"


def test_format_id_keyed_no_answers_emits_header_only() -> None:
    questions = [_id_keyed_q("q1", "First?", [("a", "Apple")])]
    result = format_id_keyed_accepted_tool_result(questions, {}, None)
    assert result == "User questions responses:"


def test_format_id_keyed_freeform_dismissal_uses_notes_without_selected_prefix() -> None:
    questions = [
        _id_keyed_q(
            "random_one",
            "Pick one",
            [("ocean", "Ocean"), ("mountains", "Mountains"), ("city", "City")],
        )
    ]
    answers = {"Pick one": ["Other"]}
    anns = {"Pick one": QuestionAnnotation(preview=None, notes="nvm")}
    result = format_id_keyed_accepted_tool_result(questions, answers, anns)
    assert result == "User questions responses:\nQuestion random_one: nvm"


def test_format_id_keyed_freeform_without_notes_is_dropped() -> None:
    questions = [_id_keyed_q("q1", "Pick", [("a", "A")])]
    answers = {"Pick": ["Other"]}
    result = format_id_keyed_accepted_tool_result(questions, answers, None)
    assert result == "User questions responses:"


# ── Path B / C / D ───────────────────────────────────────────────────────


def test_format_chat_about_this_mixed() -> None:
    questions = [
        _make_question("Which database?", ["Redis", "Postgres"]),
        _make_question("Which framework?", ["React", "Vue"]),
    ]
    partial = {"Which database?": "Redis"}
    result = format_chat_about_this(questions, partial)
    expected = (
        "The user wants to clarify these questions.\n"
        "    This means they may have additional information, context or questions for you.\n"
        "    Take their response into account and then reformulate the questions if appropriate.\n"
        "    Start by asking them what they would like to clarify.\n"
        "\n"
        "    Questions asked:\n"
        '- "Which database?"\n'
        "  Answer: Redis\n"
        '- "Which framework?"\n'
        "  (No answer provided)"
    )
    assert result == expected


def test_format_chat_about_this_all_answered() -> None:
    questions = [_make_question("Which database?", ["Redis"])]
    partial = {"Which database?": "Redis"}
    result = format_chat_about_this(questions, partial)
    assert "Answer: Redis" in result
    assert "(No answer provided)" not in result


def test_format_chat_about_this_none_answered() -> None:
    questions = [
        _make_question("Q1?", ["A"]),
        _make_question("Q2?", ["B"]),
    ]
    result = format_chat_about_this(questions, {})
    assert '- "Q1?"\n  (No answer provided)' in result
    assert '- "Q2?"\n  (No answer provided)' in result


def test_format_skip_interview_all_answered() -> None:
    questions = [
        _make_question("Which database?", ["Redis", "Postgres"]),
        _make_question("Which framework?", ["React", "Vue"]),
    ]
    partial = {
        "Which database?": "Redis",
        "Which framework?": "React",
    }
    result = format_skip_interview(questions, partial)
    expected = (
        "The user has indicated they have provided enough answers for the plan interview.\n"
        "Stop asking clarifying questions and proceed to finish the plan with the information you have.\n"
        "\n"
        "Questions asked and answers provided:\n"
        '- "Which database?"\n'
        "  Answer: Redis\n"
        '- "Which framework?"\n'
        "  Answer: React"
    )
    assert result == expected


def test_format_skip_interview_mixed() -> None:
    questions = [
        _make_question("Which database?", ["Redis"]),
        _make_question("Which framework?", ["React"]),
    ]
    partial = {"Which database?": "Redis"}
    result = format_skip_interview(questions, partial)
    assert "Answer: Redis" in result
    assert '- "Which framework?"\n  (No answer provided)' in result


def test_format_skip_interview_no_indentation() -> None:
    questions = [_make_question("Q?", ["A"])]
    result = format_skip_interview(questions, {})
    first_line = result.splitlines()[0]
    assert not first_line.startswith(" ")
    second_line = result.splitlines()[1]
    assert not second_line.startswith(" ")
    assert "\nQuestions asked and answers provided:\n" in result


def test_format_cancel() -> None:
    assert CANCEL_TEXT == (
        "User declined to answer the questions. Continue with the task using your "
        "best judgment, or ask different questions."
    )


# ── Tool surface: description + schema ───────────────────────────────────


def test_tool_name_and_description() -> None:
    tool = AskUserQuestionTool()
    assert tool.id() == "ask_user_question"
    desc = tool.description(None).description
    assert "Ask the user" in desc
    assert "Other" in desc
    assert "(Recommended)" in desc


def test_schema_has_preview_and_snake_case_multi_select() -> None:
    schema = AskUserQuestionTool().parameters_schema()
    text = str(schema)
    assert "multi_select" in text
    assert "multiSelect" not in text
    assert "preview" in text
    assert "A few words at most" in text
    assert "or implies" in text


# ── Tool run paths ───────────────────────────────────────────────────────


def test_empty_questions_handled(tmp_path: Path) -> None:
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    out = tools.call("ask_user_question", {"questions": []}, ctx)
    assert "No questions provided" in out


def test_duplicate_question_text(tmp_path: Path) -> None:
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    with pytest.raises(ToolError) as ei:
        tools.call(
            "ask_user_question",
            {
                "questions": [
                    {
                        "question": "Same question?",
                        "options": [
                            {"label": "A", "description": "a"},
                        ],
                    },
                    {
                        "question": "Same question?",
                        "options": [
                            {"label": "B", "description": "b"},
                        ],
                    },
                ]
            },
            ctx,
        )
    msg = str(ei.value)
    assert "Duplicate question text" in msg
    assert "Same question?" in msg


def test_fallback_without_host(tmp_path: Path) -> None:
    tools = _tools()
    ctx = ToolCallContext(cwd=tmp_path, extra={})
    out = tools.call(
        "ask_user_question",
        {
            "questions": [
                {
                    "question": "Which database?",
                    "options": [
                        {"label": "Redis (Recommended)", "description": "in-mem"},
                        {"label": "Memcached", "description": "cache"},
                    ],
                }
            ]
        },
        ctx,
    )
    assert "Your questions have been presented to the user for answering:" in out
    assert "Which database?" in out
    assert "Redis (Recommended)" in out
    assert ctx.extra.get("pending_user_questions")


def test_host_accepted_formats_path_a(tmp_path: Path) -> None:
    tools = _tools()

    def answer(_questions):
        return {
            "outcome": "accepted",
            "answers": {"Which database?": ["Redis"]},
        }

    ctx = ToolCallContext(cwd=tmp_path, extra={"ask_user_fn": answer})
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


def test_host_cancelled_returns_cancel_text(tmp_path: Path) -> None:
    tools = _tools()

    def answer(_questions):
        return {"outcome": "cancelled"}

    ctx = ToolCallContext(cwd=tmp_path, extra={"ask_user_fn": answer})
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


def test_host_chat_about_this(tmp_path: Path) -> None:
    tools = _tools()

    def answer(_questions):
        return {
            "outcome": "chat_about_this",
            "partial_answers": {"Which database?": "Redis"},
        }

    ctx = ToolCallContext(cwd=tmp_path, extra={"ask_user_fn": answer})
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
    assert "The user wants to clarify these questions." in out
    assert "Answer: Redis" in out
    assert "(No answer provided)" in out


def test_host_shorthand_answers_map(tmp_path: Path) -> None:
    tools = _tools()

    def answer(_questions):
        return {"answers": {"Pick one?": "A (Recommended)"}}

    ctx = ToolCallContext(cwd=tmp_path, extra={"ask_user_fn": answer})
    out = tools.call(
        "ask_user_question",
        {
            "questions": [
                {
                    "question": "Pick one?",
                    "options": [
                        {"label": "A (Recommended)", "description": "first"},
                        {"label": "B", "description": "second"},
                    ],
                }
            ]
        },
        ctx,
    )
    assert '"Pick one?"="A (Recommended)"' in out


def test_host_legacy_list_answers_json(tmp_path: Path) -> None:
    """Legacy host returning non-Grok shape still surfaces JSON."""
    tools = _tools()

    def answer(_questions):
        return {"answers": ["opt-a"]}

    ctx = ToolCallContext(cwd=tmp_path, extra={"ask_user_fn": answer})
    out = tools.call(
        "ask_user_question",
        {
            "questions": [
                {
                    "question": "Pick one?",
                    "options": [
                        {"label": "A (Recommended)", "description": "first"},
                        {"label": "B", "description": "second"},
                    ],
                }
            ]
        },
        ctx,
    )
    assert "opt-a" in out


def test_host_typed_response_objects() -> None:
    qs = [_make_question("Which database?", ["Redis", "Postgres"])]
    out = coerce_host_result(
        UserQuestionResponseAccepted(
            answers={"Which database?": ["Redis"]}, annotations=None
        ),
        qs,
    )
    assert '"Which database?"="Redis"' in out
    assert coerce_host_result(UserQuestionResponseCancelled(), qs) == CANCEL_TEXT
    out_chat = coerce_host_result(
        UserQuestionResponseChatAboutThis(questions=qs, partial_answers={}),
        qs,
    )
    assert "clarify these questions" in out_chat
    out_skip = coerce_host_result(
        UserQuestionResponseSkipInterview(
            questions=qs, partial_answers={"Which database?": "Redis"}
        ),
        qs,
    )
    assert "enough answers for the plan interview" in out_skip


def test_deserialize_accepted_old_string_format() -> None:
    qs = [_make_question("Which cache?", ["Only hot-path caches"])]
    out = coerce_host_result(
        {
            "outcome": "accepted",
            "answers": {"Which cache?": "Only hot-path caches"},
        },
        qs,
    )
    assert '"Which cache?"="Only hot-path caches"' in out


def test_accepts_multi_select_and_preview_in_args(tmp_path: Path) -> None:
    tools = _tools()
    seen: list = []

    def answer(questions):
        seen.extend(questions)
        return UserQuestionResponseCancelled()

    ctx = ToolCallContext(cwd=tmp_path, extra={"ask_user_fn": answer})
    out = tools.call(
        "ask_user_question",
        {
            "questions": [
                {
                    "question": "Pick DB?",
                    "options": [
                        {
                            "label": "Postgres",
                            "description": "Relational DB",
                        },
                        {
                            "label": "SQLite",
                            "description": "Embedded SQL database",
                            "preview": "```\nSELECT 1;\n```",
                        },
                    ],
                    "multi_select": False,
                }
            ]
        },
        ctx,
    )
    assert out == CANCEL_TEXT
    assert seen[0]["multi_select"] is False
    assert seen[0]["options"][1]["preview"] == "```\nSELECT 1;\n```"
