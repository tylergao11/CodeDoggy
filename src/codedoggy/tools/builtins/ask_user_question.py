"""ask_user_question — structured multi-choice questions (Grok AskUserQuestion).

Ported from:
  crates/codegen/xai-grok-tools/src/implementations/grok_build/ask_user_question/
    mod.rs      — description_template, Question/QuestionOption schema, run flow
    format.rs   — CANCEL_TEXT, format_accepted_*, format_chat_about_this,
                  format_skip_interview, format_id_keyed_accepted_tool_result
    types.rs    — QuestionAnnotation, UserQuestionResponse, UserQuestionError,
                  AskUserQuestionMode / ExtRequest / ExtResponse (host contract)

Runtime UI is host-injected (``extra['ask_user_fn']``). CodeDoggy does not
invent a pager/TUI; without a host hook we use Grok's migration fallback
(QuestionsSent summary). Blocking ACP/oneshot coordinator is shell-side (X).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from codedoggy.tools.kinds import ToolKind, ToolNamespace
from codedoggy.tools.runtime import (
    ListToolsContext,
    Tool,
    ToolCallContext,
    ToolDescription,
    ToolError,
    ToolId,
)

# ── Grok description_template (mod.rs ToolMetadata) ──────────────────────

_DESC = """\
Ask the user one or more multiple-choice questions.

- Every question automatically gets an "Other" choice where the user can type their own answer.
- Put your recommended option first and append "(Recommended)" to its label.
"""

# Migration fallback: when no UserQuestionSender / ask_user_fn is wired,
# fire-and-forget QuestionsSent (Grok MIGRATION_FALLBACK = true).
MIGRATION_FALLBACK = True

# Default questionnaire wait (Grok RESPONSE_TIMEOUT). Host may enforce;
# sync Python path does not block on a timer.
RESPONSE_TIMEOUT_SECS = 30 * 60
DEFAULT_ASK_USER_QUESTION_TIMEOUT_ENABLED = True
RESPONSE_TIMEOUT_ENV = "GROK_ASK_USER_QUESTION_TIMEOUT_SECS"


# ── Types (mod.rs + types.rs) ────────────────────────────────────────────


@dataclass
class QuestionOption:
    """A single option within a question (Grok QuestionOption)."""

    label: str
    description: str
    preview: str | None = None
    id: str | None = None  # opaque; hidden from model schema


@dataclass
class Question:
    """A single question with its options (Grok Question)."""

    question: str
    options: list[QuestionOption]
    multi_select: bool | None = None
    id: str | None = None  # opaque; hidden from model schema


@dataclass
class QuestionAnnotation:
    """Annotation on a single question's answer (types.rs)."""

    preview: str | None = None
    notes: str | None = None


class AskUserQuestionMode(str, Enum):
    """Mode context for the question UI (types.rs)."""

    DEFAULT = "default"
    PLAN = "plan"


@dataclass
class UserQuestionResponseAccepted:
    answers: dict[str, list[str]]  # insertion-ordered
    annotations: dict[str, QuestionAnnotation] | None = None


@dataclass
class UserQuestionResponseChatAboutThis:
    questions: list[Question]
    partial_answers: dict[str, str] = field(default_factory=dict)


@dataclass
class UserQuestionResponseSkipInterview:
    questions: list[Question]
    partial_answers: dict[str, str] = field(default_factory=dict)


@dataclass
class UserQuestionResponseCancelled:
    pass


UserQuestionResponse = (
    UserQuestionResponseAccepted
    | UserQuestionResponseChatAboutThis
    | UserQuestionResponseSkipInterview
    | UserQuestionResponseCancelled
)


class UserQuestionErrorKind(str, Enum):
    TRANSPORT = "transport"
    MALFORMED = "malformed"


@dataclass
class UserQuestionError:
    kind: UserQuestionErrorKind
    message: str


# ── Format helpers (format.rs) ───────────────────────────────────────────

# Path D: Cancel
CANCEL_TEXT = (
    "User declined to answer the questions. Continue with the task using your "
    "best judgment, or ask different questions."
)


def format_accepted_tool_result(
    answers: dict[str, list[str]],
    annotations: dict[str, QuestionAnnotation] | None,
) -> str:
    """Path A — accepted answers (format.rs::format_accepted_tool_result)."""
    entries: list[str] = []
    for question_text, selected_labels in answers.items():
        selected_label = ", ".join(selected_labels)
        parts = [f'"{question_text}"="{selected_label}"']
        if annotations is not None:
            ann = annotations.get(question_text)
            if ann is not None:
                if ann.preview is not None:
                    parts.append(f"selected preview:\n{ann.preview}")
                if ann.notes is not None:
                    parts.append(f"user notes: {ann.notes}")
        entries.append(" ".join(parts))
    return (
        "User has answered your questions: "
        f"{', '.join(entries)}. "
        "You can now continue with the user's answers in mind."
    )


def format_id_keyed_accepted_tool_result(
    input_questions: list[Question],
    answers: dict[str, list[str]],
    annotations: dict[str, QuestionAnnotation] | None,
) -> str:
    """Alternate id-keyed accepted shape (format.rs)."""
    lines: list[str] = []
    for q in input_questions:
        if q.id is None:
            continue
        labels = answers.get(q.question)
        if labels is None:
            continue
        oids: list[str] = []
        for label in labels:
            for o in q.options:
                if o.label == label and o.id is not None:
                    oids.append(o.id)
                    break
        if not oids:
            notes = None
            if annotations is not None:
                ann = annotations.get(q.question)
                if ann is not None and ann.notes is not None:
                    trimmed = ann.notes.strip()
                    if trimmed:
                        notes = trimmed
            if notes is None:
                continue
            lines.append(f"Question {q.id}: {notes}")
            continue
        lines.append(f"Question {q.id}: Selected option(s) {', '.join(oids)}")
    if not lines:
        return "User questions responses:"
    return "User questions responses:\n" + "\n".join(lines)


def format_chat_about_this(
    questions: list[Question],
    partial_answers: dict[str, str],
) -> str:
    """Path B — plan-mode 'Chat about this' (format.rs)."""
    question_lines: list[str] = []
    for q in questions:
        if q.question in partial_answers:
            question_lines.append(
                f'- "{q.question}"\n  Answer: {partial_answers[q.question]}'
            )
        else:
            question_lines.append(f'- "{q.question}"\n  (No answer provided)')
    # Whitespace intentional: lines 2–4 and "Questions asked:" have 4-space indent.
    return (
        "The user wants to clarify these questions.\n"
        "    This means they may have additional information, context or questions for you.\n"
        "    Take their response into account and then reformulate the questions if appropriate.\n"
        "    Start by asking them what they would like to clarify.\n"
        "\n"
        "    Questions asked:\n"
        f"{chr(10).join(question_lines)}"
    )


def format_skip_interview(
    questions: list[Question],
    partial_answers: dict[str, str],
) -> str:
    """Path C — plan-mode 'Skip interview' (format.rs)."""
    question_lines: list[str] = []
    for q in questions:
        if q.question in partial_answers:
            question_lines.append(
                f'- "{q.question}"\n  Answer: {partial_answers[q.question]}'
            )
        else:
            question_lines.append(f'- "{q.question}"\n  (No answer provided)')
    return (
        "The user has indicated they have provided enough answers for the plan interview.\n"
        "Stop asking clarifying questions and proceed to finish the plan with the information you have.\n"
        "\n"
        "Questions asked and answers provided:\n"
        f"{chr(10).join(question_lines)}"
    )


def format_user_question_response(
    response: UserQuestionResponse,
    *,
    use_id_keyed_format: bool = False,
    input_questions: list[Question] | None = None,
) -> str:
    """Map UserQuestionResponse → model-visible string (mod.rs step 7)."""
    if isinstance(response, UserQuestionResponseAccepted):
        if use_id_keyed_format and input_questions is not None:
            return format_id_keyed_accepted_tool_result(
                input_questions, response.answers, response.annotations
            )
        return format_accepted_tool_result(response.answers, response.annotations)
    if isinstance(response, UserQuestionResponseChatAboutThis):
        return format_chat_about_this(response.questions, response.partial_answers)
    if isinstance(response, UserQuestionResponseSkipInterview):
        return format_skip_interview(response.questions, response.partial_answers)
    if isinstance(response, UserQuestionResponseCancelled):
        return CANCEL_TEXT
    raise TypeError(f"unknown UserQuestionResponse: {type(response)!r}")


# ── Parse helpers (host / ACP wire → types) ──────────────────────────────


def _annotation_from_raw(raw: Any) -> QuestionAnnotation:
    if not isinstance(raw, dict):
        return QuestionAnnotation()
    preview = raw.get("preview")
    notes = raw.get("notes")
    return QuestionAnnotation(
        preview=str(preview) if preview is not None else None,
        notes=str(notes) if notes is not None else None,
    )


def _normalize_answers(raw: Any) -> dict[str, list[str]]:
    """Accept string or list per answer entry (types.rs deserialize_string_or_vec)."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for k, v in raw.items():
        key = str(k)
        if isinstance(v, list):
            out[key] = [str(x) for x in v]
        else:
            out[key] = [str(v)]
    return out


def parse_question_option(raw: Any) -> QuestionOption:
    if not isinstance(raw, dict):
        raise ToolError.invalid_arguments("each option must be an object")
    label = str(raw.get("label") or "").strip()
    if not label:
        raise ToolError.invalid_arguments("option label is required")
    desc = raw.get("description")
    if desc is None:
        desc = ""
    preview = raw.get("preview")
    oid = raw.get("id")
    return QuestionOption(
        label=label,
        description=str(desc),
        preview=str(preview) if preview is not None else None,
        id=str(oid) if oid is not None else None,
    )


def parse_question(raw: Any, *, index: int | None = None) -> Question:
    if not isinstance(raw, dict):
        where = f"questions[{index}]" if index is not None else "question"
        raise ToolError.invalid_arguments(f"{where} must be an object")
    text = str(raw.get("question") or "").strip()
    if not text:
        where = f"questions[{index}].question" if index is not None else "question"
        raise ToolError.invalid_arguments(f"{where} is required")
    opts_raw = raw.get("options")
    if not isinstance(opts_raw, list) or not opts_raw:
        where = f"questions[{index}].options" if index is not None else "options"
        raise ToolError.invalid_arguments(f"{where} must be a non-empty array")
    options = [parse_question_option(o) for o in opts_raw]
    # Model schema is snake_case; also accept legacy ACP multiSelect.
    multi = raw.get("multi_select")
    if multi is None and "multiSelect" in raw:
        multi = raw.get("multiSelect")
    multi_select: bool | None
    if multi is None:
        multi_select = None
    else:
        multi_select = bool(multi)
    qid = raw.get("id")
    return Question(
        question=text,
        options=options,
        multi_select=multi_select,
        id=str(qid) if qid is not None else None,
    )


def parse_questions(raw: Any) -> list[Question]:
    if not isinstance(raw, list):
        raise ToolError.invalid_arguments("questions must be a non-empty array")
    return [parse_question(q, index=i) for i, q in enumerate(raw)]


def ext_response_to_user_response(
    outcome: str,
    payload: dict[str, Any],
    questions: list[Question],
) -> UserQuestionResponse:
    """Grok AskUserQuestionExtResponse::into_response."""
    if outcome == "accepted":
        answers = _normalize_answers(payload.get("answers"))
        anns_raw = payload.get("annotations")
        annotations: dict[str, QuestionAnnotation] | None = None
        if isinstance(anns_raw, dict) and anns_raw:
            annotations = {str(k): _annotation_from_raw(v) for k, v in anns_raw.items()}
        return UserQuestionResponseAccepted(answers=answers, annotations=annotations)
    if outcome == "chat_about_this":
        partial = payload.get("partial_answers") or {}
        if not isinstance(partial, dict):
            partial = {}
        return UserQuestionResponseChatAboutThis(
            questions=questions,
            partial_answers={str(k): str(v) for k, v in partial.items()},
        )
    if outcome == "skip_interview":
        partial = payload.get("partial_answers") or {}
        if not isinstance(partial, dict):
            partial = {}
        return UserQuestionResponseSkipInterview(
            questions=questions,
            partial_answers={str(k): str(v) for k, v in partial.items()},
        )
    if outcome == "cancelled":
        return UserQuestionResponseCancelled()
    raise ValueError(f"unknown outcome: {outcome!r}")


def coerce_host_result(
    result: Any,
    questions: list[Question],
    *,
    use_id_keyed_format: bool = False,
) -> str:
    """Normalize host callback return value to model-visible text."""
    if isinstance(
        result,
        (
            UserQuestionResponseAccepted,
            UserQuestionResponseChatAboutThis,
            UserQuestionResponseSkipInterview,
            UserQuestionResponseCancelled,
        ),
    ):
        return format_user_question_response(
            result,
            use_id_keyed_format=use_id_keyed_format,
            input_questions=questions,
        )
    if isinstance(result, str):
        return result
    if isinstance(result, UserQuestionError):
        if result.kind == UserQuestionErrorKind.TRANSPORT:
            raise ToolError(
                f"Failed to reach the client for user question: {result.message}",
                code="ask_user_failed",
            )
        raise ToolError(
            f"Client returned an invalid response to user question: {result.message}",
            code="ask_user_failed",
        )
    if isinstance(result, dict):
        outcome = result.get("outcome")
        if isinstance(outcome, str):
            try:
                resp = ext_response_to_user_response(outcome, result, questions)
            except ValueError as e:
                raise ToolError(
                    f"Client returned an invalid response to user question: {e}",
                    code="ask_user_failed",
                ) from e
            return format_user_question_response(
                resp,
                use_id_keyed_format=use_id_keyed_format,
                input_questions=questions,
            )
        # Shorthand accepted: {"answers": {q: label|list}, "annotations"?: ...}
        if "answers" in result and isinstance(result.get("answers"), dict):
            answers = _normalize_answers(result["answers"])
            anns_raw = result.get("annotations")
            annotations = None
            if isinstance(anns_raw, dict) and anns_raw:
                annotations = {
                    str(k): _annotation_from_raw(v) for k, v in anns_raw.items()
                }
            return format_user_question_response(
                UserQuestionResponseAccepted(answers=answers, annotations=annotations),
                use_id_keyed_format=use_id_keyed_format,
                input_questions=questions,
            )
        # Unrecognized structured payload — surface as JSON (legacy hosts)
        return json.dumps(result, ensure_ascii=False, indent=2)
    return str(result)


def _fallback_questions_sent(questions: list[Question]) -> str:
    """Grok fallback_fire_and_forget QuestionsSent message."""
    question_summary: list[str] = []
    for i, q in enumerate(questions):
        options = ", ".join(o.label for o in q.options)
        question_summary.append(f"{i + 1}. {q.question} [options: {options}]")
    return (
        "Your questions have been presented to the user for answering:\n"
        + "\n".join(question_summary)
    )


# ── Tool ─────────────────────────────────────────────────────────────────


class AskUserQuestionTool(Tool):
    def id(self) -> ToolId:
        return ToolId("ask_user_question")

    def tool_namespace(self) -> ToolNamespace:
        return ToolNamespace.Doggy

    def kind(self) -> ToolKind:
        return ToolKind.AskUser

    def description(self, _ctx: ListToolsContext | None = None) -> ToolDescription:
        return ToolDescription(name="ask_user_question", description=_DESC.strip())

    def parameters_schema(self) -> dict[str, Any]:
        # Grok AskUserQuestionInput / Question / QuestionOption schemars.
        # multi_select is snake_case on the model schema; id/preview id fields
        # that are #[schemars(skip)] stay out of the schema.
        return {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "description": "The questions to ask, each with its own options.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": (
                                    "The question to ask, phrased as a full question."
                                ),
                            },
                            "options": {
                                "type": "array",
                                "description": "The choices for this question.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {
                                            "type": "string",
                                            "description": (
                                                "Option text shown to the user. "
                                                "A few words at most."
                                            ),
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": (
                                                "What picking this option means "
                                                "or implies."
                                            ),
                                        },
                                        "preview": {
                                            "type": "string",
                                            "description": (
                                                "Optional content shown while the "
                                                "option is focused — mockups, code "
                                                "snippets, anything the user should "
                                                "compare. Single-select questions only."
                                            ),
                                        },
                                    },
                                    "required": ["label", "description"],
                                },
                            },
                            "multi_select": {
                                "type": "boolean",
                                "description": (
                                    "Let the user pick more than one option "
                                    "(default false)."
                                ),
                            },
                        },
                        "required": ["question", "options"],
                    },
                },
            },
            "required": ["questions"],
        }

    def run(self, ctx: ToolCallContext, args: dict[str, Any]) -> str:
        raw_questions = args.get("questions")
        # Grok: empty list is Ok(QuestionsSent "No questions provided..."), not Err.
        if raw_questions is None:
            raise ToolError.invalid_arguments("questions must be a non-empty array")
        if not isinstance(raw_questions, list):
            raise ToolError.invalid_arguments("questions must be a non-empty array")
        if len(raw_questions) == 0:
            return "No questions provided. Continue with the task."

        questions = parse_questions(raw_questions)

        # Step 1: unique question text (mod.rs)
        seen: set[str] = set()
        for q in questions:
            if q.question in seen:
                raise ToolError.invalid_arguments(
                    f'Duplicate question text: "{q.question}"'
                )
            seen.add(q.question)

        use_id_keyed = bool(args.get("use_id_keyed_format"))

        # Host callback (CLI / TUI / tests) — honest X surface
        fn = (ctx.extra or {}).get("ask_user_fn")
        if callable(fn):
            try:
                # Pass model-shaped dicts (stable host contract)
                host_payload = [
                    {
                        "question": q.question,
                        "options": [
                            {
                                "label": o.label,
                                "description": o.description,
                                **(
                                    {"preview": o.preview}
                                    if o.preview is not None
                                    else {}
                                ),
                                **({"id": o.id} if o.id is not None else {}),
                            }
                            for o in q.options
                        ],
                        **(
                            {"multi_select": q.multi_select}
                            if q.multi_select is not None
                            else {}
                        ),
                        **({"id": q.id} if q.id is not None else {}),
                    }
                    for q in questions
                ]
                result = fn(host_payload)
            except ToolError:
                raise
            except Exception as e:  # noqa: BLE001
                raise ToolError(
                    f"Failed to reach the client for user question: {e}",
                    code="ask_user_failed",
                ) from e
            return coerce_host_result(
                result, questions, use_id_keyed_format=use_id_keyed
            )

        if not MIGRATION_FALLBACK:
            raise ToolError(
                "UserQuestionSender",
                code="missing_resource",
            )

        # Stash for session hosts that poll pending questions
        bag = ctx.extra if ctx.extra is not None else {}
        bag["pending_user_questions"] = [
            {
                "question": q.question,
                "options": [
                    {
                        "label": o.label,
                        "description": o.description,
                        **({"preview": o.preview} if o.preview is not None else {}),
                    }
                    for o in q.options
                ],
                **(
                    {"multi_select": q.multi_select}
                    if q.multi_select is not None
                    else {}
                ),
            }
            for q in questions
        ]
        return _fallback_questions_sent(questions)
