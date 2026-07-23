"""Read-only Markdown highlighting for the plan detail surface."""

from __future__ import annotations

import re
from collections.abc import Callable

from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.lexers import Lexer


_FENCE_RE = re.compile(r"^(\s*)(```|~~~)(.*)$")
_HEADING_RE = re.compile(r"^(\s*)(#{1,6})(\s+)(.*)$")
_CHECKBOX_RE = re.compile(r"^(\s*)([-*+])(\s+)(\[[ xX]\])(\s*)(.*)$")
_BULLET_RE = re.compile(r"^(\s*)([-*+])(\s+)(.*)$")
_ORDERED_RE = re.compile(r"^(\s*)(\d+[.)])(\s+)(.*)$")
_QUOTE_RE = re.compile(r"^(\s*)(>+)(\s*)(.*)$")
_RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_INLINE_RE = re.compile(
    r"(`[^`]+`|\[[^\]]+\]\([^)]+\)|\*\*[^*\n]+\*\*|__[^_\n]+__)"
)


def _inline_fragments(text: str, base_style: str) -> StyleAndTextTuples:
    """Highlight inline Markdown while preserving every source character."""

    fragments: StyleAndTextTuples = []
    cursor = 0
    for match in _INLINE_RE.finditer(text):
        if match.start() > cursor:
            fragments.append((base_style, text[cursor : match.start()]))
        token = match.group(0)
        if token.startswith("`"):
            style = "class:plan.code.inline"
        elif token.startswith("["):
            style = "class:plan.link"
        else:
            style = "class:plan.strong"
        fragments.append((style, token))
        cursor = match.end()
    if cursor < len(text):
        fragments.append((base_style, text[cursor:]))
    return fragments


def _render_plan_line(line: str, *, in_fence: bool) -> StyleAndTextTuples:
    fence = _FENCE_RE.match(line)
    if fence:
        return [
            ("class:plan.code.fence", fence.group(1)),
            ("class:plan.code.fence", fence.group(2)),
            ("class:plan.code.fence", fence.group(3)),
        ]
    if in_fence:
        return [("class:plan.code", line)]

    heading = _HEADING_RE.match(line)
    if heading:
        level = min(3, len(heading.group(2)))
        style = f"class:plan.heading.h{level}"
        return [
            ("class:plan.body", heading.group(1)),
            ("class:plan.marker", heading.group(2)),
            ("class:plan.body", heading.group(3)),
            *_inline_fragments(heading.group(4), style),
        ]

    checkbox = _CHECKBOX_RE.match(line)
    if checkbox:
        checked = "x" in checkbox.group(4).lower()
        checkbox_style = (
            "class:plan.checkbox.done"
            if checked
            else "class:plan.checkbox.pending"
        )
        return [
            ("class:plan.body", checkbox.group(1)),
            ("class:plan.marker", checkbox.group(2)),
            ("class:plan.body", checkbox.group(3)),
            (checkbox_style, checkbox.group(4)),
            ("class:plan.body", checkbox.group(5)),
            *_inline_fragments(checkbox.group(6), "class:plan.body"),
        ]

    bullet = _BULLET_RE.match(line)
    if bullet:
        return [
            ("class:plan.body", bullet.group(1)),
            ("class:plan.marker", bullet.group(2)),
            ("class:plan.body", bullet.group(3)),
            *_inline_fragments(bullet.group(4), "class:plan.body"),
        ]

    ordered = _ORDERED_RE.match(line)
    if ordered:
        return [
            ("class:plan.body", ordered.group(1)),
            ("class:plan.marker", ordered.group(2)),
            ("class:plan.body", ordered.group(3)),
            *_inline_fragments(ordered.group(4), "class:plan.body"),
        ]

    quote = _QUOTE_RE.match(line)
    if quote:
        return [
            ("class:plan.body", quote.group(1)),
            ("class:plan.quote.marker", quote.group(2)),
            ("class:plan.quote", quote.group(3)),
            *_inline_fragments(quote.group(4), "class:plan.quote"),
        ]

    if _RULE_RE.match(line):
        return [("class:plan.rule", line)]
    return _inline_fragments(line, "class:plan.body")


class PlanMarkdownLexer(Lexer):
    """Small deterministic Markdown lexer for a selectable read-only TextArea."""

    def lex_document(
        self, document: Document
    ) -> Callable[[int], StyleAndTextTuples]:
        rendered: list[StyleAndTextTuples] = []
        in_fence = False
        for line in document.lines:
            is_fence = _FENCE_RE.match(line) is not None
            rendered.append(_render_plan_line(line, in_fence=in_fence))
            if is_fence:
                in_fence = not in_fence

        def get_line(line_number: int) -> StyleAndTextTuples:
            if 0 <= line_number < len(rendered):
                return rendered[line_number]
            return []

        return get_line


__all__ = ["PlanMarkdownLexer"]
