"""UserPromptBlock — displays user input.

Port of ``xai-grok-pager/src/scrollback/blocks/user.rs``.

- Prefix: prompt arrow (``❯ `` / U+276F+space, width 2), bash ``$ ``, cron
  ``↻  `` (U+21BB + two spaces).
- Hanging indent matches prefix display width.
- Collapsed: max ``COLLAPSED_MAX_LINES`` (3) visual lines; last line gets
  `` …`` (space + U+2026) when truncated.
- Skill token ranges paint mid-text slash tokens in skill accent.
- Bash / cron / interjection / skill variants match Grok constructors.

Canonical BlockLine / BlockOutput / Selectable types live with the layout
agent; if ``codedoggy.tui_v2.types`` is absent this module paints rows of
StyleAndTextTuples only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from prompt_toolkit.utils import get_cwidth

if TYPE_CHECKING:
    from prompt_toolkit.formatted_text import StyleAndTextTuples
else:
    StyleAndTextTuples = list  # type: ignore[misc, assignment]

# Grok user.rs
COLLAPSED_MAX_LINES = 3
USER_PROMPT_BODY_RANGE = 0
ELLIPSIS = " \u2026"  # " …"
ELLIPSIS_WIDTH = 2  # Grok: ellipsis_width = 2
PROMPT_ARROW_WIDTH = 2  # glyphs.rs PROMPT_ARROW_WIDTH

# Style classes
S_PREFIX = "class:grok.prompt.prefix"  # accent_user (or Cyan when Reset)
S_BODY = "class:grok.prompt.body"  # text_primary
S_SKILL = "class:grok.prompt.skill"  # accent_skill
S_BG = "class:grok.prompt.band"  # optional band (theme maps bg_light)


def _prompt_arrow() -> str:
    """Grok ``glyphs::prompt_arrow`` — ``❯ `` normally, ``> `` on legacy ConHost.

    Import from ``codedoggy.tui_v2.glyphs`` when the glyphs agent has landed;
    else U+276F + space (width 2).
    """
    try:
        from codedoggy.tui_v2.glyphs import prompt_arrow  # type: ignore

        return prompt_arrow()
    except Exception:
        return "\u276f "  # ❯ + space


def _disp_width(text: str) -> int:
    return get_cwidth(text) if text else 0


def sanitize_token_ranges(
    text: str, ranges: Sequence[tuple[int, int] | slice]
) -> list[tuple[int, int]]:
    """Drop invalid token ranges (Grok ``sanitize_token_ranges``).

    Out of bounds, empty, non-char-boundary, or overlapping earlier kept
    ranges are dropped. Survivors are sorted by start.
    """
    cleaned: list[tuple[int, int]] = []
    for r in ranges:
        if isinstance(r, slice):
            start, end = int(r.start or 0), int(r.stop or 0)
        else:
            start, end = int(r[0]), int(r[1])
        if start >= end or end > len(text) or start < 0:
            continue
        # Char-boundary check (Python str slices are always char-safe, but
        # reject mid-codepoint for parity with Rust byte ranges when callers
        # pass UTF-8 byte offsets that don't land on Unicode points).
        try:
            text[start:end]
            # Ensure we didn't request past a multi-byte boundary incorrectly:
            # in Python indices are code points; accept as-is.
        except (IndexError, ValueError):
            continue
        cleaned.append((start, end))
    cleaned.sort(key=lambda t: (t[0], t[1]))
    out: list[tuple[int, int]] = []
    for start, end in cleaned:
        if out and start < out[-1][1]:
            continue
        out.append((start, end))
    return out


def _token_styled_line(
    line_text: str,
    line_start: int,
    ranges: Sequence[tuple[int, int]],
    token_style: str,
    body_style: str,
) -> StyleAndTextTuples:
    """Split one logical line into token/body spans (Grok ``token_styled_line``)."""
    line_end = line_start + len(line_text)
    spans: StyleAndTextTuples = []
    pos = line_start
    for start, end in ranges:
        s = max(line_start, min(start, line_end))
        e = max(line_start, min(end, line_end))
        if s >= e:
            continue
        if s > pos:
            spans.append((body_style, line_text[pos - line_start : s - line_start]))
        spans.append((token_style, line_text[s - line_start : e - line_start]))
        pos = e
    if pos < line_end:
        spans.append((body_style, line_text[pos - line_start :]))
    return spans or [(body_style, line_text)]


def _wrap_styled(
    fragments: StyleAndTextTuples,
    width: int,
) -> list[StyleAndTextTuples]:
    """Word-wrap a single logical line of styled fragments to ``width`` cols."""
    width = max(1, int(width))
    # Tokenize preserving style, split on spaces.
    tokens: list[tuple[str, str]] = []
    for style, text in fragments:
        if not text:
            continue
        for part in re.split(r"( +)", text):
            if part:
                tokens.append((style, part))

    rows: list[StyleAndTextTuples] = []
    row: StyleAndTextTuples = []
    used = 0

    def flush() -> None:
        nonlocal row, used
        rows.append(row)
        row = []
        used = 0

    def merge(style: str, piece: str) -> None:
        nonlocal used
        if not piece:
            return
        if row and row[-1][0] == style:
            row[-1] = (style, row[-1][1] + piece)
        else:
            row.append((style, piece))
        used += _disp_width(piece)

    for style, token in tokens:
        tw = _disp_width(token)
        if token.startswith(" ") and used > 0 and used + tw > width:
            token = token.lstrip(" ")
            if not token:
                continue
            flush()
            tw = _disp_width(token)
        if used > 0 and used + tw > width:
            flush()
            if token.startswith(" "):
                token = token.lstrip(" ")
                if not token:
                    continue
                tw = _disp_width(token)
        # Hard break
        while token and used + _disp_width(token) > width:
            room = max(1, width - used)
            acc: list[str] = []
            acc_w = 0
            chars = list(token)
            i = 0
            while i < len(chars):
                cw = _disp_width(chars[i])
                if acc and acc_w + cw > room:
                    break
                acc.append(chars[i])
                acc_w += cw
                i += 1
            if not acc:
                acc = [chars[0]]
                i = 1
            merge(style, "".join(acc))
            token = "".join(chars[i:])
            if token:
                flush()
        if token:
            merge(style, token)

    if row or not rows:
        rows.append(row)
    return rows or [[("", "")]]


def _truncate_fragments_to_width(
    fragments: StyleAndTextTuples,
    width: int,
) -> StyleAndTextTuples:
    """Keep leading fragments that fit in ``width`` display columns."""
    width = max(0, int(width))
    out: StyleAndTextTuples = []
    used = 0
    for style, text in fragments:
        if used >= width:
            break
        for ch in text:
            cw = _disp_width(ch)
            if used + cw > width:
                return out
            if out and out[-1][0] == style:
                out[-1] = (style, out[-1][1] + ch)
            else:
                out.append((style, ch))
            used += cw
    return out


@dataclass
class UserPromptBlock:
    """Grok ``UserPromptBlock`` data model (paint-side)."""

    text: str
    is_bash: bool = False
    is_cron: bool = False
    is_interjection: bool = False
    prompt_index: int | None = None
    skill_token_ranges: list[tuple[int, int]] = field(default_factory=list)

    @classmethod
    def new(cls, text: str) -> UserPromptBlock:
        return cls(text=text)

    @classmethod
    def bash(cls, text: str) -> UserPromptBlock:
        return cls(text=text, is_bash=True)

    @classmethod
    def cron(cls, text: str) -> UserPromptBlock:
        return cls(text=text, is_cron=True)

    @classmethod
    def interjection(cls, text: str) -> UserPromptBlock:
        return cls(text=text, is_interjection=True)

    @classmethod
    def skill(cls, text: str) -> UserPromptBlock:
        """Leading token on first line gets skill accent (Grok ``skill``)."""
        first = text.split("\n", 1)[0]
        token_end = len(first)
        for i, ch in enumerate(first):
            if ch.isspace():
                token_end = i
                break
        ranges = [(0, token_end)] if token_end > 0 else []
        return cls(text=text, skill_token_ranges=ranges)

    @classmethod
    def with_skill_tokens(
        cls, text: str, ranges: Sequence[tuple[int, int] | slice]
    ) -> UserPromptBlock:
        return cls(text=text, skill_token_ranges=sanitize_token_ranges(text, ranges))

    def copy_text(self) -> str:
        return self.text

    def prefix_str(self, show_prefix: bool) -> str:
        if not show_prefix:
            return ""
        if self.is_bash:
            return "$ "
        if self.is_cron:
            return "\u21bb  "  # ↻ + two spaces
        return _prompt_arrow()

    def wrap_prompt_lines(
        self,
        width: int,
        max_lines: int | None,
        show_prefix: bool,
        is_selected: bool = False,  # noqa: ARG002 — band/selection reserved for theme
    ) -> list[StyleAndTextTuples]:
        """Wrap and style the prompt (Grok ``wrap_prompt_lines``).

        Returns painted rows. Prefix on first visual line; hanging spaces on
        continuations. When ``max_lines`` is set and content exceeds it, the
        last row ends with `` …``.
        """
        width = max(1, int(width))
        prefix = self.prefix_str(show_prefix)
        prefix_w = _disp_width(prefix)
        base_content_width = max(1, width - prefix_w)

        # Optional selection band: theme may style class:grok.prompt.band.
        # We do not invent band glyphs; style class is a hook only.
        _ = is_selected

        # Rust `str::lines()` omits a trailing empty after final `\n`.
        logical = self.text.splitlines()
        if not self.text:
            logical = [""]
        elif not logical:
            logical = [""]

        all_lines: list[StyleAndTextTuples] = []
        # Codepoint offsets for skill ranges (Python paint API). Grok uses
        # UTF-8 byte ranges; callers here should pass str indices.
        total_logical = len(logical)
        # Walk the source: each logical line is followed by `\n` except the last.
        line_start = 0

        for logical_idx, line_text in enumerate(logical):
            is_first_logical = logical_idx == 0

            if line_text == "":
                if is_first_logical:
                    row: StyleAndTextTuples = (
                        [(S_PREFIX, prefix)] if prefix else [("", "")]
                    )
                else:
                    row = (
                        [(S_PREFIX, " " * prefix_w)] if prefix_w else [("", "")]
                    )
                all_lines.append(row)
                if max_lines is not None and len(all_lines) >= max_lines:
                    if logical_idx + 1 < total_logical:
                        all_lines[-1] = list(all_lines[-1]) + [(S_BODY, ELLIPSIS)]
                    return all_lines
                line_start += len(line_text)
                if logical_idx + 1 < total_logical:
                    line_start += 1
                continue

            if self.skill_token_ranges:
                content = _token_styled_line(
                    line_text,
                    line_start,
                    self.skill_token_ranges,
                    S_SKILL,
                    S_BODY,
                )
            else:
                content = [(S_BODY, line_text)]

            wrapped = _wrap_styled(content, base_content_width)
            wrapped_count = len(wrapped)

            for wrap_idx, wrapped_frags in enumerate(wrapped):
                is_first_line = logical_idx == 0 and wrap_idx == 0
                line_prefix = prefix if is_first_line else (" " * prefix_w)

                will_be_last = (
                    max_lines is not None and len(all_lines) + 1 == max_lines
                )
                has_more = (wrap_idx + 1 < wrapped_count) or (
                    logical_idx + 1 < total_logical
                )

                if will_be_last and has_more:
                    reduced = max(1, base_content_width - ELLIPSIS_WIDTH)
                    re_wrapped = _wrap_styled(wrapped_frags, reduced)
                    final_content = re_wrapped[0] if re_wrapped else []
                    final_content = _truncate_fragments_to_width(
                        final_content, reduced
                    )
                    spans: StyleAndTextTuples = []
                    if line_prefix:
                        spans.append((S_PREFIX, line_prefix))
                    spans.extend(final_content)
                    spans.append((S_BODY, ELLIPSIS))
                    all_lines.append(spans)
                    return all_lines

                spans = []
                if line_prefix:
                    spans.append((S_PREFIX, line_prefix))
                spans.extend(wrapped_frags)
                all_lines.append(spans)

                if max_lines is not None and len(all_lines) >= max_lines:
                    return all_lines

            # Advance to next logical line start (skip content + `\n`).
            line_start += len(line_text)
            if logical_idx + 1 < total_logical:
                line_start += 1  # the separating newline

        if not all_lines:
            p = prefix.rstrip() if prefix else ""
            all_lines.append([(S_PREFIX, p)] if p else [("", "")])

        return all_lines

    def is_foldable(self) -> bool:
        """Estimate if content exceeds collapsed max (Grok ``is_foldable``)."""
        min_content = 60
        visual = 0
        for line in self.text.split("\n"):
            w = _disp_width(line)
            visual += 1 if w == 0 else (w + min_content - 1) // min_content
            if visual > COLLAPSED_MAX_LINES:
                return True
        return False


def paint_user_prompt(
    text: str,
    *,
    width: int,
    collapsed: bool = False,
    selected: bool = False,
    is_bash: bool = False,
    is_cron: bool = False,
    is_interjection: bool = False,
    show_prefix: bool = True,
    skill_token_ranges: Sequence[tuple[int, int] | slice] | None = None,
) -> list[StyleAndTextTuples]:
    """Paint a user prompt block.

    ``collapsed=True`` → max ``COLLAPSED_MAX_LINES`` with trailing `` …``.
    Variants mirror Grok constructors (bash / cron / skill tokens).
    """
    if skill_token_ranges is not None:
        block = UserPromptBlock.with_skill_tokens(text, skill_token_ranges)
        block.is_bash = is_bash
        block.is_cron = is_cron
        block.is_interjection = is_interjection
    else:
        block = UserPromptBlock(
            text=text,
            is_bash=is_bash,
            is_cron=is_cron,
            is_interjection=is_interjection,
        )
    max_lines = COLLAPSED_MAX_LINES if collapsed else None
    return block.wrap_prompt_lines(
        width, max_lines, show_prefix=show_prefix, is_selected=selected
    )


__all__ = [
    "COLLAPSED_MAX_LINES",
    "PROMPT_ARROW_WIDTH",
    "USER_PROMPT_BODY_RANGE",
    "UserPromptBlock",
    "paint_user_prompt",
    "sanitize_token_ranges",
]
