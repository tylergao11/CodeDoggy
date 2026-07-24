"""Markdown content block — faithful subset of Grok ``markdown_content.rs``.

Port of ``xai-grok-pager/src/scrollback/blocks/markdown_content.rs`` and the
blockquote bar decoration from ``quote_bar.rs``. Full Grok uses
``StreamingMarkdownRenderer`` + syntect; this module renders a deterministic
pretty subset suitable for pager scrollback:

- ATX headings ``#``–``######``
- unordered / ordered lists
- blockquotes rewritten to ``│`` bar (U+2502) with muted style
- fenced code (``` / ~~~)
- horizontal rules
- paragraphs
- inline bold / italic / ``code``

Style classes use the ``class:grok.md.*`` prefix (see PORT.md). Layout/theme
agents own canonical Theme colors; these are paint-edge class names only.

Canonical layout types (BlockLine / BlockOutput) live with the layout agent;
this module returns rows of StyleAndTextTuples.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from prompt_toolkit.utils import get_cwidth

if TYPE_CHECKING:
    from prompt_toolkit.formatted_text import StyleAndTextTuples
else:
    StyleAndTextTuples = list  # type: ignore[misc, assignment]

# ---------------------------------------------------------------------------
# Style classes (class:grok.* — theme maps these to colors)
# ---------------------------------------------------------------------------

S_TEXT = "class:grok.md.text"
S_H = (
    "class:grok.md.h1",
    "class:grok.md.h2",
    "class:grok.md.h3",
    "class:grok.md.h4",
    "class:grok.md.h5",
    "class:grok.md.h6",
)
S_STRONG = "class:grok.md.strong"
S_EM = "class:grok.md.em"
S_CODE = "class:grok.md.code"
S_CODE_BLOCK = "class:grok.md.code.block"
S_CODE_LANG = "class:grok.md.code.lang"
S_QUOTE_BAR = "class:grok.md.quote.bar"  # mirrors blockquote_outer = md_muted+dim
S_QUOTE = "class:grok.md.quote"
S_LIST = "class:grok.md.list.marker"  # mirrors list_item = md_muted
S_RULE = "class:grok.md.rule"  # mirrors rule = md_muted
S_LINK = "class:grok.md.link"

# Grok quote_bar.rs: parser rewrites `>` to U+2502 bar.
QUOTE_BAR = "\u2502"

MARKDOWN_BODY_RANGE = 0  # Grok markdown_content.rs MARKDOWN_BODY_RANGE

_FENCE_RE = re.compile(r"^(\s*)(```|~~~)(.*)$")
_HEADING_RE = re.compile(r"^(#{1,6})(\s+)(.*)$")
_HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_UL_RE = re.compile(r"^(\s*)([-*+])(\s+)(.*)$")
_OL_RE = re.compile(r"^(\s*)(\d+[.)])(\s+)(.*)$")
_QUOTE_RE = re.compile(r"^(\s*)(>+)(\s?)(.*)$")
_CHECKBOX_RE = re.compile(r"^(\[[ xX]\])(\s*)(.*)$")

# Inline: code first so * inside backticks is not emphasized.
_INLINE_RE = re.compile(
    r"(`[^`\n]+`)"
    r"|(\*\*[^*\n]+\*\*|__[^_\n]+__)"
    r"|(\*[^*\n]+\*|_[^_\n]+_)"
    r"|(\[[^\]]+\]\([^)]+\))"
)


# ---------------------------------------------------------------------------
# Width / wrap helpers
# ---------------------------------------------------------------------------


def _disp_width(text: str) -> int:
    return get_cwidth(text) if text else 0


def _append_merge(row: StyleAndTextTuples, style: str, piece: str) -> None:
    if not piece:
        return
    if row and row[-1][0] == style:
        row[-1] = (style, row[-1][1] + piece)
    else:
        row.append((style, piece))


def _wrap_fragments(
    fragments: StyleAndTextTuples,
    width: int,
    *,
    subsequent_prefix: StyleAndTextTuples | None = None,
) -> list[StyleAndTextTuples]:
    """Wrap styled spans by terminal cell width.

    Soft-wraps at spaces when possible; hard-breaks overlong tokens.
    ``subsequent_prefix`` (e.g. quote bar) is prepended to continuation rows —
    mirrors Grok blockquote prefix reinjection on wrap
    (``wrapping.rs`` ``blockquote_prefix_len``).
    """
    width = max(1, int(width))
    prefix: StyleAndTextTuples = list(subsequent_prefix or [])
    prefix_w = sum(_disp_width(t) for _, t in prefix)
    cont_width = max(1, width - prefix_w)

    rows: list[StyleAndTextTuples] = []
    row: StyleAndTextTuples = []
    used = 0
    line_width = width  # first visual line uses full width

    def start_continuation() -> None:
        nonlocal row, used, line_width
        rows.append(row)
        row = list(prefix)
        used = prefix_w
        line_width = cont_width + prefix_w  # total row budget incl. prefix

    # Flatten into (style, token) keeping spaces as break opportunities.
    tokens: list[tuple[str, str]] = []
    for style, text in fragments:
        if not text:
            continue
        for part in re.split(r"( +)", text):
            if part:
                tokens.append((style, part))

    for style, token in tokens:
        # Spaces at wrap boundary are dropped (mirrors wrap_ranges_trim).
        if token.startswith(" ") and used > prefix_w and used + _disp_width(token) > line_width:
            token = token.lstrip(" ")
            if not token:
                continue
            start_continuation()

        tw = _disp_width(token)
        if used > 0 and used + tw > line_width and used > prefix_w:
            start_continuation()
            if token.startswith(" "):
                token = token.lstrip(" ")
                if not token:
                    continue
                tw = _disp_width(token)

        # Hard-break overlong token across rows.
        while token and used + _disp_width(token) > line_width:
            room = max(1, line_width - used)
            acc: list[str] = []
            acc_w = 0
            i = 0
            chars = list(token)
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
            _append_merge(row, style, "".join(acc))
            used += _disp_width("".join(acc))
            token = "".join(chars[i:])
            if token:
                start_continuation()

        if token:
            _append_merge(row, style, token)
            used += _disp_width(token)

    rows.append(row)
    return rows or [[(S_TEXT, "")]]


def _inline(text: str, base: str = S_TEXT) -> StyleAndTextTuples:
    """Render inline markdown: code, bold, italic, links (label only).

    Outer markers are stripped (Grok ``strong_outer`` / ``inline_code_outer``
    are hidden).
    """
    if not text:
        return [(base, "")]
    if not any(ch in text for ch in ("`", "*", "_", "[")):
        return [(base, text)]

    out: StyleAndTextTuples = []
    pos = 0
    for m in _INLINE_RE.finditer(text):
        if m.start() > pos:
            out.append((base, text[pos : m.start()]))
        if m.group(1) is not None:
            out.append((S_CODE, m.group(1)[1:-1]))
        elif m.group(2) is not None:
            raw = m.group(2)
            out.append((S_STRONG, raw[2:-2]))
        elif m.group(3) is not None:
            raw = m.group(3)
            out.append((S_EM, raw[1:-1]))
        else:
            link = m.group(4) or ""
            lm = re.match(r"\[([^\]]+)\]\(([^)]+)\)", link)
            label = lm.group(1) if lm else link
            out.append((S_LINK, label))
        pos = m.end()
    if pos < len(text):
        out.append((base, text[pos:]))
    return out or [(base, text)]


def _quote_depth_and_body(line: str) -> tuple[int, str]:
    """Count leading ``>`` markers and return remaining body text."""
    m = _QUOTE_RE.match(line)
    if not m:
        return 0, line
    return len(m.group(2)), m.group(4)


def _quote_bar_prefix(depth: int) -> StyleAndTextTuples:
    """Build ``│ │ … `` nesting prefix (``depth`` levels).

    Shape matches Grok quote_bar tests: ``│ text`` → depth 1, ``│ │ deep`` → 2.
    """
    if depth <= 0:
        return []
    frags: StyleAndTextTuples = []
    for i in range(depth):
        frags.append((S_QUOTE_BAR, QUOTE_BAR))
        frags.append((S_QUOTE_BAR, " "))
    return frags


def render_markdown(text: str, *, width: int) -> list[StyleAndTextTuples]:
    """Render markdown source to painted rows (Grok pretty mode, subset).

    Returns a list of rows; each row is StyleAndTextTuples (no trailing
    newline — the scrollback painter appends them). Empty input yields a
    single empty row, matching Grok ``MarkdownContent::output`` placeholder.
    """
    width = max(1, int(width))
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw:
        return [[(S_TEXT, "")]]

    rows: list[StyleAndTextTuples] = []
    in_fence = False
    fence_ch = ""

    for line in raw.split("\n"):
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(2)
            if in_fence and marker[0] == fence_ch:
                in_fence = False
                fence_ch = ""
                continue
            if not in_fence:
                in_fence = True
                fence_ch = marker[0]
                lang = (fence.group(3) or "").strip()
                # Language tag (Grok code_language style).
                if lang:
                    rows.extend(_wrap_fragments([(S_CODE_LANG, lang)], width))
                continue

        if in_fence:
            body = line if line else " "
            rows.extend(_wrap_fragments([(S_CODE_BLOCK, body)], width))
            continue

        if _HR_RE.match(line):
            # Grok paints exactly three heavy bars (U+2501), not full-width light ─.
            rows.append([(S_RULE, "\u2501" * 3)])
            continue

        depth, body = _quote_depth_and_body(line)
        if depth > 0:
            if not body.strip():
                # Bar-only blank quote line (Grok ``│`` / ``│ │``).
                blank: StyleAndTextTuples = []
                for d in range(depth):
                    blank.append((S_QUOTE_BAR, QUOTE_BAR))
                    if d + 1 < depth:
                        blank.append((S_QUOTE_BAR, " "))
                rows.append(blank)
            else:
                bar = _quote_bar_prefix(depth)
                frags: StyleAndTextTuples = list(bar) + list(_inline(body, S_QUOTE))
                rows.extend(
                    _wrap_fragments(frags, width, subsequent_prefix=bar)
                )
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            level = min(6, len(heading.group(1)))
            style = S_H[level - 1]
            title = re.sub(r"\s+#+\s*$", "", heading.group(3).rstrip())
            rows.extend(_wrap_fragments(_inline(title, style), width))
            continue

        ul = _UL_RE.match(line)
        if ul:
            indent, marker, _sp, body = ul.groups()
            cb = _CHECKBOX_RE.match(body)
            if cb:
                box, box_sp, rest = cb.groups()
                checked = "x" in box.lower()
                box_style = (
                    "class:grok.md.task.checked"
                    if checked
                    else "class:grok.md.task.unchecked"
                )
                frags = [
                    (S_LIST, indent),
                    (box_style, box),
                    (S_TEXT, box_sp if box_sp else " "),
                    *_inline(rest, S_TEXT),
                ]
                hang = [(S_TEXT, indent + "  ")]
            else:
                # Only -/* → •; keep + as + (Grok fidelity).
                bullet = "+ " if marker == "+" else "• "
                frags = [
                    (S_LIST, indent),
                    (S_LIST, bullet),
                    *_inline(body, S_TEXT),
                ]
                hang = [(S_TEXT, indent + "  ")]
            rows.extend(_wrap_fragments(frags, width, subsequent_prefix=hang))
            continue

        ol = _OL_RE.match(line)
        if ol:
            indent, num, _sp, body = ol.groups()
            frags = [
                (S_LIST, indent),
                (S_LIST, f"{num} "),
                *_inline(body, S_TEXT),
            ]
            hang = [(S_TEXT, indent + (" " * (len(num) + 1)))]
            rows.extend(_wrap_fragments(frags, width, subsequent_prefix=hang))
            continue

        if not line:
            rows.append([(S_TEXT, "")])
            continue

        rows.extend(_wrap_fragments(_inline(line, S_TEXT), width))

    return rows or [[(S_TEXT, "")]]


__all__ = [
    "MARKDOWN_BODY_RANGE",
    "QUOTE_BAR",
    "render_markdown",
]
