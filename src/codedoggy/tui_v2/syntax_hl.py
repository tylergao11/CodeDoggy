"""Per-line syntax highlight for edit/read bodies (Grok syntect spirit).

Prefer Pygments when a path or lexer is known; fall back to the shared
heuristic in ``codedoggy.tui.syntax``. Style classes: ``class:grok.syn.*``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import PurePosixPath
from typing import Sequence

Fragment = tuple[str, str]

# Token → theme class (registered in theme_style_dict)
_TOKEN_CLASS = {
    "kw": "class:grok.syn.kw",
    "fn": "class:grok.syn.fn",
    "type": "class:grok.syn.type",
    "str": "class:grok.syn.str",
    "cmt": "class:grok.syn.cmt",
    "num": "class:grok.syn.num",
    "sym": "class:grok.syn.sym",
    "plain": "class:grok.syn.plain",
}


@lru_cache(maxsize=64)
def _lexer_for_path(path: str):
    try:
        from pygments.lexers import get_lexer_for_filename
        from pygments.util import ClassNotFound
    except Exception:  # noqa: BLE001
        return None
    name = PurePosixPath(path.replace("\\", "/")).name or path
    try:
        return get_lexer_for_filename(name, stripnl=False, ensurenl=False)
    except ClassNotFound:
        return None
    except Exception:  # noqa: BLE001
        return None


def _pygments_token_kind(ttype) -> str:
    names = [str(p) for p in ttype.split()]
    joined = " ".join(names).lower()
    # Walk most-specific first via string.
    s = str(ttype)
    if "Comment" in s:
        return "cmt"
    if "String" in s or "Char" in s or "Literal.String" in s:
        return "str"
    if "Keyword" in s:
        return "kw"
    if "Name.Function" in s or "Name.Decorator" in s:
        return "fn"
    if "Name.Class" in s or "Name.Builtin" in s or "Keyword.Type" in s:
        return "type"
    if "Number" in s or "Literal.Number" in s:
        return "num"
    if "Operator" in s or "Punctuation" in s:
        return "sym"
    if "Name" in s:
        return "plain"
    _ = joined
    return "plain"


def _highlight_pygments(line: str, path: str | None) -> list[Fragment] | None:
    if not line:
        return [(_TOKEN_CLASS["plain"], "")]
    lexer = _lexer_for_path(path) if path else None
    if lexer is None:
        return None
    try:
        from pygments import lex
    except Exception:  # noqa: BLE001
        return None
    # Pygments often wants trailing newline for state; strip from segments.
    src = line if line.endswith("\n") else line + "\n"
    out: list[Fragment] = []
    try:
        for ttype, value in lex(src, lexer):
            if not value:
                continue
            text = value.replace("\r", "").replace("\n", "")
            if not text:
                continue
            kind = _pygments_token_kind(ttype)
            out.append((_TOKEN_CLASS.get(kind, _TOKEN_CLASS["plain"]), text))
    except Exception:  # noqa: BLE001
        return None
    return out or [(_TOKEN_CLASS["plain"], line)]


def _highlight_heuristic(line: str) -> list[Fragment]:
    try:
        from codedoggy.tui.syntax import highlight_code_line

        frags = highlight_code_line(line, style_prefix="grok.syn")
        # Ensure class: prefix
        fixed: list[Fragment] = []
        for st, tx in frags:
            if st.startswith("class:"):
                fixed.append((st, tx))
            else:
                fixed.append((f"class:{st}" if st else _TOKEN_CLASS["plain"], tx))
        return fixed or [(_TOKEN_CLASS["plain"], line)]
    except Exception:  # noqa: BLE001
        return [(_TOKEN_CLASS["plain"], line)]


def highlight_code_line(
    line: str,
    *,
    path: str | None = None,
    fallback_style: str | None = None,
) -> list[Fragment]:
    """Highlight one source line into style/text fragments (no trailing newline)."""
    text = line.rstrip("\r\n")
    if not text:
        style = fallback_style or _TOKEN_CLASS["plain"]
        return [(style, "")]
    frags = _highlight_pygments(text, path)
    if frags is None:
        frags = _highlight_heuristic(text)
    if fallback_style and fallback_style not in {
        _TOKEN_CLASS["plain"],
        "class:grok.diff.equal",
        "class:grok.primary",
    }:
        # Tint every fragment with an extra role (e.g. keep delete FG band readable).
        # For solid-override themes we skip; here we leave syn classes as-is.
        pass
    return frags


def highlight_or_solid(
    line: str,
    *,
    path: str | None,
    solid_style: str,
    enable: bool = True,
) -> list[Fragment]:
    """HL when enabled; else a single solid-style span (Grok bandless line FG)."""
    text = line.rstrip("\r\n")
    if not enable:
        return [(solid_style, text if text else " ")]
    frags = highlight_code_line(text, path=path, fallback_style=solid_style)
    if not frags:
        return [(solid_style, text if text else " ")]
    # Empty line → keep a space so bg band paints.
    if len(frags) == 1 and frags[0][1] == "":
        return [(frags[0][0], " ")]
    return frags


def slice_fragments(frags: Sequence[Fragment], start: int, end: int) -> list[Fragment]:
    """Slice fragments by character offset [start, end)."""
    if start >= end:
        return []
    out: list[Fragment] = []
    pos = 0
    for st, tx in frags:
        n = len(tx)
        a = max(start, pos)
        b = min(end, pos + n)
        if a < b:
            out.append((st, tx[a - pos : b - pos]))
        pos += n
        if pos >= end:
            break
    return out


def highlight_file_lines(
    file_text: str,
    *,
    path: str | None = None,
    max_line: int | None = None,
) -> dict[int, list[Fragment]]:
    """Full-file progressive HL map: 1-based line number → fragments.

    Grok ``compute_file_scoped_styles`` spirit: one lexer walk so multi-line
    constructs don't reset per hunk. Falls back to per-line HL when the
    whole-file lex fails.
    """
    lines = file_text.splitlines()
    if max_line is not None:
        lines = lines[: max(0, int(max_line))]
    out: dict[int, list[Fragment]] = {}
    lexer = _lexer_for_path(path) if path else None
    if lexer is not None:
        try:
            from pygments import lex

            # Ensure trailing newline for line-oriented lexers.
            src = file_text if file_text.endswith("\n") else file_text + "\n"
            # Cap extremely large files (Grok has HL size caps).
            if len(src) > 2_000_000:
                lexer = None
            else:
                line_no = 1
                buf: list[Fragment] = []
                for ttype, value in lex(src, lexer):
                    if not value:
                        continue
                    parts = value.split("\n")
                    for pi, part in enumerate(parts):
                        if pi > 0:
                            # Flush previous line
                            if line_no <= (max_line or line_no):
                                cleaned = [
                                    (st, tx)
                                    for st, tx in buf
                                    if tx
                                ]
                                out[line_no] = cleaned or [
                                    (_TOKEN_CLASS["plain"], "")
                                ]
                            line_no += 1
                            buf = []
                            if max_line is not None and line_no > max_line:
                                return out
                        if part:
                            kind = _pygments_token_kind(ttype)
                            buf.append(
                                (_TOKEN_CLASS.get(kind, _TOKEN_CLASS["plain"]), part)
                            )
                if buf and (max_line is None or line_no <= max_line):
                    out[line_no] = [(st, tx) for st, tx in buf if tx] or [
                        (_TOKEN_CLASS["plain"], "")
                    ]
                # Verify coverage for lines we care about
                if out:
                    return out
        except Exception:  # noqa: BLE001
            pass

    # Per-line fallback
    for i, line in enumerate(lines, start=1):
        if max_line is not None and i > max_line:
            break
        out[i] = highlight_code_line(line, path=path)
    return out


__all__ = [
    "highlight_code_line",
    "highlight_file_lines",
    "highlight_or_solid",
    "slice_fragments",
]
