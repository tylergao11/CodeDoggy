"""Small shared syntax palette adapter for every reading surface.

This is intentionally a heuristic highlighter, not a parser.  Its job is to
make code structure scannable in the TUI while keeping one token taxonomy for
the homepage, plan approval, and tool preview surfaces.
"""

from __future__ import annotations

import re

from prompt_toolkit.formatted_text import StyleAndTextTuples


_KEYWORDS = frozenset(
    {
        "False",
        "None",
        "True",
        "Self",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "case",
        "catch",
        "chan",
        "class",
        "const",
        "continue",
        "crate",
        "default",
        "defer",
        "def",
        "del",
        "do",
        "elif",
        "else",
        "enum",
        "except",
        "export",
        "extends",
        "finally",
        "fn",
        "for",
        "from",
        "func",
        "function",
        "global",
        "go",
        "if",
        "impl",
        "import",
        "in",
        "interface",
        "is",
        "lambda",
        "let",
        "loop",
        "map",
        "match",
        "mod",
        "move",
        "mut",
        "new",
        "nonlocal",
        "not",
        "null",
        "or",
        "package",
        "pass",
        "private",
        "protected",
        "pub",
        "public",
        "raise",
        "range",
        "return",
        "select",
        "self",
        "static",
        "struct",
        "super",
        "switch",
        "this",
        "throw",
        "try",
        "type",
        "typeof",
        "undefined",
        "use",
        "var",
        "void",
        "while",
        "with",
        "yield",
    }
)

_TYPES = frozenset(
    {
        "Any",
        "Array",
        "Boolean",
        "Buffer",
        "Callable",
        "Dict",
        "Error",
        "Iterable",
        "List",
        "Literal",
        "Map",
        "Never",
        "Number",
        "Object",
        "Optional",
        "Path",
        "Promise",
        "Record",
        "Sequence",
        "Set",
        "String",
        "Tuple",
        "Union",
        "Unknown",
        "Vec",
        "bool",
        "byte",
        "bytes",
        "char",
        "dict",
        "double",
        "float",
        "int",
        "list",
        "long",
        "object",
        "set",
        "short",
        "str",
        "string",
        "tuple",
        "uint",
        "usize",
    }
)

_TOKEN_RE = re.compile(
    r"(?P<cmt>//.*?$|#.*?$|/\*.*?\*/)"
    r"|(?P<str>'''(?:\\.|[^'\\])*'''|\"\"\"(?:\\.|[^\"\\])*\"\"\""
    r"|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:\\.|[^`\\])*`)"
    r"|(?P<num>\b(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b)"
    r"|(?P<id>\b[A-Za-z_][A-Za-z0-9_]*\b)"
    r"|(?P<sym>[^\sA-Za-z0-9_]+)"
    r"|(?P<ws>\s+)",
    re.MULTILINE,
)
_CALL_SUFFIX_RE = re.compile(r"\s*\(")


def highlight_code_line(
    line: str,
    *,
    style_prefix: str,
) -> StyleAndTextTuples:
    """Color one code line using semantic ``kw/fn/type/...`` style suffixes."""

    def style(kind: str) -> str:
        return f"class:{style_prefix}.{kind}"

    if not line:
        return [(style("plain"), "")]
    stripped = line.lstrip()
    if stripped.startswith(("#", "//")):
        return [(style("cmt"), line)]

    fragments: StyleAndTextTuples = []
    pos = 0
    for match in _TOKEN_RE.finditer(line):
        if match.start() > pos:
            fragments.append((style("plain"), line[pos : match.start()]))
        token = match.group(0)
        if match.group("cmt") is not None:
            kind = "cmt"
        elif match.group("str") is not None:
            kind = "str"
        elif match.group("num") is not None:
            kind = "num"
        elif match.group("id") is not None:
            ident = token
            if ident in _KEYWORDS:
                kind = "kw"
            elif ident in _TYPES or (ident[:1].isupper() and ident != "_"):
                kind = "type"
            elif _CALL_SUFFIX_RE.match(line, match.end()) is not None:
                kind = "fn"
            else:
                kind = "plain"
        elif match.group("sym") is not None:
            kind = "sym"
        else:
            kind = "plain"
        fragments.append((style(kind), token))
        pos = match.end()
    if pos < len(line):
        fragments.append((style("plain"), line[pos:]))
    return fragments or [(style("plain"), line)]


__all__ = ["highlight_code_line"]
