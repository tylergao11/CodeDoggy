"""Bounded inline previews + session artifacts for web_fetch overflow.

Ported from:
  grok-build/.../implementations/grok_build/web_fetch/overflow.rs
    inline_budget, OverflowHandler::process (core path)
    recovery_footer, bounded_output, render_with_hint, bounded_generic_marker
    PayloadClassification (format + extension)
  artifact.rs (simplified: no fs2 exclusive lock — A-grade locking)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from codedoggy.tools.grok_build.web_fetch_config import (
    BYTES_PER_TOKEN,
    WEB_FETCH_CONTEXT_PERCENT,
)

ARTIFACT_DIR = "web_fetch"
RECOVERY_FOOTER_PREFIX = "\n\n[web_fetch content truncated:"
LONG_LINE_BYTES = 2_000


class PayloadFormat(Enum):
    Markdown = auto()
    Json = auto()
    JsonLines = auto()
    Text = auto()

    def extension(self) -> str:
        return {
            PayloadFormat.Markdown: "md",
            PayloadFormat.Json: "json",
            PayloadFormat.JsonLines: "jsonl",
            PayloadFormat.Text: "txt",
        }[self]


@dataclass(frozen=True)
class PayloadClassification:
    format: PayloadFormat
    has_long_line: bool

    @staticmethod
    def classify(content_type: str, text: str) -> PayloadClassification:
        mime = content_type.split(";", 1)[0].strip().lower()
        if mime in {"markdown", "text/markdown"}:
            fmt = PayloadFormat.Markdown
        elif mime in {
            "application/x-ndjson",
            "application/ndjson",
            "application/jsonl",
            "text/jsonl",
            "text/x-jsonl",
        }:
            fmt = PayloadFormat.JsonLines
        elif (
            mime in {"application/json", "text/json"}
            or mime.endswith("+json")
            or _looks_like_json(text)
        ):
            fmt = PayloadFormat.Json
        else:
            fmt = PayloadFormat.Text
        has_long = max((len(line) for line in text.splitlines()), default=0) > LONG_LINE_BYTES
        return PayloadClassification(format=fmt, has_long_line=has_long)


def _looks_like_json(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    try:
        json.loads(t)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


@dataclass(frozen=True)
class InlineBudget:
    preview_bytes: int
    output_bytes: int


@dataclass
class OverflowResult:
    content: str
    was_truncated: bool
    artifact_path: Path | None
    path_free_content: str | None


@dataclass(frozen=True)
class RecoveryTools:
    read: str | None = None
    execute: str | None = None


def estimate_chars(tokens: int) -> int:
    """Grok / xai_token_estimation::estimate_chars — tokens * 4."""
    return int(tokens) * BYTES_PER_TOKEN


def inline_budget(context_window_tokens: int, max_markdown_length: int) -> InlineBudget:
    context_budget = int(estimate_chars(context_window_tokens) * WEB_FETCH_CONTEXT_PERCENT)
    return InlineBudget(
        preview_bytes=min(context_budget, max_markdown_length),
        output_bytes=max_markdown_length,
    )


def truncate_str(s: str, max_bytes: int) -> str:
    """UTF-8-safe byte truncate (Grok util/truncate.rs truncate_str)."""
    if max_bytes <= 0:
        return ""
    raw = s.encode("utf-8")
    if len(raw) <= max_bytes:
        return s
    end = max_bytes
    # Floor to char boundary: while not at start of a code unit sequence.
    while end > 0 and (raw[end] & 0xC0) == 0x80:
        end -= 1
    return raw[:end].decode("utf-8")


def _utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))


def recovery_footer(shown_bytes: int, total_bytes: int, file_hint: str) -> str:
    return (
        f"{RECOVERY_FOOTER_PREFIX} showing first {shown_bytes} of "
        f"{total_bytes} bytes.{file_hint}]"
    )


def read_steer(read_tool: str | None) -> str:
    if not read_tool:
        return ""
    return f" Use `{read_tool}` with offsets and limits to read it in chunks."


def web_fetch_steer(
    classification: PayloadClassification,
    tools: RecoveryTools,
) -> str:
    """Simplified steer (no QueryTools detect — A for jq/sed clause)."""
    if classification.has_long_line:
        if not tools.execute:
            return ""
        if classification.format is PayloadFormat.Json:
            fmt, action = "valid JSON", "query"
        elif classification.format is PayloadFormat.JsonLines:
            fmt, action = "JSON Lines", "query"
        elif classification.format is PayloadFormat.Markdown:
            fmt, action = "Markdown", "slice or search"
        else:
            fmt, action = "text", "slice or search"
        return (
            f" The saved file is {fmt} with a very long line, so "
            f"line-oriented read and search tools are ineffective on it — use "
            f"`{tools.execute}` to {action} it."
        )

    if classification.format is PayloadFormat.Json:
        if tools.execute:
            return f" The saved file is valid JSON; use `{tools.execute}` to query it."
        return read_steer(tools.read)
    if classification.format is PayloadFormat.JsonLines:
        if tools.execute:
            return f" The saved file is JSON Lines; use `{tools.execute}` to query it."
        return read_steer(tools.read)
    # Markdown | Text
    read = read_steer(tools.read)
    if read:
        return read
    if tools.execute:
        return f" Use `{tools.execute}` to inspect it."
    return ""


def render_with_hint(
    content: str,
    budget: InlineBudget,
    total_bytes: int,
    file_hint: str,
) -> str | None:
    provisional = recovery_footer(budget.preview_bytes, total_bytes, file_hint)
    if _utf8_len(provisional) > budget.output_bytes:
        return None
    preview_bytes = min(budget.preview_bytes, budget.output_bytes - _utf8_len(provisional))
    preview = truncate_str(content, preview_bytes)
    footer = recovery_footer(_utf8_len(preview), total_bytes, file_hint)
    output = f"{preview}{footer}"
    if _utf8_len(output) <= budget.output_bytes:
        return output
    return None


def bounded_generic_marker(content: str, budget: InlineBudget) -> str:
    for marker in ("\n\n[web_fetch output truncated]", "[truncated]", "..."):
        if _utf8_len(marker) <= budget.output_bytes:
            preview_bytes = min(
                budget.preview_bytes, budget.output_bytes - _utf8_len(marker)
            )
            return f"{truncate_str(content, preview_bytes)}{marker}"
    return ""


def bounded_output(
    content: str,
    budget: InlineBudget,
    total_bytes: int,
    saved_path: Path | None,
    classification: PayloadClassification,
    tools: RecoveryTools,
) -> str:
    if saved_path is not None:
        path_hint = f" Full content saved to: {saved_path}."
        steer = web_fetch_steer(classification, tools)
        if steer:
            full_hint = f"{path_hint}{steer}"
            out = render_with_hint(content, budget, total_bytes, full_hint)
            if out is not None:
                return out
        out = render_with_hint(content, budget, total_bytes, path_hint)
        if out is not None:
            return out
    else:
        out = render_with_hint(content, budget, total_bytes, "")
        if out is not None:
            return out
    return bounded_generic_marker(content, budget)


def save_artifact(session_folder: Path, data: bytes, extension: str) -> Path:
    """Simplified artifact writer (no exclusive lock — A vs artifact.rs)."""
    dir_path = session_folder / ARTIFACT_DIR
    dir_path.mkdir(parents=True, exist_ok=True)
    # Next number: max existing N.ext + 1
    max_n = 0
    for p in dir_path.iterdir():
        if p.name.startswith("."):
            continue
        stem = p.name.split(".", 1)[0]
        if stem.isdigit():
            max_n = max(max_n, int(stem))
    number = max_n + 1
    path = dir_path / f"{number}.{extension}"
    path.write_bytes(data)
    return path


def process_overflow(
    content: str,
    budget: InlineBudget,
    session_folder: Path | None,
    content_type: str,
    tools: RecoveryTools | None = None,
) -> OverflowResult:
    """Grok ``OverflowHandler::process`` (sync, simplified artifact I/O)."""
    tools = tools or RecoveryTools()
    limit = min(budget.preview_bytes, budget.output_bytes)
    if _utf8_len(content) <= limit:
        return OverflowResult(
            content=content,
            was_truncated=False,
            artifact_path=None,
            path_free_content=None,
        )

    total_bytes = _utf8_len(content)
    classification = PayloadClassification.classify(content_type, content)
    extension = classification.format.extension()
    saved_path: Path | None = None
    if session_folder is not None:
        try:
            saved_path = save_artifact(
                session_folder, content.encode("utf-8"), extension
            )
        except OSError:
            saved_path = None

    output = bounded_output(
        content, budget, total_bytes, saved_path, classification, tools
    )
    path_free = bounded_output(
        content, budget, total_bytes, None, classification, tools
    )
    return OverflowResult(
        content=output,
        was_truncated=True,
        artifact_path=saved_path,
        path_free_content=path_free,
    )
