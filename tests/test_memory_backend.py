"""Unit tests for host SimpleMemoryStoreBackend (C/A curated search)."""

from __future__ import annotations

from pathlib import Path

from codedoggy.host.memory_backend import (
    DEFAULT_MAX_RESULTS,
    DEFAULT_MIN_SCORE,
    SimpleMemoryStoreBackend,
    build_memory_backend,
    _entry_line_span,
    _term_overlap_score,
)
from codedoggy.memory.defaults import ENTRY_DELIMITER
from codedoggy.memory.store import MemoryStore
from codedoggy.tools import ToolRegistryBuilder
from codedoggy.tools.runtime import ToolCallContext


def test_build_memory_backend_none() -> None:
    assert build_memory_backend(None) is None


def test_build_memory_backend_wraps_store(tmp_path: Path) -> None:
    store = MemoryStore(memory_dir=tmp_path / "mem")
    store.load_from_disk()
    backend = build_memory_backend(store)
    assert isinstance(backend, SimpleMemoryStoreBackend)


def test_term_overlap_phrase_and_partial() -> None:
    assert _term_overlap_score("ripgrep code search", "prefer ripgrep over shell") == 1.0 / 3.0
    assert _term_overlap_score("prefer ripgrep", "prefer ripgrep over shell") == 1.0
    assert _term_overlap_score("zzzz missing", "prefer ripgrep") == 0.0
    assert _term_overlap_score("", "anything") == 0.0


def test_entry_line_span_multiline() -> None:
    entries = ["alpha", "beta\ngamma", "delta"]
    assert _entry_line_span(entries, 0) == (1, 1)
    # after "alpha\n§\n" → start at line 3
    start, end = _entry_line_span(entries, 1)
    assert start == 3
    assert end == 4  # two lines: beta, gamma
    start2, end2 = _entry_line_span(entries, 2)
    assert start2 == end2
    assert start2 > end


def test_search_hits_memory_and_user(tmp_path: Path) -> None:
    store = MemoryStore(memory_dir=tmp_path / "mem")
    store.load_from_disk()
    store.add("memory", "prefer ripgrep over shell grep for code search")
    store.add("user", "user likes concise answers")
    store.add("memory", "never commit secrets to the repo")

    backend = build_memory_backend(store)
    assert backend is not None

    hits = backend.search("ripgrep code search")
    assert hits
    top = hits[0]
    assert set(top) >= {"score", "source", "path", "start_line", "end_line", "snippet"}
    assert top["source"] == "memory"
    assert top["path"].endswith("MEMORY.md")
    assert "ripgrep" in top["snippet"].lower()
    assert top["score"] >= DEFAULT_MIN_SCORE
    assert top["start_line"] >= 1
    assert top["end_line"] >= top["start_line"]

    user_hits = backend.search("concise answers")
    assert any(h["source"] == "user" for h in user_hits)
    assert any("USER.md" in h["path"] for h in user_hits)


def test_search_min_score_and_max_results(tmp_path: Path) -> None:
    store = MemoryStore(memory_dir=tmp_path / "mem")
    store.load_from_disk()
    store.add("memory", "alpha beta gamma unique_token_one")
    store.add("memory", "alpha beta delta unique_token_two")
    store.add("memory", "epsilon zeta unique_token_three")

    backend = build_memory_backend(store)
    assert backend is not None

    # high floor → only strong matches
    strict = backend.search("alpha beta", min_score=0.99)
    assert all(h["score"] >= 0.99 for h in strict)

    limited = backend.search("unique_token", max_results=1)
    assert len(limited) <= 1

    empty = backend.search("unique_token", max_results=0)
    assert empty == []


def test_search_empty_query_and_no_match(tmp_path: Path) -> None:
    store = MemoryStore(memory_dir=tmp_path / "mem")
    store.load_from_disk()
    store.add("memory", "only known facts live here")
    backend = build_memory_backend(store)
    assert backend is not None
    assert backend.search("   ") == []
    assert backend.search("zzzznotpresentatall") == []


def test_line_numbers_match_joined_file(tmp_path: Path) -> None:
    """start/end lines align with §-joined on-disk layout for memory_get."""
    store = MemoryStore(memory_dir=tmp_path / "mem")
    store.load_from_disk()
    e0 = "first entry single line"
    e1 = "second entry\nhas two lines"
    store.add("memory", e0)
    store.add("memory", e1)

    backend = build_memory_backend(store)
    assert backend is not None
    hits = backend.search("second entry")
    assert hits
    h = hits[0]
    # Reconstruct file text the same way MemoryStore writes it
    body = ENTRY_DELIMITER.join([e0, e1])
    lines = body.split("\n")
    # 1-based slice
    snippet_from_file = "\n".join(lines[h["start_line"] - 1 : h["end_line"]])
    assert "second entry" in snippet_from_file
    assert "has two lines" in snippet_from_file


def test_memory_search_tool_integration(tmp_path: Path) -> None:
    """Host backend plugged into extra['memory_backend'] drives the tool shell."""
    store = MemoryStore(memory_dir=tmp_path / "mem")
    store.load_from_disk()
    store.add("memory", "prefer ripgrep over shell grep for code search")

    backend = build_memory_backend(store)
    from codedoggy.tools.builtins import register_optional_grok_memory_tools

    b = ToolRegistryBuilder.new()
    register_optional_grok_memory_tools(b)
    tools = b.finalize(product_surface=False)
    ctx = ToolCallContext(
        cwd=tmp_path,
        extra={"memory_store": store, "memory_backend": backend},
    )
    out = tools.call("memory_search", {"query": "ripgrep code search"}, ctx)
    assert "Found " in out
    assert "### Result" in out
    assert "ripgrep" in out.lower()
    assert "score:" in out.lower()


def test_defaults_exported() -> None:
    assert DEFAULT_MAX_RESULTS > 0
    assert 0.0 <= DEFAULT_MIN_SCORE < 1.0
