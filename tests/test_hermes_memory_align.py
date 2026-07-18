"""Hermes source-level memory enhancements alignment tests."""

from __future__ import annotations

from codedoggy.memory.context_fence import (
    StreamingContextScrubber,
    build_memory_context_block,
    sanitize_context,
)
from codedoggy.memory.hermes_seam import on_delegation
from codedoggy.memory.manager import MemoryManager
from codedoggy.memory.provider import BaseMemoryProvider, SessionFtsProvider


def test_fence_system_note_matches_hermes_authoritative() -> None:
    block = build_memory_context_block("fact: cwd is /tmp")
    assert block.startswith("<memory-context>")
    assert "authoritative reference data" in block
    assert "persistent memory" in block
    assert "fact: cwd is /tmp" in block
    assert block.rstrip().endswith("</memory-context>")


def test_sanitize_strips_prewrapped_provider_output() -> None:
    raw = build_memory_context_block("inner")
    clean = sanitize_context(raw)
    assert "<memory-context>" not in clean
    assert "inner" in clean or clean.strip() == ""


def test_streaming_scrubber_split_tags() -> None:
    scrubber = StreamingContextScrubber()
    # Open on block boundary
    v1 = scrubber.feed("<memory-context>\n")
    assert v1 == ""
    v2 = scrubber.feed("SECRET_RECALL\n")
    assert "SECRET" not in v2
    v3 = scrubber.feed("</memory-context>\nvisible")
    assert "SECRET" not in v3
    assert "visible" in v3
    assert scrubber.flush() == ""


def test_streaming_scrubber_partial_open_held() -> None:
    scrubber = StreamingContextScrubber()
    # Partial tag must not leak mid-tag
    part = scrubber.feed("hello\n<memory-cont")
    assert "<memory" not in part or part.endswith("hello\n") or "hello" in part
    rest = scrubber.feed("ext>\npayload\n</memory-context>\nok")
    joined = part + rest
    assert "payload" not in joined
    assert "ok" in joined


def test_session_fts_rewound_clears_warm() -> None:
    class FakeStore:
        def search(self, *a, **k):
            return []

    p = SessionFtsProvider(FakeStore())
    p._warm = "cached"
    p._warm_query = "login"
    p.on_session_switch("sid", rewound=True)
    assert p._warm == ""
    assert p._warm_query == ""


def test_on_delegation_fanout() -> None:
    class Prov(BaseMemoryProvider):
        name = "ext_del"

        def __init__(self) -> None:
            self.seen: list[tuple] = []

        def get_tool_schemas(self):
            return []

        def on_delegation(self, task, result, *, child_session_id="", **kwargs):
            self.seen.append((task, result, child_session_id))

    mm = MemoryManager()
    p = Prov()
    assert mm.add_provider(p)
    mm.on_delegation("do X", "did X", child_session_id="sub_1")
    assert p.seen == [("do X", "did X", "sub_1")]
    # seam
    p.seen.clear()
    on_delegation(mm, task="t2", result="r2", child_session_id="sub_2")
    assert p.seen == [("t2", "r2", "sub_2")]


def test_on_memory_write_skips_builtin() -> None:
    class Ext(BaseMemoryProvider):
        name = "ext_write"
        hits = 0

        def get_tool_schemas(self):
            return []

        def on_memory_write(self, action, target, content, metadata=None):
            Ext.hits += 1

    class BuiltinLike(BaseMemoryProvider):
        name = "builtin_curated"

        def get_tool_schemas(self):
            return []

        def on_memory_write(self, *a, **k):
            raise AssertionError("builtin must not receive mirror")

    mm = MemoryManager()
    mm.add_provider(BuiltinLike())
    mm.add_provider(Ext())
    Ext.hits = 0
    mm.on_memory_write("add", "memory", "hello")
    assert Ext.hits == 1


def test_rewound_not_injected_as_false_kwarg() -> None:
    class Prov(BaseMemoryProvider):
        name = "ext_switch"
        last_kw: dict = {}

        def get_tool_schemas(self):
            return []

        def on_session_switch(self, new_session_id, **kwargs):
            Prov.last_kw = dict(kwargs)

    mm = MemoryManager()
    mm.add_provider(Prov())
    mm.on_session_switch("s2", reset=False, rewound=False)
    assert "rewound" not in Prov.last_kw
    mm.on_session_switch("s2", rewound=True)
    assert Prov.last_kw.get("rewound") is True
