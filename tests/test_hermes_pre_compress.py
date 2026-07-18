"""Hermes on_pre_compress + load_on_disk_store + notes plugin."""

from __future__ import annotations

from pathlib import Path

from codedoggy.context.budget import ContextBudget
from codedoggy.context.compactor import ContextCompactor
from codedoggy.memory.manager import MemoryManager
from codedoggy.memory.plugins import load_memory_provider
from codedoggy.memory.store import MemoryStore, load_on_disk_store
from codedoggy.turn.types import Message, Role


def test_on_pre_compress_collects_provider_text(tmp_path: Path) -> None:
    class Prov:
        name = "external_pre"

        def system_prompt_block(self) -> str:
            return ""

        def prefetch(self, *a, **k) -> str:
            return ""

        def queue_prefetch(self, *a, **k) -> None:
            return None

        def sync_turn(self, *a, **k) -> None:
            return None

        def get_tool_schemas(self):
            return []

        def on_pre_compress(self, messages):
            return "KEEP_THIS_FACT"

    store = MemoryStore(memory_dir=tmp_path)
    store.load_from_disk()
    mm = MemoryManager.create_default(curated=store)
    mm.add_provider(Prov())
    out = mm.on_pre_compress([Message(role=Role.USER, content="hi")])
    assert "KEEP_THIS_FACT" in out


def test_fold_calls_on_pre_compress(tmp_path: Path) -> None:
    called: list[int] = []

    class Prov:
        name = "external_pre2"

        def system_prompt_block(self) -> str:
            return ""

        def prefetch(self, *a, **k) -> str:
            return ""

        def queue_prefetch(self, *a, **k) -> None:
            return None

        def sync_turn(self, *a, **k) -> None:
            return None

        def get_tool_schemas(self):
            return []

        def on_pre_compress(self, messages):
            called.append(len(messages))
            return "fact-from-provider"

    store = MemoryStore(memory_dir=tmp_path)
    store.load_from_disk()
    mm = MemoryManager.create_default(curated=store)
    mm.add_provider(Prov())

    budget = ContextBudget.from_max_chars(3_000, threshold_percent=20)
    c = ContextCompactor(budget=budget, memory_manager=mm)
    msgs = [Message(role=Role.SYSTEM, content="sys")]
    for i in range(20):
        msgs.append(Message(role=Role.USER, content=f"u{i} " + ("x" * 80)))
        msgs.append(Message(role=Role.ASSISTANT, content=f"a{i} " + ("y" * 80)))
    res = c.ensure(msgs)
    assert called, "on_pre_compress should run before fold"
    # Fold may reject or succeed; provider must have been notified when folding
    assert res.messages


def test_load_on_disk_store(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CODEDOGGY_HOME", str(tmp_path))
    # memories dir under home
    mem_dir = tmp_path / "memories"
    mem_dir.mkdir()
    (mem_dir / "MEMORY.md").write_text("disk-fact\n", encoding="utf-8")
    store = load_on_disk_store()
    assert any("disk-fact" in e for e in store.memory_entries)


def test_notes_plugin_loads_and_prefetches(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CODEDOGGY_HOME", str(tmp_path))
    (tmp_path / "memories").mkdir(parents=True)
    notes = load_memory_provider("notes")
    assert notes is not None
    assert notes.name == "notes"
    notes.initialize(session_id="s1")
    path = Path(notes.path)
    path.write_text("auth login cookie ttl is 3600\n", encoding="utf-8")
    hit = notes.prefetch("auth cookie")
    assert "3600" in hit or "auth" in hit.lower()
    pre = notes.on_pre_compress(
        [Message(role=Role.USER, content="remember the cookie ttl")]
    )
    assert "cookie" in pre.lower()
    schemas = notes.get_tool_schemas()
    assert any(s["name"] == "notes_append" for s in schemas)
    out = notes.handle_tool_call("notes_append", {"content": "DECIDE: use httponly"})
    assert "success" in out
