"""Full Grok↔Hermes lifecycle through hermes_seam (single owner)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codedoggy.memory.context_fence import build_memory_context_block
from codedoggy.memory.hermes_seam import (
    bind_session,
    build_system_memory_block,
    commit_session_boundary,
    notify_curated_write,
    on_pre_compress,
    on_session_close,
    on_transcript_rewound,
    on_turn_begin,
    on_turn_end,
    prefetch_fenced,
    sample_messages_with_memory,
)
from codedoggy.memory.manager import MemoryManager
from codedoggy.memory.store import MemoryStore
from codedoggy.session.kernel import RuntimeKernel
from codedoggy.turn.types import Message, Role


class _RecordingProv:
    """Minimal external provider that records lifecycle hooks."""

    name = "seam_recorder"

    def __init__(self) -> None:
        self.events: list[str] = []
        self._session_id = ""

    def is_available(self) -> bool:
        return True

    def system_prompt_block(self) -> str:
        return "PROV_STATIC"

    def prefetch(self, query: str = "", **kwargs: Any) -> str:
        self.events.append(f"prefetch:{query[:20]}")
        return f"recall:{query[:40]}" if query else ""

    def queue_prefetch(self, *a: Any, **k: Any) -> None:
        self.events.append("queue_prefetch")

    def sync_turn(self, *a: Any, **k: Any) -> None:
        self.events.append("sync")

    def get_tool_schemas(self) -> list:
        return []

    def initialize(self, **kwargs: Any) -> None:
        self._session_id = str(kwargs.get("session_id") or "")
        self.events.append(f"init:{self._session_id}")

    def on_turn_start(self, turn_number: int, message: str = "", **kwargs: Any) -> None:
        self.events.append(f"turn_start:{turn_number}")

    def on_session_end(self, messages: list[Any] | None = None) -> None:
        self.events.append(f"end:{len(messages or [])}")

    def on_session_switch(self, new_session_id: str, **kwargs: Any) -> None:
        rewound = bool(kwargs.get("rewound"))
        self._session_id = new_session_id
        tag = "rewound" if rewound else "switch"
        self.events.append(f"{tag}:{new_session_id}")

    def on_pre_compress(self, messages: list[Any] | None = None) -> str:
        self.events.append("pre_compress")
        return "KEEP_FACT"

    def notify_memory_write(self, target: str = "memory") -> None:
        self.events.append(f"notify:{target}")

    def shutdown(self) -> None:
        self.events.append("shutdown")


def _mm_with_prov(tmp_path: Path) -> tuple[MemoryManager, _RecordingProv]:
    store = MemoryStore(memory_dir=tmp_path)
    store.load_from_disk()
    store.memory_entries = ["curated-note"]
    store.refresh_system_prompt_snapshot()
    mm = MemoryManager.create_default(curated=store)
    prov = _RecordingProv()
    mm.add_provider(prov)
    return mm, prov


def test_bind_and_system_block(tmp_path: Path) -> None:
    mm, prov = _mm_with_prov(tmp_path)
    bind_session(mm, session_id="s-bind", cwd=str(tmp_path))
    assert any(e.startswith("init:s-bind") for e in prov.events)
    block = build_system_memory_block(mm)
    assert "PROV_STATIC" in block or "curated" in block.lower() or "curated-note" in block
    # curated freeze should surface somehow
    assert block.strip()


def test_prefetch_fenced_wraps(tmp_path: Path) -> None:
    mm, _ = _mm_with_prov(tmp_path)
    bind_session(mm, session_id="s1", cwd=str(tmp_path))
    fenced = prefetch_fenced(mm, user_text="auth cookie", session_id="s1")
    assert fenced is not None
    assert fenced.startswith("<memory-context>")
    assert "recall:" in fenced or "auth" in fenced


def test_turn_begin_end_order(tmp_path: Path) -> None:
    mm, prov = _mm_with_prov(tmp_path)
    bind_session(mm, session_id="s1")
    on_turn_begin(mm, mm.curated_store, turn_number=3, user_text="hi")
    on_turn_end(
        mm,
        user_text="hi",
        assistant_text="yo",
        session_id="s1",
        messages=[{"role": "user", "content": "hi"}],
    )
    # sync_all is background — drain worker (warm owned by sync_turn, not
    # a second queue_prefetch_all that would overwrite blended warm)
    assert mm.flush_pending(timeout=3.0) is True
    assert "turn_start:3" in prov.events
    assert "sync" in prov.events


def test_sample_messages_ephemeral_only() -> None:
    msgs = [
        Message(role=Role.SYSTEM, content="sys"),
        Message(role=Role.USER, content="do it"),
    ]
    fence = build_memory_context_block("hit")
    sample = sample_messages_with_memory(msgs, fence)
    assert msgs[1].content == "do it"
    assert sample[1].content is not None
    assert "<memory-context>" in sample[1].content
    assert sample_messages_with_memory(msgs, None)[1].content == "do it"


def test_pre_compress_via_seam(tmp_path: Path) -> None:
    mm, prov = _mm_with_prov(tmp_path)
    out = on_pre_compress(mm, [Message(role=Role.USER, content="x")])
    assert "KEEP_FACT" in out
    assert "pre_compress" in prov.events


def test_transcript_rewound(tmp_path: Path) -> None:
    mm, prov = _mm_with_prov(tmp_path)
    bind_session(mm, session_id="same")
    on_transcript_rewound(mm, session_id="same")
    assert any(e.startswith("rewound:same") for e in prov.events)


def test_session_rewind_notifies_rewound_once(tmp_path: Path) -> None:
    """Session.rewind via runner must not double-fire on_transcript_rewound."""
    from codedoggy.context.compactor import ContextCompactor
    from codedoggy.context.segments import write_segment
    from codedoggy.session.extensions import SessionExtensions
    from codedoggy.session.session import Session
    from codedoggy.turn.runner import AgentTurnRunner
    from codedoggy.tools import ToolRegistryBuilder

    mid = [
        Message(role=Role.USER, content="recover me auth JWT"),
        Message(role=Role.ASSISTANT, content="jwt 15m"),
    ]
    path = write_segment(mid, home=tmp_path, note="ckpt")
    mm, prov = _mm_with_prov(tmp_path)
    bind_session(mm, session_id="rw-once")

    class _Sampler:
        def sample(self, messages, tools):
            from codedoggy.turn.types import SampleResult

            return SampleResult(content="ok")

    runner = AgentTurnRunner(
        sampler=_Sampler(),
        tools=ToolRegistryBuilder.new().finalize(),
        context_compactor=ContextCompactor.from_env(),
    )
    runner.live_messages = [
        Message(role=Role.SYSTEM, content="sys"),
        Message(role=Role.USER, content="latest"),
    ]
    runner.context_compactor.last_checkpoint_path = str(path)  # type: ignore[union-attr]
    s = Session.create(
        tmp_path,
        session_id="rw-once",
        extensions=SessionExtensions(
            turn_runner=runner,
            memory_manager=mm,
            context=runner.context_compactor,
        ),
    )
    try:
        r = s.rewind_context()
        assert r.get("ok") is True
        rewound = [e for e in prov.events if e.startswith("rewound:")]
        assert rewound == ["rewound:rw-once"], f"expected single rewound, got {rewound}"
    finally:
        s.close()


def test_legacy_session_close_uses_seam(tmp_path: Path) -> None:
    """No-kernel Session.close must still run on_session_end via seam."""
    from codedoggy.session.extensions import SessionExtensions
    from codedoggy.session.session import Session

    mm, prov = _mm_with_prov(tmp_path)
    bind_session(mm, session_id="legacy-close")

    class _Live:
        live_messages = [Message(role=Role.USER, content="last turn")]

        def clear_live_history(self) -> None:
            self.live_messages = []

    s = Session.create(
        tmp_path,
        extensions=SessionExtensions(
            turn_runner=_Live(),
            memory_manager=mm,
        ),
    )
    # No kernel bound
    assert s._kernel is None
    s.close()
    assert any(e.startswith("end:") for e in prov.events)


def test_commit_session_boundary_end_then_switch(tmp_path: Path) -> None:
    mm, prov = _mm_with_prov(tmp_path)
    bind_session(mm, session_id="old")
    commit_session_boundary(
        mm,
        [{"role": "user", "content": "bye"}],
        new_session_id="new",
        parent_session_id="old",
        reason="new_session",
    )
    assert mm.flush_pending(timeout=3.0) is True
    # end before switch
    end_i = next(i for i, e in enumerate(prov.events) if e.startswith("end:"))
    switch_i = next(i for i, e in enumerate(prov.events) if e.startswith("switch:new"))
    assert end_i < switch_i


def test_kernel_new_session_uses_seam(tmp_path: Path) -> None:
    mm, prov = _mm_with_prov(tmp_path)
    bind_session(mm, session_id="k-old")
    k = RuntimeKernel(cwd=tmp_path, session_id="k-old", memory_manager=mm)
    new_id = k.new_session(title="t")
    assert new_id != "k-old"
    assert k.session_id == new_id
    assert mm.flush_pending(timeout=3.0) is True
    assert any(e.startswith("switch:") for e in prov.events)


def test_session_close_flush_shutdown(tmp_path: Path) -> None:
    mm, prov = _mm_with_prov(tmp_path)
    bind_session(mm, session_id="close-me")
    on_session_close(mm, messages=[{"role": "user", "content": "last"}], timeout_s=2.0)
    assert any(e.startswith("end:") for e in prov.events)
    # shutdown_all should have been attempted (provider may not implement all hooks)
    # curated path still safe with None manager
    on_session_close(None)


def test_notify_curated_write_null_safe() -> None:
    notify_curated_write(None)
    bind_session(None, session_id="x")
    assert build_system_memory_block(None) == ""
    assert prefetch_fenced(None, user_text="q") is None
    assert on_pre_compress(None) == ""


def test_full_lifecycle_sequence(tmp_path: Path) -> None:
    """init → turn → sample inject → pre_compress → rewound → boundary → close."""
    mm, prov = _mm_with_prov(tmp_path)
    bind_session(mm, session_id="life", cwd=str(tmp_path))
    sys_block = build_system_memory_block(mm)
    assert sys_block
    fence = prefetch_fenced(mm, user_text="question about life", session_id="life")
    on_turn_begin(mm, mm.curated_store, turn_number=1, user_text="question about life")
    live = [
        Message(role=Role.SYSTEM, content=sys_block),
        Message(role=Role.USER, content="question about life"),
    ]
    sample = sample_messages_with_memory(live, fence)
    assert fence is None or "<memory-context>" in (sample[1].content or "")
    on_turn_end(
        mm,
        user_text="question about life",
        assistant_text="answer",
        session_id="life",
        messages=live,
    )
    mm.flush_pending(timeout=3.0)
    assert on_pre_compress(mm, live)
    on_transcript_rewound(mm, session_id="life")
    commit_session_boundary(
        mm, live, new_session_id="life-2", parent_session_id="life"
    )
    mm.flush_pending(timeout=3.0)
    on_session_close(mm, messages=live, timeout_s=2.0)

    # Required lifecycle markers present
    joined = " ".join(prov.events)
    assert "init:life" in joined
    assert "turn_start:1" in joined
    assert "sync" in joined
    assert "pre_compress" in joined
    assert "rewound:life" in joined
    assert "switch:life-2" in joined
    assert "end:" in joined
