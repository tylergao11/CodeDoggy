"""Submit + projection regression tests for tui_v2.

Covers pure project/scrollback helpers and a mock-light GrokShellApp submit
path that records ``handle_prompt`` without needing a real TTY session.
Existing paint coverage lives in ``test_tui_v2_port.py`` and is left intact.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from codedoggy.session.types import SessionPhase, TurnResult, TurnStatus
from codedoggy.tui_v2.glyphs import accent_bar
from codedoggy.tui_v2.project import ScrollItem, project_message
from codedoggy.tui_v2.scrollback import (
    ScrollbackState,
    reconcile_turn_finish,
    render_scrollback,
    seed_scrollback,
)


def _flat_scroll(state: ScrollbackState, *, width: int = 56, height: int = 16) -> str:
    return "".join(t for _, t in render_scrollback(state, width=width, height=height))


def _id_factory() -> Any:
    n = {"i": 0}

    def factory(prefix: str) -> str:
        n["i"] += 1
        return f"{prefix}_{n['i']}"

    return factory


# ── project_message pure projection ─────────────────────────────────────────


def test_project_message_user_assistant_tool_call_and_result() -> None:
    tool_open: dict[str, ScrollItem] = {}
    factory = _id_factory()

    user = project_message(
        {"role": "user", "content": "read a.py"},
        tool_open=tool_open,
        id_factory=factory,
    )
    assert len(user) == 1
    assert user[0].kind == "user"
    assert user[0].text == "read a.py"

    asst = project_message(
        {
            "role": "assistant",
            "content": "opening",
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "read_file",
                    "arguments": {"target_file": "a.py"},
                }
            ],
        },
        tool_open=tool_open,
        id_factory=factory,
    )
    assert [i.kind for i in asst] == ["assistant", "tool"]
    assert asst[0].text == "opening"
    assert asst[0].status == "running"  # has tool_calls
    tool = asst[1]
    assert tool.tool_name == "read_file"
    assert tool.tool_args.get("target_file") == "a.py"
    assert tool.status == "running"
    assert tool_open["c1"] is tool

    # Matching tool result mutates open call in place; returns no new items.
    result_items = project_message(
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "read_file",
            "content": "print(1)",
        },
        tool_open=tool_open,
        id_factory=factory,
    )
    assert result_items == []
    assert "c1" not in tool_open
    assert tool.tool_result == "print(1)"
    assert tool.status == "completed"


def test_project_message_orphan_tool_result_and_failed() -> None:
    tool_open: dict[str, ScrollItem] = {}
    factory = _id_factory()
    items = project_message(
        {
            "role": "tool",
            "tool_call_id": "missing",
            "name": "read_file",
            "content": "Error: not found",
        },
        tool_open=tool_open,
        id_factory=factory,
    )
    assert len(items) == 1
    assert items[0].kind == "tool"
    assert items[0].status == "failed"
    assert items[0].tool_name == "read_file"


def test_project_message_thinking_from_reasoning() -> None:
    tool_open: dict[str, ScrollItem] = {}
    items = project_message(
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "let me think",
        },
        tool_open=tool_open,
        id_factory=_id_factory(),
    )
    kinds = [i.kind for i in items]
    assert kinds == ["thinking", "assistant"]
    assert items[0].text == "let me think"
    assert items[0].collapsed is True


# ── append_message paints Read (submit/projection regression) ───────────────


def test_append_message_user_assistant_tool_paints_read() -> None:
    """User + assistant tool_call + tool result still paint Read/path chrome."""
    s = ScrollbackState()
    s.append_message({"role": "user", "content": "hello chrome"})
    s.append_message(
        {
            "role": "assistant",
            "content": "done",
            "tool_calls": [
                {
                    "id": "1",
                    "name": "read_file",
                    "arguments": {"target_file": "a.py"},
                }
            ],
        }
    )
    s.append_message(
        {
            "role": "tool",
            "tool_call_id": "1",
            "name": "read_file",
            "content": "x = 1",
        }
    )
    text = _flat_scroll(s, width=48, height=12)
    assert accent_bar() in text
    assert "hello chrome" in text
    assert "done" in text
    assert "Read" in text
    assert "a.py" in text
    assert "✦" not in text
    assert "💭" not in text


# ── seed_scrollback ──────────────────────────────────────────────────────────


def test_seed_scrollback_projects_full_transcript() -> None:
    messages = [
        {"role": "user", "content": "seed me"},
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [
                {
                    "id": "t1",
                    "name": "read_file",
                    "arguments": {"target_file": "seed.py"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "t1",
            "name": "read_file",
            "content": "seeded",
        },
    ]
    s = ScrollbackState()
    # Pre-existing junk must be cleared.
    s.append_message({"role": "user", "content": "stale"})
    seed_scrollback(s, messages)

    kinds = [i.kind for i in s.items]
    assert kinds == ["user", "assistant", "tool"]
    assert s.items[0].text == "seed me"
    assert s.items[2].tool_name == "read_file"
    assert s.items[2].tool_result == "seeded"
    assert s.items[2].status == "completed"
    assert s.tool_open == {}

    text = _flat_scroll(s)
    assert "seed me" in text
    assert "Read" in text
    assert "seed.py" in text


def test_seed_scrollback_method_and_empty() -> None:
    s = ScrollbackState()
    s.append_message({"role": "user", "content": "x"})
    s.seed_from_messages([])
    assert s.items == []
    assert s.selected == -1

    seed_scrollback(s, None)
    assert s.items == []


# ── finish reconciliation ────────────────────────────────────────────────────


def test_reconcile_turn_finish_completes_running_tools_and_clears_draft() -> None:
    s = ScrollbackState()
    s.append_message({"role": "user", "content": "go"})
    s.append_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "orphan",
                    "name": "read_file",
                    "arguments": {"target_file": "z.py"},
                }
            ],
        }
    )
    # No matching tool result — tool stays running until finish.
    running = [i for i in s.items if i.kind == "tool"]
    assert len(running) == 1
    assert running[0].status == "running"
    s.set_draft("partial stream…")

    reconcile_turn_finish(s, result_status=TurnStatus.COMPLETED)

    assert not any(i.meta.get("draft") for i in s.items)
    tools = [i for i in s.items if i.kind == "tool"]
    assert tools and all(t.status == "completed" for t in tools)
    assert not any(i.kind == "system" for i in s.items)


def test_reconcile_turn_finish_appends_final_text_when_assistant_missing() -> None:
    s = ScrollbackState()
    s.append_message({"role": "user", "content": "hi"})
    s.set_draft("half…")
    turn_start = 0

    reconcile_turn_finish(
        s,
        result_status=TurnStatus.COMPLETED,
        final_text="Hello from the model",
        turn_scroll_start=turn_start,
    )

    assert not any(i.meta.get("draft") for i in s.items)
    assistants = [i for i in s.items if i.kind == "assistant"]
    assert len(assistants) == 1
    assert assistants[0].text == "Hello from the model"
    assert assistants[0].status == "done"


def test_reconcile_turn_finish_does_not_duplicate_existing_assistant() -> None:
    s = ScrollbackState()
    s.append_message({"role": "user", "content": "hi"})
    s.append_message({"role": "assistant", "content": "already there"})
    before = len(s.items)

    reconcile_turn_finish(
        s,
        result_status=TurnStatus.COMPLETED,
        final_text="already there",
        turn_scroll_start=0,
    )
    assert len(s.items) == before
    assert sum(1 for i in s.items if i.kind == "assistant") == 1


def test_reconcile_turn_finish_projects_missing_live_messages() -> None:
    s = ScrollbackState()
    s.append_message({"role": "user", "content": "use tools"})
    # Live transcript has assistant+tool+result that never hit on_live_message.
    live = [
        {"role": "user", "content": "use tools"},
        {
            "role": "assistant",
            "content": "opening",
            "tool_calls": [
                {
                    "id": "c9",
                    "name": "read_file",
                    "arguments": {"target_file": "z.py"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c9",
            "name": "read_file",
            "content": "body",
        },
        {"role": "assistant", "content": "done reading"},
    ]
    reconcile_turn_finish(
        s,
        result_status=TurnStatus.COMPLETED,
        final_text="done reading",
        live_messages=live,
        turn_scroll_start=0,
    )
    kinds = [i.kind for i in s.items]
    # user (once) + assistant + tool + assistant
    assert kinds.count("user") == 1
    assert "tool" in kinds
    tools = [i for i in s.items if i.kind == "tool"]
    assert tools[0].tool_result == "body"
    assert tools[0].status == "completed"
    assert any(
        i.kind == "assistant" and "done reading" in i.text for i in s.items
    )


def test_reconcile_turn_finish_appends_error_on_failed() -> None:
    s = ScrollbackState()
    s.append_message({"role": "user", "content": "boom"})
    reconcile_turn_finish(
        s,
        result_status="failed",
        result_error="sampler exploded",
    )
    systems = [i for i in s.items if i.kind == "system"]
    assert len(systems) == 1
    assert "sampler exploded" in systems[0].text

    # Empty error → no system row
    s2 = ScrollbackState()
    s2.append_message({"role": "user", "content": "x"})
    reconcile_turn_finish(s2, result_status=TurnStatus.ERROR, result_error="")
    assert not any(i.kind == "system" for i in s2.items)


# ── GrokShellApp submit → handle_prompt ──────────────────────────────────────


class _FakeSession:
    """Minimal session surface for GrokShellApp without real model auth."""

    def __init__(self) -> None:
        self.phase = SessionPhase.IDLE
        self.id = "test-session"
        self.cwd = Path.cwd()
        self.extensions = SimpleNamespace(turn_runner=None, kernel=None)
        self.handle_prompt_calls: list[dict[str, Any]] = []
        self.interject_calls: list[dict[str, Any]] = []

    def handle_prompt(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        attachments: Any = (),
    ) -> TurnResult:
        self.handle_prompt_calls.append(
            {
                "text": text,
                "metadata": dict(metadata or {}),
                "attachments": attachments,
            }
        )
        return TurnResult(status=TurnStatus.COMPLETED, final_text="ok")

    def interject(
        self, text: str, attachments: Any = ()
    ) -> None:  # pragma: no cover - shape only
        self.interject_calls.append({"text": text, "attachments": attachments})


def test_submit_calls_handle_prompt_when_ready() -> None:
    from codedoggy.tui_v2.app import GrokShellApp

    session = _FakeSession()
    with patch("codedoggy.tui.surface.ready_to_sample", return_value=True):
        app = GrokShellApp(session)
        app._submit("  please read foo  ")
        assert app._worker is not None
        app._worker.join(timeout=5.0)
        assert not app._worker.is_alive()

    assert len(session.handle_prompt_calls) == 1
    call = session.handle_prompt_calls[0]
    assert call["text"] == "please read foo"
    assert call["metadata"].get("stream_sample") is True
    assert call["metadata"].get("on_live_message") is not None
    assert call["metadata"].get("on_sample_delta") is not None
    # User row painted into scrollback immediately.
    assert any(i.kind == "user" and "please read foo" in i.text for i in app._scroll.items)
    # Finish path must surface final_text when no on_live assistant arrived.
    assert any(
        i.kind == "assistant" and i.text == "ok" for i in app._scroll.items
    )


def test_start_turn_finish_projects_on_live_and_skips_duplicate_final() -> None:
    from codedoggy.tui_v2.app import GrokShellApp

    class _LiveSession(_FakeSession):
        def handle_prompt(
            self,
            text: str,
            metadata: dict[str, Any] | None = None,
            attachments: Any = (),
        ) -> TurnResult:
            super().handle_prompt(text, metadata=metadata, attachments=attachments)
            on_live = (metadata or {}).get("on_live_message")
            on_delta = (metadata or {}).get("on_sample_delta")
            if callable(on_delta):
                on_delta("Hel")
                on_delta("lo world")
            if callable(on_live):
                on_live({"role": "assistant", "content": "Hello world"})
            return TurnResult(status=TurnStatus.COMPLETED, final_text="Hello world")

    session = _LiveSession()
    with patch("codedoggy.tui.surface.ready_to_sample", return_value=True):
        app = GrokShellApp(session)
        app._start_turn("say hi")
        assert app._worker is not None
        app._worker.join(timeout=5.0)

    assistants = [i for i in app._scroll.items if i.kind == "assistant"]
    assert len(assistants) == 1
    assert assistants[0].text == "Hello world"
    assert not any(i.meta.get("draft") for i in app._scroll.items)


def test_on_live_user_does_not_duplicate_optimistic_user_row() -> None:
    """Runner archives USER at loop start; UI already painted optimistic row.

    Regression: one prompt must paint exactly one user ScrollItem.
    """
    from codedoggy.tui_v2.app import GrokShellApp

    class _ArchiveUserSession(_FakeSession):
        def handle_prompt(
            self,
            text: str,
            metadata: dict[str, Any] | None = None,
            attachments: Any = (),
        ) -> TurnResult:
            super().handle_prompt(text, metadata=metadata, attachments=attachments)
            on_live = (metadata or {}).get("on_live_message")
            if callable(on_live):
                # Same text the optimistic row already shows (stripped chips).
                on_live({"role": "user", "content": text})
                on_live({"role": "assistant", "content": "one reply"})
            return TurnResult(status=TurnStatus.COMPLETED, final_text="one reply")

    session = _ArchiveUserSession()
    with patch("codedoggy.tui.surface.ready_to_sample", return_value=True):
        app = GrokShellApp(session)
        app._start_turn("hello once")
        assert app._worker is not None
        app._worker.join(timeout=5.0)

    users = [i for i in app._scroll.items if i.kind == "user"]
    assert len(users) == 1
    assert users[0].text == "hello once"
    assert users[0].meta.get("optimistic_user") is True
    assert sum(1 for i in app._scroll.items if i.kind == "assistant") == 1
    assert any(
        i.kind == "assistant" and i.text == "one reply" for i in app._scroll.items
    )


def test_on_live_framed_interject_skips_when_optimistic_interject_exists() -> None:
    """Interject paints plain user text; runner archives format_interjection frame.

    Live path must not paint a second user row for the framed content.
    """
    from codedoggy.tui_v2.app import GrokShellApp

    framed = (
        "The user sent a message while you were working:\n"
        "<user_query>\n"
        "stop that\n"
        "</user_query>"
    )

    # Mid-turn interject: UI already has plain optimistic interject row;
    # runner then fires on_live with the framed USER archive.
    class _MidTurnInterjectSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.app: Any = None

        def handle_prompt(
            self,
            text: str,
            metadata: dict[str, Any] | None = None,
            attachments: Any = (),
        ) -> TurnResult:
            super().handle_prompt(text, metadata=metadata, attachments=attachments)
            on_live = (metadata or {}).get("on_live_message")
            if not callable(on_live):
                return TurnResult(status=TurnStatus.COMPLETED, final_text="done")
            # Optimistic user for the original prompt (same text as strip).
            on_live({"role": "user", "content": text})
            # UI-side interject paint (what _submit does while running).
            assert self.app is not None
            self.app._scroll.items.append(
                ScrollItem(
                    kind="user",
                    id=self.app._scroll.new_id("user"),
                    text="stop that",
                    meta={"interject": True, "optimistic_user": True},
                )
            )
            # Runner archives framed interjection — must not double-paint.
            on_live({"role": "user", "content": framed})
            on_live({"role": "assistant", "content": "acknowledged"})
            return TurnResult(
                status=TurnStatus.COMPLETED, final_text="acknowledged"
            )

    session = _MidTurnInterjectSession()
    with patch("codedoggy.tui.surface.ready_to_sample", return_value=True):
        app = GrokShellApp(session)
        session.app = app
        app._start_turn("working on it")
        assert app._worker is not None
        app._worker.join(timeout=5.0)

    users = [i for i in app._scroll.items if i.kind == "user"]
    # Original prompt + plain interject only — framed archive skipped.
    assert len(users) == 2
    assert users[0].text == "working on it"
    assert users[1].text == "stop that"
    assert users[1].meta.get("interject") is True
    assert not any("while you were working" in (i.text or "") for i in users)
    assert not any("<user_query>" in (i.text or "") for i in users)

    # Unit-level guard: framed content skips when optimistic interject exists.
    assert app._should_skip_live_message(
        {"role": "user", "content": framed}, turn_scroll_start=0
    )
    assert app._should_skip_live_message(
        {"role": "user", "content": "stop that"}, turn_scroll_start=0
    )


def test_start_turn_finish_reconciles_from_runner_live_messages() -> None:
    from codedoggy.tui_v2.app import GrokShellApp

    class _RunnerSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self._live: list[Any] = []
            self.extensions = SimpleNamespace(
                turn_runner=SimpleNamespace(
                    live_messages=self._live, on_live_message=None
                ),
                kernel=None,
            )

        def handle_prompt(
            self,
            text: str,
            metadata: dict[str, Any] | None = None,
            attachments: Any = (),
        ) -> TurnResult:
            super().handle_prompt(text, metadata=metadata, attachments=attachments)
            # Simulate runner writing live_messages without calling on_live.
            self._live[:] = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "from live list"},
            ]
            # Keep runner.live_messages pointing at the same list object.
            self.extensions.turn_runner.live_messages = self._live
            return TurnResult(
                status=TurnStatus.COMPLETED, final_text="from live list"
            )

    session = _RunnerSession()
    with patch("codedoggy.tui.surface.ready_to_sample", return_value=True):
        app = GrokShellApp(session)
        app._start_turn("ping")
        assert app._worker is not None
        app._worker.join(timeout=5.0)

    assert any(
        i.kind == "assistant" and i.text == "from live list" for i in app._scroll.items
    )
    assert sum(1 for i in app._scroll.items if i.kind == "user") == 1


def test_live_messages_since_empty_when_start_equals_len() -> None:
    """Regression: start == len(live) must not re-project seeded history.

    After handle_prompt with no new live rows, ``_live_messages_since`` used
    to return the full list (older ``start >= n`` branch), which re-seeded
    history and duplicated user/assistant rows on every turn.
    """
    from codedoggy.tui_v2.app import GrokShellApp

    history = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
    ]

    class _HistorySession(_FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self._live: list[Any] = list(history)
            self.extensions = SimpleNamespace(
                turn_runner=SimpleNamespace(
                    live_messages=self._live, on_live_message=None
                ),
                kernel=None,
            )

    session = _HistorySession()
    with patch("codedoggy.tui.surface.ready_to_sample", return_value=True):
        app = GrokShellApp(session)
        # __init__ seeds scroll from runner live_messages.
        before_kinds = [i.kind for i in app._scroll.items]
        assert before_kinds == ["user", "assistant"]
        n = len(history)
        assert len(app._live_messages_snapshot()) == n

        # Unchanged live list after handle_prompt: nothing new.
        assert app._live_messages_since(n) == []
        # start past end (shorter replacement) still returns full list.
        assert app._live_messages_since(n + 1) == list(history)
        # Mid-slice still works.
        assert app._live_messages_since(1) == [history[1]]

        # Reconcile with empty live_messages + final_text must not duplicate
        # the already-seeded user/assistant rows.
        turn_start = len(app._scroll.items)
        app._scroll.items.append(
            ScrollItem(
                kind="user",
                id=app._scroll.new_id("user"),
                text="new turn",
            )
        )
        reconcile_turn_finish(
            app._scroll,
            result_status=TurnStatus.COMPLETED,
            final_text="new answer",
            live_messages=[],
            turn_scroll_start=turn_start,
        )
        kinds = [i.kind for i in app._scroll.items]
        assert kinds.count("user") == 2  # seed + new turn only
        assert kinds.count("assistant") == 2  # seed + final_text only
        assert sum(1 for i in app._scroll.items if i.text == "old user") == 1
        assert sum(1 for i in app._scroll.items if i.text == "old assistant") == 1
        assert any(
            i.kind == "assistant" and i.text == "new answer" for i in app._scroll.items
        )


def test_on_accept_strips_and_submits_when_ready() -> None:
    from codedoggy.tui_v2.app import GrokShellApp

    session = _FakeSession()
    with patch("codedoggy.tui.surface.ready_to_sample", return_value=True):
        app = GrokShellApp(session)
        # Bypass Application accept path: exercise buffer-shaped accept handler.
        buf = SimpleNamespace(text="  validate path  ")
        assert app._on_accept(buf) is True
        assert app._worker is not None
        app._worker.join(timeout=5.0)

    assert len(session.handle_prompt_calls) == 1
    assert session.handle_prompt_calls[0]["text"] == "validate path"


def test_on_accept_empty_does_not_submit() -> None:
    from codedoggy.tui_v2.app import GrokShellApp

    session = _FakeSession()
    with patch("codedoggy.tui.surface.ready_to_sample", return_value=True):
        app = GrokShellApp(session)
        assert app._on_accept(SimpleNamespace(text="   ")) is True
        assert app._worker is None
    assert session.handle_prompt_calls == []


def test_submit_defers_to_auth_when_not_ready() -> None:
    from codedoggy.tui_v2.app import GrokShellApp

    session = _FakeSession()
    with patch("codedoggy.tui.surface.ready_to_sample", return_value=False):
        app = GrokShellApp(session)
        app._submit("need login")
        assert app._worker is None
        assert app._pending_prompt == "need login"
        assert app._auth_open is True
    assert session.handle_prompt_calls == []
