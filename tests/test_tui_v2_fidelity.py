"""Residual paint / fold fidelity for tui_v2 (bg_task, session_event, fold, md).

Complements ``test_tui_v2_port.py`` (core port) and ``test_tui_v2_submit.py``
(product path). Keep these assertions tight to Grok string fidelity.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from codedoggy.session.types import SessionPhase
from codedoggy.tools.task_manager import BackgroundTaskManager, TaskSnapshot
from codedoggy.tui_v2.blocks.bg_task import paint_bg_task
from codedoggy.tui_v2.blocks.markdown import render_markdown
from codedoggy.tui_v2.blocks.session_event import paint_session_event
from codedoggy.tui_v2.blocks.tool import paint_tool
from codedoggy.tui_v2.blocks.tool.common import ELLIPSIS, HEADER_READ
from codedoggy.tui_v2.project import ScrollItem
from codedoggy.tui_v2.text_selection import (
    TextSel,
    reconstruct_selection_text,
    strip_quote_bar_prefix,
)
from codedoggy.tui_v2.verb_group import classify_verb


def _flat(rows: list) -> str:
    return "".join(t for row in rows for _, t in row)


# ── 1. ScrollItem.cycle_fold ─────────────────────────────────────────────────


def test_cycle_fold_tool_collapsed_truncated_expanded() -> None:
    """Tool: collapsed → truncated → expanded → collapsed."""
    item = ScrollItem(
        kind="tool",
        id="t1",
        tool_name="read_file",
        tool_args={"target_file": "a.py"},
        tool_result="body",
        collapsed=True,
        truncated=False,
    )
    assert item.collapsed is True and item.truncated is False
    assert item.tool_display_collapsed() is True
    assert item.tool_display_truncated() is False

    item.cycle_fold()  # open truncated first
    assert item.collapsed is False and item.truncated is True
    assert item.tool_display_collapsed() is False
    assert item.tool_display_truncated() is True

    item.cycle_fold()  # full expanded
    assert item.collapsed is False and item.truncated is False
    assert item.tool_display_truncated() is False

    item.cycle_fold()  # back to collapsed
    assert item.collapsed is True and item.truncated is False


def test_cycle_fold_non_tool_is_binary() -> None:
    item = ScrollItem(kind="thinking", id="th1", text="hmm", collapsed=True)
    item.cycle_fold()
    assert item.collapsed is False
    item.cycle_fold()
    assert item.collapsed is True


# ── 2. paint_tool truncated vs full body ─────────────────────────────────────


def test_paint_tool_truncated_vs_full_body() -> None:
    """truncated=True caps body; truncated=False shows every line."""
    body = "\n".join(f"line_{i}" for i in range(1, 21))
    full = paint_tool(
        "read_file",
        {"target_file": "big.py"},
        body,
        width=60,
        collapsed=False,
        status="completed",
        truncated=False,
    )
    full_text = _flat(full)
    assert HEADER_READ in full_text or "Read " in full_text
    for i in range(1, 21):
        assert f"line_{i}" in full_text
    assert ELLIPSIS not in full_text

    short = paint_tool(
        "read_file",
        {"target_file": "big.py"},
        body,
        width=60,
        collapsed=False,
        status="completed",
        truncated=True,
    )
    short_text = _flat(short)
    assert "line_1" in short_text and "line_5" in short_text
    assert "line_18" in short_text and "line_20" in short_text
    assert "line_10" not in short_text
    assert ELLIPSIS in short_text


# ── 3. bg_task paint strings ─────────────────────────────────────────────────


def test_bg_task_paint_task_started_and_completed() -> None:
    started = _flat(paint_bg_task("compile assets", width=48, status="running"))
    assert "Task started" in started or (
        "Task " in started and "started:" in started
    )
    assert "compile assets" in started

    done = _flat(
        paint_bg_task(
            "compile assets", width=48, status="completed", elapsed_ms=1200
        )
    )
    assert "Task " in done
    assert "completed" in done
    assert "1.2s" in done

    # Host ScrollItem path
    running = ScrollItem(
        kind="bg_task", id="bg1", text="compile assets", status="running"
    )
    rtext = _flat(running.paint(width=48))
    assert "Task " in rtext and "started:" in rtext

    finished = ScrollItem(
        kind="bg_task",
        id="bg2",
        text="compile assets",
        status="completed",
        elapsed_ms=500,
    )
    ftext = _flat(finished.paint(width=48))
    assert "completed" in ftext and "0.5s" in ftext


# ── 4. session_event paint ───────────────────────────────────────────────────


def test_session_event_paint_worked_for_and_turn_failed() -> None:
    worked = _flat(paint_session_event("worked", detail="3.4s", width=40))
    assert "Worked for" in worked
    assert "3.4s" in worked

    failed = _flat(
        paint_session_event("turn_failed", detail="timeout", width=40)
    )
    assert "Turn failed" in failed
    assert "timeout" in failed

    bare = _flat(paint_session_event("turn_failed", width=40))
    assert "Turn failed" in bare

    item = ScrollItem(
        kind="session_event",
        id="se1",
        text="2.1s",
        meta={"event": "turn_completed"},
    )
    itext = _flat(item.paint(width=40))
    assert "Worked for" in itext and "2.1s" in itext

    item_f = ScrollItem(
        kind="session_event",
        id="se2",
        text="sampler down",
        meta={"event": "turn_failed"},
    )
    ftext = _flat(item_f.paint(width=48))
    assert "Turn failed" in ftext and "sampler down" in ftext


# ── 5. markdown HR ━━━ and + list marker ─────────────────────────────────────


def test_markdown_hr_heavy_bars_and_plus_list_marker() -> None:
    """HR → U+2501×3 (━━━); ``+`` stays ``+`` (``-``/``*`` become •)."""
    hr = _flat(render_markdown("---", width=40))
    assert "\u2501" * 3 in hr  # ━━━

    hr2 = _flat(render_markdown("***", width=40))
    assert "\u2501" * 3 in hr2

    plus = _flat(render_markdown("+ first item\n+ second", width=40))
    assert "+ " in plus
    assert "first item" in plus
    assert "second" in plus

    dash = _flat(render_markdown("- dashed item", width=40))
    assert "• " in dash
    assert "dashed item" in dash


# ── 6. verb_group execute still None ─────────────────────────────────────────


def test_verb_group_execute_still_not_groupable() -> None:
    assert classify_verb("execute") is None
    assert classify_verb("run_terminal_cmd") is None
    assert classify_verb("run_terminal_command") is None
    assert classify_verb("bash") is None
    assert classify_verb("shell") is None
    # groupable control
    assert classify_verb("read_file") == "file"


# ── 7. quote-bar strip on copy ───────────────────────────────────────────────


def test_strip_quote_bar_prefix_strips_nested_bars() -> None:
    """Nested ``│ `` / bare ``│`` peel fully; mid-line bars stay."""
    bar = "\u2502"  # │
    assert strip_quote_bar_prefix(f"{bar} deep") == "deep"
    assert strip_quote_bar_prefix(f"{bar} {bar} deep") == "deep"
    # Bare stacked bars then space: ││ nested → peel bare │ → │ nested → peel │  → nested
    assert strip_quote_bar_prefix(f"{bar}{bar} nested") == "nested"
    assert strip_quote_bar_prefix(f"{bar} {bar} {bar} x") == "x"
    # Mid-line quote bar is not a prefix — left intact
    assert strip_quote_bar_prefix(f"keep {bar} mid") == f"keep {bar} mid"
    assert strip_quote_bar_prefix("plain") == "plain"
    assert strip_quote_bar_prefix("") == ""


def test_reconstruct_selection_text_strips_quote_bars() -> None:
    """Copy path peels leading quote-bar chrome from each selected line."""
    bar = "\u2502"
    rows = [
        [("", f"{bar} quoted line")],
        [("", f"{bar} {bar} nested")],
        [("", "plain line")],
    ]
    # Full first line
    assert reconstruct_selection_text(rows, TextSel(0, 0, 0, 100)) == "quoted line"
    # Nested second line
    assert reconstruct_selection_text(rows, TextSel(1, 0, 1, 100)) == "nested"
    # Multi-line selection strips each line independently
    text = reconstruct_selection_text(rows, TextSel(0, 0, 2, 100))
    assert text == "quoted line\nnested\nplain line"


# ── 8. session_event model_unavailable / max_turns ───────────────────────────


def test_session_event_model_unavailable_and_max_turns_labels() -> None:
    bare = _flat(paint_session_event("model_unavailable", width=48))
    assert "Model unavailable" in bare

    with_detail = _flat(
        paint_session_event(
            "model_unavailable", detail="rate limited", width=48
        )
    )
    assert "Model unavailable" in with_detail
    assert "rate limited" in with_detail

    max_t = _flat(paint_session_event("max_turns", width=40))
    assert "Max turns reached" in max_t

    # ScrollItem host path (meta.event)
    item_mu = ScrollItem(
        kind="session_event",
        id="se_mu",
        text="quota",
        meta={"event": "model_unavailable"},
    )
    assert "Model unavailable" in _flat(item_mu.paint(width=48))
    assert "quota" in _flat(item_mu.paint(width=48))

    item_mt = ScrollItem(
        kind="session_event",
        id="se_mt",
        text="",
        meta={"event": "max_turns"},
    )
    assert "Max turns reached" in _flat(item_mt.paint(width=40))


# ── 9. bg_task failed exit_code → "exit N" ───────────────────────────────────


def test_bg_task_failed_paints_exit_code() -> None:
    """failed + exit_code paints muted `` exit N`` (Grok bg_task.rs)."""
    failed = _flat(
        paint_bg_task(
            "build",
            width=56,
            status="failed",
            elapsed_ms=500,
            exit_code=1,
        )
    )
    assert "Task " in failed
    assert "failed" in failed
    assert "exit 1" in failed
    assert "build" in failed

    no_code = _flat(
        paint_bg_task(
            "build",
            width=56,
            status="failed",
            elapsed_ms=500,
            exit_code=None,
        )
    )
    assert "failed" in no_code
    assert "exit " not in no_code

    # Host ScrollItem path via meta.exit_code
    item = ScrollItem(
        kind="bg_task",
        id="bg_fail",
        text="compile",
        status="failed",
        elapsed_ms=1200,
        meta={"exit_code": 2},
    )
    itext = _flat(item.paint(width=56))
    assert "failed" in itext and "exit 2" in itext


# ── 10. BackgroundTaskManager listener + notify ──────────────────────────────


def test_background_task_manager_listener_fires_started() -> None:
    """add_listener receives events from _notify (stable; no process spawn)."""
    mgr = BackgroundTaskManager()
    events: list[tuple[str, str]] = []

    def listener(event: str, snap: TaskSnapshot) -> None:
        events.append((event, snap.task_id))

    mgr.add_listener(listener)
    snap = TaskSnapshot(
        task_id="task_unit1",
        command="echo hi",
        cwd=str(Path.cwd()),
        start_time=0.0,
        description="unit",
    )
    mgr._notify("started", snap)
    assert events == [("started", "task_unit1")]

    # Second notify still delivers
    mgr._notify("failed", snap)
    assert events == [("started", "task_unit1"), ("failed", "task_unit1")]

    mgr.remove_listener(listener)
    mgr._notify("started", snap)
    assert len(events) == 2  # no further delivery after remove


# ── 11. App _apply_bg_task_snap upserts by task_id ───────────────────────────


def test_apply_bg_task_snap_upserts_by_task_id() -> None:
    """Same task_id updates one row; new id appends another."""
    from codedoggy.tui_v2.app import GrokShellApp

    session = SimpleNamespace(
        phase=SessionPhase.IDLE,
        id="test-bg",
        cwd=Path.cwd(),
        extensions=SimpleNamespace(turn_runner=None, kernel=None),
        handle_prompt=lambda *a, **k: None,
    )
    app = GrokShellApp(session)
    app._request_redraw = MagicMock()  # type: ignore[method-assign]

    snap_run = TaskSnapshot(
        task_id="task_abc",
        command="python -c pass",
        cwd=str(Path.cwd()),
        start_time=1.0,
        description="first desc",
        completed=False,
    )
    app._apply_bg_task_snap(snap_run)
    bg = [i for i in app._scroll.items if i.kind == "bg_task"]
    assert len(bg) == 1
    assert bg[0].text == "first desc"
    assert bg[0].status == "running"
    assert bg[0].meta.get("task_id") == "task_abc"

    snap_fail = TaskSnapshot(
        task_id="task_abc",
        command="python -c pass",
        cwd=str(Path.cwd()),
        start_time=1.0,
        end_time=2.5,
        description="first desc",
        completed=True,
        exit_code=7,
    )
    app._apply_bg_task_snap(snap_fail)
    bg2 = [i for i in app._scroll.items if i.kind == "bg_task"]
    assert len(bg2) == 1  # upsert, not duplicate
    assert bg2[0].status == "failed"
    assert bg2[0].meta.get("exit_code") == 7
    assert bg2[0].elapsed_ms is not None and bg2[0].elapsed_ms >= 1000

    snap_other = TaskSnapshot(
        task_id="task_other",
        command="true",
        cwd=str(Path.cwd()),
        start_time=3.0,
        description="other",
        completed=False,
    )
    app._apply_bg_task_snap(snap_other)
    bg3 = [i for i in app._scroll.items if i.kind == "bg_task"]
    assert len(bg3) == 2
    ids = {i.meta.get("task_id") for i in bg3}
    assert ids == {"task_abc", "task_other"}


# ── 12. paint_bg_task collapsed header only; expanded shows output ───────────


def test_paint_bg_task_collapsed_header_only_expanded_shows_output() -> None:
    """collapsed=True → lifecycle header only; expanded → output body lines."""
    output = "stdout_line_a\nstdout_line_b\nstdout_line_c"
    collapsed = paint_bg_task(
        "npm build",
        width=48,
        status="running",
        collapsed=True,
        output=output,
    )
    ctext = _flat(collapsed)
    assert "Task " in ctext and "started:" in ctext
    assert "npm build" in ctext
    assert "stdout_line_a" not in ctext
    assert "stdout_line_b" not in ctext
    # Header is a single content row (no blank separator / body)
    assert len(collapsed) == 1

    expanded = paint_bg_task(
        "npm build",
        width=48,
        status="completed",
        elapsed_ms=900,
        collapsed=False,
        truncated=False,
        output=output,
    )
    etext = _flat(expanded)
    assert "Task " in etext and "completed" in etext
    assert "stdout_line_a" in etext
    assert "stdout_line_b" in etext
    assert "stdout_line_c" in etext
    assert len(expanded) > 1  # header + separator + body

    # Empty expanded output → muted "(no output)"
    empty = _flat(
        paint_bg_task(
            "quiet",
            width=40,
            status="completed",
            collapsed=False,
            output="",
        )
    )
    assert "(no output)" in empty


# ── 13. paint_bg_task truncated mid-ellipsis for long output ─────────────────


def test_paint_bg_task_truncated_mid_ellipsis_long_output() -> None:
    """truncated=True keeps first 5 + last 3 with mid ellipsis (+N lines)."""
    body = "\n".join(f"bg_line_{i}" for i in range(1, 16))  # 15 lines > 5+3
    rows = paint_bg_task(
        "long job",
        width=60,
        status="completed",
        elapsed_ms=1000,
        collapsed=False,
        truncated=True,
        output=body,
    )
    text = _flat(rows)
    assert "Task " in text and "completed" in text
    # first 5 kept
    for i in range(1, 6):
        assert f"bg_line_{i}" in text
    # last 3 kept
    for i in range(13, 16):
        assert f"bg_line_{i}" in text
    # middle dropped
    assert "bg_line_8" not in text
    assert "bg_line_10" not in text
    # mid ellipsis with hidden count (body_lines_with_ellipsis show_hidden_count)
    assert ELLIPSIS in text
    assert "lines" in text

    # truncated=False shows middle lines (no mid ellipsis for 15-line body)
    full = _flat(
        paint_bg_task(
            "long job",
            width=60,
            status="completed",
            collapsed=False,
            truncated=False,
            output=body,
        )
    )
    assert "bg_line_8" in full and "bg_line_10" in full


# ── 14. ScrollItem bg_task cycle_fold + paint expanded with tool_result ──────


def test_scrollitem_bg_task_cycle_fold_and_paint_expanded() -> None:
    """bg_task three-state fold; expanded paint surfaces tool_result stdout."""
    item = ScrollItem(
        kind="bg_task",
        id="bg_fold",
        text="watch files",
        status="running",
        tool_result="out_alpha\nout_beta",
    )
    # __post_init__ forces collapsed for bg_task
    assert item.collapsed is True
    assert item.truncated is False

    collapsed_text = _flat(item.paint(width=48))
    assert "Task " in collapsed_text and "started:" in collapsed_text
    assert "out_alpha" not in collapsed_text

    item.cycle_fold()  # → truncated (open first)
    assert item.collapsed is False and item.truncated is True
    trunc_text = _flat(item.paint(width=48))
    assert "out_alpha" in trunc_text
    assert "out_beta" in trunc_text

    item.cycle_fold()  # → full expanded
    assert item.collapsed is False and item.truncated is False
    exp_text = _flat(item.paint(width=48))
    assert "out_alpha" in exp_text and "out_beta" in exp_text

    item.cycle_fold()  # → collapsed again
    assert item.collapsed is True and item.truncated is False
    assert "out_alpha" not in _flat(item.paint(width=48))


# ── 15. session_event turn_queued / compact_failed with detail ───────────────


def test_session_event_turn_queued_and_compact_failed() -> None:
    queued = _flat(paint_session_event("turn_queued", width=40))
    assert "Turn queued" in queued

    bare_fail = _flat(paint_session_event("compact_failed", width=48))
    assert "Compaction failed" in bare_fail

    with_detail = _flat(
        paint_session_event(
            "compact_failed", detail="token budget exceeded", width=56
        )
    )
    assert "Compaction failed" in with_detail
    assert "token budget exceeded" in with_detail

    # Host ScrollItem path
    item_q = ScrollItem(
        kind="session_event",
        id="se_q",
        text="",
        meta={"event": "turn_queued"},
    )
    assert "Turn queued" in _flat(item_q.paint(width=40))

    item_cf = ScrollItem(
        kind="session_event",
        id="se_cf",
        text="disk full",
        meta={"event": "compact_failed"},
    )
    ctext = _flat(item_cf.paint(width=48))
    assert "Compaction failed" in ctext and "disk full" in ctext


# ── 16. App._bg_task_output_from_snap (memory + temp file) ───────────────────


def test_bg_task_output_from_snap_memory_and_temp_file(tmp_path: Path) -> None:
    """Prefer snap.output; else read head of output_file (capped)."""
    from codedoggy.tui_v2.app import GrokShellApp

    session = SimpleNamespace(
        phase=SessionPhase.IDLE,
        id="test-bg-out",
        cwd=Path.cwd(),
        extensions=SimpleNamespace(turn_runner=None, kernel=None),
        handle_prompt=lambda *a, **k: None,
    )
    app = GrokShellApp(session)

    # In-memory output wins even if a file is also present
    f_unused = tmp_path / "unused.log"
    f_unused.write_text("from_file_should_not_win", encoding="utf-8")
    mem = SimpleNamespace(output="from_memory\nline2", output_file=str(f_unused))
    assert app._bg_task_output_from_snap(mem) == "from_memory\nline2"

    # Empty / whitespace output → fall through to file
    log = tmp_path / "task.out"
    log.write_text("file_stdout_a\nfile_stdout_b\n", encoding="utf-8")
    from_file = SimpleNamespace(output="", output_file=str(log))
    assert "file_stdout_a" in app._bg_task_output_from_snap(from_file)
    assert "file_stdout_b" in app._bg_task_output_from_snap(from_file)

    # Missing file → empty string
    missing = SimpleNamespace(output="", output_file=str(tmp_path / "nope.log"))
    assert app._bg_task_output_from_snap(missing) == ""

    # No path and no output
    bare = SimpleNamespace(output=None, output_file=None)
    assert app._bg_task_output_from_snap(bare) == ""
