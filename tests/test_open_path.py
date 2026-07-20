"""Click-to-open image paths for the TUI (Grok-style link affordance)."""

from __future__ import annotations

import json
from pathlib import Path

from codedoggy.tui.agent_detail import (
    DetailBlock,
    DetailRecord,
    render_detail_body,
    snapshot_from_messages,
)
from codedoggy.tui.open_path import (
    extract_image_paths,
    open_local_path,
    paths_from_detail_record,
    resolve_openable_path,
)
from codedoggy.turn.types import Message, Role, ToolCall


def test_extract_image_paths_from_media_gen_json() -> None:
    payload = {
        "path": r"C:\work\images\1.jpg",
        "filename": "1.jpg",
        "session_folder": "images",
        "message": "Image generated and saved to C:\\work\\images\\1.jpg.",
    }
    text = json.dumps(payload, ensure_ascii=False)
    paths = extract_image_paths(text)
    assert any(p.endswith("1.jpg") for p in paths)


def test_extract_relative_images_path() -> None:
    text = "see images/3.jpg for the asset"
    assert "images/3.jpg" in extract_image_paths(text)


def test_resolve_and_open_local_path(tmp_path: Path, monkeypatch: object) -> None:
    img = tmp_path / "images" / "1.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"\xff\xd8\xff\xd9")
    resolved = resolve_openable_path("images/1.jpg", cwd=tmp_path)
    assert resolved is not None
    assert resolved == img.resolve()

    opened: list[str] = []

    def _fake_startfile(path: str) -> None:
        opened.append(path)

    monkeypatch.setattr("os.startfile", _fake_startfile, raising=False)
    monkeypatch.setattr(
        "codedoggy.tui.open_path.sys.platform",
        "win32",
        raising=False,
    )
    # Force win32 branch
    import codedoggy.tui.open_path as op

    monkeypatch.setattr(op.sys, "platform", "win32")
    monkeypatch.setattr(op.os, "startfile", _fake_startfile, raising=False)

    ok, msg = open_local_path("images/1.jpg", cwd=tmp_path)
    assert ok is True
    assert opened and opened[0].endswith("1.jpg")
    assert "已打开" in msg


def test_detail_body_emits_clickable_image_line(tmp_path: Path) -> None:
    abs_path = str((tmp_path / "images" / "2.jpg").resolve())
    payload = json.dumps(
        {
            "path": abs_path,
            "filename": "2.jpg",
            "session_folder": "images",
            "message": f"Image generated and saved to {abs_path}.",
        },
        ensure_ascii=False,
    )
    messages = [
        Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="ig1", name="image_gen", arguments={"prompt": "dog"})],
        ),
        Message(role=Role.TOOL, name="image_gen", tool_call_id="ig1", content=payload),
    ]
    snap = snapshot_from_messages(
        messages,
        task_id="t1",
        agent_id="t1:main",
        agent_label="MAIN",
        task_title="gen",
        status="completed",
    )
    clicks: list[str] = []

    def path_mouse(path: str):
        def handler(_event: object) -> None:
            clicks.append(path)

        return handler

    frags = render_detail_body(
        snap, 80, active_filter="tool", path_mouse=path_mouse
    )
    text = "".join(f[1] for f in frags)
    assert "点击打开" in text
    # At least one fragment carries a mouse handler (3-tuple)
    assert any(len(f) >= 3 and f[2] is not None for f in frags if isinstance(f, tuple))
