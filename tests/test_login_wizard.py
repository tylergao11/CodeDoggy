"""Auth wizard state machine — keyboard menu contract."""

from __future__ import annotations

from unittest.mock import patch

from codedoggy.model.auth.base import AUTH_OAUTH, AuthStatus
from codedoggy.tui.login_wizard import AuthWizard, WizardStep, hud_snapshot


def test_wizard_home_has_imperial_and_moves() -> None:
    w = AuthWizard()
    w.open()
    assert w.step == WizardStep.HOME
    assert any(i.id == "grok" for i in w.items)
    assert any(i.id == "claude" for i in w.items)
    assert any(i.id == "codex" for i in w.items)
    start = w.cursor
    w.move(1)
    assert w.cursor != start or len(w.items) == 1
    w.move(-1)
    assert w.cursor == start


def test_wizard_enter_provider_and_back() -> None:
    w = AuthWizard()
    w.open()
    # select grok
    for i, item in enumerate(w.items):
        if item.id == "grok":
            w.cursor = i
            break
    action = w.activate()
    assert w.step == WizardStep.PROVIDER
    assert w.provider == "grok"
    assert action.kind == "none"
    assert any(i.id == "login" for i in w.items)
    # back
    for i, item in enumerate(w.items):
        if item.id == "back":
            w.cursor = i
            break
    w.activate()
    assert w.step == WizardStep.HOME


def test_wizard_login_starts_waiting() -> None:
    w = AuthWizard()
    w.open()
    w.provider = "grok"
    w.step = WizardStep.PROVIDER
    w._rebuild()
    for i, item in enumerate(w.items):
        if item.id == "login":
            w.cursor = i
            break
    action = w.activate()
    assert action.kind == "start_login"
    assert action.provider == "grok"
    assert w.step == WizardStep.WAITING
    assert w.busy is True


def test_wizard_login_finished_success() -> None:
    w = AuthWizard()
    w.provider = "grok"
    w.step = WizardStep.WAITING
    w.busy = True
    st = AuthStatus(
        provider="grok",
        kind=AUTH_OAUTH,
        logged_in=True,
        source="file:test",
        detail="signed in",
    )
    action = w.on_login_finished(st)
    assert w.step == WizardStep.RESULT
    assert w.result_ok is True
    assert action.kind == "reload_client"


def test_wizard_paste_sets_env(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    w = AuthWizard()
    w.open()
    w.provider = "deepseek"
    w.step = WizardStep.PASTE
    w._rebuild()
    action = w.submit_paste_text("sk-test-deepseek")
    assert action.kind in {"reload_client", "blur_input"}
    import os

    assert os.environ.get("DEEPSEEK_API_KEY") == "sk-test-deepseek"


def test_hud_snapshot_shape() -> None:
    snap = hud_snapshot("grok")
    assert "rows" in snap
    assert len(snap["rows"]) == 3
    assert snap["provider"] == "grok"


def test_go_back_from_home_closes() -> None:
    w = AuthWizard()
    w.open()
    action = w.go_back()
    assert action.kind == "close"


def test_wizard_pick_model_emits_apply_action() -> None:
    w = AuthWizard()
    w.open(active_provider="ollama", active_model="qwen3:8b")
    w.provider = "ollama"
    w.step = WizardStep.PROVIDER
    w._rebuild()
    assert any(i.id == "pick_model" for i in w.items)
    for i, item in enumerate(w.items):
        if item.id == "pick_model":
            w.cursor = i
            break
    w.activate()
    assert w.step == WizardStep.MODEL
    assert any(i.id.startswith("model:") for i in w.items)
    for i, item in enumerate(w.items):
        if item.id.startswith("model:") and "qwen" not in item.id:
            w.cursor = i
            mid = item.id.split(":", 1)[1]
            break
    else:
        # fall back to first catalog row
        for i, item in enumerate(w.items):
            if item.id.startswith("model:"):
                w.cursor = i
                mid = item.id.split(":", 1)[1]
                break
    action = w.activate()
    assert action.kind == "reload_client"
    assert action.provider == "ollama"
    assert action.model == mid


def test_wizard_custom_model_paste() -> None:
    w = AuthWizard()
    w.provider = "ollama"
    w.step = WizardStep.MODEL
    w._rebuild()
    for i, item in enumerate(w.items):
        if item.id == "custom_model":
            w.cursor = i
            break
    action = w.activate()
    assert action.kind == "focus_input"
    assert w.paste_kind == "model"
    action = w.submit_paste_text("my-custom:7b")
    assert action.kind == "reload_client"
    assert action.model == "my-custom:7b"
    assert action.provider == "ollama"
