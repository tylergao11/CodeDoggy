"""Read-only session surface for the TUI — single fan-in over session truth.

All display paths that need provider/model/auth/budget/mode go through here.
View-local state (ledger, modals, selection) stays in ``CodeDoggyTUI``.
"""

from __future__ import annotations

from typing import Any

from codedoggy.model.auth import auth_status
from codedoggy.model.connection import (
    ActiveConnection,
    ConnectionService,
    connection_of,
)


def get_connection(session: Any) -> ConnectionService | None:
    return connection_of(session)


def active_connection(session: Any) -> ActiveConnection | None:
    svc = connection_of(session)
    if svc is None:
        return None
    return svc.snapshot()


def require_connection(session: Any) -> ConnectionService:
    svc = connection_of(session)
    if svc is None:
        raise RuntimeError(
            "session has no ConnectionService — build_session must attach "
            "extensions.connection (unified model truth)"
        )
    return svc


def provider_id(session: Any) -> str:
    snap = active_connection(session)
    if snap is not None and snap.provider:
        return snap.provider
    return ""


def model_id(session: Any) -> str:
    snap = active_connection(session)
    if snap is not None and snap.model:
        return snap.model
    return ""


def ready_to_sample(session: Any) -> bool:
    """Gate for starting a turn — prefers connection truth, refreshes auth lightly."""
    svc = connection_of(session)
    if svc is not None:
        snap = svc.refresh_auth()
        return bool(snap.ready_to_sample)
    # Legacy / unit-test sessions without ConnectionService.
    from codedoggy.model.auth import auth_status, is_imperial
    from codedoggy.model.registry import model_config_from_env

    try:
        cfg = model_config_from_env()
    except Exception:  # noqa: BLE001
        return True
    pid = (cfg.provider or "ollama").strip().lower()
    if pid == "ollama" or not is_imperial(pid):
        if pid == "ollama":
            return True
        return bool(auth_status(pid).logged_in)
    return bool(auth_status(pid).logged_in)


def session_mode_label(session: Any) -> str:
    kernel = getattr(getattr(session, "extensions", None), "kernel", None)
    mode_state = getattr(kernel, "session_mode_state", None)
    raw_mode = getattr(getattr(mode_state, "mode", None), "value", None)
    return {
        "normal": "auto",
        "goal": "goal",
        "plan": "plan",
    }.get(str(raw_mode or "normal"), str(raw_mode or "auto"))


def model_and_mode_text(session: Any) -> str:
    snap = active_connection(session)
    model = snap.model if snap is not None else "model"
    return f"{model} · {session_mode_label(session)}"


def budget_text(session: Any) -> str:
    """Token budget line for header — context stats + connection window."""
    context = getattr(getattr(session, "extensions", None), "context", None)
    budget = getattr(context, "budget", None)
    used = getattr(budget, "last_prompt_tokens", None)
    total = getattr(budget, "context_window", None)
    if not total:
        snap = active_connection(session)
        total = snap.context_window if snap is not None else None
    if not total:
        return ""
    used_text = "—" if used is None else _compact_number(int(used))
    return f"{used_text} / {_compact_number(int(total))}"


def hud_projection(session: Any) -> dict[str, Any]:
    """Street AUTH panel data — current provider from connection, not env."""
    rows = []
    for pid in ("grok", "claude", "codex"):
        st = auth_status(pid)
        rows.append(
            {
                "id": pid,
                "logged_in": st.logged_in,
                "detail": st.detail,
                "source": st.source,
            }
        )
    snap = active_connection(session)
    if snap is not None:
        cur = snap.provider
        current_ok = bool(snap.logged_in) or cur == "ollama"
    else:
        cur = ""
        current_ok = False
    any_in = any(r["logged_in"] for r in rows)
    return {
        "provider": cur,
        "model": snap.model if snap is not None else "",
        "any_logged_in": any_in,
        "rows": rows,
        "current_ok": current_ok,
        "generation": snap.generation if snap is not None else 0,
        "label": snap.label if snap is not None else "",
    }


def apply_connection(
    session: Any,
    *,
    provider: str | None = None,
    model: str | None = None,
    require_auth: bool = True,
    source: str = "panel",
) -> ActiveConnection:
    """TUI write path — only through ConnectionService.apply."""
    svc = connection_of(session)
    if svc is None:
        svc = _attach_ephemeral_connection(session)
    return svc.apply(
        provider=provider,
        model=model,
        require_auth=require_auth,
        source=source,  # type: ignore[arg-type]
    )


def _attach_ephemeral_connection(session: Any) -> ConnectionService:
    """Last-resort attach for sessions built outside ``build_session``."""
    from codedoggy.model.profiles import model_profiles_from_env
    from codedoggy.model.registry import create_client
    from codedoggy.session.extensions import SessionExtensions

    prof = model_profiles_from_env()
    ext = getattr(session, "extensions", None)
    runner = getattr(ext, "turn_runner", None) if ext is not None else None
    client = getattr(getattr(runner, "sampler", None), "client", None)
    if client is None:
        client = create_client(prof.main, require_auth=False)
    svc = ConnectionService.bootstrap(
        prof.main, aux=prof.aux, client=client, runner=runner
    )
    if ext is not None and hasattr(session, "bind_extensions"):
        if isinstance(ext, SessionExtensions):
            session.bind_extensions(ext.with_connection(svc))
        else:
            try:
                ext.connection = svc
            except Exception:  # noqa: BLE001
                pass
    elif ext is not None:
        try:
            ext.connection = svc
        except Exception:  # noqa: BLE001
            pass
    else:
        session.extensions = SessionExtensions(connection=svc, turn_runner=runner)
    return svc


def _compact_number(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(value)
