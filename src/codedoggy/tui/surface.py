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


def active_connection(session: Any) -> ActiveConnection | None:
    svc = connection_of(session)
    if svc is None:
        return None
    return svc.snapshot()


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
    """Session mode chip: plan wins over goal when both flags conflict.

    Grok modes are exclusive; prefer live plan_phase, then goal, then mode enum.
    """
    kernel = getattr(getattr(session, "extensions", None), "kernel", None)
    mode_state = getattr(kernel, "session_mode_state", None)
    if mode_state is not None:
        phase = str(getattr(mode_state, "plan_phase", "") or "")
        if phase == "pending":
            return "plan…"
        if phase == "exit_pending":
            return "plan↓"
        if phase == "active" or getattr(mode_state, "is_plan", lambda: False)():
            return "plan"
        if getattr(mode_state, "is_goal", lambda: False)():
            return "goal"
    raw_mode = getattr(getattr(mode_state, "mode", None), "value", None)
    return {
        "normal": "auto",
        "goal": "goal",
        "plan": "plan",
    }.get(str(raw_mode or "normal"), str(raw_mode or "auto"))


def model_and_mode_text(session: Any) -> str:
    """Prompt caption: ``model · 推理:high · auto|plan``."""
    snap = active_connection(session)
    model = snap.model if snap is not None else "model"
    mode = session_mode_label(session)
    reason = snap.reasoning_label if snap is not None else ""
    parts = [model]
    if reason:
        parts.append(reason)
    parts.append(mode)
    return " · ".join(parts)


# Sticky last-known usage so the header does not flash "—" while the model is
# sampling / awaiting real usage (compactor clears last_prompt_tokens then).
_budget_sticky: dict[str, tuple[int | None, int | None]] = {}


def budget_text(session: Any) -> str:
    """Token budget line for header — context stats + connection window.

    When ``last_prompt_tokens`` is temporarily ``None`` (reasoning / compact /
    provider switch), keep showing the last good used count instead of ``—``.
    """
    context = getattr(getattr(session, "extensions", None), "context", None)
    budget = getattr(context, "budget", None)
    used = getattr(budget, "last_prompt_tokens", None)
    total = getattr(budget, "context_window", None)
    if not total:
        snap = active_connection(session)
        total = snap.context_window if snap is not None else None

    sid = str(getattr(session, "id", "") or id(session))
    prev_used, prev_total = _budget_sticky.get(sid, (None, None))

    if total is not None:
        try:
            total_i = int(total)
        except (TypeError, ValueError):
            total_i = prev_total
    else:
        total_i = prev_total
    if not total_i:
        return ""

    used_i: int | None
    if used is not None:
        try:
            used_i = int(used)
        except (TypeError, ValueError):
            used_i = prev_used
    else:
        used_i = prev_used

    _budget_sticky[sid] = (used_i, total_i)

    if used_i is None:
        # Unknown used: still show window size, never a bare dash.
        return f"… / {_compact_number(total_i)}"
    return f"{_compact_number(used_i)} / {_compact_number(total_i)}"


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
    kernel = getattr(getattr(session, "extensions", None), "kernel", None)
    runtime = getattr(kernel, "mcp_runtime", None)
    statuses = list(getattr(runtime, "statuses", []) or []) if runtime is not None else []
    mcp_ready = sum(
        1 for item in statuses
        if isinstance(item, dict) and item.get("status") == "ready"
    )
    mcp_bad = sum(
        1 for item in statuses
        if isinstance(item, dict)
        and item.get("status") in {"unavailable", "needs_auth"}
    )
    mcp_connecting = sum(
        1 for item in statuses
        if isinstance(item, dict) and item.get("status") == "initializing"
    )
    return {
        "provider": cur,
        "model": snap.model if snap is not None else "",
        "reasoning": snap.reasoning_label if snap is not None else "",
        "reasoning_effort": snap.reasoning_effort if snap is not None else "",
        "reasoning_enabled": bool(snap.reasoning_enabled) if snap is not None else False,
        "mode": session_mode_label(session),
        "any_logged_in": any_in,
        "rows": rows,
        "current_ok": current_ok,
        "generation": snap.generation if snap is not None else 0,
        "label": snap.label if snap is not None else "",
        "mcp": {
            "ready": mcp_ready,
            "bad": mcp_bad,
            "connecting": mcp_connecting,
            "configured": bool(getattr(runtime, "servers", []) or statuses)
            if runtime is not None
            else False,
        },
    }


def apply_connection(
    session: Any,
    *,
    provider: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    reasoning_enabled: bool | None = None,
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
        reasoning_effort=reasoning_effort,
        reasoning_enabled=reasoning_enabled,
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
