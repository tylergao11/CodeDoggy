"""Auth + model connection wizard — Hermes-style menu, keyboard + mouse friendly.

Steps:
  home       → pick Grok / Claude / Codex / API-key providers / refresh
  provider   → login / paste token / select model / apply / back
  model      → pick from catalog (or custom id)
  waiting    → background browser login in progress
  paste      → enter API key / token / custom model id
  result     → success or failure summary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from codedoggy.model.auth import auth_status, begin_login, is_imperial
from codedoggy.model.auth.base import AuthStatus
from codedoggy.model.catalog import suggested_models
from codedoggy.model.profile_registry import get_profile
from codedoggy.model.registry import create_client, model_config_from_env


class WizardStep(str, Enum):
    HOME = "home"
    PROVIDER = "provider"
    MODEL = "model"
    WAITING = "waiting"
    PASTE = "paste"
    RESULT = "result"


@dataclass(slots=True)
class MenuItem:
    id: str
    label: str
    hint: str = ""
    enabled: bool = True
    style: str = "normal"  # normal | accent | muted | danger | ok


@dataclass
class WizardAction:
    """Side effect the TUI host should perform after activate()."""

    kind: Literal[
        "none",
        "start_login",
        "cancel_login",
        "reload_client",
        "close",
        "focus_input",
        "blur_input",
    ] = "none"
    provider: str | None = None
    model: str | None = None
    message: str = ""
    feedback_kind: str = "info"


@dataclass
class AuthWizard:
    step: WizardStep = WizardStep.HOME
    cursor: int = 0
    provider: str | None = None
    items: list[MenuItem] = field(default_factory=list)
    title: str = "AUTH GATE"
    subtitle: str = "选择身份 · ↑↓ 移动 · Enter 确认 · Esc 返回"
    body_note: str = ""
    paste_buffer: str = ""
    paste_prompt: str = ""
    paste_kind: str = "token"  # token | model
    last_status: AuthStatus | None = None
    busy: bool = False
    result_ok: bool | None = None
    # Session connection truth (set by TUI on open)
    active_provider: str = ""
    active_model: str = ""

    def open(
        self,
        *,
        active_provider: str | None = None,
        active_model: str | None = None,
    ) -> None:
        self.step = WizardStep.HOME
        self.cursor = 0
        self.provider = None
        self.paste_buffer = ""
        self.paste_kind = "token"
        self.busy = False
        self.result_ok = None
        self.body_note = ""
        if active_provider is not None:
            self.active_provider = str(active_provider or "").strip()
        if active_model is not None:
            self.active_model = str(active_model or "").strip()
        self._rebuild()

    def _rebuild(self) -> None:
        if self.step == WizardStep.HOME:
            self._build_home()
        elif self.step == WizardStep.PROVIDER:
            self._build_provider()
        elif self.step == WizardStep.MODEL:
            self._build_model()
        elif self.step == WizardStep.WAITING:
            self.items = [
                MenuItem("cancel", "取消等待", "Esc", enabled=True, style="danger"),
            ]
        elif self.step == WizardStep.PASTE:
            self.items = [
                MenuItem("submit", "确认提交", "Enter", style="accent"),
                MenuItem("back", "返回", "Esc", style="muted"),
            ]
        elif self.step == WizardStep.RESULT:
            self.items = [
                MenuItem("done", "完成", "Enter 关闭", style="ok" if self.result_ok else "accent"),
                MenuItem("again", "继续配置", "", style="normal"),
            ]
        self.cursor = max(0, min(self.cursor, max(0, len(self.items) - 1)))

    def _build_home(self) -> None:
        self.title = "连接"
        cur = ""
        if self.active_provider or self.active_model:
            cur = f"当前 {self.active_provider or '—'}/{self.active_model or '—'}"
        self.subtitle = cur or "Provider · Model · 登录 · ↑↓ Enter · Esc"
        items: list[MenuItem] = []
        active_provider = self.active_provider.strip().lower()

        def status_style(provider: str, *, available: bool) -> str:
            if provider == active_provider:
                return "active"
            return "logged" if available else "offline"

        for pid, label, hint in (
            ("grok", "Grok (xAI)", "浏览器 device-code · 订阅优先"),
            ("claude", "Claude", "打开网页 · 需 token/凭证文件"),
            ("codex", "Codex", "打开网页 · 需 ~/.codex 或 Key"),
        ):
            st = auth_status(pid)
            badge = "✓ 已登录" if st.logged_in else "○ 未登录"
            active = " · 使用中" if pid == self.active_provider else ""
            items.append(
                MenuItem(
                    pid,
                    f"{label}  {badge}{active}",
                    st.detail or hint,
                    style=status_style(pid, available=st.logged_in),
                )
            )
        for pid in ("deepseek", "openai", "ollama", "custom"):
            prof = get_profile(pid)
            if prof is None:
                continue
            st = auth_status(pid)
            badge = "✓ Key" if st.logged_in else "○ 无 Key"
            if pid == "ollama":
                badge = "✓ 本地"
            active = " · 使用中" if pid == self.active_provider else ""
            items.append(
                MenuItem(
                    pid,
                    f"{prof.display_name or pid}  {badge}{active}",
                    "API Key 路径" if pid != "ollama" else "本机 Ollama",
                    style=status_style(
                        pid,
                        available=st.logged_in or pid == "ollama",
                    ),
                )
            )
        items.append(MenuItem("refresh", "刷新状态", "重新探测本机凭证", style="muted"))
        items.append(MenuItem("close", "关闭", "Esc", style="muted"))
        self.items = items

    def _build_provider(self) -> None:
        pid = self.provider or "grok"
        st = auth_status(pid)
        self.last_status = st
        prof = get_profile(pid)
        name = (prof.display_name if prof else pid) or pid
        self.title = f"连接 · {name.upper()}"
        imperial = is_imperial(pid)
        model_hint = self.active_model if pid == self.active_provider else (
            (prof.default_model if prof else "") or ""
        )
        if st.logged_in or pid == "ollama":
            self.subtitle = f"已就绪 · model={model_hint or '—'}"
            self.body_note = st.detail or ""
        else:
            self.subtitle = f"未登录 · 可选 model={model_hint or '—'}"
            self.body_note = st.detail or ""

        items: list[MenuItem] = []
        if imperial and pid == "grok":
            items.append(
                MenuItem(
                    "login",
                    "浏览器登录 (device-code)",
                    "打开网页授权 · 完整闭环",
                    style="accent",
                )
            )
        elif imperial:
            items.append(
                MenuItem(
                    "login",
                    "打开登录网页",
                    "不会自动拿 token · 需本机凭证或下方粘贴",
                    style="accent",
                )
            )
            items.append(
                MenuItem(
                    "paste",
                    "粘贴 Token / API Key",
                    "ANTHROPIC_* / OPENAI_* / 会话 token",
                    style="normal",
                )
            )
        elif pid != "ollama":
            items.append(
                MenuItem(
                    "paste",
                    "粘贴 API Key",
                    f"写入环境供本会话使用 · {pid}",
                    style="accent",
                )
            )

        items.append(
            MenuItem(
                "pick_model",
                "选择模型",
                f"当前候选 · {model_hint or 'catalog'}",
                style="accent",
            )
        )

        can_apply = bool(st.logged_in) or pid == "ollama"
        items.append(
            MenuItem(
                "reload",
                "应用此 Provider",
                "切换连接真源 · 使用当前/默认 model",
                enabled=can_apply,
                style="ok" if can_apply else "muted",
            )
        )
        items.append(MenuItem("back", "返回列表", "Esc", style="muted"))
        self.items = items

    def _build_model(self) -> None:
        pid = self.provider or self.active_provider or "ollama"
        prof = get_profile(pid)
        name = (prof.display_name if prof else pid) or pid
        self.title = f"模型 · {name.upper()}"
        current = self.active_model if pid == self.active_provider else ""
        if not current and prof is not None:
            current = prof.default_model or ""
        self.subtitle = f"选择 model · 当前 {current or '—'}"
        self.body_note = "选中后立即写入连接真源并热切换"

        items: list[MenuItem] = []
        for mid in suggested_models(pid):
            mark = "✓ " if mid == self.active_model and pid == self.active_provider else ""
            items.append(
                MenuItem(
                    f"model:{mid}",
                    f"{mark}{mid}",
                    "Apply 到本会话",
                    style="ok" if mark else "normal",
                )
            )
        items.append(
            MenuItem(
                "custom_model",
                "自定义 model id…",
                "输入任意模型名",
                style="accent",
            )
        )
        items.append(MenuItem("back", "返回", "Esc", style="muted"))
        self.items = items

    def move(self, delta: int) -> None:
        if not self.items or self.busy:
            return
        n = len(self.items)
        self.cursor = (self.cursor + delta) % n
        for _ in range(n):
            if self.items[self.cursor].enabled:
                break
            self.cursor = (self.cursor + delta) % n

    def set_cursor(self, index: int) -> None:
        if 0 <= index < len(self.items) and self.items[index].enabled:
            self.cursor = index

    def activate(self) -> WizardAction:
        if not self.items:
            return WizardAction()
        item = self.items[self.cursor]
        if self.busy and not (
            self.step is WizardStep.WAITING and item.id == "cancel"
        ):
            return WizardAction()
        if not item.enabled:
            return WizardAction()

        if self.step == WizardStep.HOME:
            if item.id == "close":
                return WizardAction(kind="close")
            if item.id == "refresh":
                self._rebuild()
                return WizardAction(message="状态已刷新", feedback_kind="info")
            self.provider = item.id
            self.step = WizardStep.PROVIDER
            self.cursor = 0
            self._rebuild()
            return WizardAction()

        if self.step == WizardStep.PROVIDER:
            if item.id == "back":
                self.step = WizardStep.HOME
                self.cursor = 0
                self.provider = None
                self._rebuild()
                return WizardAction()
            if item.id == "login":
                self.step = WizardStep.WAITING
                self.busy = True
                self.body_note = "正在打开浏览器，请在网页完成授权…"
                self._rebuild()
                return WizardAction(
                    kind="start_login",
                    provider=self.provider,
                    message="浏览器登录中…",
                )
            if item.id == "paste":
                self.step = WizardStep.PASTE
                self.paste_kind = "token"
                self.paste_buffer = ""
                self.paste_prompt = (
                    "粘贴 API Key / OAuth Token，然后 Enter"
                    if self.provider
                    else "粘贴凭证"
                )
                self.cursor = 0
                self._rebuild()
                return WizardAction(kind="focus_input")
            if item.id == "pick_model":
                self.step = WizardStep.MODEL
                self.cursor = 0
                self._rebuild()
                return WizardAction()
            if item.id == "reload":
                return WizardAction(
                    kind="reload_client",
                    provider=self.provider,
                    model=None,
                    message="已应用 Provider（默认/当前 model）",
                    feedback_kind="success",
                )

        if self.step == WizardStep.MODEL:
            if item.id == "back":
                self.step = WizardStep.PROVIDER
                self.cursor = 0
                self._rebuild()
                return WizardAction()
            if item.id == "custom_model":
                self.step = WizardStep.PASTE
                self.paste_kind = "model"
                self.paste_buffer = ""
                self.paste_prompt = "输入 model id，然后 Enter"
                self.cursor = 0
                self._rebuild()
                return WizardAction(kind="focus_input")
            if item.id.startswith("model:"):
                mid = item.id[len("model:") :]
                self.active_model = mid
                if self.provider:
                    self.active_provider = self.provider
                return WizardAction(
                    kind="reload_client",
                    provider=self.provider,
                    model=mid,
                    message=f"已切换模型 · {self.provider}/{mid}",
                    feedback_kind="success",
                )

        if self.step == WizardStep.WAITING:
            if item.id == "cancel":
                self.busy = False
                self.step = WizardStep.PROVIDER
                self.cursor = 0
                self._rebuild()
                return WizardAction(
                    kind="cancel_login",
                    message="已取消等待",
                    feedback_kind="warning",
                )

        if self.step == WizardStep.PASTE:
            if item.id == "back":
                self.step = (
                    WizardStep.MODEL if self.paste_kind == "model" else WizardStep.PROVIDER
                )
                self.cursor = 0
                self.paste_kind = "token"
                self._rebuild()
                return WizardAction(kind="blur_input")
            if item.id == "submit":
                return self._submit_paste()

        if self.step == WizardStep.RESULT:
            if item.id == "done":
                return WizardAction(kind="close", message=self.body_note or "完成")
            if item.id == "again":
                self.step = WizardStep.HOME
                self.cursor = 0
                self.result_ok = None
                self._rebuild()
                return WizardAction(kind="blur_input")

        return WizardAction()

    def submit_paste_text(self, text: str) -> WizardAction:
        self.paste_buffer = text.strip()
        return self._submit_paste()

    def _submit_paste(self) -> WizardAction:
        token = self.paste_buffer.strip()
        # Secrets must not remain on the wizard object after dispatch.
        self.paste_buffer = ""
        if not token:
            self.body_note = "输入为空"
            return WizardAction(message="请输入非空内容", feedback_kind="warning")

        if self.paste_kind == "model":
            mid = token
            self.active_model = mid
            if self.provider:
                self.active_provider = self.provider
            self.step = WizardStep.RESULT
            self.result_ok = True
            self.body_note = f"模型 {self.provider}/{mid}"
            self.paste_kind = "token"
            self.cursor = 0
            self._rebuild()
            return WizardAction(
                kind="reload_client",
                provider=self.provider,
                model=mid,
                message=f"已切换模型 · {self.provider}/{mid}",
                feedback_kind="success",
            )

        pid = self.provider or "custom"
        env_map = {
            "grok": "XAI_API_KEY",
            "xai": "XAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "codex": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "openai": "OPENAI_API_KEY",
            "ollama": "OLLAMA_API_KEY",
            "custom": "CODEDOGGY_API_KEY",
        }
        import os

        if pid in {"claude", "anthropic"} and not token.startswith("sk-ant-api"):
            os.environ["ANTHROPIC_TOKEN"] = token
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            key = env_map.get(pid, "CODEDOGGY_API_KEY")
            os.environ[key] = token
            if pid in {"claude", "anthropic"}:
                os.environ.pop("ANTHROPIC_TOKEN", None)

        st = auth_status(pid)
        self.result_ok = st.logged_in
        self.step = WizardStep.RESULT
        self.body_note = (
            f"已写入环境 · {st.source or 'env'}"
            if st.logged_in
            else "写入后仍未解析到凭证"
        )
        self.cursor = 0
        self._rebuild()
        return WizardAction(
            kind="reload_client" if st.logged_in else "blur_input",
            provider=pid,
            message=self.body_note,
            feedback_kind="success" if st.logged_in else "warning",
        )

    def on_login_finished(self, status: AuthStatus) -> WizardAction:
        self.busy = False
        self.last_status = status
        self.result_ok = bool(status.logged_in)
        self.step = WizardStep.RESULT
        self.body_note = status.detail or (
            "登录成功" if status.logged_in else "登录未完成"
        )
        self.cursor = 0
        self._rebuild()
        return WizardAction(
            kind="reload_client" if status.logged_in else "none",
            provider=self.provider,
            message=self.body_note,
            feedback_kind="success" if status.logged_in else "warning",
        )

    def go_back(self) -> WizardAction:
        if self.busy and self.step is WizardStep.WAITING:
            self.busy = False
            self.step = WizardStep.PROVIDER
            self.cursor = 0
            self._rebuild()
            return WizardAction(
                kind="cancel_login",
                message="已取消等待",
                feedback_kind="warning",
            )
        if self.busy:
            return WizardAction()
        if self.step == WizardStep.HOME:
            return WizardAction(kind="close")
        if self.step == WizardStep.MODEL:
            self.step = WizardStep.PROVIDER
            self.cursor = 0
            self._rebuild()
            return WizardAction(kind="blur_input")
        if self.step in {WizardStep.PROVIDER, WizardStep.RESULT}:
            self.step = WizardStep.HOME
            self.cursor = 0
            self.provider = None
            self._rebuild()
            return WizardAction(kind="blur_input")
        if self.step == WizardStep.PASTE:
            self.step = (
                WizardStep.MODEL if self.paste_kind == "model" else WizardStep.PROVIDER
            )
            self.cursor = 0
            self.paste_kind = "token"
            self._rebuild()
            return WizardAction(kind="blur_input")
        if self.step == WizardStep.WAITING:
            self.step = WizardStep.PROVIDER
            self.cursor = 0
            self._rebuild()
            return WizardAction()
        return WizardAction(kind="close")


def hud_snapshot(current_provider: str | None = None) -> dict[str, Any]:
    """Legacy helper — prefer ``tui.surface.hud_projection(session)``."""
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
    cur = (current_provider or "").strip().lower()
    any_in = any(r["logged_in"] for r in rows)
    return {
        "provider": cur,
        "any_logged_in": any_in,
        "rows": rows,
        "current_ok": auth_status(cur).logged_in if cur else False,
    }


def run_browser_login(provider: str, *, cancel_event: Any | None = None) -> AuthStatus:
    """Blocking browser login — call from worker thread."""
    return begin_login(provider, cancel_event=cancel_event)


def reload_chat_client(provider: str | None = None) -> Any:
    """Legacy: build client without ConnectionService (tests / scripts)."""
    cfg = model_config_from_env(provider=provider)
    return create_client(cfg, require_auth=True)
