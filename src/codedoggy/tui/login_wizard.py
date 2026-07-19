"""Auth login wizard — Hermes-style menu, keyboard + mouse friendly.

Steps:
  home       → pick Grok / Claude / Codex / API-key providers / refresh
  provider   → login / paste token / status / back
  waiting    → background browser login in progress
  paste      → enter API key / token
  result     → success or failure summary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal

from codedoggy.model.auth import auth_status, begin_login, is_imperial
from codedoggy.model.auth.base import AuthStatus
from codedoggy.model.profile_registry import get_profile
from codedoggy.model.registry import create_client, model_config_from_env


class WizardStep(str, Enum):
    HOME = "home"
    PROVIDER = "provider"
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
        "reload_client",
        "close",
        "focus_input",
        "blur_input",
    ] = "none"
    provider: str | None = None
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
    last_status: AuthStatus | None = None
    busy: bool = False
    result_ok: bool | None = None

    def open(self) -> None:
        self.step = WizardStep.HOME
        self.cursor = 0
        self.provider = None
        self.paste_buffer = ""
        self.busy = False
        self.result_ok = None
        self.body_note = ""
        self._rebuild()

    def _rebuild(self) -> None:
        if self.step == WizardStep.HOME:
            self._build_home()
        elif self.step == WizardStep.PROVIDER:
            self._build_provider()
        elif self.step == WizardStep.WAITING:
            self.items = [
                MenuItem("cancel", "取消等待", "Esc", enabled=not self.busy, style="danger"),
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
        self.title = "AUTH GATE"
        self.subtitle = "Grok / Claude / Codex 网页授权 · 其它走 API Key · ↑↓ Enter · 点击"
        items: list[MenuItem] = []
        for pid, label, hint in (
            ("grok", "Grok (xAI)", "浏览器 device-code · 订阅优先"),
            ("claude", "Claude", "打开网页 · 需 token/凭证文件"),
            ("codex", "Codex", "打开网页 · 需 ~/.codex 或 Key"),
        ):
            st = auth_status(pid)
            badge = "✓ 已登录" if st.logged_in else "○ 未登录"
            items.append(
                MenuItem(
                    pid,
                    f"{label}  {badge}",
                    st.detail or hint,
                    style="ok" if st.logged_in else "accent",
                )
            )
        # API-key family
        for pid in ("deepseek", "openai", "ollama", "custom"):
            prof = get_profile(pid)
            if prof is None:
                continue
            st = auth_status(pid)
            badge = "✓ Key" if st.logged_in else "○ 无 Key"
            items.append(
                MenuItem(
                    pid,
                    f"{prof.display_name or pid}  {badge}",
                    "API Key 路径",
                    style="ok" if st.logged_in else "muted",
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
        self.title = f"AUTH · {name.upper()}"
        imperial = is_imperial(pid)
        if st.logged_in:
            self.subtitle = f"已登录 · source={st.source}"
            self.body_note = st.detail or ""
        else:
            self.subtitle = "未登录"
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
        else:
            items.append(
                MenuItem(
                    "paste",
                    "粘贴 API Key",
                    f"写入环境供本会话使用 · {pid}",
                    style="accent",
                )
            )
        if st.logged_in:
            items.append(
                MenuItem(
                    "reload",
                    "重新加载客户端",
                    "用当前凭证热切换 sampler",
                    style="ok",
                )
            )
        items.append(MenuItem("back", "返回列表", "Esc", style="muted"))
        self.items = items

    def move(self, delta: int) -> None:
        if not self.items or self.busy:
            return
        n = len(self.items)
        self.cursor = (self.cursor + delta) % n
        # skip disabled
        for _ in range(n):
            if self.items[self.cursor].enabled:
                break
            self.cursor = (self.cursor + delta) % n

    def set_cursor(self, index: int) -> None:
        if 0 <= index < len(self.items) and self.items[index].enabled:
            self.cursor = index

    def activate(self) -> WizardAction:
        if not self.items or self.busy:
            return WizardAction()
        item = self.items[self.cursor]
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
                self.paste_buffer = ""
                self.paste_prompt = (
                    "粘贴 API Key / OAuth Token，然后 Enter"
                    if self.provider
                    else "粘贴凭证"
                )
                self.cursor = 0
                self._rebuild()
                return WizardAction(kind="focus_input")
            if item.id == "reload":
                return WizardAction(
                    kind="reload_client",
                    provider=self.provider,
                    message="客户端已按当前凭证重载",
                    feedback_kind="success",
                )

        if self.step == WizardStep.WAITING:
            if item.id == "cancel":
                self.busy = False
                self.step = WizardStep.PROVIDER
                self.cursor = 0
                self._rebuild()
                return WizardAction(message="已取消等待", feedback_kind="warning")

        if self.step == WizardStep.PASTE:
            if item.id == "back":
                self.step = WizardStep.PROVIDER
                self.cursor = 0
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
        if not token:
            self.body_note = "凭证为空"
            return WizardAction(message="请粘贴非空凭证", feedback_kind="warning")
        pid = self.provider or "custom"
        # Store on process env so resolve() picks it up
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

        # Prefer OAuth-shaped env for claude when token looks like oauth
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
        if self.busy:
            return WizardAction()
        if self.step == WizardStep.HOME:
            return WizardAction(kind="close")
        if self.step in {WizardStep.PROVIDER, WizardStep.RESULT}:
            self.step = WizardStep.HOME
            self.cursor = 0
            self.provider = None
            self._rebuild()
            return WizardAction(kind="blur_input")
        if self.step == WizardStep.PASTE:
            self.step = WizardStep.PROVIDER
            self.cursor = 0
            self._rebuild()
            return WizardAction(kind="blur_input")
        if self.step == WizardStep.WAITING:
            self.step = WizardStep.PROVIDER
            self.cursor = 0
            self._rebuild()
            return WizardAction()
        return WizardAction(kind="close")


def hud_snapshot(current_provider: str | None = None) -> dict[str, Any]:
    """Data for street HUD auth panel."""
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
    if not cur:
        try:
            cfg = model_config_from_env()
            cur = cfg.provider
        except Exception:  # noqa: BLE001
            cur = ""
    any_in = any(r["logged_in"] for r in rows)
    return {
        "provider": cur,
        "any_logged_in": any_in,
        "rows": rows,
        "current_ok": auth_status(cur).logged_in if cur else False,
    }


def run_browser_login(provider: str) -> AuthStatus:
    """Blocking browser login — call from worker thread."""
    return begin_login(provider)


def reload_chat_client(provider: str | None = None) -> Any:
    """Build a ChatClient with hard auth for the given/current provider."""
    cfg = model_config_from_env(provider=provider)
    return create_client(cfg, require_auth=True)
