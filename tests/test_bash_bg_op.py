"""Grok bash background-operator detection (bash_bg_op / bash_params)."""

from __future__ import annotations

from codedoggy.tools.grok_build.bash_bg_op import (
    AmpersandSemantics,
    contains_background_operator,
    contains_unwaited_background_operator,
    ends_with_wait_builtin,
    has_trailing_background_operator,
    should_reject_background_op,
)
from codedoggy.tools.grok_build.bash_params import (
    effective_auto_bg_wait_ms,
    resolve_fg_timeout_ms,
)
from codedoggy.tools.util.shell_state import shell_env_overrides


def test_trailing_ampersand() -> None:
    assert has_trailing_background_operator("sleep 1 &")
    assert not has_trailing_background_operator("true && false")
    assert not has_trailing_background_operator("echo hi")


def test_contains_bg_op_quotes_and_redirects() -> None:
    assert contains_background_operator("cmd1 & cmd2")
    assert not contains_background_operator("true && false")
    assert not contains_background_operator("cmd &> out.txt")
    assert not contains_background_operator("echo 'a & b'")
    assert not contains_background_operator('echo "a & b"')


def test_wait_builtin_allows_parallel() -> None:
    assert ends_with_wait_builtin("cmd1 & cmd2 & wait")
    assert contains_unwaited_background_operator("cmd1 &")
    assert not contains_unwaited_background_operator("cmd1 & wait")


def test_should_reject_grok_defaults() -> None:
    # Grok default: allow=true, enabled=true → never reject
    assert (
        should_reject_background_op(
            is_background=False,
            allow_background_operator=True,
            background_enabled=True,
            semantics=AmpersandSemantics.PosixBackground,
            command="sleep 1 &",
        )
        is None
    )
    # Explicit background=true also bypasses
    assert (
        should_reject_background_op(
            is_background=True,
            allow_background_operator=False,
            background_enabled=True,
            semantics=AmpersandSemantics.PosixBackground,
            command="sleep 1 &",
        )
        is None
    )
    # allow=false + enabled → reject
    v = should_reject_background_op(
        is_background=False,
        allow_background_operator=False,
        background_enabled=True,
        semantics=AmpersandSemantics.PosixBackground,
        command="sleep 1 &",
    )
    assert v is not None


def test_auto_bg_wait_min_budget() -> None:
    assert effective_auto_bg_wait_ms(120_000, auto_background_on_timeout=False) == 120_000
    assert effective_auto_bg_wait_ms(120_000, auto_background_on_timeout=True) == 15_000
    assert effective_auto_bg_wait_ms(5_000, auto_background_on_timeout=True) == 5_000


def test_resolve_fg_timeout() -> None:
    assert resolve_fg_timeout_ms(None) == 120_000
    assert resolve_fg_timeout_ms(0) == 120_000
    assert resolve_fg_timeout_ms(500_000) == 300_000


def test_shell_env_overrides() -> None:
    o = shell_env_overrides()
    assert o["TERM"] == "dumb"
    assert o["NO_COLOR"] == "1"
    assert o["FORCE_COLOR"] == "0"
