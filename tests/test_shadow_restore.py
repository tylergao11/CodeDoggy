"""Shadow soft restore on P0 (critical) mutations."""

from __future__ import annotations

from pathlib import Path

from codedoggy.audit.hooks import ResidentAuditHooks
from codedoggy.audit.restore import restore_mutation_before, shadow_restore_enabled
from codedoggy.audit.services import AuditServices
from codedoggy.audit.trajectory import MutationTrajectory
from codedoggy.audit.types import (
    AuditContext,
    AuditFinding,
    AuditVerdict,
    FindingSeverity,
    MutationEvent,
)
from codedoggy.session import Session
from codedoggy.tools import ToolRegistryBuilder
from codedoggy.turn import Role, SampleResult, ToolCall, run_agent_loop
from codedoggy.turn.hooks import HookContext
from codedoggy.turn.types import FileMutation, ToolResultRecord


class CriticalPathAuditor:
    """Always emit critical finding for matching path substring."""

    def __init__(self, needle: str = "", message: str = "P0 stop") -> None:
        self.needle = needle
        self.message = message

    def review(self, ctx: AuditContext) -> AuditVerdict:
        path = ctx.mutation.path
        if self.needle and self.needle not in path and self.needle not in (
            ctx.mutation.after or ""
        ):
            return AuditVerdict.pass_silent()
        return AuditVerdict.fail(
            [
                AuditFinding(
                    message=self.message,
                    severity=FindingSeverity.CRITICAL,
                    path=path,
                    code="test_p0",
                )
            ]
        )


class ScriptedSampler:
    def __init__(self, script: list[SampleResult]) -> None:
        self.script = list(script)
        self.n = 0

    def sample(self, messages, tools):
        if self.n >= len(self.script):
            return SampleResult(content="done")
        out = self.script[self.n]
        self.n += 1
        return out


def test_restore_mutation_before_edit(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("NEW", encoding="utf-8")
    mut = FileMutation(
        path="a.txt",
        tool_name="search_replace",
        call_id="1",
        before="OLD\n",
        after="NEW",
    )
    r = restore_mutation_before(tmp_path, mut)
    assert r["ok"] is True
    assert r["reason"] == "before_restored"
    assert f.read_text(encoding="utf-8") == "OLD\n"


def test_restore_mutation_before_create_deletes(tmp_path: Path) -> None:
    f = tmp_path / "created.txt"
    f.write_text("brand new", encoding="utf-8")
    mut = MutationEvent(
        path="created.txt",
        tool_name="search_replace",
        call_id="1",
        before=None,
        after="brand new",
        is_create=True,
    )
    r = restore_mutation_before(tmp_path, mut)
    assert r["ok"] is True
    assert r["reason"] == "create_undone"
    assert not f.exists()


def test_restore_mutation_delete_rewrites(tmp_path: Path) -> None:
    mut = FileMutation(
        path="gone.txt",
        tool_name="search_replace",
        call_id="1",
        before="was here",
        after=None,
        is_delete=True,
    )
    r = restore_mutation_before(tmp_path, mut)
    assert r["ok"] is True
    assert r["reason"] == "delete_restored"
    assert (tmp_path / "gone.txt").read_text(encoding="utf-8") == "was here"


def test_restore_skips_outside_cwd(tmp_path: Path) -> None:
    mut = FileMutation(
        path="../escape.txt",
        tool_name="t",
        call_id="1",
        before="x",
        after="y",
    )
    r = restore_mutation_before(tmp_path, mut)
    assert r["ok"] is False
    assert r["reason"] == "path_outside_cwd"


def test_restore_no_before_without_create(tmp_path: Path) -> None:
    mut = FileMutation(
        path="a.txt",
        tool_name="t",
        call_id="1",
        before=None,
        after="y",
        is_create=False,
    )
    r = restore_mutation_before(tmp_path, mut)
    assert r["ok"] is False
    assert r["reason"] == "no_before"


def test_shadow_restore_env_gate(monkeypatch) -> None:
    monkeypatch.delenv("CODEDOGGY_SHADOW_RESTORE", raising=False)
    assert shadow_restore_enabled() is True
    monkeypatch.setenv("CODEDOGGY_SHADOW_RESTORE", "1")
    assert shadow_restore_enabled() is True
    monkeypatch.setenv("CODEDOGGY_SHADOW_RESTORE", "0")
    assert shadow_restore_enabled() is False
    monkeypatch.setenv("CODEDOGGY_SHADOW_RESTORE", "off")
    assert shadow_restore_enabled() is False


def test_p0_after_mutation_restores_before(tmp_path: Path) -> None:
    """Integration: search_replace mutates; P0 triggers soft restore to before."""
    target = tmp_path / "goal.txt"
    target.write_text("hello world\n", encoding="utf-8")

    traj = MutationTrajectory()
    audit = AuditServices.create(
        auditor=CriticalPathAuditor(needle="goal.txt", message="critical: bad edit"),
        trajectory=traj,
    )
    hooks = ResidentAuditHooks(audit)
    tools = ToolRegistryBuilder.new().finalize()

    sampler = ScriptedSampler(
        [
            SampleResult(
                content="edit",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="search_replace",
                        arguments={
                            "file_path": "goal.txt",
                            "old_string": "hello world",
                            "new_string": "DESTROYED",
                        },
                    )
                ],
            ),
            SampleResult(content="should not need second round"),
        ]
    )
    result = run_agent_loop(
        user_text="edit",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        hooks=hooks,
        session=Session.create(tmp_path, goal="keep hello"),
    )
    # Soft restore: file content back to before
    assert target.read_text(encoding="utf-8") == "hello world\n"
    tool_msg = next(m for m in result.messages if m.role is Role.TOOL)
    assert "P0" in (tool_msg.content or "")
    # Loop aborted remaining writes due to P0
    assert result.aborted or not result.completed or "P0" in (tool_msg.content or "")


def test_p0_restore_metadata_on_decision(tmp_path: Path) -> None:
    f = tmp_path / "m.txt"
    f.write_text("after", encoding="utf-8")
    traj = MutationTrajectory()
    audit = AuditServices.create(
        auditor=CriticalPathAuditor(needle="m.txt"),
        trajectory=traj,
    )
    hooks = ResidentAuditHooks(audit)
    from codedoggy.turn.types import ToolCall as TC

    call = TC(id="c1", name="search_replace", arguments={"file_path": "m.txt"})
    mut = FileMutation(
        path="m.txt",
        tool_name="search_replace",
        call_id="c1",
        before="before content",
        after="after",
    )
    record = ToolResultRecord(
        call=call,
        content="ok",
        ok=True,
        mutation=mut,
        mutations=[mut],
    )
    ctx = HookContext(cwd=tmp_path, round_index=0, session=None)
    decision = hooks.after_mutation(record, ctx)
    assert decision is not None
    assert decision.abort is True
    assert decision.metadata.get("p0_count", 0) >= 1
    restored = decision.metadata.get("restored") or []
    assert any(r.get("ok") and r.get("path") == "m.txt" for r in restored)
    assert f.read_text(encoding="utf-8") == "before content"


def test_restore_disabled_by_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CODEDOGGY_SHADOW_RESTORE", "0")
    f = tmp_path / "m.txt"
    f.write_text("after", encoding="utf-8")
    traj = MutationTrajectory()
    audit = AuditServices.create(
        auditor=CriticalPathAuditor(needle="m.txt"),
        trajectory=traj,
    )
    hooks = ResidentAuditHooks(audit)
    from codedoggy.turn.types import ToolCall as TC

    call = TC(id="c1", name="search_replace", arguments={})
    mut = FileMutation(
        path="m.txt",
        tool_name="search_replace",
        call_id="c1",
        before="before content",
        after="after",
    )
    record = ToolResultRecord(
        call=call, content="ok", ok=True, mutation=mut, mutations=[mut]
    )
    decision = hooks.after_mutation(
        record, HookContext(cwd=tmp_path, round_index=0)
    )
    assert decision is not None
    assert decision.abort is True
    assert "restored" not in decision.metadata
    # File left as after — restore skipped
    assert f.read_text(encoding="utf-8") == "after"


def test_p0_create_restored_by_delete(tmp_path: Path) -> None:
    traj = MutationTrajectory()
    audit = AuditServices.create(
        auditor=CriticalPathAuditor(needle="new.txt"),
        trajectory=traj,
    )
    hooks = ResidentAuditHooks(audit)
    tools = ToolRegistryBuilder.new().finalize()
    sampler = ScriptedSampler(
        [
            SampleResult(
                content="create",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="search_replace",
                        arguments={
                            "file_path": "new.txt",
                            "old_string": "",
                            "new_string": "oops created",
                        },
                    )
                ],
            ),
            SampleResult(content="done"),
        ]
    )
    run_agent_loop(
        user_text="create",
        sampler=sampler,
        tools=tools,
        cwd=tmp_path,
        hooks=hooks,
        session=Session.create(tmp_path, goal="do not create new.txt"),
    )
    assert not (tmp_path / "new.txt").exists()
