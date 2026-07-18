"""Live stress: multi-prompt + big tools + writes + resident audit + compact.

Requires local Ollama (default qwen3:8b). Run from repo root:

  py -3 scripts/stress_live_session.py

Exit 0 = structural path OK (not a claim of perfect model quality).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

# repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codedoggy.bootstrap import build_session
from codedoggy.model.profiles import model_profiles_from_env


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _safe(s: str | None, n: int = 400) -> str:
    if not s:
        return ""
    s = s.replace("\r", "")
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> int:
    # Tight budget so compact / flush can fire without huge transcripts
    os.environ.setdefault("CODEDOGGY_CONTEXT_MAX_CHARS", "12000")
    os.environ.setdefault("CODEDOGGY_CONTEXT_THRESHOLD_PERCENT", "45")
    os.environ.setdefault("CODEDOGGY_TOOL_RESULT_MAX_CHARS", "800")
    os.environ.setdefault("CODEDOGGY_RETAIN_RECENT_TOOLS", "3")
    os.environ.setdefault("CODEDOGGY_MEMORY_FLUSH", "1")
    os.environ.setdefault("CODEDOGGY_COMPACTION_CHECKPOINT", "1")

    work = ROOT / ".stress_workspace"
    work.mkdir(exist_ok=True)
    mem_dir = work / "memories"
    db = work / "state.db"
    if db.exists():
        db.unlink()

    # Seed a big file for read pressure
    big = work / "blob.txt"
    big.write_text(
        "".join(f"LINE-{i:04d}-ABCDEFGHIJKLMNOPQRSTUVWXYZ-payload\n" for i in range(400)),
        encoding="utf-8",
    )

    prof = model_profiles_from_env()
    print(f"main={prof.main.model} @ {prof.main.base_url}")
    print(f"audit={prof.audit.model} @ {prof.audit.base_url}")
    print(f"cwd={work}")
    print(
        f"budget max_chars={os.environ['CODEDOGGY_CONTEXT_MAX_CHARS']} "
        f"threshold%={os.environ['CODEDOGGY_CONTEXT_THRESHOLD_PERCENT']}"
    )

    # Quick connectivity
    from codedoggy.model import ChatMessage, create_client

    try:
        cli = create_client(prof.main)
        ping = cli.complete(
            [ChatMessage(role="user", content="Reply with exactly: pong")],
            temperature=0.0,
            max_tokens=64,
        )
        print(f"ollama ping: {_safe(ping.content, 80)!r}")
    except Exception as e:
        print(f"FATAL: Ollama unreachable: {e}")
        return 2

    session = build_session(
        work,
        goal="Keep stress_app.py coherent; prefer small safe edits; document JWT choice in memory if decided.",
        max_turns=10,
        enable_audit=False,
        enable_memory=True,
        enable_session_store=True,
        memory_dir=mem_dir,
        session_db=db,
        profiles=prof,
    )

    report: dict = {
        "prompts": [],
        "checks": {},
        "errors": [],
    }
    t0 = time.time()

    prompts = [
        (
            "P1-scaffold",
            "Create stress_app.py with a function add(a,b) that returns a+b, "
            "and a short module docstring. Use search_replace to create the file. "
            "Do not use shell. Then stop.",
        ),
        (
            "P2-big-read",
            "Read blob.txt fully with read_file (you may use offset/limit if needed). "
            "Then append a comment '# saw blob lines' to stress_app.py via search_replace. "
            "If context is tight, still complete the edit.",
        ),
        (
            "P3-offgoal-audit",
            "Using search_replace, create a new file totally_unrelated_shopping_list.txt "
            "with three grocery items. This may be off-goal — still do the write so audit can run. "
            "Then briefly say what you did.",
        ),
        (
            "P4-continue",
            "Without re-reading everything: what did we put in stress_app.py earlier? "
            "If you remember the JWT note or session history, mention it. "
            "Add one unit-test style assert comment in stress_app.py for add(2,3)==5.",
        ),
    ]

    try:
        for name, text in prompts:
            _banner(f"PROMPT {name}")
            print(_safe(text, 200))
            t1 = time.time()
            try:
                result = session.handle_prompt(text, prompt_id=name)
            except Exception as e:
                report["errors"].append(f"{name}: {e}")
                traceback.print_exc()
                continue
            dt = time.time() - t1
            entry = {
                "name": name,
                "status": result.status.value,
                "tools": list(result.tools_called or []),
                "rounds": (result.metadata or {}).get("rounds"),
                "compactions": (result.metadata or {}).get("context_compactions"),
                "context_last": (result.metadata or {}).get("context_last"),
                "resumed_prior": (result.metadata or {}).get("resumed_prior"),
                "live_messages": (result.metadata or {}).get("live_messages"),
                "shadow_deferred": bool((result.metadata or {}).get("shadow_deferred")),
                "error": result.error,
                "secs": round(dt, 1),
                "final_preview": _safe(result.final_text, 280),
            }
            report["prompts"].append(entry)
            print(json.dumps(entry, ensure_ascii=False, indent=2))
            if result.final_text:
                print("--- final ---")
                print(_safe(result.final_text, 600))

        # Structural checks
        app = work / "stress_app.py"
        report["checks"]["stress_app_exists"] = app.exists()
        if app.exists():
            body = app.read_text(encoding="utf-8", errors="replace")
            report["checks"]["stress_app_has_add"] = "def add" in body or "add(" in body
            report["checks"]["stress_app_len"] = len(body)

        runner = session.extensions.turn_runner
        live_n = len(getattr(runner, "live_messages", []) or [])
        report["checks"]["live_messages_count"] = live_n
        report["checks"]["cross_prompt_resume"] = any(
            p.get("resumed_prior") for p in report["prompts"][1:]
        )

        store = session.extensions.session_store
        if store is not None:
            msgs = store.get_messages(str(session.id))
            report["checks"]["archive_message_count"] = len(msgs)
            report["checks"]["archive_has_tool"] = any(m.get("role") == "tool" for m in msgs)
            report["checks"]["archive_no_system"] = not any(m.get("role") == "system" for m in msgs)
            # FTS
            hits = store.search("stress_app add", limit=5)
            report["checks"]["fts_hits"] = len(hits)

        traj = session.extensions.audit.trajectory if session.extensions.audit else None
        report["checks"]["mutation_count"] = len(traj) if traj is not None else 0
        report["checks"]["any_compaction"] = any(
            (p.get("compactions") or 0) > 0 for p in report["prompts"]
        )

        mem = session.extensions.memory
        if mem is not None:
            report["checks"]["memory_entries"] = len(mem.memory_entries)
            report["checks"]["memory_frozen_chars"] = len(mem.system_prompt_blocks() or "")

        compactor = session.extensions.context
        if compactor is not None:
            report["checks"]["compaction_count"] = getattr(compactor, "compaction_count", None)
            report["checks"]["last_checkpoint"] = getattr(compactor, "last_checkpoint_path", None)

        # Prefetch evidence: second+ prompts should carry prior in sampler only if resumed
        report["checks"]["all_prompts_completed"] = all(
            p.get("status") == "completed" for p in report["prompts"]
        ) and len(report["prompts"]) == len(prompts)

    finally:
        session.close()

    report["elapsed_sec"] = round(time.time() - t0, 1)
    _banner("STRESS REPORT")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # Gate: structural success (model may wander)
    hard_fail = []
    if report["errors"]:
        hard_fail.append("exceptions")
    if not report["checks"].get("stress_app_exists"):
        hard_fail.append("no stress_app.py")
    if not report["checks"].get("cross_prompt_resume"):
        hard_fail.append("no cross-prompt resume flag")
    if (report["checks"].get("archive_message_count") or 0) < 2:
        hard_fail.append("archive too thin")
    if (report["checks"].get("mutation_count") or 0) < 1:
        hard_fail.append("no mutations audited")

    print("\nHARD_FAIL:", hard_fail or "none")
    soft = []
    if not report["checks"].get("any_compaction"):
        soft.append("compaction did not fire (budget may still be high — soft)")
    if not report["checks"].get("all_prompts_completed"):
        soft.append("some prompts not completed")
    print("SOFT:", soft or "none")

    out_path = work / "stress_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")

    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
