"""CLI entry: model-brained session (main + audit on Ollama by default)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from codedoggy.bootstrap import build_session
from codedoggy.model.profiles import model_profiles_from_env


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="codedoggy", description="CodeDoggy coding agent")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="User prompt (if omitted, smoke-print session wiring only)",
    )
    parser.add_argument("--cwd", default=".", help="Workspace root")
    parser.add_argument("--goal", default=None, help="Session-level goal for audit")
    parser.add_argument("--max-turns", type=int, default=16, help="Max sampling rounds")
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Disable resident model auditor",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable curated MEMORY.md / USER.md",
    )
    args = parser.parse_args(argv)

    cwd = Path(args.cwd).resolve()
    profiles = model_profiles_from_env()
    print(
        f"CodeDoggy models: main={profiles.main.provider}/{profiles.main.model} "
        f"@ {profiles.main.base_url}"
    )
    print(
        f"                 audit={profiles.audit.provider}/{profiles.audit.model} "
        f"@ {profiles.audit.base_url}"
    )

    session = build_session(
        cwd,
        goal=args.goal,
        max_turns=args.max_turns,
        enable_audit=not args.no_audit,
        enable_memory=not args.no_memory,
        profiles=profiles,
    )

    try:
        if not args.prompt:
            print(repr(session))
            print("extensions: tools=", bool(session.extensions.tools),
                  "audit=", bool(session.extensions.audit),
                  "memory=", bool(session.extensions.memory))
            print("No prompt given — wiring OK. Pass a prompt to run.")
            return

        print(f"goal={session.goal!r}")
        print(f"prompt={args.prompt!r}")
        result = session.handle_prompt(args.prompt)
        print("---")
        print("status:", result.status.value)
        if result.final_text:
            print("final:", result.final_text)
        if result.error:
            print("error:", result.error)
        print("tools_called:", result.tools_called)
        print("metadata:", result.metadata)
        if session.extensions.audit is not None:
            traj = session.extensions.audit.trajectory
            print(f"mutations logged: {len(traj)}")
    finally:
        session.close()


if __name__ == "__main__":
    main(sys.argv[1:])
