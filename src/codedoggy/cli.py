"""CodeDoggy CLI: parallel-first task cockpit with a plain fallback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from codedoggy.bootstrap import build_session
from codedoggy.model.profiles import model_profiles_from_env


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="codedoggy",
        description="CodeDoggy parallel coding agent",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Initial boss task (the TUI stays open after it finishes)",
    )
    parser.add_argument("--cwd", default=".", help="Workspace root")
    parser.add_argument("--goal", default=None, help="Session-level objective")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Optional sampling-round limit (default: unlimited)",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable curated MEMORY.md / USER.md",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Run one prompt with machine-friendly text output instead of the TUI",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Build the session, print its wiring, and exit",
    )
    args = parser.parse_args(argv)

    cwd = Path(args.cwd).resolve()
    profiles = model_profiles_from_env()
    session = build_session(
        cwd,
        goal=args.goal,
        max_turns=args.max_turns,
        enable_memory=not args.no_memory,
        profiles=profiles,
    )

    try:
        if args.smoke:
            _print_wiring(session, profiles)
            return
        interactive = not args.plain and sys.stdin.isatty() and sys.stdout.isatty()
        if interactive:
            from codedoggy.tui import run_tui

            run_tui(session, initial_prompt=args.prompt)
            return
        if args.prompt:
            _run_plain(session, args.prompt, profiles)
            return
        _print_wiring(session, profiles)
        print("No TTY detected. Run `codedoggy` in a terminal or pass a prompt.")
    finally:
        session.close()


def _run_plain(session: object, prompt: str, profiles: object) -> None:
    _print_models(profiles)
    result = session.handle_prompt(prompt)  # type: ignore[attr-defined]
    print("status:", result.status.value)
    if result.final_text:
        print(result.final_text)
    if result.error:
        print("error:", result.error)


def _print_models(profiles: object) -> None:
    main_profile = profiles.main  # type: ignore[attr-defined]
    print(
        f"CodeDoggy model: {main_profile.provider}/{main_profile.model} "
        f"@ {main_profile.base_url}"
    )


def _print_wiring(session: object, profiles: object) -> None:
    _print_models(profiles)
    print(repr(session))
    extensions = session.extensions  # type: ignore[attr-defined]
    kernel = getattr(extensions, "kernel", None)
    print(
        "extensions: tools=",
        bool(extensions.tools),
        "parallel=",
        bool(getattr(kernel, "subagent_coordinator", None)),
        "memory=",
        bool(extensions.memory),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
