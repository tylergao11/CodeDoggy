"""Path resolution for model-supplied paths."""

from __future__ import annotations

from pathlib import Path


def resolve_model_path(cwd: Path, target: str) -> Path:
    """Resolve relative paths against cwd; keep absolute paths as-is (then resolve)."""
    p = Path(target)
    if p.is_absolute():
        return p.resolve()
    return (cwd / p).resolve()


from codedoggy.tools.defaults import PATH_COMPONENT_NAME_MAX

NAME_MAX = PATH_COMPONENT_NAME_MAX


def validate_path_component_lengths(file_path: str) -> str | None:
    """Return an error message if any path component exceeds NAME_MAX."""
    for part in Path(file_path).parts:
        if part in (".", "..", "/", "\\"):
            continue
        # Windows drive like "C:"
        if len(part) == 2 and part[1] == ":":
            continue
        if len(part) > NAME_MAX:
            return (
                f"Error: file name exceeds the {NAME_MAX}-character limit "
                f"({len(part)} characters). Please use a shorter file name."
            )
    return None
