"""list_dir tree + budget BFS — source port from Grok.

Ported from:
  grok-build/.../implementations/grok_build/list_dir/mod.rs

Functions / constants mapped 1:1 where practical:
  DEFAULT_MAX_OUTPUT_CHARS, TOP_K_EXTENSIONS, MAX_GLOBAL_ITEMS, MAX_SEED_ITEMS
  DirAccum, DirNode, seed_depth1_children, build_tree*, budget_expand,
  render_truncated_root, compute_display_path, root_truncation_notice

Walk uses pathlib (no `ignore` crate). Dotfiles always hidden.
Gitignore: simple path/name match from cwd `.gitignore` when respect_gitignore.
"""

from __future__ import annotations

import fnmatch
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

# ── constants (list_dir/mod.rs) ──────────────────────────────────────
DEFAULT_MAX_OUTPUT_CHARS: int = 10_000
TOP_K_EXTENSIONS: int = 3
MAX_GLOBAL_ITEMS: int = 100_000
MAX_SEED_ITEMS: int = 100_000
assert MAX_SEED_ITEMS == MAX_GLOBAL_ITEMS

ROOT_TRUNCATION_NOTICE_FALLBACK: str = (
    "    ...\n\n"
    "Note: this directory is too large to list fully. Try list_dir on a narrower path, or "
    "use grep / bash."
)

GLOBAL_CUTOFF_NOTICE = (
    f"\nNote: there are more than {MAX_GLOBAL_ITEMS} items in the directory, "
    "so not all files may be shown.\n"
)


def compute_display_path(display_base: Path, target: str) -> Path:
    """Special-case ``.`` / ``./`` / empty so header has no ugly ``/./``.

    Grok: ``target.trim().trim_start_matches("./")`` then empty or ``.`` → base.
    """
    t = target.strip()
    while t.startswith("./"):
        t = t[2:]
    if not t or t == ".":
        return Path(display_base)
    return Path(display_base) / t


def root_truncation_notice() -> str:
    return ROOT_TRUNCATION_NOTICE_FALLBACK


# ── DirAccum ─────────────────────────────────────────────────────────
@dataclass
class DirAccum:
    total_files: int = 0
    by_ext: dict[str, int] = field(default_factory=dict)

    def add_ext(self, ext: str) -> None:
        self.total_files += 1
        self.by_ext[ext] = self.by_ext.get(ext, 0) + 1

    def to_summary(self, top_n: int) -> str:
        if not self.by_ext:
            return ""
        items = sorted(self.by_ext.items(), key=lambda kv: (-kv[1], kv[0]))
        parts: list[str] = []
        top_sum = 0
        for ext, count in items[:top_n]:
            top_sum += count
            if ext == "no-ext":
                parts.append(f"{count} *no-ext")
            else:
                parts.append(f"{count} *.{ext}")
        ellipsis = ", ..." if top_sum < self.total_files else ""
        file_word = "file" if self.total_files == 1 else "files"
        return f"[{self.total_files} {file_word} in subtree: {', '.join(parts)}{ellipsis}]"


def ext_key_from_path(name: str) -> str:
    p = Path(name)
    if p.suffix:
        return p.suffix.lstrip(".").lower()
    return "no-ext"


# ── DirNode ──────────────────────────────────────────────────────────
@dataclass
class DirNode:
    depth: int
    files: list[str] = field(default_factory=list)
    subdirs: list[str] = field(default_factory=list)
    children: dict[str, "DirNode"] = field(default_factory=dict)
    subtree: DirAccum = field(default_factory=DirAccum)
    is_expanded: bool = False

    def add_item(self, rel_parts: list[str], is_dir: bool) -> None:
        if not rel_parts:
            return
        if len(rel_parts) == 1:
            name = rel_parts[0]
            if is_dir:
                key = f"{name}/"
                if key not in self.children:
                    self.children[key] = DirNode(depth=self.depth + 1)
                    self.subdirs.append(key)
            else:
                ext = ext_key_from_path(name)
                self.files.append(name)
                self.subtree.add_ext(ext)
            return
        subdir = rel_parts[0]
        key = f"{subdir}/"
        if key not in self.children:
            self.children[key] = DirNode(depth=self.depth + 1)
            self.subdirs.append(key)
        child = self.children[key]
        child.add_item(rel_parts[1:], is_dir)
        if not is_dir:
            ext = ext_key_from_path(rel_parts[-1])
            self.subtree.add_ext(ext)

    def sort_recursive(self) -> None:
        self.files.sort(key=lambda a: a.lower())
        self.subdirs.sort(key=lambda a: a.lower())
        for child in self.children.values():
            child.sort_recursive()

    def all_subitems_sorted(self) -> list[str]:
        items = list(self.files) + list(self.subdirs)
        items.sort(key=lambda a: a.lower())
        return items

    def subitem_line(self, name: str) -> str:
        indent = "  " * (self.depth + 1)
        return f"{indent}- {name}"

    def summary_str(self, top_k: int) -> str:
        return self.subtree.to_summary(top_k)

    def summary_char_cost(self, top_k: int) -> int:
        s = self.summary_str(top_k)
        if not s:
            return 0
        return (self.depth + 1) * 2 + len(s) + 1

    def render_expanded(self, top_k: int) -> str:
        out: list[str] = []
        for name in self.all_subitems_sorted():
            out.append(self.subitem_line(name))
            out.append("\n")
            child = self.children.get(name)
            if child is not None:
                out.append(child.render_subtree(top_k))
        return "".join(out)

    def render_subtree(self, top_k: int) -> str:
        if self.is_expanded:
            return self.render_expanded(top_k)
        summary = self.summary_str(top_k)
        if not summary:
            return ""
        indent = "  " * (self.depth + 1)
        return f"{indent}{summary}\n"


# ── walk / seed / build ──────────────────────────────────────────────
def _is_hidden_name(name: str) -> bool:
    return name.startswith(".")


def load_gitignore_patterns(root: Path) -> list[str]:
    """Minimal gitignore patterns (not full git rules)."""
    patterns: list[str] = []
    gi = root / ".gitignore"
    if not gi.is_file():
        return patterns
    try:
        text = gi.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return patterns
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("!"):
            continue
        patterns.append(s.rstrip("/"))
    return patterns


def _is_ignored(rel_posix: str, name: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if not pat:
            continue
        if name == pat or name == pat.rstrip("/"):
            return True
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel_posix, pat):
            return True
        if fnmatch.fnmatch(rel_posix, pat.rstrip("/") + "/*"):
            return True
        if rel_posix == pat or rel_posix.startswith(pat.rstrip("/") + "/"):
            return True
    return False


def seed_depth1_children(
    root: Path,
    root_node: DirNode,
    *,
    ignore_patterns: list[str],
    max_seed: int = MAX_SEED_ITEMS,
) -> bool:
    """Seed depth-1. Returns True if seed hit max_seed (more exist)."""
    try:
        children = list(root.iterdir())
    except OSError:
        return False
    seed_count = 0
    # stable-ish: sort by name for determinism (Grok walk order may differ)
    for child in sorted(children, key=lambda p: p.name.lower()):
        name = child.name
        if _is_hidden_name(name):
            continue
        if _is_ignored(name, name, ignore_patterns):
            continue
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        seed_count += 1
        if seed_count > max_seed:
            return True
        root_node.add_item([name], is_dir)
    return False


def _walk_depth_ge2(
    root: Path,
    root_node: DirNode,
    *,
    ignore_patterns: list[str],
    max_items: int,
) -> bool:
    """BFS walk depth ≥ 2. Returns walk_truncated."""
    item_count = 0
    # queue of (abs_path, rel_parts)
    q: deque[tuple[Path, list[str]]] = deque()
    try:
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if _is_hidden_name(child.name):
                continue
            if _is_ignored(child.name, child.name, ignore_patterns):
                continue
            try:
                if child.is_dir():
                    q.append((child, [child.name]))
            except OSError:
                continue
    except OSError:
        return False

    while q:
        dir_path, rel_parts = q.popleft()
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for entry in entries:
            name = entry.name
            if _is_hidden_name(name):
                continue
            rel = "/".join([*rel_parts, name])
            if _is_ignored(rel, name, ignore_patterns):
                continue
            try:
                is_dir = entry.is_dir()
            except OSError:
                continue
            # depth of this entry = len(rel_parts)+1; only count depth ≥ 2
            depth = len(rel_parts) + 1
            if depth <= 1:
                continue
            item_count += 1
            if item_count > max_items:
                return True
            parts = [*rel_parts, name]
            root_node.add_item(parts, is_dir)
            if is_dir:
                q.append((entry, parts))
    return False


def build_tree(root: Path, *, respect_gitignore: bool = True) -> tuple[DirNode, bool]:
    return build_tree_with_limit(root, respect_gitignore=respect_gitignore, max_items=MAX_GLOBAL_ITEMS)


def build_tree_with_limit(
    root: Path,
    *,
    respect_gitignore: bool = True,
    max_items: int = MAX_GLOBAL_ITEMS,
) -> tuple[DirNode, bool]:
    root_node = DirNode(depth=0)
    ignore_patterns = load_gitignore_patterns(root) if respect_gitignore else []
    # Also load from parent workspace roots? Grok uses WalkBuilder from root only.
    seed_truncated = seed_depth1_children(
        root, root_node, ignore_patterns=ignore_patterns, max_seed=MAX_SEED_ITEMS
    )
    walk_truncated = _walk_depth_ge2(
        root, root_node, ignore_patterns=ignore_patterns, max_items=max_items
    )
    root_node.sort_recursive()
    return root_node, seed_truncated or walk_truncated


def navigate_mut(root: DirNode, path: list[str]) -> DirNode | None:
    node = root
    for key in path:
        child = node.children.get(key)
        if child is None:
            return None
        node = child
    return node


def render_truncated_root(
    root: DirNode,
    max_chars: int,
    top_k: int,
    notice: str,
) -> str:
    out = ""
    remaining = max_chars
    child_summary_indent = "  " * (root.depth + 2)
    for name in root.all_subitems_sorted():
        chunk = root.subitem_line(name) + "\n"
        child = root.children.get(name)
        if child is not None:
            summary = child.summary_str(top_k)
            if summary:
                chunk += f"{child_summary_indent}{summary}\n"
        if len(chunk) > remaining:
            break
        out += chunk
        remaining -= len(chunk)
    out += notice
    return out


def budget_expand(
    root: DirNode,
    max_chars: int,
    top_k: int = TOP_K_EXTENSIONS,
    truncated: bool = False,
    truncation_notice: str | None = None,
) -> str:
    """BFS-expand directories within character budget; return rendered body."""
    notice = truncation_notice if truncation_notice is not None else root_truncation_notice()
    cutoff_msg = GLOBAL_CUTOFF_NOTICE if truncated else ""
    if not root.files and not root.subdirs:
        return cutoff_msg

    root.is_expanded = True
    root_expanded = root.render_expanded(top_k)
    if len(root_expanded) > max_chars:
        out = render_truncated_root(root, max_chars, top_k, notice)
        out += cutoff_msg
        return out

    remaining = max_chars - len(root_expanded)
    queue: deque[list[str]] = deque()
    for name in root.subdirs:
        queue.append([name])

    while queue:
        node_path = queue.popleft()
        node = navigate_mut(root, node_path)
        if node is None:
            continue
        node.is_expanded = True
        expanded = node.render_expanded(top_k)
        summary_cost = node.summary_char_cost(top_k)
        if len(expanded) > remaining + summary_cost:
            node.is_expanded = False
            continue
        remaining += summary_cost
        remaining -= len(expanded)
        for child_name in list(node.subdirs):
            child_path = [*node_path, child_name]
            queue.append(child_path)

    out = root.render_expanded(top_k)
    out += cutoff_msg
    return out


def render_list_dir(
    path: Path,
    display_path: Path | str,
    *,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    respect_gitignore: bool = True,
) -> str:
    """Full list_dir body with header (non-legacy path)."""
    tree, truncated = build_tree(path, respect_gitignore=respect_gitignore)
    body = budget_expand(
        tree,
        max_output_chars,
        TOP_K_EXTENSIONS,
        truncated,
        root_truncation_notice(),
    )
    trimmed = body.rstrip("\n") if body.endswith("\n") else body
    # Grok: body.trim_end() then format
    trimmed = body.rstrip()
    return f"- {display_path}/\n{trimmed}" if trimmed else f"- {display_path}/"
