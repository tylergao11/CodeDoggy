"""IndexBuilder — port of ``xai-codebase-graph`` manager/builder.rs.

Source strengths we implement (not API cosplay):

1. **Two-phase build** (``build_fast``):
   - Phase 1: parallel parse/extract into lightweight ``FileSymbols``
   - Phase 2: sequential merge into one ``ScopeGraphIndex``
2. **Bounded merge batches** (``build_batch_size``): cap peak memory
3. **Chunked parallel work** (``chunk_size``): locality / thread pool efficiency
4. **Thread count** default N-1 cores (``with_threads`` is real, not a no-op)
5. Relative paths, size/binary gates, git ls-files then walk

Language extract: tree-sitter queries (crate path). Pipeline + query_version
match the crate.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from codedoggy.graph.index import ScopeGraphIndex
from codedoggy.graph.languages import FileExtract, LanguageRegistry
from codedoggy.graph.types import FileMeta, SymbolAlias, SymbolOccurrence

logger = logging.getLogger(__name__)

# index_manager.rs
MAX_INDEXABLE_FILE_SIZE: int = 5 * 1024 * 1024

_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".codedoggy",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        ".venv",
        "venv",
        ".idea",
        ".vscode",
    }
)


class IndexError(Exception):
    """Mirror builder IndexError."""


def _default_num_threads() -> int:
    """builder.rs: N-1 cores, minimum 1."""
    n = os.cpu_count() or 2
    return max(1, n - 1)


@dataclass
class FileSymbols:
    """Lightweight per-file extract — builder.rs ``FileSymbols``.

    Lightweight per-file extract; merge phase owns the global index.
    """

    path: str  # relative to root
    definitions: list[SymbolOccurrence] = field(default_factory=list)
    references: list[SymbolOccurrence] = field(default_factory=list)
    aliases: list[SymbolAlias] = field(default_factory=list)
    file_meta: FileMeta | None = None


class IndexBuilder:
    """Build a ``ScopeGraphIndex`` — parallel extract, sequential merge."""

    def __init__(
        self,
        *,
        registry: LanguageRegistry | None = None,
        respect_gitignore: bool = True,
        skip_hidden: bool = True,
        num_threads: int | None = None,
        chunk_size: int = 100,
        build_batch_size: int = 5_000,
    ) -> None:
        self.registry = registry or LanguageRegistry()
        self.respect_gitignore = respect_gitignore
        self.skip_hidden = skip_hidden
        self.num_threads = (
            num_threads if num_threads is not None else _default_num_threads()
        )
        self.chunk_size = max(1, int(chunk_size))
        # Clamp batch size against chunk_size (builder.rs order-independent clamp)
        self.build_batch_size = max(self.chunk_size, int(build_batch_size))

    def with_threads(self, count: int) -> IndexBuilder:
        """builder.rs ``with_threads`` — real pool size, not a stub."""
        self.num_threads = max(1, int(count))
        return self

    def with_chunk_size(self, size: int) -> IndexBuilder:
        self.chunk_size = max(1, int(size))
        self.build_batch_size = max(self.chunk_size, self.build_batch_size)
        return self

    def with_build_batch_size(self, size: int) -> IndexBuilder:
        self.build_batch_size = max(self.chunk_size, int(size))
        return self

    def build(self, root_path: str | Path) -> ScopeGraphIndex:
        root = Path(root_path).resolve()
        if not root.is_dir():
            raise IndexError(f"not a directory: {root}")
        files = self._collect_files(root)
        if not files:
            index = ScopeGraphIndex()
            index.set_query_version(self.registry.compute_query_hash())
            return index
        return self._build_fast(root, files)

    def _build_fast(self, root: Path, file_paths: list[Path]) -> ScopeGraphIndex:
        """Two-phase: parallel extract → sequential merge (builder.rs build_fast)."""
        index = ScopeGraphIndex()
        workers = self.num_threads
        batch_size = self.build_batch_size

        # Process in bounded merge-batches (peak memory O(batch) not O(all files))
        for batch_start in range(0, len(file_paths), batch_size):
            batch = file_paths[batch_start : batch_start + batch_size]
            batch_symbols = self._parallel_extract(root, batch, workers)
            # Phase 2: sequential merge into single index (one "interner" owner)
            for file_syms in batch_symbols:
                if file_syms is None:
                    continue
                path_str = file_syms.path
                index.add_definitions(path_str, file_syms.definitions)
                index.add_references(path_str, file_syms.references)
                index.add_aliases(path_str, file_syms.aliases)
                if file_syms.file_meta is not None:
                    index.set_file_meta(path_str, file_syms.file_meta)
            # batch_symbols dropped → free before next batch

        # builder.rs: set query_version so cache invalidates when queries change
        index.set_query_version(self.registry.compute_query_hash())
        return index

    def _parallel_extract(
        self, root: Path, batch: list[Path], workers: int
    ) -> list[FileSymbols | None]:
        """Phase 1: parse/extract files in a thread pool (rayon spirit)."""
        if not batch:
            return []
        # Single-threaded if only one file or workers==1
        if workers <= 1 or len(batch) == 1:
            return [self._process_file_fast(p, root) for p in batch]

        results: list[FileSymbols | None] = [None] * len(batch)
        # Chunk for locality like par_chunks(chunk_size)
        chunks: list[list[tuple[int, Path]]] = []
        for i in range(0, len(batch), self.chunk_size):
            chunk = [(i + j, batch[i + j]) for j in range(min(self.chunk_size, len(batch) - i))]
            chunks.append(chunk)

        def _run_one(idx: int, path: Path) -> tuple[int, FileSymbols | None]:
            try:
                return idx, self._process_file_fast(path, root)
            except Exception:  # noqa: BLE001 — per-file isolation
                logger.exception("extract failed for %s", path)
                return idx, None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = []
            for ch in chunks:
                for idx, path in ch:
                    futs.append(pool.submit(_run_one, idx, path))
            for fut in as_completed(futs):
                try:
                    idx, syms = fut.result()
                    results[idx] = syms
                except Exception:  # noqa: BLE001
                    continue
        return results

    def _process_file_fast(self, path: Path, root: Path) -> FileSymbols | None:
        """process_file_fast — tree-sitter extract for one file."""
        if not self.registry.is_supported(path):
            return None
        try:
            st = path.stat()
        except OSError:
            return None
        if st.st_size == 0 or st.st_size > MAX_INDEXABLE_FILE_SIZE:
            return None
        # Prefix binary check (8KB) then full read
        try:
            with path.open("rb") as f:
                head = f.read(8000)
                if b"\x00" in head:
                    return None
                rest = f.read()
            raw = head + rest
        except OSError:
            return None
        try:
            source = raw.decode("utf-8")
        except UnicodeDecodeError:
            source = raw.decode("utf-8", errors="replace")

        try:
            rel = path.resolve().relative_to(root).as_posix()
        except ValueError:
            rel = path.name

        extracted: FileExtract = self.registry.extract(path, source)
        return FileSymbols(
            path=rel,
            definitions=list(extracted.definitions),
            references=list(extracted.references),
            aliases=list(extracted.aliases),
            file_meta=FileMeta.from_path(path),
        )

    def _collect_files(self, root: Path) -> list[Path]:
        """Collect indexable source files.

        When ``respect_gitignore`` is True:
          1. Prefer ``git ls-files --cached --others --exclude-standard``
             (tracked + untracked, minus ignore rules).
          2. Else walk the tree applying ``.gitignore`` patterns.
        When False: plain walk (skip only known junk dirs / hidden).
        """
        if self.respect_gitignore:
            git_files = self._collect_files_git(root)
            if git_files is not None:
                return git_files
            return self._collect_files_walk(root, use_gitignore=True)
        return self._collect_files_walk(root, use_gitignore=False)

    def _collect_files_git(self, root: Path) -> list[Path] | None:
        """Return supported paths via git, or None if git unavailable / not a repo."""
        try:
            import subprocess

            r = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "ls-files",
                    "-z",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                ],
                capture_output=True,
                timeout=60,
                check=False,
            )
            if r.returncode != 0:
                return None
            out: list[Path] = []
            for raw in r.stdout.split(b"\0"):
                if not raw:
                    continue
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError:
                    line = raw.decode("utf-8", errors="replace")
                line = line.strip()
                if not line:
                    continue
                if self.registry.is_supported(line):
                    out.append(root / line)
            return out
        except (OSError, subprocess.TimeoutExpired):
            return None
        except Exception:
            return None

    def _collect_files_walk(
        self, root: Path, *, use_gitignore: bool = False
    ) -> list[Path]:
        """Filesystem walk; optionally honor .gitignore files found along the way."""
        out: list[Path] = []
        # dir_rel (posix, "" for root) -> stacked ignore matchers from root→dir
        ignore_stack: dict[str, list[_GitIgnoreFile]] = {"": []}
        if use_gitignore:
            root_gi = root / ".gitignore"
            if root_gi.is_file():
                ignore_stack[""] = [_GitIgnoreFile.load(root_gi, base_rel="")]

        for dirpath, dirnames, filenames in os.walk(root):
            dir_path = Path(dirpath)
            try:
                rel_dir = dir_path.resolve().relative_to(root).as_posix()
            except ValueError:
                rel_dir = ""
            if rel_dir == ".":
                rel_dir = ""

            if use_gitignore and rel_dir:
                parent_key = str(Path(rel_dir).parent.as_posix())
                if parent_key == ".":
                    parent_key = ""
                inherited = list(ignore_stack.get(parent_key, ignore_stack[""]))
                local_gi = dir_path / ".gitignore"
                if local_gi.is_file():
                    inherited = inherited + [
                        _GitIgnoreFile.load(local_gi, base_rel=rel_dir)
                    ]
                ignore_stack[rel_dir] = inherited

            matchers = ignore_stack.get(rel_dir, ignore_stack[""]) if use_gitignore else []

            keep: list[str] = []
            for d in dirnames:
                if d in _SKIP_DIR_NAMES:
                    continue
                if self.skip_hidden and d.startswith("."):
                    continue
                child_rel = f"{rel_dir}/{d}" if rel_dir else d
                if use_gitignore and _is_ignored(child_rel, is_dir=True, matchers=matchers):
                    continue
                keep.append(d)
            dirnames[:] = keep

            for name in filenames:
                if self.skip_hidden and name.startswith("."):
                    continue
                child_rel = f"{rel_dir}/{name}" if rel_dir else name
                if use_gitignore and _is_ignored(
                    child_rel, is_dir=False, matchers=matchers
                ):
                    continue
                path = dir_path / name
                if self.registry.is_supported(path):
                    out.append(path)
        return out


@dataclass
class _GitIgnoreRule:
    """One .gitignore line (simplified: glob, optional dir-only, optional negation)."""

    pattern: str
    negated: bool = False
    dir_only: bool = False
    anchored: bool = False  # leading slash → relative to this ignore file base


@dataclass
class _GitIgnoreFile:
    base_rel: str  # directory containing this .gitignore, relative to root
    rules: list[_GitIgnoreRule]

    @classmethod
    def load(cls, path: Path, *, base_rel: str) -> _GitIgnoreFile:
        rules: list[_GitIgnoreRule] = []
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return cls(base_rel=base_rel, rules=rules)
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            negated = False
            if line.startswith("!"):
                negated = True
                line = line[1:]
            dir_only = line.endswith("/")
            if dir_only:
                line = line.rstrip("/")
            anchored = line.startswith("/")
            if anchored:
                line = line.lstrip("/")
            if not line:
                continue
            rules.append(
                _GitIgnoreRule(
                    pattern=line,
                    negated=negated,
                    dir_only=dir_only,
                    anchored=anchored,
                )
            )
        return cls(base_rel=base_rel, rules=rules)

    def match(self, rel_path: str, *, is_dir: bool) -> bool | None:
        """Return True if ignored, False if explicitly un-ignored, None if no match."""
        # Path relative to this ignore file's directory
        if self.base_rel:
            prefix = self.base_rel + "/"
            if not rel_path.startswith(prefix) and rel_path != self.base_rel:
                return None
            local = (
                rel_path[len(prefix) :]
                if rel_path.startswith(prefix)
                else ""
            )
        else:
            local = rel_path
        if not local and rel_path != self.base_rel:
            return None

        result: bool | None = None
        name = local.rsplit("/", 1)[-1] if local else ""
        for rule in self.rules:
            if rule.dir_only and not is_dir:
                continue
            pat = rule.pattern
            matched = False
            if rule.anchored or "/" in pat.rstrip("/"):
                # Match against full path relative to ignore base
                matched = fnmatch.fnmatch(local, pat) or fnmatch.fnmatch(
                    local, pat.rstrip("/")
                )
                # Also allow ** style suffix: pat matches any depth if unanchored w/ slash
                if not matched and not rule.anchored and "/" not in pat:
                    matched = fnmatch.fnmatch(name, pat)
            else:
                # Basename match at any level under this base
                matched = fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(local, pat)
                if not matched and "/" in local:
                    # e.g. pattern "*.pyc" matches nested files
                    matched = fnmatch.fnmatch(name, pat)
            if matched:
                result = not rule.negated
        return result


def _is_ignored(
    rel_path: str, *, is_dir: bool, matchers: list[_GitIgnoreFile]
) -> bool:
    """Last matching rule across stacked matchers wins (gitignore semantics)."""
    ignored = False
    for m in matchers:
        hit = m.match(rel_path, is_dir=is_dir)
        if hit is not None:
            ignored = hit
    return ignored
