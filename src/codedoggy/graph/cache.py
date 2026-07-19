"""Index cache — manager/cache.rs spirit as portable JSON + query_version.

Grok keeps codebase indexes outside the workspace under
``~/.grok/indexes/{encoded_cwd}``.  CodeDoggy mirrors that ownership under
``CODEDOGGY_HOME`` (or ``~/.codedoggy``) so a read-only graph query never
creates a file in the repository being inspected.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

from codedoggy.graph.index import ScopeGraphIndex
from codedoggy.graph.types import FileMeta, QueryVersion

# cache.rs CACHE_FILE_NAME — json for portable Python (not .goto_index.bin)
CACHE_FILE_NAME = ".goto_index.json"
CACHE_FORMAT_VERSION = 2  # bump when wire format changes


def get_cache_path(root_path: Path | str) -> Path:
    """Return the profile-owned cache path for a canonical workspace root.

    Source analogue:
    ``xai-grok-workspace/file_system/codebase_index.rs::get_index_cache_path``.
    The public function signature is unchanged; only cache ownership moves
    from the repository to the CodeDoggy profile.
    """
    root = Path(root_path).expanduser().resolve()
    home_env = os.environ.get("CODEDOGGY_HOME", "").strip()
    home = (
        Path(home_env).expanduser().resolve()
        if home_env
        else (Path.home() / ".codedoggy").resolve()
    )
    encoded = quote(str(root), safe="")
    return home / "indexes" / encoded / CACHE_FILE_NAME


def save_index(path: Path | str, index: ScopeGraphIndex) -> None:
    """Atomically replace a cache snapshot.

    The temporary file lives beside the destination so ``os.replace`` stays
    on one filesystem.  Readers therefore observe either the previous complete
    JSON snapshot or the new complete snapshot, never a truncated write.
    """
    path = Path(path)
    data = {
        "format_version": CACHE_FORMAT_VERSION,
        "query_version": index.query_version.to_wire(),
        "definitions": {k: list(v) for k, v in index.definitions.items()},
        "references": {k: list(v) for k, v in index.references.items()},
        "aliases": dict(index.aliases),
        "file_meta": {
            k: {"size": m.size, "mtime_secs": m.mtime_secs}
            for k, m in index.file_meta.items()
        },
        "files": list(index._files),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class CacheFormatError(ValueError):
    """Raised when on-disk cache format_version does not match this build."""


def load_index(path: Path | str) -> ScopeGraphIndex:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    ver = raw.get("format_version")
    if ver != CACHE_FORMAT_VERSION:
        raise CacheFormatError(
            f"cache format_version mismatch: got {ver!r}, want {CACHE_FORMAT_VERSION}"
        )
    index = ScopeGraphIndex()
    index.query_version = QueryVersion.from_wire(raw.get("query_version"))
    for name, locs in (raw.get("definitions") or {}).items():
        for item in locs:
            p, line = item[0], int(item[1])
            index.definitions[name].append((p, line))
            index._files.add(p)
    for name, locs in (raw.get("references") or {}).items():
        for item in locs:
            p, line = item[0], int(item[1])
            index.references[name].append((p, line))
            index._files.add(p)
    for alias, original in (raw.get("aliases") or {}).items():
        # Path-scoped keys ("rel/path::alias") are stored as-is for remove_file cleanup.
        if "::" in alias and not alias.startswith("::"):
            # Restore path-scoped + bare without double-scoping
            path_part, bare = alias.split("::", 1)
            if path_part and bare:
                index.add_alias(bare, original, path=path_part)
                continue
        index.add_alias(alias, original)
    for p, m in (raw.get("file_meta") or {}).items():
        index.file_meta[p] = FileMeta(
            size=int(m["size"]), mtime_secs=float(m["mtime_secs"])
        )
        index._files.add(p)
    for p in raw.get("files") or []:
        index._files.add(p)
    return index
