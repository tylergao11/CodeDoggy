"""Index cache — manager/cache.rs spirit as portable JSON + query_version.

Rust cache file is ``.goto_index.bin`` (SGIX). Here: ``.goto_index.json`` with
the same semantic fields (defs/refs/aliases/file_meta/query_version).
"""

from __future__ import annotations

import json
from pathlib import Path

from codedoggy.graph.index import ScopeGraphIndex
from codedoggy.graph.types import FileMeta, QueryVersion

# cache.rs CACHE_FILE_NAME — json for portable Python (not .goto_index.bin)
CACHE_FILE_NAME = ".goto_index.json"
CACHE_FORMAT_VERSION = 2  # bump when wire format changes


def get_cache_path(root_path: Path | str) -> Path:
    return Path(root_path).resolve() / CACHE_FILE_NAME


def save_index(path: Path | str, index: ScopeGraphIndex) -> None:
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
    path.write_text(json.dumps(data), encoding="utf-8")


def load_index(path: Path | str) -> ScopeGraphIndex:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
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
        index.add_alias(alias, original)
    for p, m in (raw.get("file_meta") or {}).items():
        index.file_meta[p] = FileMeta(
            size=int(m["size"]), mtime_secs=float(m["mtime_secs"])
        )
        index._files.add(p)
    for p in raw.get("files") or []:
        index._files.add(p)
    return index
