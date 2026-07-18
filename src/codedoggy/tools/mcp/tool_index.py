"""BM25 ToolSearchIndex — source-level port of Grok shell tool_index.rs.

Ported from:
  crates/codegen/xai-grok-shell/src/session/tool_index.rs
    split_identifier, normalize_query, ToolMetadata.to_document
    Bm25ToolSearchIndex::search_snapshot (exact-match fast path + BM25 rank)
  crates/codegen/xai-grok-tools/src/types/tool_index.rs
    ToolSearchResult, SearchSnapshot, ServerSummary, ToolSearchIndex trait

BM25 ranking is a pure-Python Okapi BM25 (k1=1.5, b=0.75) over the same
document construction as Grok — no Rust bm25 crate dependency.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

from codedoggy.tools.mcp.types import (
    SearchSnapshot,
    ServerSummary,
    ToolIndex,
    ToolSearchIndex,
    ToolSearchResult,
    unwrap_tool_index,
)


@dataclass
class ToolMetadata:
    """Grok shell ``ToolMetadata`` — one MCP tool document."""

    qualified_name: str
    server_name: str
    tool_name: str
    description: str = ""
    parameters: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)

    def to_document(self) -> str:
        """Grok ``ToolMetadata::to_document``."""
        params = " ".join(self.parameters)
        doc = f"{self.server_name} {self.tool_name} {self.description} {params}"
        extra = " ".join(
            [
                *split_identifier(self.server_name),
                *split_identifier(self.tool_name),
                *[w for p in self.parameters for w in split_identifier(p)],
            ]
        )
        return f"{doc} {extra}".strip()


@dataclass
class ServerMetadata:
    """Grok shell ``ServerMetadata``."""

    name: str
    description: str | None = None


@dataclass
class ToolMetadataSnapshot:
    """Grok shell ``ToolMetadataSnapshot``."""

    tools: list[ToolMetadata] = field(default_factory=list)
    servers: list[ServerMetadata] = field(default_factory=list)
    mcp_initialized: bool = True


def split_identifier(s: str) -> list[str]:
    """Grok ``split_identifier`` — __, _, -, camelCase."""
    words: list[str] = []
    for part in s.replace("__", " ").replace("_", " ").replace("-", " ").split():
        if not part:
            continue
        # camelCase split
        buf = [part[0]] if part else []
        for i in range(1, len(part)):
            prev, ch = part[i - 1], part[i]
            if prev.islower() and ch.isupper():
                words.append("".join(buf))
                buf = [ch]
            else:
                buf.append(ch)
        if buf:
            words.append("".join(buf))
    return words


def normalize_query(query: str) -> str:
    """Grok ``normalize_query``."""
    needs = (
        "__" in query
        or "_" in query
        or "-" in query
        or any(
            query[i - 1].islower() and query[i].isupper()
            for i in range(1, len(query))
        )
    )
    if not needs:
        return query
    extra = [w for tok in query.split() for w in split_identifier(tok)]
    if not extra:
        return query
    return f"{query} {' '.join(extra)}"


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _bm25_scores(
    documents: list[str],
    query: str,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Okapi BM25 scores for each document (same ranking role as Grok crate)."""
    docs_tokens = [_tokenize(d) for d in documents]
    n = len(docs_tokens)
    if n == 0:
        return []
    avgdl = sum(len(t) for t in docs_tokens) / n
    # df
    df: dict[str, int] = {}
    for toks in docs_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    q_terms = _tokenize(query)
    scores = [0.0] * n
    for i, toks in enumerate(docs_tokens):
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        dl = len(toks) or 1
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            n_q = df.get(term, 0)
            idf = math.log(1.0 + (n - n_q + 0.5) / (n_q + 0.5))
            freq = tf[term]
            denom = freq + k1 * (1.0 - b + b * dl / (avgdl or 1.0))
            score += idf * (freq * (k1 + 1.0) / denom)
        scores[i] = score
    return scores


class Bm25ToolSearchIndex:
    """Grok shell ``Bm25ToolSearchIndex`` — implements ``ToolSearchIndex``."""

    def __init__(
        self,
        tools: list[ToolMetadata] | None = None,
        *,
        servers: list[ServerMetadata | tuple[str, str | None]] | None = None,
        mcp_initialized: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self._snapshot = ToolMetadataSnapshot(
            tools=list(tools or []),
            servers=_coerce_servers(servers),
            mcp_initialized=mcp_initialized,
        )

    def update(
        self,
        tools: list[ToolMetadata],
        *,
        servers: list[ServerMetadata | tuple[str, str | None]] | None = None,
        mcp_initialized: bool = True,
    ) -> None:
        with self._lock:
            self._snapshot = ToolMetadataSnapshot(
                tools=list(tools),
                servers=_coerce_servers(servers),
                mcp_initialized=mcp_initialized,
            )

    def search(self, query: str, limit: int = 5) -> SearchSnapshot:
        return self.search_snapshot(query, limit)

    def search_snapshot(self, query: str, limit: int = 5) -> SearchSnapshot:
        with self._lock:
            snap = ToolMetadataSnapshot(
                tools=list(self._snapshot.tools),
                servers=list(self._snapshot.servers),
                mcp_initialized=self._snapshot.mcp_initialized,
            )
        is_ready = snap.mcp_initialized
        total = len(snap.tools)
        if not snap.tools:
            return SearchSnapshot(results=[], total_hidden_tools=total, is_ready=is_ready)

        q = (query or "").strip()
        query_lower = q.lower()
        for t in snap.tools:
            if (
                t.qualified_name.lower() == query_lower
                or t.tool_name.lower() == query_lower
            ):
                return SearchSnapshot(
                    results=[
                        ToolSearchResult(
                            tool_name=t.qualified_name,
                            server_name=t.server_name,
                            description=t.description,
                            score=1.0,
                            parameters=list(t.parameters),
                            input_schema=dict(t.input_schema),
                        )
                    ],
                    total_hidden_tools=total,
                    is_ready=is_ready,
                )

        documents = [t.to_document() for t in snap.tools]
        scores = _bm25_scores(documents, normalize_query(q))
        ranked = sorted(
            ((scores[i], i) for i in range(len(scores)) if scores[i] > 0),
            key=lambda x: -x[0],
        )
        lim = max(0, int(limit))
        results: list[ToolSearchResult] = []
        for score, i in ranked[:lim]:
            meta = snap.tools[i]
            results.append(
                ToolSearchResult(
                    tool_name=meta.qualified_name,
                    server_name=meta.server_name,
                    description=meta.description,
                    score=float(score),
                    parameters=list(meta.parameters),
                    input_schema=dict(meta.input_schema),
                )
            )
        return SearchSnapshot(
            results=results,
            total_hidden_tools=total,
            is_ready=is_ready,
        )

    def list_server_summaries(self) -> list[ServerSummary]:
        with self._lock:
            tools = list(self._snapshot.tools)
            servers = list(self._snapshot.servers)
        counts: dict[str, list[str]] = {}
        for t in tools:
            counts.setdefault(t.server_name, []).append(t.tool_name)
        desc_map = {s.name: s.description for s in servers}
        out: list[ServerSummary] = []
        for name in sorted(set(counts) | set(desc_map)):
            names = sorted(set(counts.get(name, [])))
            out.append(
                ServerSummary(
                    name=name,
                    tool_count=len(names),
                    description=desc_map.get(name),
                    tool_names=names,
                )
            )
        return out

    def get(self, tool_name: str) -> ToolSearchResult | None:
        """Resolve one tool by qualified or bare name (use_tool schema path)."""
        q = (tool_name or "").strip().lower()
        if not q:
            return None
        with self._lock:
            tools = list(self._snapshot.tools)
        for t in tools:
            if t.qualified_name.lower() == q or t.tool_name.lower() == q:
                return ToolSearchResult(
                    tool_name=t.qualified_name,
                    server_name=t.server_name,
                    description=t.description,
                    score=1.0,
                    parameters=list(t.parameters),
                    input_schema=dict(t.input_schema),
                )
        return None

    def lookup(self, tool_name: str) -> ToolSearchResult | None:
        return self.get(tool_name)

    def get_schema(self, tool_name: str) -> dict[str, Any] | None:
        r = self.get(tool_name)
        return dict(r.input_schema) if r and r.input_schema else None

    def schema_for(self, tool_name: str) -> dict[str, Any] | None:
        return self.get_schema(tool_name)


def _coerce_servers(
    servers: list[ServerMetadata | tuple[str, str | None]] | None,
) -> list[ServerMetadata]:
    out: list[ServerMetadata] = []
    for s in servers or []:
        if isinstance(s, ServerMetadata):
            out.append(s)
        elif isinstance(s, tuple) and s:
            out.append(ServerMetadata(name=str(s[0]), description=s[1] if len(s) > 1 else None))
        elif isinstance(s, dict) and s.get("name"):
            out.append(
                ServerMetadata(name=str(s["name"]), description=s.get("description"))
            )
    return out


def parameter_names_from_schema(schema: dict[str, Any] | None) -> list[str]:
    if not isinstance(schema, dict):
        return []
    props = schema.get("properties")
    if isinstance(props, dict):
        return [str(k) for k in props.keys()]
    return []


def tools_from_mcp_catalog(tools: Iterable[Any]) -> list[ToolMetadata]:
    """Build ToolMetadata list from host ``mcp_tools`` catalog dicts."""
    out: list[ToolMetadata] = []
    for raw in tools:
        if not isinstance(raw, dict):
            continue
        qname = str(raw.get("name") or "").strip()
        if not qname:
            continue
        if "__" in qname:
            server, _, bare = qname.partition("__")
        else:
            server = str(raw.get("server") or raw.get("server_name") or "mcp")
            bare = qname
        schema = raw.get("parameters") or raw.get("input_schema") or {}
        if not isinstance(schema, dict):
            schema = {}
        out.append(
            ToolMetadata(
                qualified_name=qname,
                server_name=str(raw.get("server") or raw.get("server_name") or server),
                tool_name=str(raw.get("tool_name") or bare),
                description=str(raw.get("description") or ""),
                parameters=parameter_names_from_schema(schema),
                input_schema=schema,
            )
        )
    return out


def index_from_mcp_tools(
    tools: list[Any] | None,
    *,
    servers: list[Any] | None = None,
    mcp_initialized: bool = True,
) -> Bm25ToolSearchIndex:
    """Construct BM25 index from catalog (default when host only has mcp_tools)."""
    meta = tools_from_mcp_catalog(tools or [])
    return Bm25ToolSearchIndex(
        meta,
        servers=_coerce_servers(list(servers or []) if servers else None),
        mcp_initialized=mcp_initialized,
    )


def ensure_mcp_tool_index(extra: dict[str, Any]) -> Any | None:
    """If host has mcp_tools but no index, install Grok ``ToolIndex`` wrapper.

    ``extra['mcp_tool_index']`` becomes ``ToolIndex(Bm25ToolSearchIndex(...))``
    matching shell injection of ``ToolIndex(Arc<dyn ToolSearchIndex>)``.
    """
    existing = extra.get("mcp_tool_index")
    if existing is not None:
        # Normalize bare implementors to ToolIndex wrapper
        if not isinstance(existing, ToolIndex) and unwrap_tool_index(existing) is not None:
            if not isinstance(existing, ToolIndex):
                try:
                    extra["mcp_tool_index"] = ToolIndex(index=existing)  # type: ignore[arg-type]
                    return extra["mcp_tool_index"]
                except Exception:  # noqa: BLE001
                    return existing
        return existing
    tools = extra.get("mcp_tools")
    if not isinstance(tools, list) or not tools:
        return None
    bm25 = index_from_mcp_tools(
        tools,
        servers=extra.get("mcp_servers") if isinstance(extra.get("mcp_servers"), list) else None,
        mcp_initialized=bool(extra.get("mcp_initialized", True)),
    )
    wrapped = ToolIndex(index=bm25)
    extra["mcp_tool_index"] = wrapped
    return wrapped
