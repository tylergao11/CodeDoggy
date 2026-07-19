"""SQLite session history + FTS5 search (Hermes-style big memory).

Stores every turn's messages so ``session_search`` and MemorySelector can
recall past work without stuffing the live prompt. Curated MEMORY.md remains
the small always-on layer; this is the unlimited on-demand layer.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codedoggy.memory.paths import default_memory_home

logger = logging.getLogger(__name__)

_SAFE_TOKEN = re.compile(r"[^\w\*\"\-]+", re.UNICODE)
# FTS5 column names for messages_fts — bare `col:term` is column filter syntax.
_FTS_COLUMNS = frozenset({"content", "role", "tool_name", "session_id"})
# Words that break MATCH when unquoted / reserved-ish
_FTS_BLOCKLIST = frozenset(
    {
        "and",
        "or",
        "not",
        "near",
        "content",
        "role",
        "tool_name",
        "session_id",
    }
)


@dataclass(slots=True)
class SearchHit:
    session_id: str
    message_id: int
    role: str
    content: str
    snippet: str
    title: str | None = None
    goal: str | None = None
    timestamp: float = 0.0
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class SessionCwdValidation:
    """Result of checking whether a session may hydrate in a workspace."""

    allowed: bool
    reason: str
    session_id: str
    requested_cwd: str
    stored_cwd: str | None = None


class SessionStore:
    """Process-local SQLite store under ``~/.codedoggy/state.db`` by default."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            home = default_memory_home()
            home.mkdir(parents=True, exist_ok=True)
            db_path = home / "state.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions
        )
        self._conn.row_factory = sqlite3.Row
        self._fts = True
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        with self._lock:
            c = self._conn
            c.execute("PRAGMA journal_mode=WAL")
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    cwd TEXT,
                    goal TEXT,
                    title TEXT,
                    created_at REAL,
                    updated_at REAL,
                    message_count INTEGER DEFAULT 0
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    tool_name TEXT,
                    tool_call_id TEXT,
                    tool_calls TEXT,
                    timestamp REAL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
                """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session "
                "ON messages(session_id, id)"
            )
            try:
                c.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                        content,
                        role,
                        tool_name,
                        session_id UNINDEXED,
                        content='messages',
                        content_rowid='id',
                        tokenize='porter unicode61'
                    )
                    """
                )
                # Keep FTS in sync via triggers
                c.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                      INSERT INTO messages_fts(rowid, content, role, tool_name, session_id)
                      VALUES (new.id, new.content, new.role, new.tool_name, new.session_id);
                    END
                    """
                )
                c.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                      INSERT INTO messages_fts(messages_fts, rowid, content, role, tool_name, session_id)
                      VALUES ('delete', old.id, old.content, old.role, old.tool_name, old.session_id);
                    END
                    """
                )
            except sqlite3.OperationalError as e:
                logger.warning("FTS5 unavailable (%s); using LIKE fallback", e)
                self._fts = False

    # ── sessions ────────────────────────────────────────────────────────

    def ensure_session(
        self,
        session_id: str,
        *,
        cwd: str | None = None,
        goal: str | None = None,
        title: str | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO sessions (id, cwd, goal, title, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, cwd, goal, title or (goal or "")[:80], now, now),
                )
            else:
                # Do NOT overwrite cwd on existing sessions (cross-project boundary).
                # Only refresh goal/title when provided; cwd is immutable after insert.
                self._conn.execute(
                    "UPDATE sessions SET updated_at = ?, "
                    "goal = COALESCE(?, goal), "
                    "title = COALESCE(?, title) "
                    "WHERE id = ?",
                    (now, goal, title, session_id),
                )

    def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Return session ownership metadata without hydrating its transcript."""
        sid = str(session_id or "").strip()
        if not sid:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT id, cwd, goal, title, created_at, updated_at, message_count "
                "FROM sessions WHERE id = ?",
                (sid,),
            ).fetchone()
        return dict(row) if row is not None else None

    def validate_session_cwd(
        self,
        session_id: str,
        cwd: str | Path,
        *,
        allow_missing: bool = False,
        allow_unbound: bool = False,
    ) -> SessionCwdValidation:
        """Validate an explicit session id before transcript hydration.

        Existing session cwd is immutable, so a mismatch is a hard boundary.
        ``allow_missing`` supports callers that intentionally use an explicit
        id to create a new session; ``allow_unbound`` is only for legacy rows.
        """
        sid = str(session_id or "").strip()
        requested = _normalize_cwd(cwd)
        if not sid:
            return SessionCwdValidation(
                False, "missing_session_id", sid, requested
            )
        metadata = self.get_session_metadata(sid)
        if metadata is None:
            return SessionCwdValidation(
                bool(allow_missing),
                "new_session" if allow_missing else "session_not_found",
                sid,
                requested,
            )
        stored_raw = metadata.get("cwd")
        if not isinstance(stored_raw, str) or not stored_raw.strip():
            return SessionCwdValidation(
                bool(allow_unbound),
                "legacy_unbound" if allow_unbound else "session_cwd_unbound",
                sid,
                requested,
                None,
            )
        stored = _normalize_cwd(stored_raw)
        matches = stored == requested
        return SessionCwdValidation(
            matches,
            "ok" if matches else "cwd_mismatch",
            sid,
            requested,
            stored,
        )

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str | None,
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_calls: Any = None,
    ) -> int:
        from codedoggy.memory.redact import redact_secrets

        self.ensure_session(session_id)
        # Redact before write — no dual-store of unredacted secrets.
        safe_content = redact_secrets(content)
        tc_json = None
        if tool_calls:
            try:
                raw_tc = json.dumps(tool_calls, ensure_ascii=False)
            except (TypeError, ValueError):
                raw_tc = str(tool_calls)
            tc_json = redact_secrets(raw_tc)
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages "
                "(session_id, role, content, tool_name, tool_call_id, tool_calls, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    role,
                    safe_content,
                    tool_name,
                    tool_call_id,
                    tc_json,
                    now,
                ),
            )
            mid = int(cur.lastrowid)
            self._conn.execute(
                "UPDATE sessions SET updated_at = ?, "
                "message_count = message_count + 1 WHERE id = ?",
                (now, session_id),
            )
            return mid

    def append_turn_messages(
        self,
        session_id: str,
        messages: list[Any],
        *,
        cwd: str | None = None,
        goal: str | None = None,
    ) -> int:
        """Persist a list of turn Message objects (or dicts). Returns count."""
        self.ensure_session(session_id, cwd=cwd, goal=goal)
        n = 0
        for m in messages:
            if hasattr(m, "role"):
                role = m.role.value if hasattr(m.role, "value") else str(m.role)
                content = m.content
                tool_name = m.name
                tool_call_id = m.tool_call_id
                tool_calls = None
                if m.tool_calls:
                    tool_calls = [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in m.tool_calls
                    ]
            else:
                role = str(m.get("role", "user"))
                content = m.get("content")
                tool_name = m.get("name") or m.get("tool_name")
                tool_call_id = m.get("tool_call_id")
                tool_calls = m.get("tool_calls")
            self.append_message(
                session_id,
                role,
                content,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_calls=tool_calls,
            )
            n += 1
        return n

    def get_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Oldest-first slice (offset from start). Prefer get_messages_tail for resume."""
        sql = (
            "SELECT id, session_id, role, content, tool_name, tool_call_id, "
            "tool_calls, timestamp FROM messages WHERE session_id = ? "
            "ORDER BY id"
        )
        params: list[Any] = [session_id]
        if limit is not None or offset:
            sql += " LIMIT ? OFFSET ?"
            params.extend([-1 if limit is None else int(limit), int(offset)])
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_msg(r) for r in rows]

    def get_messages_tail(
        self,
        session_id: str,
        *,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Newest ``limit`` messages, returned oldest→newest (Grok resume)."""
        limit = max(1, int(limit))
        sql = (
            "SELECT id, session_id, role, content, tool_name, tool_call_id, "
            "tool_calls, timestamp FROM messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, (session_id, limit)).fetchall()
        msgs = [self._row_to_msg(r) for r in rows]
        msgs.reverse()
        return msgs

    def get_messages_around(
        self,
        session_id: str,
        around_message_id: int,
        *,
        window: int = 5,
    ) -> dict[str, Any]:
        window = max(0, int(window))
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM messages WHERE id = ? AND session_id = ?",
                (around_message_id, session_id),
            ).fetchone()
            if not exists:
                return {"window": [], "messages_before": 0, "messages_after": 0}
            before = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id < ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, around_message_id, window),
            ).fetchall()
            anchor = self._conn.execute(
                "SELECT * FROM messages WHERE id = ?", (around_message_id,)
            ).fetchone()
            after = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id > ? "
                "ORDER BY id ASC LIMIT ?",
                (session_id, around_message_id, window),
            ).fetchall()
        before_msgs = [self._row_to_msg(r) for r in reversed(before)]
        after_msgs = [self._row_to_msg(r) for r in after]
        win = before_msgs + ([self._row_to_msg(anchor)] if anchor else []) + after_msgs
        return {
            "window": win,
            "messages_before": len(before_msgs),
            "messages_after": len(after_msgs),
        }

    def list_recent_sessions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, cwd, goal, title, created_at, updated_at, message_count "
                "FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            # preview: last user/assistant content
            prev = self._conn.execute(
                "SELECT content, role FROM messages WHERE session_id = ? "
                "AND role IN ('user','assistant') ORDER BY id DESC LIMIT 1",
                (d["id"],),
            ).fetchone()
            d["preview"] = (prev["content"][:160] if prev and prev["content"] else "")
            out.append(d)
        return out

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        exclude_session_id: str | None = None,
        session_id: str | None = None,
        cwd: str | None = None,
        roles: list[str] | None = None,
    ) -> list[SearchHit]:
        """FTS with optional session/cwd/role scope (Grok memory boundary)."""
        q = (query or "").strip()
        if not q:
            return []
        limit = max(1, min(int(limit), 100))
        if self._fts:
            return self._search_fts(
                q,
                limit=limit,
                exclude_session_id=exclude_session_id,
                session_id=session_id,
                cwd=cwd,
                roles=roles,
            )
        return self._search_like(
            q,
            limit=limit,
            exclude_session_id=exclude_session_id,
            session_id=session_id,
            cwd=cwd,
            roles=roles,
        )

    def _search_fts(
        self,
        query: str,
        *,
        limit: int,
        exclude_session_id: str | None,
        session_id: str | None = None,
        cwd: str | None = None,
        roles: list[str] | None = None,
    ) -> list[SearchHit]:
        fts_q = self._sanitize_fts_query(query)
        if not fts_q:
            return []
        sql = (
            "SELECT m.id, m.session_id, m.role, m.content, m.timestamp, "
            "s.title, s.goal, "
            "snippet(messages_fts, 0, '«', '»', '…', 24) AS snip, "
            "bm25(messages_fts) AS score "
            "FROM messages_fts "
            "JOIN messages m ON m.id = messages_fts.rowid "
            "JOIN sessions s ON s.id = m.session_id "
            "WHERE messages_fts MATCH ? "
        )
        params: list[Any] = [fts_q]
        if session_id:
            sql += "AND m.session_id = ? "
            params.append(session_id)
        if exclude_session_id:
            sql += "AND m.session_id != ? "
            params.append(exclude_session_id)
        if cwd:
            sql += "AND s.cwd = ? "
            params.append(str(cwd))
        if roles:
            placeholders = ",".join("?" for _ in roles)
            sql += f"AND m.role IN ({placeholders}) "
            params.extend(list(roles))
        # bm25 lower-is-better; pull a wider pool then re-rank with recency.
        pool = max(limit * 3, limit + 5)
        sql += "ORDER BY score ASC, m.timestamp DESC LIMIT ?"
        params.append(pool)
        with self._lock:
            try:
                rows = self._conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as e:
                logger.warning("FTS query failed (%s); LIKE fallback", e)
                return self._search_like(
                    query,
                    limit=limit,
                    exclude_session_id=exclude_session_id,
                    session_id=session_id,
                    cwd=cwd,
                    roles=roles,
                )
        now = time.time()
        hits: list[SearchHit] = []
        for r in rows:
            bm25 = float(r["score"] or 0.0)
            relevance = -bm25
            ts = float(r["timestamp"] or 0.0)
            age_h = max(0.0, (now - ts) / 3600.0) if ts else 1e9
            if age_h < 1:
                recency = 2.0
            elif age_h < 24:
                recency = 1.2
            elif age_h < 24 * 7:
                recency = 0.5
            else:
                recency = 0.0
            role_boost = 0.3 if r["role"] in {"user", "assistant"} else 0.0
            hits.append(
                SearchHit(
                    session_id=r["session_id"],
                    message_id=r["id"],
                    role=r["role"],
                    content=r["content"] or "",
                    snippet=r["snip"] or (r["content"] or "")[:120],
                    title=r["title"],
                    goal=r["goal"],
                    timestamp=ts,
                    score=relevance + recency + role_boost,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def _search_like(
        self,
        query: str,
        *,
        limit: int,
        exclude_session_id: str | None,
        session_id: str | None = None,
        cwd: str | None = None,
        roles: list[str] | None = None,
    ) -> list[SearchHit]:
        tokens = [t for t in re.split(r"\s+", query.strip()) if t][:8]
        if not tokens:
            return []
        clauses = ["m.content LIKE ?" for _ in tokens]
        params: list[Any] = [f"%{t}%" for t in tokens]
        sql = (
            "SELECT m.id, m.session_id, m.role, m.content, m.timestamp, "
            "s.title, s.goal FROM messages m "
            "JOIN sessions s ON s.id = m.session_id WHERE "
            + " AND ".join(clauses)
        )
        if session_id:
            sql += " AND m.session_id = ?"
            params.append(session_id)
        if cwd:
            sql += " AND s.cwd = ?"
            params.append(str(cwd))
        if roles:
            placeholders = ",".join("?" for _ in roles)
            sql += f" AND m.role IN ({placeholders})"
            params.extend(list(roles))
        if exclude_session_id:
            sql += " AND m.session_id != ?"
            params.append(exclude_session_id)
        sql += " ORDER BY m.timestamp DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        hits = []
        for r in rows:
            content = r["content"] or ""
            hits.append(
                SearchHit(
                    session_id=r["session_id"],
                    message_id=r["id"],
                    role=r["role"],
                    content=content,
                    snippet=content[:120],
                    title=r["title"],
                    goal=r["goal"],
                    timestamp=r["timestamp"] or 0.0,
                )
            )
        return hits

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Turn free text into a safe FTS5 MATCH query.

        Free text is tokenized and each token is double-quoted so FTS5 never
        treats a bare word as a *column filter* (``col:term``). That bug
        surfaced live as ``no such column: reading`` when natural-language
        hints were passed to search.
        """
        q = (query or "").strip()[:2048]
        if not q:
            return ""
        # Split on whitespace and punctuation (keep underscores for read_file).
        raw_tokens = re.split(r"[\s,;|/\\]+", q)
        safe: list[str] = []
        for t in raw_tokens:
            if not t:
                continue
            # Kill column-filter syntax and other punctuation
            t = t.replace(":", " ").replace(".", " ").replace("(", " ").replace(")", " ")
            t = _SAFE_TOKEN.sub("", t).strip()
            if not t or len(t) < 2:
                continue
            low = t.lower()
            if low in _FTS_BLOCKLIST or low in _FTS_COLUMNS:
                continue
            # Quote so MATCH never parses as column:term or bare operator
            if '"' in t:
                t = t.replace('"', "")
            if not t:
                continue
            safe.append(f'"{t}"')
            if len(safe) >= 12:
                break
        if not safe:
            return ""
        if len(safe) == 1:
            return safe[0]
        return " OR ".join(safe)

    @staticmethod
    def _row_to_msg(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        if d.get("tool_calls"):
            try:
                d["tool_calls"] = json.loads(d["tool_calls"])
            except (json.JSONDecodeError, TypeError):
                d["tool_calls"] = []
        return d


def _normalize_cwd(cwd: str | Path) -> str:
    """Canonical, case-normalized workspace identity for ownership checks."""
    raw = Path(cwd).expanduser().resolve(strict=False)
    return os.path.normcase(os.path.normpath(str(raw)))


def default_session_db_path() -> Path:
    return default_memory_home() / "state.db"
