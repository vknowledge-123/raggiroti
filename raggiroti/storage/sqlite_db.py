from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS transcripts (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  language TEXT,
  tags_json TEXT,
  content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_chunks (
  id TEXT PRIMARY KEY,
  transcript_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  embedding_json TEXT,
  FOREIGN KEY(transcript_id) REFERENCES transcripts(id)
);

CREATE TABLE IF NOT EXISTS rule_proposals (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  source_transcript_id TEXT,
  proposal_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'draft'  -- draft|approved|rejected|merged
);

CREATE TABLE IF NOT EXISTS llm_cache (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  model TEXT NOT NULL,
  request_hash TEXT NOT NULL,
  response_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rulebook_index (
  rulebook_version TEXT NOT NULL,
  rule_id TEXT NOT NULL,
  category TEXT,
  name TEXT,
  condition TEXT,
  action TEXT,
  tags_json TEXT,
  PRIMARY KEY(rulebook_version, rule_id)
);
"""


@dataclass(frozen=True)
class Transcript:
    id: str
    created_at: str
    language: str | None
    tags: list[str]
    content: str


class SqliteStore:
    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def add_transcript(self, transcript: Transcript) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO transcripts(id, created_at, language, tags_json, content) VALUES(?,?,?,?,?)",
            (
                transcript.id,
                transcript.created_at,
                transcript.language,
                json.dumps(transcript.tags, ensure_ascii=False),
                transcript.content,
            ),
        )
        self._conn.commit()

    def add_chunks(
        self,
        transcript_id: str,
        chunks: Iterable[tuple[str, int, str]],
    ) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO transcript_chunks(id, transcript_id, chunk_index, content) VALUES(?,?,?,?)",
            [(cid, transcript_id, idx, content) for (cid, idx, content) in chunks],
        )
        self._conn.commit()

    def set_chunk_embedding(self, chunk_id: str, embedding: list[float]) -> None:
        self._conn.execute(
            "UPDATE transcript_chunks SET embedding_json=? WHERE id=?",
            (json.dumps(embedding), chunk_id),
        )
        self._conn.commit()

    def iter_chunks_with_embeddings(self) -> Iterable[tuple[str, str, list[float]]]:
        cur = self._conn.execute(
            "SELECT id, content, embedding_json FROM transcript_chunks WHERE embedding_json IS NOT NULL"
        )
        for chunk_id, content, emb_json in cur.fetchall():
            yield chunk_id, content, json.loads(emb_json)

    def add_rule_proposal(self, proposal_id: str, created_at: str, source_transcript_id: str | None, proposal: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO rule_proposals(id, created_at, source_transcript_id, proposal_json, status) VALUES(?,?,?,?,?)",
            (proposal_id, created_at, source_transcript_id, json.dumps(proposal, ensure_ascii=False), "draft"),
        )
        self._conn.commit()

    def list_transcripts(self, limit: int = 50, offset: int = 0) -> list[dict]:
        cur = self._conn.execute(
            "SELECT id, created_at, language, tags_json FROM transcripts ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (int(limit), int(offset)),
        )
        out: list[dict] = []
        for tid, created_at, language, tags_json in cur.fetchall():
            out.append(
                {
                    "id": tid,
                    "created_at": created_at,
                    "language": language,
                    "tags": [] if not tags_json else json.loads(tags_json),
                }
            )
        return out

    def get_transcript(self, transcript_id: str) -> dict | None:
        cur = self._conn.execute(
            "SELECT id, created_at, language, tags_json, content FROM transcripts WHERE id=?",
            (transcript_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "created_at": row[1],
            "language": row[2],
            "tags": [] if not row[3] else json.loads(row[3]),
            "content": row[4],
        }

    def list_rule_proposals(self, limit: int = 50, offset: int = 0, status: str | None = None) -> list[dict]:
        if status:
            cur = self._conn.execute(
                "SELECT id, created_at, source_transcript_id, status FROM rule_proposals WHERE status=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, int(limit), int(offset)),
            )
        else:
            cur = self._conn.execute(
                "SELECT id, created_at, source_transcript_id, status FROM rule_proposals ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (int(limit), int(offset)),
            )
        return [
            {"id": rid, "created_at": created_at, "source_transcript_id": stid, "status": st}
            for (rid, created_at, stid, st) in cur.fetchall()
        ]

    def get_rule_proposal(self, proposal_id: str) -> dict | None:
        cur = self._conn.execute(
            "SELECT id, created_at, source_transcript_id, proposal_json, status FROM rule_proposals WHERE id=?",
            (proposal_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "created_at": row[1],
            "source_transcript_id": row[2],
            "proposal": json.loads(row[3]),
            "status": row[4],
        }

    def set_rule_proposal_status(self, proposal_id: str, status: str) -> None:
        if status not in {"draft", "approved", "rejected", "merged"}:
            raise ValueError("invalid status")
        self._conn.execute("UPDATE rule_proposals SET status=? WHERE id=?", (status, proposal_id))
        self._conn.commit()

    def rebuild_rulebook_index(self, *, rulebook_version: str, rules: list[dict]) -> int:
        self._conn.execute("DELETE FROM rulebook_index WHERE rulebook_version=?", (rulebook_version,))
        rows = []
        for r in rules or []:
            rows.append(
                (
                    rulebook_version,
                    str(r.get("id") or ""),
                    str(r.get("category") or ""),
                    str(r.get("name") or ""),
                    str(r.get("condition") or ""),
                    str(r.get("action") or ""),
                    json.dumps(r.get("tags") or [], ensure_ascii=False),
                )
            )
        self._conn.executemany(
            "INSERT OR REPLACE INTO rulebook_index(rulebook_version, rule_id, category, name, condition, action, tags_json) VALUES(?,?,?,?,?,?,?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def get_indexed_rules(self, *, rulebook_version: str, limit: int = 5000) -> list[dict]:
        cur = self._conn.execute(
            "SELECT rule_id, category, name, condition, action, tags_json FROM rulebook_index WHERE rulebook_version=? LIMIT ?",
            (rulebook_version, int(limit)),
        )
        out: list[dict] = []
        for rid, cat, name, cond, act, tags_json in cur.fetchall():
            out.append(
                {
                    "id": rid,
                    "category": cat,
                    "name": name,
                    "condition": cond,
                    "action": act,
                    "tags": [] if not tags_json else json.loads(tags_json),
                }
            )
        return out

    def get_llm_cache(self, request_hash: str, model: str) -> dict | None:
        cur = self._conn.execute(
            "SELECT response_json FROM llm_cache WHERE request_hash=? AND model=?",
            (request_hash, model),
        )
        row = cur.fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def set_llm_cache(self, cache_id: str, created_at: str, model: str, request_hash: str, response: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache(id, created_at, model, request_hash, response_json) VALUES(?,?,?,?,?)",
            (cache_id, created_at, model, request_hash, json.dumps(response, ensure_ascii=False)),
        )
        self._conn.commit()
