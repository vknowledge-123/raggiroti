from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone

from raggiroti.config import get_settings
from raggiroti.rag.chunking import chunk_text
from raggiroti.rag.embeddings import EmbeddingClient
from raggiroti.storage.sqlite_db import SqliteStore, Transcript


def _id_from_text(prefix: str, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Transcript text file path (utf-8)")
    ap.add_argument("--language", default="hi", help="Language tag (e.g., hi, hinglish, en)")
    ap.add_argument("--tags", default="", help="Comma-separated tags")
    ap.add_argument("--embed", action="store_true", help="Create embeddings now (requires OPENAI_API_KEY)")
    args = ap.parse_args()

    settings = get_settings()
    with open(args.file, "r", encoding="utf-8") as f:
        content = f.read()

    created_at = datetime.now(timezone.utc).isoformat()
    transcript_id = _id_from_text("transcript", content)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    store = SqliteStore(settings.db_path)
    store.add_transcript(
        Transcript(
            id=transcript_id,
            created_at=created_at,
            language=args.language,
            tags=tags,
            content=content,
        )
    )

    chunks = chunk_text(content)
    chunk_rows = []
    for idx, ch in enumerate(chunks):
        chunk_id = _id_from_text(f"chunk{idx}", transcript_id + ch)
        chunk_rows.append((chunk_id, idx, ch))
    store.add_chunks(transcript_id, chunk_rows)

    if args.embed:
        if not settings.openai_api_key:
            raise SystemExit("OPENAI_API_KEY is required for --embed")
        embedder = EmbeddingClient(api_key=settings.openai_api_key, model=settings.openai_embed_model)
        embeddings = embedder.embed([c[2] for c in chunk_rows])
        for (chunk_id, _, _), emb in zip(chunk_rows, embeddings):
            store.set_chunk_embedding(chunk_id, emb)

    store.close()
    print(f"OK: ingested transcript {transcript_id} with {len(chunk_rows)} chunks into {settings.db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
