from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone

from raggiroti.config import get_settings
from raggiroti.rag.rule_extractor import RuleExtractor
from raggiroti.storage.sqlite_db import SqliteStore


def _proposal_id(text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"proposal_{h}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Transcript text file path (utf-8)")
    ap.add_argument("--source-transcript-id", default=None)
    args = ap.parse_args()

    settings = get_settings()
    api_key = settings.llm_api_key or settings.openai_api_key
    base_url = settings.llm_base_url
    if not api_key and not base_url:
        raise SystemExit("Set OPENAI_API_KEY (OpenAI) or LLM_BASE_URL+LLM_API_KEY (local) for rule extraction")
    if not api_key and base_url:
        api_key = "local"

    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()

    extractor = RuleExtractor(api_key=api_key, base_url=base_url, model=settings.openai_rule_extract_model)
    proposal = extractor.extract_rules(text)

    created_at = datetime.now(timezone.utc).isoformat()
    proposal_id = _proposal_id(created_at + text)
    store = SqliteStore(settings.db_path)
    store.add_rule_proposal(proposal_id, created_at, args.source_transcript_id, proposal)
    store.close()

    print(f"OK: stored proposal {proposal_id} (draft) into {settings.db_path}")
    print("Next: review proposal JSON in DB, then manually merge into rulebook with version bump + backtest gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
