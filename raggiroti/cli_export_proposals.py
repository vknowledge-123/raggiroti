from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

from raggiroti.config import get_settings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./rulebook/proposals", help="Directory to write proposal JSON files")
    ap.add_argument("--status", default="draft", help="Filter by status (draft|approved|rejected|merged)")
    args = ap.parse_args()

    settings = get_settings()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(settings.db_path)
    cur = conn.execute(
        "SELECT id, created_at, proposal_json FROM rule_proposals WHERE status=? ORDER BY created_at ASC",
        (args.status,),
    )
    rows = cur.fetchall()
    for pid, created_at, proposal_json in rows:
        payload = json.loads(proposal_json)
        payload["_meta"] = {"proposal_id": pid, "created_at": created_at, "status": args.status}
        path = outdir / f"{pid}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(path))
    conn.close()
    if not rows:
        print("No proposals found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

