from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path


def _parse_semver(v: str) -> tuple[int, int, int]:
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", (v or "").strip())
    if not m:
        return 0, 0, 0
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _bump_patch(v: str) -> str:
    a, b, c = _parse_semver(v)
    return f"{a}.{b}.{c+1}"


def _max_dt_sl(raw: dict) -> int:
    mx = 0
    for r in raw.get("rules", []):
        rid = str(r.get("id") or "")
        m = re.match(r"^DT-SL-(\d+)$", rid)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx


@dataclass(frozen=True)
class MergeResult:
    ok: bool
    old_version: str
    new_version: str
    added: int
    skipped: int
    path: str


def merge_rule_proposal_into_rulebook(
    *,
    rulebook_path: str,
    proposal: dict,
    source_transcript_id: str | None,
    today: str | None = None,
) -> MergeResult:
    """
    Merges proposal rules into the canonical rulebook JSON.

    - Assigns fresh DT-SL ids sequentially to avoid collisions.
    - Adds a changelog entry.
    - Updates updated_at and version (patch bump).
    """
    p = Path(rulebook_path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    old_version = str(raw.get("version") or "0.0.0")
    new_version = _bump_patch(old_version)
    raw["version"] = new_version
    raw["updated_at"] = today or date.today().isoformat()

    existing_fingerprints: set[str] = set()
    for r in raw.get("rules", []):
        fp = (str(r.get("category")), str(r.get("name")), str(r.get("condition")), str(r.get("action")))
        existing_fingerprints.add("|".join(fp))

    next_id = _max_dt_sl(raw) + 1
    added = 0
    skipped = 0

    rules = proposal.get("rules") if isinstance(proposal, dict) else None
    if not isinstance(rules, list):
        return MergeResult(ok=False, old_version=old_version, new_version=old_version, added=0, skipped=0, path=str(p))

    for rr in rules:
        if not isinstance(rr, dict):
            continue
        cat = str(rr.get("category") or "").strip()
        name = str(rr.get("name") or "").strip()
        cond = str(rr.get("condition") or "").strip()
        interp = str(rr.get("interpretation") or "").strip()
        act = str(rr.get("action") or "").strip()
        tags = rr.get("tags") or []
        if not cat or not name or not cond or not interp or not act:
            skipped += 1
            continue
        fp = "|".join((cat, name, cond, act))
        if fp in existing_fingerprints:
            skipped += 1
            continue
        existing_fingerprints.add(fp)

        raw.setdefault("rules", []).append(
            {
                "id": f"DT-SL-{next_id}",
                "category": cat,
                "name": name,
                "condition": cond,
                "interpretation": interp,
                "action": act,
                "tags": list(tags) if isinstance(tags, list) else [],
                "source_transcript_id": source_transcript_id,
            }
        )
        next_id += 1
        added += 1

    raw.setdefault("changelog", []).append(
        {
            "version": new_version,
            "date": raw["updated_at"],
            "summary": f"Merged rule proposal from transcript {source_transcript_id or 'unknown'}: +{added} rules (skipped {skipped}).",
        }
    )

    p.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return MergeResult(ok=True, old_version=old_version, new_version=new_version, added=added, skipped=skipped, path=str(p))

