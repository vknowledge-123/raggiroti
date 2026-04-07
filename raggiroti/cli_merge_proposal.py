from __future__ import annotations

import argparse
import json
from datetime import date


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rulebook", required=True, help="Path to canonical rulebook JSON")
    ap.add_argument("--proposal", required=True, help="Path to a proposal JSON (exported)")
    ap.add_argument("--bump", default="patch", choices=["patch", "minor", "major"], help="Semver bump")
    args = ap.parse_args()

    with open(args.rulebook, "r", encoding="utf-8") as f:
        rb = json.load(f)
    with open(args.proposal, "r", encoding="utf-8") as f:
        prop = json.load(f)

    existing_ids = {r["id"] for r in rb.get("rules", [])}
    new_rules = []
    for r in prop.get("rules", []):
        if r["id"] in existing_ids:
            continue
        new_rules.append(r)

    rb.setdefault("rules", [])
    rb["rules"].extend(new_rules)

    # Merge conflicts into integration_conflicts as notes (optional)
    if prop.get("conflicts"):
        rb.setdefault("integration_conflicts", [])
        base = len(rb["integration_conflicts"]) + 1
        for i, c in enumerate(prop["conflicts"], start=0):
            rb["integration_conflicts"].append(
                {
                    "id": f"CONFLICT-PROP-{base+i:03d}",
                    "topic": c["topic"],
                    "note": c["note"],
                }
            )

    # Bump version
    old = rb.get("version", "0.0.0")
    major, minor, patch = [int(x) for x in old.split(".")]
    if args.bump == "patch":
        patch += 1
    elif args.bump == "minor":
        minor += 1
        patch = 0
    else:
        major += 1
        minor = 0
        patch = 0
    rb["version"] = f"{major}.{minor}.{patch}"
    rb["updated_at"] = date.today().isoformat()

    rb.setdefault("changelog", [])
    rb["changelog"].append(
        {
            "version": rb["version"],
            "date": rb["updated_at"],
            "summary": f"Merged proposal {prop.get('_meta', {}).get('proposal_id', '')}".strip(),
        }
    )

    with open(args.rulebook, "w", encoding="utf-8") as f:
        json.dump(rb, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"OK: merged {len(new_rules)} new rules into {args.rulebook} and bumped version to {rb['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

