from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Rulebook:
    name: str
    version: str
    raw: dict


def load_rulebook(path: str) -> Rulebook:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return Rulebook(name=raw["name"], version=raw["version"], raw=raw)

