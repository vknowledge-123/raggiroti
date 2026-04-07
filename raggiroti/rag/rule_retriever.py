from __future__ import annotations

from dataclasses import dataclass

from raggiroti.rules.rulebook_loader import load_rulebook
from raggiroti.storage.sqlite_db import SqliteStore
from raggiroti.config import get_settings


@dataclass(frozen=True)
class RetrievedRules:
    rulebook_version: str
    rules: list[dict]


def _match_rule(rule: dict, state: dict) -> bool:
    tags = set(rule.get("tags") or [])
    zone = state.get("zone")
    structure = state.get("structure")
    gap = state.get("gap_type")
    market_type = state.get("market_type")
    validity_ok = state.get("validity_ok")
    has_swings = bool(state.get("last_swing_high_1m") or state.get("last_swing_low_1m") or state.get("last_swing_high_5m") or state.get("last_swing_low_5m"))
    has_oi = bool(isinstance(state.get("oi"), dict) and state.get("oi", {}).get("ok"))

    # Very small deterministic retrieval; extend as you add more signals.
    if zone in ("discount", "inflated") and "zone_engine" in tags:
        return True
    if gap in ("gap_up", "gap_down") and ("gap" in tags or "opening_filter" in tags or "opening_candle" in tags):
        return True
    if state.get("confirmed_long") and ("reentry" in tags or "confirmation" in tags or "sl_hunting" in tags):
        return True
    if state.get("confirmed_short") and ("confirmation" in tags or "sl_hunting" in tags):
        return True
    if structure in ("bull", "bear") and ("dow_theory" in tags or "structure" in tags):
        return True
    if has_swings and ("swing" in tags or "sl_pools" in tags or "liquidity" in tags):
        return True
    if any(state.get(k) for k in ("reclaimed_last_swing_high", "reclaimed_last_swing_low", "reclaimed_prev_pdh", "reclaimed_prev_pdl")) and ("reclaim" in tags or "micro_sweep" in tags or "trap" in tags or "confirmation" in tags):
        return True
    if market_type in ("trap", "chop") and ("trap" in tags or "overtrading" in tags or "wait" in tags or "discipline" in tags):
        return True
    if validity_ok is False and ("validity" in tags or "no_trade" in tags or "wait" in tags):
        return True
    if has_oi and ("options" in tags or "open_interest" in tags or "change_in_oi" in tags or "writer_pressure" in tags):
        return True
    # Fallback: always include risk/disciplines by category match
    if rule.get("category") in ("Risk Management", "Risk Control", "Discipline", "Execution"):
        return True
    return False


def retrieve_rulebook_rules(rulebook_path: str, state: dict, limit: int = 25) -> RetrievedRules:
    rb = load_rulebook(rulebook_path)
    rules = rb.raw.get("rules", [])
    # Fast path: if a rulebook index exists for this version, use it.
    try:
        settings = get_settings()
        store = SqliteStore(settings.db_path)
        indexed = store.get_indexed_rules(rulebook_version=rb.version)
        store.close()
        if indexed:
            rules = indexed
    except Exception:
        pass

    matched = []
    for r in rules:
        if _match_rule(r, state):
            matched.append(r)
        if len(matched) >= limit:
            break
    return RetrievedRules(rulebook_version=rb.version, rules=matched)
