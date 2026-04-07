from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import time

import httpx

from raggiroti.backtest.day_split import group_by_date
from raggiroti.backtest.prev_day_planner import compute_prev_day_levels
from raggiroti.backtest.state_builder import StateBuilder
from raggiroti.dhan.historical import DhanIntradayRequest, fetch_intraday_candles
from raggiroti.dhan.option_chain import derive_oi_features
from raggiroti.rag.rule_retriever import retrieve_rulebook_rules
from raggiroti.storage.sqlite_db import SqliteStore


def _hash_request(payload: dict) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")
    # 1) direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) try to extract the first balanced JSON object substring
    def _balanced_object(s: str) -> str | None:
        i = s.find("{")
        if i < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(s)):
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[i : j + 1]
        return None

    cand = _balanced_object(text)
    if cand:
        try:
            return json.loads(cand)
        except Exception:
            text = cand

    # 3) heuristic repairs for common Gemini "almost JSON" mistakes
    repaired = text
    # Quote unquoted keys: {foo: 1} -> {"foo": 1}
    repaired = re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', repaired)
    # Replace Python-ish tokens
    repaired = repaired.replace(": None", ": null").replace(": True", ": true").replace(": False", ": false")
    # Remove trailing commas
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try:
        return json.loads(repaired)
    except Exception:
        # 4) last resort: regex object grab
        m = re.search(r"\{.*\}", repaired, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def _extract_candidate_text(data: dict) -> str:
    try:
        cand = (data.get("candidates") or [{}])[0] or {}
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        texts: list[str] = []
        for p in parts:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                texts.append(p["text"])
            inline = p.get("inlineData") if isinstance(p, dict) else None
            if isinstance(inline, dict) and isinstance(inline.get("data"), str):
                texts.append(inline["data"])
        return "\n".join(texts).strip()
    except Exception:
        return ""


def _normalize_model_id(model: str) -> str:
    m = (model or "").strip()
    if m.startswith("models/"):
        m = m[len("models/") :]
    return m


def _sanitize_error(s: str) -> str:
    s = (s or "").replace("\n", " ").strip()
    # Avoid leaking Gemini API keys (httpx may include full URL with ?key=...).
    s = re.sub(r"(key=)[^&\s]+", r"\1***", s)
    return s


def _gemini_feedback(data: dict) -> dict:
    try:
        pf = data.get("promptFeedback") or {}
        cands = data.get("candidates") or []
        cand0 = (cands[0] if cands else {}) or {}
        return {
            "candidates_count": int(len(cands)),
            "prompt_block_reason": pf.get("blockReason"),
            "prompt_safety_ratings": pf.get("safetyRatings"),
            "finish_reason": cand0.get("finishReason"),
            "candidate_safety_ratings": cand0.get("safetyRatings"),
        }
    except Exception:
        return {}

def _single_line(s: str) -> str:
    return (s or "").replace("\r", " ").replace("\n", " ").strip()


def _sanitize_reason_for_bias(reason: str, bias: str) -> str:
    """
    Ensure reason_points do not suggest both-side entries.
    - If bias=BUY, remove short/sell language.
    - If bias=SELL, remove long/buy language.
    - If bias=WAIT, caller should replace reasons with WAIT-only reasons.
    """
    txt = _single_line(reason)
    if not txt:
        return ""

    lower = txt.lower()
    # If a sentence explicitly contains both sides (common: "... for longs OR ... for shorts"),
    # select the side consistent with bias.
    if (" or " in lower) and ("long" in lower) and ("short" in lower):
        # Split once; keep the clause that best matches bias.
        parts = re.split(r"\bor\b", txt, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            left, right = parts[0].strip(" ,.;"), parts[1].strip(" ,.;")
            llow, rlow = left.lower(), right.lower()
            if bias == "BUY":
                txt = left if ("reclaim" in llow or "long" in llow or "buy" in llow) else right
            elif bias == "SELL":
                txt = left if ("reject" in llow or "short" in llow or "sell" in llow) else right

    # Remove explicit opposite-side guidance.
    if bias == "BUY":
        txt = re.sub(r"\bfor\s+shorts?\b", "", txt, flags=re.IGNORECASE).strip()
        txt = re.sub(r"\bshorts?\b", "", txt, flags=re.IGNORECASE).strip()
        txt = re.sub(r"\bsell\b", "", txt, flags=re.IGNORECASE).strip()
    elif bias == "SELL":
        txt = re.sub(r"\bfor\s+longs?\b", "", txt, flags=re.IGNORECASE).strip()
        txt = re.sub(r"\blongs?\b", "", txt, flags=re.IGNORECASE).strip()
        txt = re.sub(r"\bbuy\b", "", txt, flags=re.IGNORECASE).strip()

    # Cleanup repeated spaces/punctuation remnants.
    txt = re.sub(r"\s{2,}", " ", txt).strip(" ,.;")
    return txt


def _sanitize_prediction_output(out: dict) -> dict:
    """
    Post-process Gemini output to enforce product constraints:
    - No both-side guidance in reason_points.
    - WAIT plans must not contain entry/SL zones.
    - Keep arrays small to reduce token/noise in UI.
    """
    if not isinstance(out, dict):
        return out

    # summary_points single-line
    if isinstance(out.get("summary_points"), list):
        out["summary_points"] = [_single_line(str(x)) for x in out["summary_points"] if _single_line(str(x))][:8]

    plans = out.get("gap_plans")
    if not isinstance(plans, list):
        return out

    new_plans = []
    for p in plans:
        if not isinstance(p, dict):
            continue
        bias = str(p.get("bias") or "WAIT").upper()
        if bias not in {"BUY", "SELL", "WAIT"}:
            bias = "WAIT"
        p["bias"] = bias

        # Normalize arrays
        if not isinstance(p.get("targets"), list):
            p["targets"] = []
        if not isinstance(p.get("liquidity_pools"), list):
            p["liquidity_pools"] = []
        p["targets"] = [float(x) for x in p["targets"][:3] if isinstance(x, (int, float))]
        p["liquidity_pools"] = [float(x) for x in p["liquidity_pools"][:3] if isinstance(x, (int, float))]

        # reasons
        reasons = p.get("reason_points")
        if not isinstance(reasons, list):
            reasons = []

        if bias == "WAIT":
            # Enforce WAIT = no entry, no SL.
            p["entry_zone"] = None
            p["operator_zone"] = None
            p["no_trade_zone"] = p.get("no_trade_zone") if isinstance(p.get("no_trade_zone"), list) else None
            p["sl"] = None
            p["reason_points"] = [
                "WAIT: no one-sided confirmation for this gap bucket.",
                "Avoid both-side entries; wait for clear acceptance/rejection at key levels.",
            ][:3]
        else:
            cleaned = []
            for r in reasons:
                rr = _sanitize_reason_for_bias(str(r), bias=bias)
                if rr:
                    cleaned.append(rr)
            cleaned = [_single_line(x) for x in cleaned if _single_line(x)][:3]
            if not cleaned:
                cleaned = [
                    (f"Bias={bias}: wait for sweep + reclaim (BUY) or sweep + reject (SELL) at key liquidity.")
                    if bias in {"BUY", "SELL"}
                    else "WAIT.",
                ]
            p["reason_points"] = cleaned

        new_plans.append(p)

    out["gap_plans"] = new_plans
    return out


def _expected_bucket_keys(gap_buckets: list[dict]) -> list[str]:
    keys: list[str] = []
    for b in gap_buckets or []:
        k = b.get("key")
        if isinstance(k, str) and k:
            keys.append(k)
    return keys


def _looks_like_single_bucket_output(out: dict) -> bool:
    if not isinstance(out, dict):
        return False
    # Common legacy/incorrect shapes we've observed from LLMs when schema isn't enforced.
    return any(k in out for k in ("gap_bucket", "bucket", "bucket_key")) and "gap_plans" not in out


def _coerce_bias(v: object) -> str:
    s = str(v or "").strip().upper()
    if s in {"BUY", "SELL", "WAIT"}:
        return s
    if s in {"BULLISH", "LONG"}:
        return "BUY"
    if s in {"BEARISH", "SHORT"}:
        return "SELL"
    return "WAIT"


def _merge_single_bucket_into_fallback(single: dict, fallback: dict, expected_keys: list[str]) -> dict:
    """
    If Gemini returns only one bucket, merge it into the fallback full-bucket structure.
    This ensures UI always gets all regimes (flat + all gaps) even when Gemini misbehaves.
    """
    if not isinstance(single, dict) or not isinstance(fallback, dict):
        return fallback
    plans = fallback.get("gap_plans")
    if not isinstance(plans, list):
        return fallback

    k = single.get("gap_bucket") or single.get("bucket_key") or single.get("bucket")
    if not isinstance(k, str) or k not in expected_keys:
        return fallback

    bias = _coerce_bias(single.get("bias"))
    # Accept common field aliases
    entry_zone = single.get("entry_zone") or single.get("entry") or None
    operator_zone = single.get("operator_zone") or single.get("operator") or None
    no_trade_zone = single.get("no_trade_zone") or single.get("no_trade") or None
    sl = single.get("sl")
    targets = single.get("targets") or []
    pools = single.get("liquidity_pools") or single.get("liquidity") or []
    reasons = single.get("reason_points") or single.get("reasons") or []

    for p in plans:
        if not isinstance(p, dict):
            continue
        if p.get("bucket_key") != k:
            continue
        p["bias"] = bias
        p["entry_zone"] = entry_zone
        p["operator_zone"] = operator_zone
        p["no_trade_zone"] = no_trade_zone
        p["sl"] = sl
        p["targets"] = targets
        p["liquidity_pools"] = pools
        p["reason_points"] = reasons
        break

    sp = fallback.get("summary_points")
    if not isinstance(sp, list):
        sp = []
    sp = list(sp)[:6]
    sp.insert(0, f"Gemini returned single-bucket output ({k}); merged into full regime plan.")
    fallback["summary_points"] = sp[:8]
    fallback["_fallback"] = True
    fallback["_partial_merge"] = True
    return fallback


def _validate_full_prediction_shape(out: dict, expected_keys: list[str]) -> str | None:
    """
    Returns None if OK, else returns a short reason string.
    """
    if not isinstance(out, dict):
        return "not_a_dict"
    if not isinstance(out.get("summary_points"), list):
        return "missing_summary_points"
    if not isinstance(out.get("base_levels"), dict):
        return "missing_base_levels"
    plans = out.get("gap_plans")
    if not isinstance(plans, list):
        return "missing_gap_plans"
    keys = []
    for p in plans:
        if isinstance(p, dict) and isinstance(p.get("bucket_key"), str):
            keys.append(p["bucket_key"])
    if len(keys) != len(expected_keys):
        return f"gap_plans_len_mismatch:{len(keys)}"
    if set(keys) != set(expected_keys):
        return "gap_plan_keys_mismatch"
    return None


def _replace_bucket_plan(out: dict, bucket_key: str, plan: dict) -> dict:
    if not isinstance(out, dict):
        return out
    plans = out.get("gap_plans")
    if not isinstance(plans, list):
        return out
    for i, p in enumerate(plans):
        if isinstance(p, dict) and p.get("bucket_key") == bucket_key:
            plans[i] = plan
            break
    out["gap_plans"] = plans
    return out


def _coerce_prediction_shape(out: dict) -> dict:
    """
    Accept common near-misses from Gemini and coerce to our expected shape:
    - "summary" -> "summary_points"
    - "summaryPoints" -> "summary_points"
    - "baseLevels" -> "base_levels"
    - "baseLevels"/"base_levels" casing variants
    - "gap_plans" as object map -> list with bucket_key
    - "gapPlans" -> "gap_plans"
    This runs BEFORE validation.
    """
    if not isinstance(out, dict):
        return out

    # Summary: accept common key variants and types.
    if "summary_points" not in out and isinstance(out.get("summaryPoints"), list):
        out["summary_points"] = out.get("summaryPoints")
    if "summary_points" not in out and isinstance(out.get("summaryPoints"), str):
        out["summary_points"] = [out.get("summaryPoints")]
    if "summary_points" not in out and isinstance(out.get("summary"), str):
        out["summary_points"] = [out.get("summary")]
    if isinstance(out.get("summary_points"), str):
        out["summary_points"] = [out.get("summary_points")]

    # Base levels: accept camelCase.
    if "base_levels" not in out and isinstance(out.get("baseLevels"), dict):
        out["base_levels"] = out.get("baseLevels")
    if "base_levels" not in out and isinstance(out.get("base_levels"), dict):
        out["base_levels"] = out.get("base_levels")

    # Gap plans: accept camelCase and object-map shapes.
    if "gap_plans" not in out and isinstance(out.get("gapPlans"), list):
        out["gap_plans"] = out.get("gapPlans")
    if "gap_plans" not in out and isinstance(out.get("gapPlans"), dict):
        out["gap_plans"] = out.get("gapPlans")

    if isinstance(out.get("gap_plans"), dict):
        plans = []
        for k, v in (out.get("gap_plans") or {}).items():
            if isinstance(v, dict):
                v2 = dict(v)
                v2.setdefault("bucket_key", str(k))
                plans.append(v2)
        out["gap_plans"] = plans

    return out


def _merge_partial_prediction_into_fallback(out: dict, fallback: dict, expected_keys: list[str]) -> dict:
    """
    Gemini sometimes returns valid JSON but misses a top-level key (often summary_points) or
    returns incomplete bucket objects. Instead of hard-failing into deterministic fallback,
    merge whatever Gemini produced into the deterministic baseline so UI still gets all regimes.

    This is NOT the same as "bucket-level planning" (which makes 11 Gemini calls). This keeps
    the original Gemini call result and only fills missing pieces.
    """
    if not isinstance(out, dict) or not isinstance(fallback, dict):
        return fallback

    merged = dict(fallback)
    changed = False

    # summary_points
    sp = out.get("summary_points")
    if isinstance(sp, list):
        merged["summary_points"] = sp
    elif isinstance(sp, str):
        merged["summary_points"] = [sp]
        changed = True
    else:
        # Auto-fill a minimal summary if Gemini omitted it but provided something else.
        if any(k in out for k in ("base_levels", "gap_plans", "gap_bucket", "bucket_key")):
            merged["summary_points"] = [
                "Gemini output missing summary_points; auto-filled.",
                "See bucket-wise plans for regime-specific levels.",
            ]
            changed = True

    # base_levels
    bl = out.get("base_levels")
    if isinstance(bl, dict):
        merged["base_levels"] = {**merged.get("base_levels", {}), **bl}

    # gap_plans
    plans_in = out.get("gap_plans")
    plan_map: dict[str, dict] = {}
    if isinstance(plans_in, list):
        for p in plans_in:
            if isinstance(p, dict):
                k = p.get("bucket_key") or p.get("gap_bucket") or p.get("bucket")
                if isinstance(k, str) and k:
                    plan_map[k] = p

    plans_out = []
    fb_plans = merged.get("gap_plans")
    fb_map: dict[str, dict] = {}
    if isinstance(fb_plans, list):
        for p in fb_plans:
            if isinstance(p, dict) and isinstance(p.get("bucket_key"), str):
                fb_map[p["bucket_key"]] = p

    for k in expected_keys:
        base = dict(fb_map.get(k) or {"bucket_key": k})
        p = plan_map.get(k)
        if isinstance(p, dict):
            base.update(p)
        base["bucket_key"] = k
        # Ensure bias is normalized even if schema was not enforced.
        base["bias"] = _coerce_bias(base.get("bias"))
        plans_out.append(base)
        if p is not None and p is not fb_map.get(k):
            changed = True

    merged["gap_plans"] = plans_out

    if changed:
        sp2 = merged.get("summary_points")
        if not isinstance(sp2, list):
            sp2 = []
        sp2 = list(sp2)[:6]
        sp2.append("Gemini output was partially merged with fallback to satisfy full regime schema.")
        merged["summary_points"] = sp2[:8]
        merged["_partial_merge"] = True

    # If we got ANY meaningful Gemini signal, this should not be considered a full deterministic fallback.
    if any(k in out for k in ("summary_points", "base_levels", "gap_plans", "gap_bucket", "bucket_key")):
        merged["_fallback"] = False

    return merged

def _truncate(s: str, n: int = 240) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _compact_rules(rules: list[dict], limit: int = 25) -> list[dict]:
    out = []
    for r in (rules or [])[: int(limit)]:
        out.append(
            {
                "id": r.get("id"),
                "category": r.get("category"),
                # Keep this extremely compact; long rule text increases token usage and can push responses to MAX_TOKENS.
                "name": _truncate(str(r.get("name") or ""), 80),
                "condition": _truncate(str(r.get("condition") or ""), 180),
                "action": _truncate(str(r.get("action") or ""), 180),
            }
        )
    return out


@dataclass(frozen=True)
class NextDayPrediction:
    ok: bool
    instrument: str
    security_id: str
    target_date: str
    training_start_date: str
    training_end_date: str
    prev_date_used: str
    prev_levels: dict
    stats: dict
    oi: dict | None
    retrieved_rules: dict
    prediction: dict


@dataclass(frozen=True)
class NextDayPredictor:
    """
    Next-day prediction engine:
    - Fetches historical 1m candles for training window ending on previous trading day.
    - Computes previous-day levels + multi-day stats + swing liquidity markers.
    - Optionally fetches live option chain OI walls (current snapshot).
    - Calls Gemini once to produce bucketed next-day levels for gap regimes.
    """

    api_key: str
    model: str
    db_path: str
    rulebook_path: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_s: float = 60.0
    max_retries: int = 4
    retry_backoff_s: float = 1.2

    def predict_next_day(
        self,
        *,
        instrument: str,
        security_id: str,
        exchange_segment: str,
        training_start_date: str,
        target_date: str,
        dhan_client_id: str | None,
        dhan_access_token: str,
        use_prev_day_oi_snapshot: bool = True,
    ) -> NextDayPrediction:
        # Resolve training end date = day before target_date (calendar day). If holiday/weekend,
        # Dhan will simply return last trading day data within the window; we pick the max date.
        t = datetime.strptime(target_date, "%Y-%m-%d")
        end_calendar = (t - timedelta(days=1)).strftime("%Y-%m-%d")

        from_dt = datetime.strptime(training_start_date + " 09:15:00", "%Y-%m-%d %H:%M:%S")
        to_dt = datetime.strptime(end_calendar + " 15:31:00", "%Y-%m-%d %H:%M:%S")
        if to_dt <= from_dt:
            raise ValueError("invalid period: training window must end after start date")

        req = DhanIntradayRequest(
            security_id=str(security_id),
            exchange_segment=str(exchange_segment),
            instrument="INDEX",
            interval="1",
            oi=False,
            from_dt=from_dt,
            to_dt=to_dt,
        )
        candles = fetch_intraday_candles(req, access_token=dhan_access_token)
        by = group_by_date(candles)
        dates = sorted(by.keys())
        if not dates:
            raise ValueError("no candles returned for training window")
        prev_date_used = dates[-1]
        prev_day_candles = sorted(by[prev_date_used], key=lambda c: c.dt)
        prev_levels = compute_prev_day_levels(prev_day_candles)

        # Multi-day stats (training_start_date .. prev_date_used)
        training_days = [d for d in dates if d >= training_start_date and d <= prev_date_used]
        all_training = []
        for d in training_days:
            all_training.extend(by[d])
        all_training = sorted(all_training, key=lambda c: c.dt)
        day_ranges = []
        for d in training_days:
            ds = by[d]
            h = max(c.high for c in ds)
            l = min(c.low for c in ds)
            day_ranges.append(float(h - l))
        avg_range = (sum(day_ranges) / len(day_ranges)) if day_ranges else None
        n_high = max(c.high for c in all_training)
        n_low = min(c.low for c in all_training)
        first_close = all_training[0].close
        last_close = all_training[-1].close
        trend_points = float(last_close - first_close)

        # Prev-day swing liquidity markers (1m + 5m)
        sb = StateBuilder()
        sb.on_new_day(prev=None, gap_up_threshold_points=30.0, gap_down_threshold_points=30.0, flat_threshold_points=15.0)
        last_state = None
        for c in prev_day_candles:
            last_state = sb.update(c)
        swings = {}
        dow = {}
        if isinstance(last_state, dict):
            swings = {
                "last_swing_high_1m": last_state.get("last_swing_high_1m"),
                "last_swing_low_1m": last_state.get("last_swing_low_1m"),
                "last_swing_high_5m": last_state.get("last_swing_high_5m"),
                "last_swing_low_5m": last_state.get("last_swing_low_5m"),
            }
            dow = {
                "dow_structure_1m": last_state.get("dow_structure_1m") or last_state.get("structure_1m") or last_state.get("structure"),
                "dow_structure_5m": last_state.get("dow_structure_5m") or last_state.get("structure_5m"),
                "market_type": last_state.get("market_type"),
                "comfort_risk": last_state.get("comfort_risk"),
                "operator_exit_risk": last_state.get("operator_exit_risk"),
                "overcrowding_risk": last_state.get("overcrowding_risk"),
            }

        stats = {
            "training_days": len(training_days),
            "n_day_high": float(n_high),
            "n_day_low": float(n_low),
            "avg_day_range": None if avg_range is None else float(avg_range),
            "trend_points": float(trend_points),
            "prev_day_range": float(prev_levels.high - prev_levels.low),
            "prev_day_close_location_pct": (
                None
                if (prev_levels.high - prev_levels.low) <= 0
                else float((prev_levels.close - prev_levels.low) / (prev_levels.high - prev_levels.low))
            ),
            "swings": swings,
            "dow": dow,
        }

        # Previous-day OI snapshot (captured and stored in DB).
        # This is the correct way to have "historical" option-chain OI before market open.
        oi_snapshot = None
        if use_prev_day_oi_snapshot:
            try:
                store0 = SqliteStore(self.db_path)
                row = store0.get_latest_oi_snapshot(date=prev_date_used, security_id=str(security_id))
                store0.close()
                if row and isinstance(row.get("snapshot"), dict):
                    oi_snapshot = row.get("snapshot")
            except Exception:
                oi_snapshot = None

        oi_features = None
        if oi_snapshot is not None:
            f = derive_oi_features(spot_price=float(prev_levels.close), snapshot=oi_snapshot)
            oi_features = {"ok": True, "snapshot": oi_snapshot, "features": f}

        # Gap buckets (points from prev_close)
        gap_buckets = [
            # Flat opening = within +/- 30 points of prev close.
            {"key": "flat_open_30", "type": "flat", "min": -30, "max": 30},
            {"key": "gap_up_50_100", "type": "gap_up", "min": 50, "max": 100},
            {"key": "gap_up_100_150", "type": "gap_up", "min": 100, "max": 150},
            {"key": "gap_up_150_200", "type": "gap_up", "min": 150, "max": 200},
            {"key": "gap_up_200_250", "type": "gap_up", "min": 200, "max": 250},
            {"key": "gap_up_250_plus", "type": "gap_up", "min": 250, "max": None},
            {"key": "gap_down_50_100", "type": "gap_down", "min": -100, "max": -50},
            {"key": "gap_down_100_150", "type": "gap_down", "min": -150, "max": -100},
            {"key": "gap_down_150_200", "type": "gap_down", "min": -200, "max": -150},
            {"key": "gap_down_200_250", "type": "gap_down", "min": -250, "max": -200},
            {"key": "gap_down_250_plus", "type": "gap_down", "min": None, "max": -250},
        ]

        # Retrieval for planning prompt
        retrieval_state = {
            "zone": "fair",
            # Multi-regime prediction: do not bias retrieval only to gap-up rules.
            "gap_type": "multi",
            "structure": "unknown",
            "market_type": "unknown",
            "validity_ok": True,
            "confirmed_long": False,
            "confirmed_short": False,
            "last_swing_high_1m": swings.get("last_swing_high_1m"),
            "last_swing_low_1m": swings.get("last_swing_low_1m"),
            "last_swing_high_5m": swings.get("last_swing_high_5m"),
            "last_swing_low_5m": swings.get("last_swing_low_5m"),
            "reclaimed_last_swing_high": False,
            "reclaimed_last_swing_low": False,
            "reclaimed_prev_pdh": False,
            "reclaimed_prev_pdl": False,
            "dow": dow,
            "oi": oi_features or {"ok": False},
        }
        retrieved = retrieve_rulebook_rules(self.rulebook_path, retrieval_state, limit=18)

        prompt_payload = {
            "instrument": instrument,
            "security_id": str(security_id),
            "target_date": target_date,
            "training_window": {"start": training_start_date, "end": prev_date_used, "days": len(training_days)},
            "prev_levels": prev_levels.__dict__,
            "stats": stats,
            "oi": oi_features,
            "gap_buckets": gap_buckets,
            "retrieved": {
                "rulebook_version": retrieved.rulebook_version,
                # Keep prompt small and stable (reduces Gemini JSON failures).
                "rules": _compact_rules(retrieved.rules, limit=12),
            },
        }

        # Bump schema_version whenever the expected output contract changes (e.g., adding flat bucket),
        # otherwise older cached predictions can be returned with the wrong shape.
        req_hash = _hash_request({"predict_next_day": prompt_payload, "schema_version": 3})
        store = SqliteStore(self.db_path)
        cached = store.get_llm_cache(req_hash, self.model)
        if cached is not None:
            # Validate cached shape; if it doesn't match the current contract, ignore cache and recompute.
            expected_keys = _expected_bucket_keys(gap_buckets)
            try:
                if isinstance(cached, dict) and isinstance(cached.get("gap_plans"), dict):
                    plans = []
                    for k, v in cached["gap_plans"].items():
                        if isinstance(v, dict):
                            v2 = dict(v)
                            v2.setdefault("bucket_key", str(k))
                            plans.append(v2)
                    cached["gap_plans"] = plans
            except Exception:
                pass
            try:
                if not (isinstance(cached, dict) and _validate_full_prediction_shape(cached, expected_keys) is None):
                    cached = None
            except Exception:
                cached = None
            if cached is not None:
                store.close()
                cached = _sanitize_prediction_output(cached)
                return NextDayPrediction(
                    ok=True,
                    instrument=instrument,
                    security_id=str(security_id),
                    target_date=target_date,
                    training_start_date=training_start_date,
                    training_end_date=prev_date_used,
                    prev_date_used=prev_date_used,
                    prev_levels={**prev_levels.__dict__, "symbol": instrument},
                    stats=stats,
                    oi=oi_features,
                    retrieved_rules={"rulebook_version": retrieved.rulebook_version, "count": len(retrieved.rules)},
                    prediction=cached,
                )
            # Cached value was invalid for the current schema; fall through to recompute with Gemini.

        bucket_schema = {
            "type": "object",
            # Keep schema minimal; some Gemini deployments reject minItems/maxItems and will 400 -> schema dropped.
            "properties": {
                "bucket_key": {"type": "string"},
                "gap_points_min": {"type": ["number", "null"]},
                "gap_points_max": {"type": ["number", "null"]},
                "bias": {"type": "string", "enum": ["BUY", "SELL", "WAIT"]},
                "entry_zone": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                },
                "operator_zone": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                },
                "no_trade_zone": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                },
                "sl": {"type": ["number", "null"]},
                "targets": {"type": "array", "items": {"type": "number"}},
                "liquidity_pools": {"type": "array", "items": {"type": "number"}},
                "reason_points": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["bucket_key", "gap_points_min", "gap_points_max", "bias", "entry_zone", "operator_zone", "no_trade_zone", "sl", "targets", "liquidity_pools", "reason_points"],
        }

        schema = {
            "type": "object",
            # Keep schema minimal for maximum compatibility with Gemini structured outputs.
            "properties": {
                "summary_points": {"type": "array", "items": {"type": "string"}},
                "base_levels": {
                    "type": "object",
                    "properties": {
                        "prev_close": {"type": "number"},
                        "PDH": {"type": "number"},
                        "PDL": {"type": "number"},
                        "operator_sell_zone": {"type": "array", "items": {"type": "number"}},
                        "operator_buy_zone": {"type": "array", "items": {"type": "number"}},
                        "no_trade_zone": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["prev_close", "PDH", "PDL", "operator_sell_zone", "operator_buy_zone", "no_trade_zone"],
                },
                "gap_plans": {"type": "array", "items": bucket_schema},
            },
            "required": ["summary_points", "base_levels", "gap_plans"],
        }

        sys = (
            "You are a next-day SL-hunting prediction engine for NIFTY/BANKNIFTY. "
            "Use ONLY the provided historical context, OI snapshot (if present), and retrieved rulebook rules. "
            "For each gap bucket, output actionable levels (operator zones, liquidity pools) and a one-sided bias for that bucket. "
            "You MUST output exactly 11 gap_plans (one per provided gap_buckets key). "
            "Be conservative: if unsure for a bucket, set bias=WAIT and set entry_zone/operator_zone/no_trade_zone/sl to null (targets/liquidity_pools/reason_points still required). "
            "Keep arrays small: targets max 3 numbers; liquidity_pools max 3 numbers; reason_points max 3 short points. "
            "Keep each string very short (<= 180 chars). Do not repeat input payload. Do not add explanations outside JSON. "
            "All strings must be SINGLE-LINE (no raw newlines). If needed, use '\\n' inside strings. "
            "Output STRICT JSON only. No markdown. No comments. No trailing commas. Use double quotes for all JSON keys/strings."
        )

        body = {
            "systemInstruction": {"parts": [{"text": sys}]},
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            ],
            "contents": [{"role": "user", "parts": [{"text": json.dumps(prompt_payload, ensure_ascii=False)}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
                # Structured outputs (JSON Schema) per Gemini API docs (REST: generationConfig.responseJsonSchema).
                "responseJsonSchema": schema,
            },
        }

        url = f"{self.base_url}/models/{_normalize_model_id(self.model)}:generateContent"
        headers = {"x-goog-api-key": self.api_key}
        def _fallback_prediction(err: str) -> dict:
            pc = float(prev_levels.close)
            pdh = float(prev_levels.high)
            pdl = float(prev_levels.low)
            rng = max(1.0, pdh - pdl)
            q1 = float(pdl + 0.25 * rng)
            q3 = float(pdl + 0.75 * rng)
            base = {
                "prev_close": pc,
                "PDH": pdh,
                "PDL": pdl,
                "operator_sell_zone": [pdh - 10.0, pdh + 25.0],
                "operator_buy_zone": [pdl - 25.0, pdl + 10.0],
                "no_trade_zone": [q1, q3],
            }

            def _bucket_plan(b: dict) -> dict:
                typ = b["type"]
                k = str(b.get("key"))
                # expected open zone for the bucket
                if typ == "gap_up":
                    gmin = float(b.get("min") or 0.0)
                    gmax = float(b.get("max") or (gmin + 50.0))
                    open_zone = [pc + gmin, pc + gmax]
                elif typ == "gap_down":
                    # negative points
                    gmin = float(b.get("min") if b.get("min") is not None else (-(abs(b.get("max") or 250.0) + 50.0)))
                    gmax = float(b.get("max") or -50.0)
                    open_zone = [pc + gmin, pc + gmax]
                else:
                    # flat within +/- 30
                    open_zone = [pc - 30.0, pc + 30.0]

                if typ == "gap_up":
                    bias = "SELL"
                    # If open is far above PDH, use open-zone rejection; else use PDH zone.
                    if open_zone[0] >= (pdh + 40.0):
                        entry_zone = [open_zone[0] - 10.0, open_zone[1] + 10.0]
                        sl = open_zone[1] + 30.0
                    else:
                        entry_zone = [pdh - 5.0, pdh + 15.0]
                        sl = pdh + 30.0
                    targets = [pc, q1, pdl]
                    pools = [pdh, pc, pdl]
                elif typ == "gap_down":
                    bias = "BUY"
                    if open_zone[1] <= (pdl - 40.0):
                        entry_zone = [open_zone[0] - 10.0, open_zone[1] + 10.0]
                        sl = open_zone[0] - 30.0
                    else:
                        entry_zone = [pdl - 15.0, pdl + 5.0]
                        sl = pdl - 30.0
                    targets = [pc, q3, pdh]
                    pools = [pdl, pc, pdh]
                else:
                    # Flat opening: derive a conservative bias from OI + prior trend/close location.
                    bias = "WAIT"
                    try:
                        b0 = (oi_features or {}).get("features", {}).get("oi_bias")
                        if str(b0).lower() == "bullish":
                            bias = "BUY"
                        elif str(b0).lower() == "bearish":
                            bias = "SELL"
                    except Exception:
                        pass

                    # Prefer mid-range scalps only after liquidity sweep; keep tight risk.
                    if bias == "BUY":
                        entry_zone = [pc - 20.0, pc - 5.0]
                        sl = pc - 40.0
                        targets = [pc + 20.0, q3, pdh]
                        pools = [pc, q3, pdh]
                    elif bias == "SELL":
                        entry_zone = [pc + 5.0, pc + 20.0]
                        sl = pc + 40.0
                        targets = [pc - 20.0, q1, pdl]
                        pools = [pc, q1, pdl]
                    else:
                        entry_zone = None
                        sl = None
                        targets = [pc, q1, q3]
                        pools = [pc, q1, q3]
                return {
                    "bucket_key": k,
                    "gap_points_min": None if b.get("min") is None else float(b.get("min")),
                    "gap_points_max": None if b.get("max") is None else float(b.get("max")),
                    "bias": bias,
                    "entry_zone": entry_zone,
                    "operator_zone": base["operator_sell_zone"] if bias == "SELL" else base["operator_buy_zone"],
                    "no_trade_zone": base["no_trade_zone"],
                    "sl": sl,
                    "targets": targets[:3],
                    "liquidity_pools": pools[:3],
                    "reason_points": [
                        "Fallback plan (Gemini output invalid).",
                        "Levels anchored to prev_close/PDH/PDL quartiles.",
                        "Wait for sweep+reclaim at PDH/PDL before commitment.",
                    ],
                }

            gap_plans = [_bucket_plan(b) for b in gap_buckets]
            return {
                "summary_points": [
                    "Fallback prediction used due to Gemini JSON error.",
                    f"Error: {err}",
                ],
                "base_levels": base,
                "gap_plans": gap_plans,
                "_fallback": True,
            }

        def _bucket_llm_plan(client: httpx.Client, bucket: dict) -> tuple[dict | None, str | None]:
            """
            More reliable fallback path: plan one bucket per request (small output -> fewer schema failures).
            Returns (plan, error). plan includes full bucket fields if successful.
            """
            bkey = str(bucket.get("key") or "")
            bp = {
                "bucket": bucket,
                "instrument": prompt_payload.get("instrument"),
                "security_id": prompt_payload.get("security_id"),
                "target_date": prompt_payload.get("target_date"),
                "prev_levels": prompt_payload.get("prev_levels"),
                "stats": prompt_payload.get("stats"),
                "oi": prompt_payload.get("oi"),
                "retrieved": prompt_payload.get("retrieved"),
            }

            bucket_only_schema = {
                "type": "object",
                "properties": {
                    "bucket_key": {"type": "string"},
                    "gap_points_min": {"type": ["number", "null"]},
                    "gap_points_max": {"type": ["number", "null"]},
                    "bias": {"type": "string", "enum": ["BUY", "SELL", "WAIT"]},
                    "entry_zone": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                    },
                    "operator_zone": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                    },
                    "no_trade_zone": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                    },
                    "sl": {"type": ["number", "null"]},
                    "targets": {"type": "array", "items": {"type": "number"}},
                    "liquidity_pools": {"type": "array", "items": {"type": "number"}},
                    "reason_points": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "bucket_key",
                    "gap_points_min",
                    "gap_points_max",
                    "bias",
                    "entry_zone",
                    "operator_zone",
                    "no_trade_zone",
                    "sl",
                    "targets",
                    "liquidity_pools",
                    "reason_points",
                ],
            }

            sys_bucket = (
                "You are a next-day SL-hunting planner. "
                "Plan ONLY the provided bucket. Output strict JSON only, matching schema. "
                "One-sided bias: BUY or SELL; if unclear use WAIT. "
                "Do not mention both longs and shorts in reasons. "
                "Use OI walls + prev-day swings + PDH/PDL and Dow structure. "
                "Keep arrays small (<=3)."
            )

            body_bucket = {
                "systemInstruction": {"parts": [{"text": sys_bucket}]},
                "safetySettings": body.get("safetySettings"),
                "contents": [{"role": "user", "parts": [{"text": json.dumps(bp, ensure_ascii=False)}]}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 700,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": bucket_only_schema,
                },
            }

            try:
                last_err: str | None = None
                for attempt in range(1, 3):
                    r = client.post(url, headers=headers, json=body_bucket)
                    if r.status_code >= 400 and ("responseJsonSchema" in body_bucket["generationConfig"]):
                        body_bucket["generationConfig"].pop("responseJsonSchema", None)
                        r = client.post(url, headers=headers, json=body_bucket)
                    r.raise_for_status()
                    data = r.json()
                    text = _extract_candidate_text(data)
                    if not text:
                        fb = _gemini_feedback(data)
                        last_err = f"empty_candidate_text:{json.dumps(fb, ensure_ascii=False)}"
                        time.sleep(0.2 * attempt)
                        continue
                    outb = _extract_json(text)
                    if not isinstance(outb, dict):
                        last_err = "bucket_output_not_object"
                        time.sleep(0.2 * attempt)
                        continue
                    outb["bucket_key"] = bkey
                    outb["gap_points_min"] = None if bucket.get("min") is None else float(bucket.get("min"))
                    outb["gap_points_max"] = None if bucket.get("max") is None else float(bucket.get("max"))
                    outb["bias"] = _coerce_bias(outb.get("bias"))
                    return outb, None
                return None, _sanitize_error(last_err or "bucket_plan_failed")
            except Exception as e:
                return None, _sanitize_error(str(e))

        def _bucket_level_plans(client: httpx.Client, why: str) -> dict:
            """
            Produce a full 11-bucket plan by running Gemini per bucket.
            Falls back per-bucket if Gemini fails for that bucket.
            """
            base_out = _fallback_prediction(f"bucket_level:{why}")
            any_llm = False
            any_bucket_fallback = False
            for b in gap_buckets:
                plan, err = _bucket_llm_plan(client, b)
                if plan is None:
                    any_bucket_fallback = True
                    continue
                any_llm = True
                _replace_bucket_plan(base_out, str(b.get("key")), plan)

            sp = [
                "Bucket-level planning used (full-output schema failed).",
                f"Reason: {why}",
                "Some buckets may still use deterministic fallback if Gemini failed for that bucket.",
            ]
            base_out["summary_points"] = sp[:8]
            base_out["_fallback"] = not any_llm
            base_out["_bucket_level"] = True
            base_out["_partial_bucket_fallback"] = bool(any_bucket_fallback)
            return base_out

        out: dict | None = None
        last_err: str | None = None
        expected_keys = _expected_bucket_keys(gap_buckets)
        try:
            data: dict | None = None
            schema_dropped = False
            with httpx.Client(timeout=self.timeout_s) as client:
                for attempt in range(1, int(self.max_retries) + 1):
                    r = client.post(url, headers=headers, json=body)
                    if r.status_code >= 400 and ("responseJsonSchema" in body["generationConfig"]):
                        # Older deployments may not support schemas. Drop and retry.
                        body["generationConfig"].pop("responseJsonSchema", None)
                        schema_dropped = True
                        r = client.post(url, headers=headers, json=body)
                    try:
                        r.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        status = getattr(getattr(e, "response", None), "status_code", None)
                        if status in (429, 500, 502, 503, 504) and attempt < int(self.max_retries):
                            time.sleep(float(self.retry_backoff_s) * (2 ** (attempt - 1)))
                            continue
                        raise
                    data = r.json()
                    text = _extract_candidate_text(data)
                    if not text:
                        if attempt < int(self.max_retries):
                            time.sleep(float(self.retry_backoff_s) * (2 ** (attempt - 1)))
                            continue
                        fb = _gemini_feedback(data)
                        raise ValueError(f"empty_candidate_text: {json.dumps(fb, ensure_ascii=False)}")
                    try:
                        out = _extract_json(text)
                        break
                    except Exception as pe:
                        # If the candidate was truncated due to max tokens, increase token budget and retry.
                        fb = _gemini_feedback(data)
                        if fb.get("finish_reason") == "MAX_TOKENS" and attempt < int(self.max_retries):
                            cur = int(body.get("generationConfig", {}).get("maxOutputTokens") or 4096)
                            body["generationConfig"]["maxOutputTokens"] = min(cur * 2, 6144)
                            time.sleep(float(self.retry_backoff_s) * (2 ** (attempt - 1)))
                            continue
                        # If schema had to be dropped, parsing can still fail; retry a couple times.
                        if attempt < int(self.max_retries):
                            time.sleep(float(self.retry_backoff_s) * (2 ** (attempt - 1)))
                            continue
                        raise pe

            if out is None:
                raise RuntimeError("gemini_error: empty_or_unparseable_response" + (" (schema_dropped)" if schema_dropped else ""))

            # Coerce common key variants before validation/repair logic.
            out = _coerce_prediction_shape(out if isinstance(out, dict) else {})

            # If Gemini returned a single-bucket legacy output, merge it into full-bucket fallback.
            if _looks_like_single_bucket_output(out):
                # Prefer bucket-level planning (11 small calls) over merging a single bucket,
                # because the user expects ALL regimes to be Gemini-driven whenever possible.
                out_bucket = _bucket_level_plans(client, "single_bucket_output")
                if isinstance(out_bucket, dict) and out_bucket.get("_fallback") is not True:
                    out = out_bucket
                else:
                    out = _merge_single_bucket_into_fallback(out, _fallback_prediction("single_bucket_output"), expected_keys)

            # Validate shape; if Gemini returned a single bucket or wrong schema, try one repair call.
            bad = _validate_full_prediction_shape(out, expected_keys)
            if bad is not None:
                # If only small parts are missing (e.g., summary_points), merge partial output into fallback first.
                merged0 = _merge_partial_prediction_into_fallback(out, _fallback_prediction(f"partial_merge:{bad}"), expected_keys)
                bad0 = _validate_full_prediction_shape(merged0, expected_keys)
                if bad0 is None:
                    out = merged0
                else:
                # Attempt a repair call (still small; reuses same prompt payload).
                    repair_payload = {
                        "note": "The previous output did not match the required schema. Re-emit strictly per schema.",
                        "expected_bucket_keys": expected_keys,
                        "gap_buckets": gap_buckets,
                        "prev_levels": prompt_payload.get("prev_levels"),
                        "stats": prompt_payload.get("stats"),
                        "oi": prompt_payload.get("oi"),
                        "retrieved": prompt_payload.get("retrieved"),
                        "previous_output": out,
                    }
                    body_repair = {
                        "systemInstruction": body.get("systemInstruction"),
                        "safetySettings": body.get("safetySettings"),
                        "contents": [{"role": "user", "parts": [{"text": json.dumps(repair_payload, ensure_ascii=False)}]}],
                        "generationConfig": {
                            "temperature": 0,
                            "maxOutputTokens": 4096,
                            "responseMimeType": "application/json",
                            "responseJsonSchema": schema,
                        },
                    }
                    r3 = client.post(url, headers=headers, json=body_repair)
                    if r3.status_code >= 400 and ("responseJsonSchema" in body_repair["generationConfig"]):
                        body_repair["generationConfig"].pop("responseJsonSchema", None)
                        r3 = client.post(url, headers=headers, json=body_repair)
                    r3.raise_for_status()
                    data3 = r3.json()
                    text3 = _extract_candidate_text(data3)
                    out3 = _extract_json(text3)
                    out3 = _coerce_prediction_shape(out3 if isinstance(out3, dict) else {})
                    # Merge partial repair output if needed (common: missing summary_points).
                    out3m = _merge_partial_prediction_into_fallback(out3, _fallback_prediction(f"partial_merge_repair:{bad}"), expected_keys)
                    bad3 = _validate_full_prediction_shape(out3m, expected_keys)
                    if bad3 is None:
                        out = out3m
                    else:
                        # Final fallback: build full plan via per-bucket Gemini calls (much more reliable).
                        out = _bucket_level_plans(client, f"gemini_invalid_shape:{bad}->{bad3}" + (" (schema_dropped)" if schema_dropped else ""))
        except Exception as e:
            last_err = _sanitize_error(str(e))
            # One parse retry: no schema.
            try:
                body2 = {
                    "systemInstruction": body.get("systemInstruction"),
                    "contents": body["contents"],
                    "generationConfig": {
                        "temperature": 0,
                        "maxOutputTokens": 4096,
                        "responseMimeType": "application/json",
                    },
                }
                # Keep the client open so we can immediately fall back to bucket-level planning
                # without accidentally using a closed httpx client.
                with httpx.Client(timeout=self.timeout_s) as client2:
                    r2 = client2.post(url, headers=headers, json=body2)
                    r2.raise_for_status()
                    data2 = r2.json()
                    text2 = _extract_candidate_text(data2)
                    out = _extract_json(text2)
                    out = _coerce_prediction_shape(out if isinstance(out, dict) else {})

                    if _looks_like_single_bucket_output(out):
                        out_bucket = _bucket_level_plans(client2, "single_bucket_output_no_schema")
                        if isinstance(out_bucket, dict) and out_bucket.get("_fallback") is not True:
                            out = out_bucket
                        else:
                            out = _merge_single_bucket_into_fallback(out, _fallback_prediction("single_bucket_output_no_schema"), expected_keys)

                    bad2 = _validate_full_prediction_shape(out, expected_keys)
                    if bad2 is not None:
                        # Try partial merge first; if still invalid, go bucket-by-bucket to guarantee full regime output.
                        merged2 = _merge_partial_prediction_into_fallback(
                            out, _fallback_prediction(f"partial_merge_no_schema:{bad2}"), expected_keys
                        )
                        bad2m = _validate_full_prediction_shape(merged2, expected_keys)
                        if bad2m is None:
                            out = merged2
                        else:
                            out = _bucket_level_plans(client2, f"gemini_invalid_shape:{bad2}")
                    last_err = None
            except Exception as e2:
                last_err = f"{last_err} / retry_failed: {_sanitize_error(str(e2))}"
                out = _fallback_prediction(last_err)

        # Cache only valid Gemini outputs. Do NOT cache fallbacks; otherwise a transient Gemini failure
        # would permanently poison future predictions for the same payload/model.
        # Always sanitize output before caching/returning.
        out = _sanitize_prediction_output(out if isinstance(out, dict) else {})

        if not (isinstance(out, dict) and out.get("_fallback") is True):
            cache_id = f"pred_{req_hash[:16]}"
            store.set_llm_cache(cache_id, datetime.now(timezone.utc).isoformat(), self.model, req_hash, out)
        store.close()

        # Normalize output shape if Gemini returns gap_plans as an object (older format).
        try:
            if isinstance(out, dict) and isinstance(out.get("gap_plans"), dict):
                plans = []
                for k, v in out["gap_plans"].items():
                    if isinstance(v, dict):
                        v2 = dict(v)
                        v2.setdefault("bucket_key", str(k))
                        plans.append(v2)
                out["gap_plans"] = plans
        except Exception:
            pass

        return NextDayPrediction(
            ok=True,
            instrument=instrument,
            security_id=str(security_id),
            target_date=target_date,
            training_start_date=training_start_date,
            training_end_date=prev_date_used,
            prev_date_used=prev_date_used,
            prev_levels={**prev_levels.__dict__, "symbol": instrument},
            stats=stats,
            oi=oi_features,
            retrieved_rules={"rulebook_version": retrieved.rulebook_version, "count": len(retrieved.rules)},
            prediction=out,
        )
