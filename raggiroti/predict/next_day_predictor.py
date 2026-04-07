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

        req_hash = _hash_request({"predict_next_day": prompt_payload, "schema_version": 1})
        store = SqliteStore(self.db_path)
        cached = store.get_llm_cache(req_hash, self.model)
        if cached is not None:
            store.close()
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

        bucket_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "bucket_key": {"type": "string"},
                "gap_points_min": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "gap_points_max": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "bias": {"type": "string", "enum": ["BUY", "SELL", "WAIT"]},
                "entry_zone": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        {"type": "null"},
                    ]
                },
                "operator_zone": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        {"type": "null"},
                    ]
                },
                "no_trade_zone": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        {"type": "null"},
                    ]
                },
                "sl": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "targets": {"type": "array", "items": {"type": "number"}, "maxItems": 5},
                "liquidity_pools": {"type": "array", "items": {"type": "number"}, "maxItems": 8},
                "reason_points": {"type": "array", "items": {"type": "string", "maxLength": 180}, "maxItems": 4},
            },
            "required": ["bucket_key", "gap_points_min", "gap_points_max", "bias", "entry_zone", "operator_zone", "no_trade_zone", "sl", "targets", "liquidity_pools", "reason_points"],
        }

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary_points": {"type": "array", "items": {"type": "string", "maxLength": 220}, "maxItems": 8},
                "base_levels": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "prev_close": {"type": "number"},
                        "PDH": {"type": "number"},
                        "PDL": {"type": "number"},
                        "operator_sell_zone": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "operator_buy_zone": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "no_trade_zone": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    },
                    "required": ["prev_close", "PDH", "PDL", "operator_sell_zone", "operator_buy_zone", "no_trade_zone"],
                },
                "gap_plans": {"type": "array", "items": bucket_schema, "minItems": 11, "maxItems": 11},
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
                # Use JSON Schema structured outputs for reliable JSON.
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

        out: dict | None = None
        last_err: str | None = None
        try:
            data: dict | None = None
            schema_dropped = False
            with httpx.Client(timeout=self.timeout_s) as client:
                for attempt in range(1, int(self.max_retries) + 1):
                    r = client.post(url, headers=headers, json=body)
                    if r.status_code >= 400 and "responseJsonSchema" in body["generationConfig"]:
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
                with httpx.Client(timeout=self.timeout_s) as client:
                    r2 = client.post(url, headers=headers, json=body2)
                    r2.raise_for_status()
                    data2 = r2.json()
                text2 = _extract_candidate_text(data2)
                out = _extract_json(text2)
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
