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
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
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
        return "\n".join(texts).strip()
    except Exception:
        return ""


def _normalize_model_id(model: str) -> str:
    m = (model or "").strip()
    if m.startswith("models/"):
        m = m[len("models/") :]
    return m


def _sanitize_error(s: str) -> str:
    s = s or ""
    return s.replace("\n", " ").strip()


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
    timeout_s: float = 25.0
    max_retries: int = 3
    retry_backoff_s: float = 0.8

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
        if isinstance(last_state, dict):
            swings = {
                "last_swing_high_1m": last_state.get("last_swing_high_1m"),
                "last_swing_low_1m": last_state.get("last_swing_low_1m"),
                "last_swing_high_5m": last_state.get("last_swing_high_5m"),
                "last_swing_low_5m": last_state.get("last_swing_low_5m"),
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
            "gap_type": "gap_up",
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
            "oi": oi_features or {"ok": False},
        }
        retrieved = retrieve_rulebook_rules(self.rulebook_path, retrieval_state, limit=35)

        prompt_payload = {
            "instrument": instrument,
            "security_id": str(security_id),
            "target_date": target_date,
            "training_window": {"start": training_start_date, "end": prev_date_used, "days": len(training_days)},
            "prev_levels": prev_levels.__dict__,
            "stats": stats,
            "oi": oi_features,
            "gap_buckets": gap_buckets,
            "retrieved": {"rulebook_version": retrieved.rulebook_version, "rules": retrieved.rules},
        }

        req_hash = _hash_request({"predict_next_day": prompt_payload, "schema_version": 1})
        store = SqliteStore(self.db_path)
        cached = store.get_llm_cache(req_hash, self.model)
        if cached is not None:
            store.close()
            return NextDayPrediction(
                ok=True,
                instrument=instrument,
                security_id=str(security_id),
                target_date=target_date,
                training_start_date=training_start_date,
                training_end_date=prev_date_used,
                prev_date_used=prev_date_used,
                prev_levels=prev_levels.__dict__,
                stats=stats,
                oi=oi_features,
                retrieved_rules={"rulebook_version": retrieved.rulebook_version, "count": len(retrieved.rules)},
                prediction=cached,
            )

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary_points": {"type": "array", "items": {"type": "string"}},
                "base_levels": {
                    "type": "object",
                    "additionalProperties": False,
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
                "gap_plans": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {k["key"]: {"type": "object"} for k in gap_buckets},
                },
            },
            "required": ["summary_points", "base_levels", "gap_plans"],
        }

        sys = (
            "You are a next-day SL-hunting prediction engine for NIFTY/BANKNIFTY. "
            "Use ONLY the provided historical context, OI snapshot (if present), and retrieved rulebook rules. "
            "For each gap bucket, output actionable levels (operator zones, liquidity pools) and a one-sided bias for that bucket. "
            "Be conservative: if unsure for a bucket, set its plan to WAIT and explain briefly. "
            "All strings must be SINGLE-LINE (no raw newlines). If needed, use '\\n' inside strings. "
            "Output STRICT JSON only. No markdown."
        )

        body = {
            "contents": [{"role": "user", "parts": [{"text": sys + "\n\n" + json.dumps(prompt_payload, ensure_ascii=False)}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 1200,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }

        url = f"{self.base_url}/models/{_normalize_model_id(self.model)}:generateContent"
        params = {"key": self.api_key}
        data: dict | None = None
        last_err: str | None = None
        with httpx.Client(timeout=self.timeout_s) as client:
            for attempt in range(1, int(self.max_retries) + 1):
                try:
                    r = client.post(url, params=params, json=body)
                    if r.status_code >= 400 and "responseSchema" in body["generationConfig"]:
                        body["generationConfig"].pop("responseSchema", None)
                        r = client.post(url, params=params, json=body)
                    try:
                        r.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        status = getattr(getattr(e, "response", None), "status_code", None)
                        if status in (429, 500, 502, 503, 504) and attempt < int(self.max_retries):
                            last_err = _sanitize_error(str(e))
                            time.sleep(float(self.retry_backoff_s) * (2 ** (attempt - 1)))
                            continue
                        raise
                    data = r.json()
                    last_err = None
                    break
                except Exception as e:
                    last_err = _sanitize_error(str(e))
                    break

        if data is None:
            store.close()
            raise RuntimeError(f"gemini_error: {last_err or 'unknown_error'}")

        # Parse response
        try:
            text = _extract_candidate_text(data)
            out = _extract_json(text)
        except Exception as e:
            # One parse retry: ask again with no schema and tighter token limit.
            try:
                body2 = {
                    "contents": body["contents"],
                    "generationConfig": {
                        "temperature": 0,
                        "maxOutputTokens": 900,
                        "responseMimeType": "application/json",
                    },
                }
                with httpx.Client(timeout=self.timeout_s) as client:
                    r2 = client.post(url, params=params, json=body2)
                    r2.raise_for_status()
                    data2 = r2.json()
                text2 = _extract_candidate_text(data2)
                out = _extract_json(text2)
            except Exception as e2:
                store.close()
                raise RuntimeError(
                    f"gemini_parse_error: {_sanitize_error(str(e))} / retry_failed: {_sanitize_error(str(e2))}"
                ) from e2

        cache_id = f"pred_{req_hash[:16]}"
        store.set_llm_cache(cache_id, datetime.now(timezone.utc).isoformat(), self.model, req_hash, out)
        store.close()

        return NextDayPrediction(
            ok=True,
            instrument=instrument,
            security_id=str(security_id),
            target_date=target_date,
            training_start_date=training_start_date,
            training_end_date=prev_date_used,
            prev_date_used=prev_date_used,
            prev_levels=prev_levels.__dict__,
            stats=stats,
            oi=oi_features,
            retrieved_rules={"rulebook_version": retrieved.rulebook_version, "count": len(retrieved.rules)},
            prediction=out,
        )
