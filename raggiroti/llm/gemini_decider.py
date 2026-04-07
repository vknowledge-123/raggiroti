from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from raggiroti.storage.sqlite_db import SqliteStore


def _hash_request(payload: dict) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _extract_candidate_text(data: dict) -> str:
    """
    Extracts text from Gemini generateContent response robustly.
    Gemini may return multiple parts; we concatenate all text parts.
    """
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

def _sanitize_error(s: str) -> str:
    # Prevent leaking API keys into UI/DB logs. httpx errors may include the full URL.
    s = s or ""
    s = re.sub(r"(key=)[^&\s]+", r"\1***", s)
    return s


def _normalize_model_id(model: str) -> str:
    """
    Accept both forms:
      - "gemini-2.0-flash-001"
      - "models/gemini-2.0-flash-001"
    because ListModels returns names prefixed with "models/".
    """
    m = (model or "").strip()
    if m.startswith("models/"):
        m = m[len("models/") :]
    return m


@dataclass(frozen=True)
class GeminiDecider:
    """
    Gemini-based per-candle decider.

    Notes:
    - Designed for small prompts: state + top 10-30 retrieved rules.
    - Uses temperature=0 for consistency.
    - Caches responses in SQLite by request hash (helps replay/backtests).
    """

    api_key: str
    model: str
    db_path: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_s: float = 15.0
    max_retries: int = 3
    retry_backoff_s: float = 0.6

    def decide(self, state: dict, retrieved: dict) -> dict:
        payload = {"state": state, "retrieved": retrieved, "schema_version": 1}
        req_hash = _hash_request(payload)
        store = SqliteStore(self.db_path)
        cached = store.get_llm_cache(req_hash, self.model)
        if cached is not None:
            store.close()
            return cached

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": ["BUY", "SELL", "WAIT", "EXIT"]},
                "sl": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "targets": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["action", "sl", "targets"],
        }

        sys = (
            "You are an intraday SL-hunting trading engine. "
            "Use ONLY the provided state and retrieved rules. "
            "Always output ONE of: BUY, SELL, WAIT, EXIT. "
            "If unclear, output WAIT. "
            "If state.daily_plan is present, prefer its bias (one-sided). "
            "Avoid overtrading: do not suggest repeated re-entries after SL unless a NEW sweep+reclaim event appears. "
            "Use swing highs/lows and sweep/reclaim levels for SL and targets (liquidity pools). "
            "Output STRICT JSON only with keys: action, sl, targets. "
            "sl and targets must be ABSOLUTE price levels for the underlying index. "
            "targets must be ordered nearest->farthest. "
            "Do not output any extra keys."
        )

        # Prefer Gemini structured output when available; fallback to prompt-only JSON.
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": sys + "\n\n" + json.dumps(payload, ensure_ascii=False)}]}
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 180,
                "responseMimeType": "application/json",
                # Some Gemini deployments support JSON schema; if rejected, we will retry without it.
                "responseSchema": schema,
            },
        }

        # Normalize model id to avoid "/models/models/..." 404s when users copy from ListModels.
        url = f"{self.base_url}/models/{_normalize_model_id(self.model)}:generateContent"
        params = {"key": self.api_key}
        data: dict | None = None
        last_err: str | None = None
        with httpx.Client(timeout=self.timeout_s) as client:
            for attempt in range(1, int(self.max_retries) + 1):
                try:
                    r = client.post(url, params=params, json=body)
                    if r.status_code >= 400 and "responseSchema" in body["generationConfig"]:
                        # Retry without schema if the endpoint does not support it.
                        body["generationConfig"].pop("responseSchema", None)
                        r = client.post(url, params=params, json=body)
                    try:
                        r.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        status = getattr(getattr(e, "response", None), "status_code", None)
                        # Retry only on transient server/rate issues.
                        if status in (429, 500, 502, 503, 504):
                            raise
                        # Non-transient: don't retry; bubble up.
                        raise
                    data = r.json()
                    last_err = None
                    break
                except httpx.HTTPStatusError as e:
                    last_err = _sanitize_error(str(e))
                    # For non-transient codes, don't retry.
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    if status not in (429, 500, 502, 503, 504):
                        break
                    if attempt < int(self.max_retries):
                        # Small exponential backoff; keep overall latency bounded.
                        time.sleep(float(self.retry_backoff_s) * (2 ** (attempt - 1)))
                        continue
                except Exception as e:
                    last_err = _sanitize_error(str(e))
                    break

        if data is None:
            store.close()
            return {"action": "WAIT", "sl": None, "targets": [], "error": f"gemini_error: {last_err or 'unknown_error'}"}

        try:
            text = _extract_candidate_text(data)
            out = _extract_json(text)
            # Basic sanity
            if out.get("action") not in {"BUY", "SELL", "WAIT", "EXIT"}:
                raise ValueError(f"invalid action: {out.get('action')}")
            if not isinstance(out.get("targets"), list):
                raise ValueError("targets must be list")
        except Exception as e:
            # One retry for parse errors: ask again with simpler settings (no schema, plain text JSON).
            try:
                body2 = {
                    "contents": body["contents"],
                    "generationConfig": {
                        "temperature": 0,
                        "maxOutputTokens": 180,
                        "responseMimeType": "application/json",
                    },
                }
                with httpx.Client(timeout=self.timeout_s) as client:
                    r2 = client.post(url, params=params, json=body2)
                    r2.raise_for_status()
                    data2 = r2.json()
                text2 = _extract_candidate_text(data2)
                out2 = _extract_json(text2)
                if out2.get("action") not in {"BUY", "SELL", "WAIT", "EXIT"}:
                    raise ValueError(f"invalid action: {out2.get('action')}")
                if not isinstance(out2.get("targets"), list):
                    raise ValueError("targets must be list")
                out = out2
            except Exception as e2:
                store.close()
                return {
                    "action": "WAIT",
                    "sl": None,
                    "targets": [],
                    "error": f"gemini_parse_error: {_sanitize_error(str(e))} / retry_failed: {_sanitize_error(str(e2))}",
                }

        created_at = datetime.now(timezone.utc).isoformat()
        cache_id = f"gem_{req_hash[:16]}"
        store.set_llm_cache(cache_id, created_at, self.model, req_hash, out)
        store.close()
        return out
