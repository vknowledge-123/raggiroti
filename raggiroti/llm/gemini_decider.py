from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

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

def _sanitize_error(s: str) -> str:
    # Prevent leaking API keys into UI/DB logs. httpx errors may include the full URL.
    s = s or ""
    s = re.sub(r"(key=)[^&\s]+", r"\1***", s)
    return s


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

        url = f"{self.base_url}/models/{self.model}:generateContent"
        params = {"key": self.api_key}
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                r = client.post(url, params=params, json=body)
                if r.status_code >= 400 and "responseSchema" in body["generationConfig"]:
                    # Retry without schema if the endpoint does not support it.
                    body["generationConfig"].pop("responseSchema", None)
                    r = client.post(url, params=params, json=body)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            store.close()
            return {"action": "WAIT", "sl": None, "targets": [], "error": f"gemini_error: {_sanitize_error(str(e))}"}

        try:
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            out = _extract_json(text)
            # Basic sanity
            if out.get("action") not in {"BUY", "SELL", "WAIT", "EXIT"}:
                raise ValueError(f"invalid action: {out.get('action')}")
            if not isinstance(out.get("targets"), list):
                raise ValueError("targets must be list")
        except Exception as e:
            store.close()
            return {"action": "WAIT", "sl": None, "targets": [], "error": f"gemini_parse_error: {_sanitize_error(str(e))}"}

        created_at = datetime.now(timezone.utc).isoformat()
        cache_id = f"gem_{req_hash[:16]}"
        store.set_llm_cache(cache_id, created_at, self.model, req_hash, out)
        store.close()
        return out
