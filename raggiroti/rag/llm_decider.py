from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
import re

from raggiroti.storage.sqlite_db import SqliteStore


def _hash_request(payload: dict) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _truncate(s: str, n: int = 240) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _compact_rules(rules: list[dict]) -> list[dict]:
    compact = []
    for r in rules or []:
        compact.append(
            {
                "id": r.get("id"),
                "category": r.get("category"),
                "name": _truncate(str(r.get("name") or ""), 120),
                "condition": _truncate(str(r.get("condition") or ""), 260),
                "action": _truncate(str(r.get("action") or ""), 260),
                "tags": r.get("tags") or [],
            }
        )
    return compact


@dataclass(frozen=True)
class LLMDecider:
    """
    Event-driven LLM decider with caching for reproducibility.

    - Use temperature=0
    - Use JSON schema output
    - Cache by request hash in SQLite so repeated backtests are deterministic
    """

    api_key: str
    base_url: str | None
    model: str
    db_path: str

    def decide(self, state: dict, retrieved: dict) -> dict:
        # Keep prompt compact for low-latency local inference.
        rules = retrieved.get("rules") if isinstance(retrieved, dict) else None
        retrieved_compact = retrieved
        if isinstance(retrieved, dict) and isinstance(rules, list):
            retrieved_compact = {
                "rulebook_version": retrieved.get("rulebook_version"),
                "rules": _compact_rules(rules),
            }

        payload = {"state": state, "retrieved": retrieved_compact, "schema_version": 1}
        req_hash = _hash_request(payload)
        store = SqliteStore(self.db_path)
        cached = store.get_llm_cache(req_hash, self.model)
        if cached is not None:
            store.close()
            return cached

        from openai import OpenAI
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30.0)
        except TypeError:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": ["BUY", "SELL", "WAIT", "EXIT"]},
                "confidence": {"type": "number"},
                "sl_points": {"type": "number"},
                "target_points": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "reason": {"type": "string"},
                "rule_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["action", "confidence", "sl_points", "target_points", "reason", "rule_ids"],
        }

        try:
            # Local servers typically support chat.completions only.
            if self.base_url:
                chat = client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=220,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a trading backtest decision assistant. "
                                "Use ONLY the provided state and retrieved rules. "
                                "If unclear, output WAIT. Avoid overtrading. "
                                "Keep reason short (<= 20 words). "
                                "Output ONLY JSON matching this schema:\n"
                                + json.dumps(schema, ensure_ascii=False)
                            ),
                        },
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                )
                text = chat.choices[0].message.content or ""
            else:
                resp = client.responses.create(
                    model=self.model,
                    temperature=0,
                    max_output_tokens=220,
                    input=[
                        {
                            "role": "system",
                            "content": (
                                "You are a trading backtest decision assistant. "
                                "Use ONLY the provided state and retrieved rules. "
                                "If unclear, output WAIT. Avoid overtrading."
                            ),
                        },
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    response_format={"type": "json_schema", "json_schema": {"name": "decision", "schema": schema}},
                )
                text = resp.output_text

            try:
                out = json.loads(text)
            except Exception:
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if not m:
                    raise ValueError(f"LLM returned non-JSON output: {text[:120]!r}")
                out = json.loads(m.group(0))

            created_at = datetime.now(timezone.utc).isoformat()
            cache_id = f"llm_{req_hash[:16]}"
            store.set_llm_cache(cache_id, created_at, self.model, req_hash, out)
            store.close()
            return out
        except Exception as e:
            # Ensure we don't crash the caller/UI; just degrade to WAIT.
            store.close()
            return {
                "action": "WAIT",
                "confidence": 0.0,
                "sl_points": 15.0,
                "target_points": None,
                "reason": f"llm_error: {e}",
                "rule_ids": [],
            }
