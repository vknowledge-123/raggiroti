from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx


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


@dataclass(frozen=True)
class GeminiRuleExtractor:
    api_key: str
    model: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_s: float = 30.0

    def extract_rules(self, transcript_text: str) -> dict:
        """
        Returns:
        {
          "summary": "...",
          "rules": [{ "category","name","condition","interpretation","action","tags":[...] }],
          "conflicts": [{ "topic","note" }]
        }

        IDs are intentionally omitted; merge step assigns DT-SL IDs safely.
        """
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "category": {"type": "string"},
                            "name": {"type": "string"},
                            "condition": {"type": "string"},
                            "interpretation": {"type": "string"},
                            "action": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["category", "name", "condition", "interpretation", "action"],
                    },
                },
                "conflicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"topic": {"type": "string"}, "note": {"type": "string"}},
                        "required": ["topic", "note"],
                    },
                },
            },
            "required": ["summary", "rules"],
        }

        sys = (
            "Extract NEW trading rules from the transcript for an SL-hunting rulebook. "
            "Be conservative: do not duplicate obvious existing rules. "
            "Write atomic rules in condition->interpretation->action form. "
            "Output ONLY JSON matching schema. No markdown."
        )

        body = {
            "contents": [{"role": "user", "parts": [{"text": sys + "\n\n" + transcript_text}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 1200,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }

        url = f"{self.base_url}/models/{self.model}:generateContent"
        params = {"key": self.api_key}

        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, params=params, json=body)
            if r.status_code >= 400:
                # Retry without schema for older deployments.
                body["generationConfig"].pop("responseSchema", None)
                r = client.post(url, params=params, json=body)
            r.raise_for_status()
            data = r.json()

        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return _extract_json(text)

