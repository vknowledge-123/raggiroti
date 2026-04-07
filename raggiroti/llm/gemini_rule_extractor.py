from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import httpx


def _extract_candidate_text(data: dict) -> str:
    """
    Gemini generateContent responses may include multiple parts. Concatenate all text-ish parts.
    """
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
                # In some cases JSON may arrive as inline data; treat it as UTF-8 text.
                texts.append(inline["data"])
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

def _normalize_model_id(model: str) -> str:
    m = (model or "").strip()
    if m.startswith("models/"):
        m = m[len("models/") :]
    return m


def _sanitize_error(s: str) -> str:
    s = (s or "").replace("\n", " ").strip()
    s = re.sub(r"(key=)[^&\s]+", r"\1***", s)
    return s


@dataclass(frozen=True)
class GeminiRuleExtractor:
    api_key: str
    model: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_s: float = 30.0
    max_retries: int = 3
    retry_backoff_s: float = 0.8

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
            "systemInstruction": {"parts": [{"text": sys}]},
            "contents": [{"role": "user", "parts": [{"text": transcript_text}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 1200,
                "responseMimeType": "application/json",
                # Use JSON Schema structured outputs.
                "responseJsonSchema": schema,
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
                    if r.status_code >= 400 and "responseJsonSchema" in body["generationConfig"]:
                        # Older deployments may not support schemas; retry without it.
                        body["generationConfig"].pop("responseJsonSchema", None)
                        r = client.post(url, params=params, json=body)
                    r.raise_for_status()
                    data = r.json()
                    if not _extract_candidate_text(data):
                        raise ValueError("empty_candidate_text")
                    last_err = None
                    break
                except httpx.HTTPStatusError as e:
                    last_err = _sanitize_error(str(e))
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    if status in (429, 500, 502, 503, 504) and attempt < int(self.max_retries):
                        time.sleep(float(self.retry_backoff_s) * (2 ** (attempt - 1)))
                        continue
                    break
                except ValueError as e:
                    last_err = _sanitize_error(str(e))
                    if str(e) == "empty_candidate_text" and attempt < int(self.max_retries):
                        time.sleep(float(self.retry_backoff_s) * (2 ** (attempt - 1)))
                        continue
                    break
                except Exception as e:
                    last_err = _sanitize_error(str(e))
                    break

        if data is None:
            raise RuntimeError(f"gemini_error: {last_err or 'unknown_error'}")

        return _extract_json(_extract_candidate_text(data))
