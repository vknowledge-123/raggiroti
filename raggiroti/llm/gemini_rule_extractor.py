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

    # 3) heuristic repairs for common "almost JSON" mistakes
    repaired = text
    repaired = re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', repaired)
    repaired = repaired.replace(": None", ": null").replace(": True", ": true").replace(": False", ": false")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try:
        return json.loads(repaired)
    except Exception:
        m = re.search(r"\{.*\}", repaired, re.DOTALL)
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


def _is_schema_related_400(resp: httpx.Response) -> bool:
    """
    Some Gemini deployments reject generationConfig.responseJsonSchema with HTTP 400.
    We only drop the schema in that specific case (not on auth/model/other errors).
    """
    try:
        if int(resp.status_code) != 400:
            return False
        j = resp.json()
        msg = ""
        if isinstance(j, dict):
            err = j.get("error")
            if isinstance(err, dict) and isinstance(err.get("message"), str):
                msg = err.get("message") or ""
            elif isinstance(j.get("message"), str):
                msg = j.get("message") or ""
        msg = (msg or "").lower()
        return any(
            k in msg
            for k in (
                "responsejsonschema",
                "jsonschema",
                "schema",
                "unknown name",
                "invalid json schema",
                "invalid schema",
            )
        )
    except Exception:
        return False


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
            "propertyOrdering": ["summary", "rules", "conflicts"],
            "properties": {
                "summary": {"type": "string"},
                "rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "propertyOrdering": ["category", "name", "condition", "interpretation", "action", "tags"],
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
                        "propertyOrdering": ["topic", "note"],
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
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            ],
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
        headers = {"x-goog-api-key": self.api_key}

        data: dict | None = None
        last_err: str | None = None
        with httpx.Client(timeout=self.timeout_s) as client:
            for attempt in range(1, int(self.max_retries) + 1):
                try:
                    r = client.post(url, headers=headers, json=body)
                    if _is_schema_related_400(r) and ("responseJsonSchema" in body["generationConfig"]):
                        # Older deployments may not support schemas; retry without it.
                        body["generationConfig"].pop("responseJsonSchema", None)
                        r = client.post(url, headers=headers, json=body)
                    r.raise_for_status()
                    data = r.json()
                    if not _extract_candidate_text(data):
                        fb = _gemini_feedback(data)
                        raise ValueError(f"empty_candidate_text: {json.dumps(fb, ensure_ascii=False)}")
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
                    if str(e).startswith("empty_candidate_text") and attempt < int(self.max_retries):
                        time.sleep(float(self.retry_backoff_s) * (2 ** (attempt - 1)))
                        continue
                    break
                except Exception as e:
                    last_err = _sanitize_error(str(e))
                    break

        if data is None:
            raise RuntimeError(f"gemini_error: {last_err or 'unknown_error'}")

        return _extract_json(_extract_candidate_text(data))
