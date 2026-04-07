from __future__ import annotations

import json
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class RuleExtractor:
    api_key: str
    base_url: str | None
    model: str

    def extract_rules(self, transcript_text: str) -> dict:
        """
        Extract atomic rules as a proposal payload:
        {
          "rules": [{ "id": "...", "category": "...", "name": "...", "condition": "...", "interpretation": "...", "action": "...", "tags": [...] }],
          "conflicts": [{ "topic": "...", "note": "..." }],
          "summary": "..."
        }
        """
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
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
                            "id": {"type": "string"},
                            "category": {"type": "string"},
                            "name": {"type": "string"},
                            "condition": {"type": "string"},
                            "interpretation": {"type": "string"},
                            "action": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["id", "category", "name", "condition", "interpretation", "action"],
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

        # Local OpenAI-compatible servers (Ollama/LM Studio) typically support /v1/chat/completions, not /v1/responses.
        if self.base_url:
            chat = client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract NEW trading rules from the transcript. "
                            "Be conservative: avoid duplicates. "
                            "Output ONLY JSON (no markdown). Must match this JSON Schema:\n"
                            + json.dumps(schema, ensure_ascii=False)
                        ),
                    },
                    {"role": "user", "content": transcript_text},
                ],
            )
            text = chat.choices[0].message.content or ""
        else:
            resp = client.responses.create(
                model=self.model,
                temperature=0,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "Extract NEW trading rules from the transcript. "
                            "Be conservative: avoid duplicates. "
                            "Output only valid JSON matching the given schema."
                        ),
                    },
                    {"role": "user", "content": transcript_text},
                ],
                response_format={"type": "json_schema", "json_schema": {"name": "rule_proposal", "schema": schema}},
            )
            text = resp.output_text

        try:
            return json.loads(text)
        except Exception:
            # Best-effort JSON extraction for models that wrap output.
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                raise
            return json.loads(m.group(0))
