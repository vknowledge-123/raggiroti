from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    llm_base_url: str | None
    llm_api_key: str | None
    openai_embed_model: str
    openai_rule_extract_model: str
    db_path: str
    rulebook_path: str


def _project_root() -> Path:
    # config.py lives at <root>/raggiroti/config.py
    return Path(__file__).resolve().parents[1]


def _resolve_path(p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((_project_root() / path).resolve())


def get_settings() -> Settings:
    db_path = os.getenv("RAGGIROTI_DB", "./data/raggiroti.sqlite")
    rulebook_path = os.getenv("RULEBOOK_PATH", "./rulebook/nexus_ultra_v2.rulebook.json")
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_api_key=os.getenv("LLM_API_KEY"),
        openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"),
        openai_rule_extract_model=os.getenv("LLM_RULE_EXTRACT_MODEL", os.getenv("OPENAI_RULE_EXTRACT_MODEL", "gpt-4.1-mini")),
        db_path=_resolve_path(db_path),
        rulebook_path=_resolve_path(rulebook_path),
    )
