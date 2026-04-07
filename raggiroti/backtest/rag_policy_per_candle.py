from __future__ import annotations

from dataclasses import dataclass

from raggiroti.config import get_settings
from raggiroti.rag.llm_decider import LLMDecider
from raggiroti.rag.rule_retriever import retrieve_rulebook_rules

from .policy import Decision, Policy


@dataclass(frozen=True)
class PerCandleRAGPolicy(Policy):
    """
    Calls LLM on *every candle* (slow, expensive), but gives you the per-candle "what is happening" reasoning.
    Use only for small ranges or debugging.
    """

    def decide(self, state: dict) -> Decision:
        settings = get_settings()
        api_key = settings.llm_api_key or settings.openai_api_key
        if not api_key and not settings.llm_base_url:
            return Decision(action="WAIT", reason="no_llm_configured")
        if not api_key and settings.llm_base_url:
            api_key = "local"

        # If a position is open, the engine only honors EXIT; don't burn LLM calls in-position.
        if state.get("position") is not None:
            return Decision(action="WAIT", reason="in_position_skip_llm")

        # Keep prompts small: per-candle LLM calls are expensive.
        retrieved = retrieve_rulebook_rules(settings.rulebook_path, state, limit=10)
        llm = LLMDecider(
            api_key=api_key,
            base_url=settings.llm_base_url,
            model=settings.openai_rule_extract_model,
            db_path=settings.db_path,
        )
        try:
            out = llm.decide(state=state, retrieved={"rulebook_version": retrieved.rulebook_version, "rules": retrieved.rules})
        except Exception as e:
            return Decision(action="WAIT", reason=f"llm_error: {e}")
        return Decision(
            action=out["action"],
            sl_points=float(out["sl_points"]),
            target_points=None if out["target_points"] is None else float(out["target_points"]),
            reason=f"LLM({out.get('confidence', 0):.2f}) {out.get('reason','')}",
        )
