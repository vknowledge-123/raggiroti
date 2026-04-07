from __future__ import annotations

from dataclasses import dataclass

from raggiroti.config import get_settings
from raggiroti.rag.llm_decider import LLMDecider
from raggiroti.rag.rule_retriever import retrieve_rulebook_rules

from .policy import Decision, Policy
from .rulebook_scoring_policy import RulebookScoringPolicy


def _is_event(state: dict) -> bool:
    # Call LLM only on events (low latency, reproducible via cache).
    if state.get("event_confirmed_long") or state.get("event_confirmed_short"):
        return True
    # Opening period: first candle logic / gap effects
    if state.get("gap_type") in ("gap_up", "gap_down") and state.get("first_candle_color") in ("red", "green"):
        # only during first ~10 minutes if caller supplies a minute index; otherwise treat as event once
        return state.get("minute_index", 0) <= 10
    return False


@dataclass(frozen=True)
class RAGPolicy(Policy):
    """
    Hybrid policy:
    - Always compute deterministic score (fast).
    - If ambiguous (WAIT) AND an event occurs, ask LLM using ONLY retrieved rules + state.
    - LLM outputs structured decision, cached in SQLite for deterministic repeats.
    """

    def decide(self, state: dict) -> Decision:
        deterministic = RulebookScoringPolicy().decide(state)
        # If a position is open, the engine only honors EXIT decisions.
        # Avoid LLM calls while in-position to keep backtests low-latency.
        if state.get("position") is not None:
            return deterministic
        if deterministic.action != "WAIT":
            return deterministic

        if not _is_event(state):
            return deterministic

        settings = get_settings()
        api_key = settings.llm_api_key or settings.openai_api_key
        if not api_key and not settings.llm_base_url:
            return deterministic
        if not api_key and settings.llm_base_url:
            api_key = "local"

        # Keep prompts small for low-latency local inference.
        retrieved = retrieve_rulebook_rules(settings.rulebook_path, state, limit=12)
        llm = LLMDecider(
            api_key=api_key,
            base_url=settings.llm_base_url,
            model=settings.openai_rule_extract_model,  # env can point to Qwen local model
            db_path=settings.db_path,
        )
        try:
            out = llm.decide(
                state=state,
                retrieved={"rulebook_version": retrieved.rulebook_version, "rules": retrieved.rules},
            )
        except Exception as e:
            # Never crash the backtest/web app due to LLM instability.
            return Decision(action="WAIT", reason=f"llm_error: {e}")
        return Decision(
            action=out["action"],
            sl_points=float(out["sl_points"]),
            target_points=None if out["target_points"] is None else float(out["target_points"]),
            reason=f"LLM({out.get('confidence', 0):.2f}) {out.get('reason','')}",
        )
