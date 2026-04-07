from __future__ import annotations

from dataclasses import dataclass

from raggiroti.llm.gemini_decider import GeminiDecider
from raggiroti.rag.rule_retriever import retrieve_rulebook_rules

from .policy import Decision, Policy


@dataclass
class PerCandleGeminiPolicy(Policy):
    """
    Gemini per-candle policy (simulation-like):
    - Calls Gemini on every 1m candle (including while in-position, so EXIT is possible).
    - Uses RAG-lite retrieval (rulebook rules filtered by state).
    - Converts Gemini absolute SL/targets to point distances so BrokerSim can execute.

    This is intentionally expensive and should be used for short ranges / validation.
    """

    api_key: str
    model: str
    db_path: str
    rulebook_path: str
    retrieve_limit: int = 25

    # Debug hook: last raw Gemini output for engine logging
    last_raw: dict | None = None

    def decide(self, state: dict) -> Decision:
        # Keep prompts compact.
        retrieved = retrieve_rulebook_rules(self.rulebook_path, state, limit=int(self.retrieve_limit))
        decider = GeminiDecider(api_key=self.api_key, model=self.model, db_path=self.db_path)
        out = decider.decide(state=state, retrieved={"rulebook_version": retrieved.rulebook_version, "rules": retrieved.rules})
        self.last_raw = {
            "provider": "gemini",
            "model": self.model,
            "rulebook_version": retrieved.rulebook_version,
            "retrieved_count": len(retrieved.rules),
            "retrieved_rule_ids": [str(r.get("id")) for r in (retrieved.rules or []) if r.get("id")][:30],
            "out": out,
        }

        action = str(out.get("action") or "WAIT").upper().strip()
        if action not in {"BUY", "SELL", "WAIT", "EXIT"}:
            return Decision(action="WAIT", reason="gemini_invalid_action")
        if action in {"WAIT", "EXIT"}:
            return Decision(action=action, reason="gemini")

        try:
            entry = float(state.get("price"))
        except Exception:
            # Fallback to close if present
            entry = float(state.get("close") or 0.0)

        sl_abs = out.get("sl")
        targets = out.get("targets") or []

        # Fallbacks (rulebook-aligned): if Gemini omits SL/targets, use liquidity pools.
        if sl_abs is None:
            if action == "BUY":
                lvl = None
                if isinstance(state.get("last_swing_low_1m"), dict):
                    lvl = state["last_swing_low_1m"].get("price")
                if lvl is None:
                    lvl = state.get("prev_pdl")
                if lvl is not None:
                    sl_abs = float(lvl) - 5.0
                else:
                    sl_abs = entry - 15.0
            else:
                lvl = None
                if isinstance(state.get("last_swing_high_1m"), dict):
                    lvl = state["last_swing_high_1m"].get("price")
                if lvl is None:
                    lvl = state.get("prev_pdh")
                if lvl is not None:
                    sl_abs = float(lvl) + 5.0
                else:
                    sl_abs = entry + 15.0

        if not targets:
            if action == "BUY":
                t = None
                if isinstance(state.get("last_swing_high_1m"), dict):
                    t = state["last_swing_high_1m"].get("price")
                if t is None:
                    t = state.get("prev_pdh") or state.get("prev_close")
                if t is not None:
                    targets = [float(t)]
            else:
                t = None
                if isinstance(state.get("last_swing_low_1m"), dict):
                    t = state["last_swing_low_1m"].get("price")
                if t is None:
                    t = state.get("prev_pdl") or state.get("prev_close")
                if t is not None:
                    targets = [float(t)]
        t1_abs = None if not targets else targets[0]

        # Convert absolute to points
        sl_points = 15.0
        try:
            slp = float(sl_abs)
            sl_points = max(0.1, (entry - slp) if action == "BUY" else (slp - entry))
        except Exception:
            sl_points = 15.0

        target_points = None
        if t1_abs is not None:
            try:
                tp = float(t1_abs)
                target_points = max(0.1, (tp - entry) if action == "BUY" else (entry - tp))
            except Exception:
                target_points = None

        return Decision(action=action, sl_points=float(sl_points), target_points=target_points, reason="gemini")
