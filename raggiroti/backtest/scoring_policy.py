from __future__ import annotations

from dataclasses import dataclass

from .policy import Decision, Policy


@dataclass(frozen=True)
class SimpleScoringPolicy(Policy):
    """
    Deterministic starter policy:
    - Avoid fair zone
    - Require confirmation flags in state (computed by your state builder)
    - This is intentionally minimal; extend it using your rulebook rules.
    """

    min_score: int = 6

    def decide(self, state: dict) -> Decision:
        zone = state.get("zone")
        if zone == "fair":
            return Decision(action="WAIT", reason="zone=fair")

        score_long = 0
        score_short = 0

        if state.get("structure") == "bull":
            score_long += 2
        if state.get("structure") == "bear":
            score_short += 2

        if state.get("sl_available") is True:
            score_long += 2
            score_short += 2

        if state.get("comfort_risk") is True:
            # comfort increases trap risk; bias to contrarian
            score_short += 2 if zone == "inflated" else 0
            score_long += 2 if zone == "discount" else 0

        if state.get("confirmed_long") is True:
            score_long += 3
        if state.get("confirmed_short") is True:
            score_short += 3

        if score_long >= self.min_score and score_long > score_short:
            return Decision(action="BUY", reason=f"score_long={score_long}")
        if score_short >= self.min_score and score_short > score_long:
            return Decision(action="SELL", reason=f"score_short={score_short}")
        return Decision(action="WAIT", reason=f"scores L{score_long}/S{score_short}")

