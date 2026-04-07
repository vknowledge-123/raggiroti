from __future__ import annotations

from dataclasses import dataclass

from .policy import Decision, Policy


@dataclass(frozen=True)
class RulebookScoringPolicy(Policy):
    """
    Deterministic, low-latency scoring policy aligned with your rulebook concepts.
    It does NOT call an LLM per candle.

    Signals expected in `state` (from StateBuilder):
    - zone: discount|fair|inflated
    - structure: bull|bear|range|unknown
    - gap_type: gap_up|gap_down|flat|None
    - first_candle_color: red|green|None
    - confirmed_long / confirmed_short: sweep+reclaim booleans
    """

    min_score: int = 6
    sl_points: float = 15.0
    target_points: float | None = 40.0

    def decide(self, state: dict) -> Decision:
        # Hard vetoes
        if state.get("validity_ok") is False:
            return Decision(action="WAIT", reason="validity_ok=false")
        if state.get("zone") == "fair":
            return Decision(action="WAIT", reason="zone=fair")
        if state.get("market_type") == "chop":
            return Decision(action="WAIT", reason="market_type=chop")

        score_long = 0
        score_short = 0

        # Zone posture
        if state.get("zone") == "discount":
            score_long += 2
        if state.get("zone") == "inflated":
            score_short += 2

        # Dow structure (with lag)
        if state.get("structure") == "bull":
            score_long += 2
        elif state.get("structure") == "bear":
            score_short += 2
        elif state.get("structure") == "range":
            # range increases trap/chop; lower score
            score_long -= 1
            score_short -= 1

        # Opening matrix (gap + first candle) only matters early; we keep it as a soft score
        gap = state.get("gap_type")
        first = state.get("first_candle_color")
        if gap == "gap_up" and first == "green":
            score_long += 1
        if gap == "gap_down" and first == "red":
            score_short += 1
        if (gap == "gap_up" and first == "red") or (gap == "gap_down" and first == "green"):
            # trap/volatile
            score_long -= 1
            score_short -= 1

        # Confirmation gates (sweep + reclaim is high weight)
        if state.get("confirmed_long") is True:
            score_long += 4
        if state.get("confirmed_short") is True:
            score_short += 4

        # Comfort / overcrowding / operator exit risk (single-instrument proxies)
        if state.get("comfort_risk") is True:
            if state.get("zone") == "inflated":
                score_short += 1
            if state.get("zone") == "discount":
                score_long += 1
        if state.get("overcrowding_risk") is True:
            score_long -= 1
            score_short -= 1
        if state.get("operator_exit_risk") is True:
            score_long -= 2
            score_short -= 2

        # No second position rule is enforced by engine; policy just chooses.
        if score_long >= self.min_score and score_long > score_short:
            return Decision(action="BUY", sl_points=self.sl_points, target_points=self.target_points, reason=f"score_long={score_long}")
        if score_short >= self.min_score and score_short > score_long:
            return Decision(action="SELL", sl_points=self.sl_points, target_points=self.target_points, reason=f"score_short={score_short}")
        return Decision(action="WAIT", reason=f"scores L{score_long}/S{score_short}")
