from __future__ import annotations

from dataclasses import dataclass

from .policy import Decision, Policy
from .prev_day_planner import PrevDayLevels


@dataclass(frozen=True)
class ScenarioPolicy(Policy):
    """
    Deterministic "next-day plan" policy driven by previous-day levels.
    It implements your rulebook constraint:
    - no second position (handled by engine/broker)
    - confirmation-first
    - gap-up: look for failed acceptance above PDH (comfort trap) => short
    - gap-down: don't sell immediately; wait pullback to prev_close area => short on rejection
    - flat: trade acceptance beyond PDH/PDL

    This is a starter. You will extend with:
    - Dow structure (HH/HL)
    - comfort engine
    - zone engine
    - validity filters
    """

    prev: PrevDayLevels
    scenario: str
    stop_buffer_points: float = 15.0
    target_step_points: float = 40.0

    def decide(self, state: dict) -> Decision:
        price = float(state["price"])
        high = float(state.get("high", price))
        low = float(state.get("low", price))
        close = float(state.get("close", price))

        pdh = self.prev.high
        pdl = self.prev.low
        prev_close = self.prev.close

        # Basic "avoid fair zone" veto if provided
        if state.get("zone") == "fair":
            return Decision(action="WAIT", reason="zone=fair")

        # GAP UP: contrarian short after failed acceptance above PDH
        if self.scenario in ("gap_up", "mild_gap_up"):
            # "was above pdh and now back below" signal
            if high >= pdh and close < pdh:
                return Decision(
                    action="SELL",
                    sl_points=(pdh + self.stop_buffer_points) - price,
                    target_points=price - pdl if price > pdl else self.target_step_points,
                    reason="gap_up_failed_acceptance_above_pdh",
                )
            return Decision(action="WAIT", reason="gap_up_wait_confirmation")

        # GAP DOWN: do not sell immediately; sell pullback rejection near prev_close
        if self.scenario in ("gap_down", "mild_gap_down"):
            anchor = min(prev_close, pdl + 15.0)
            # if price trades above anchor and then closes back below => rejection
            if high >= anchor and close < anchor:
                return Decision(
                    action="SELL",
                    sl_points=(anchor + self.stop_buffer_points) - price,
                    target_points=price - (pdl - self.target_step_points),
                    reason="gap_down_pullback_rejection",
                )
            return Decision(action="WAIT", reason="gap_down_wait_pullback")

        # FLAT: acceptance beyond PDH/PDL
        if self.scenario == "flat":
            if close > pdh:
                return Decision(
                    action="BUY",
                    sl_points=price - (pdh - self.stop_buffer_points),
                    target_points=self.target_step_points,
                    reason="flat_accept_above_pdh",
                )
            if close < pdl:
                return Decision(
                    action="SELL",
                    sl_points=(pdl + self.stop_buffer_points) - price,
                    target_points=self.target_step_points,
                    reason="flat_accept_below_pdl",
                )
            return Decision(action="WAIT", reason="flat_inside_range")

        return Decision(action="WAIT", reason=f"unknown_scenario={self.scenario}")

