from __future__ import annotations

from dataclasses import dataclass

from .broker_sim import BrokerSim
from .csv_loader import Candle
from .policy import Policy
from .state_builder import StateBuilder
from .prev_day_planner import PrevDayLevels


@dataclass(frozen=True)
class BacktestResult:
    realized_pnl_points: float
    fills: list


def run_backtest(
    candles: list[Candle],
    policy: Policy,
    qty: int = 65,
    prev: PrevDayLevels | None = None,
    gap_threshold_points: float = 30.0,
    flat_threshold_points: float = 15.0,
) -> BacktestResult:
    broker = BrokerSim()
    state_builder = StateBuilder()
    state_builder.on_new_day(prev=prev, gap_threshold_points=gap_threshold_points, flat_threshold_points=flat_threshold_points)

    for c in candles:
        state = state_builder.update(c)
        broker.on_candle(state["dt"], high=c.high, low=c.low)

        # Always keep analytics running; allow policy to request an exit even while a position is open.
        state["position"] = None if broker.position is None else {
            "side": broker.position.side,
            "entry": broker.position.entry,
            "sl": broker.position.sl,
            "target": broker.position.target,
            "qty": broker.position.qty,
        }

        decision = policy.decide(state)
        if broker.position is not None:
            if decision.action == "EXIT":
                broker.exit(state["dt"], c.close, decision.reason or "EXIT")
            continue
        if decision.action == "BUY":
            broker.enter(
                dt=state["dt"],
                side="LONG",
                price=c.close,
                sl=c.close - decision.sl_points,
                target=None if decision.target_points is None else (c.close + decision.target_points),
                qty=qty,
                reason=decision.reason,
            )
        if decision.action == "SELL":
            broker.enter(
                dt=state["dt"],
                side="SHORT",
                price=c.close,
                sl=c.close + decision.sl_points,
                target=None if decision.target_points is None else (c.close - decision.target_points),
                qty=qty,
                reason=decision.reason,
            )

    # Flatten at end of file
    if broker.position is not None:
        broker.exit(candles[-1].dt.isoformat(timespec="minutes"), candles[-1].close, "EOD")

    return BacktestResult(realized_pnl_points=broker.realized_pnl_points, fills=broker.fills)
