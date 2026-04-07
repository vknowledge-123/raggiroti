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
    decisions: list | None = None


def run_backtest(
    candles: list[Candle],
    policy: Policy,
    qty: int = 65,
    prev: PrevDayLevels | None = None,
    gap_threshold_points: float = 30.0,
    flat_threshold_points: float = 15.0,
    include_decisions: bool = False,
    max_decisions: int = 1200,
) -> BacktestResult:
    broker = BrokerSim()
    state_builder = StateBuilder()
    state_builder.on_new_day(prev=prev, gap_threshold_points=gap_threshold_points, flat_threshold_points=flat_threshold_points)
    decisions: list[dict] | None = [] if include_decisions else None

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
        if decisions is not None and len(decisions) < int(max_decisions):
            # Compute absolute SL/target from points for easier debugging.
            entry = float(c.close)
            sl_abs = None
            t_abs = None
            if decision.action == "BUY":
                sl_abs = entry - float(decision.sl_points)
                if decision.target_points is not None:
                    t_abs = entry + float(decision.target_points)
            elif decision.action == "SELL":
                sl_abs = entry + float(decision.sl_points)
                if decision.target_points is not None:
                    t_abs = entry - float(decision.target_points)
            raw = getattr(policy, "last_raw", None)
            decisions.append(
                {
                    "dt": state["dt"],
                    "action": decision.action,
                    "sl_points": float(decision.sl_points),
                    "target_points": None if decision.target_points is None else float(decision.target_points),
                    "sl_abs": sl_abs,
                    "t1_abs": t_abs,
                    "reason": decision.reason,
                    "raw": raw,
                }
            )
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

    return BacktestResult(realized_pnl_points=broker.realized_pnl_points, fills=broker.fills, decisions=decisions)
