from __future__ import annotations

from dataclasses import dataclass

from .broker_sim import BrokerSim
from .csv_loader import Candle
from .policy import Decision, Policy
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
    gap_up_threshold_points: float = 30.0,
    gap_down_threshold_points: float = 30.0,
    flat_threshold_points: float = 15.0,
    include_decisions: bool = False,
    max_decisions: int = 1200,
    max_entries_per_day: int | None = None,
    cooldown_after_sl_candles: int = 0,
    lock_direction_after_first_entry: bool = False,
) -> BacktestResult:
    broker = BrokerSim()
    state_builder = StateBuilder()
    state_builder.on_new_day(
        prev=prev,
        gap_up_threshold_points=gap_up_threshold_points,
        gap_down_threshold_points=gap_down_threshold_points,
        flat_threshold_points=flat_threshold_points,
    )
    decisions: list[dict] | None = [] if include_decisions else None

    entries_taken = 0
    locked_direction: str | None = None  # "BUY" | "SELL"
    cooldown_until_minute = 0

    for c in candles:
        state = state_builder.update(c)
        fills_before = len(broker.fills)
        broker.on_candle(state["dt"], high=c.high, low=c.low)
        # Apply cooldown after an SL exit (prevents repeated re-entries on the same chop).
        if cooldown_after_sl_candles > 0 and len(broker.fills) > fills_before:
            for f in broker.fills[fills_before:]:
                if f.side.startswith("EXIT_") and f.reason == "SL":
                    cooldown_until_minute = int(state.get("minute_index") or 0) + int(cooldown_after_sl_candles)

        # Always keep analytics running; allow policy to request an exit even while a position is open.
        state["position"] = None if broker.position is None else {
            "side": broker.position.side,
            "entry": broker.position.entry,
            "sl": broker.position.sl,
            "target": broker.position.target,
            "qty": broker.position.qty,
        }

        decision = policy.decide(state)

        # ---------------- Guardrails to increase trade quality ----------------
        if broker.position is None and decision.action in {"BUY", "SELL"}:
            mi = int(state.get("minute_index") or 0)
            if max_entries_per_day is not None and entries_taken >= int(max_entries_per_day):
                decision = Decision(action="WAIT", reason="guardrail:max_entries_per_day")
            elif cooldown_after_sl_candles > 0 and mi < int(cooldown_until_minute):
                decision = Decision(action="WAIT", reason="guardrail:cooldown_after_sl")
            elif lock_direction_after_first_entry and locked_direction is not None and decision.action != locked_direction:
                decision = Decision(action="WAIT", reason="guardrail:direction_lock")

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
            plan_bias = None
            try:
                if isinstance(state.get("daily_plan"), dict):
                    plan_bias = state["daily_plan"].get("bias")
            except Exception:
                plan_bias = None
            decisions.append(
                {
                    "dt": state["dt"],
                    "action": decision.action,
                    "sl_points": float(decision.sl_points),
                    "target_points": None if decision.target_points is None else float(decision.target_points),
                    "sl_abs": sl_abs,
                    "t1_abs": t_abs,
                    "reason": decision.reason,
                    "scenario": state.get("scenario"),
                    "zone": state.get("zone"),
                    "plan_bias": plan_bias,
                    "prev_pdh": state.get("prev_pdh"),
                    "prev_pdl": state.get("prev_pdl"),
                    "last_swing_high_1m": None if not isinstance(state.get("last_swing_high_1m"), dict) else state["last_swing_high_1m"].get("price"),
                    "last_swing_low_1m": None if not isinstance(state.get("last_swing_low_1m"), dict) else state["last_swing_low_1m"].get("price"),
                    "reclaimed_prev_pdh": state.get("reclaimed_prev_pdh"),
                    "reclaimed_prev_pdl": state.get("reclaimed_prev_pdl"),
                    "reclaimed_last_swing_high": state.get("reclaimed_last_swing_high"),
                    "reclaimed_last_swing_low": state.get("reclaimed_last_swing_low"),
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
            entries_taken += 1
            if lock_direction_after_first_entry and locked_direction is None:
                locked_direction = "BUY"
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
            entries_taken += 1
            if lock_direction_after_first_entry and locked_direction is None:
                locked_direction = "SELL"

    # Flatten at end of file
    if broker.position is not None:
        broker.exit(candles[-1].dt.isoformat(timespec="minutes"), candles[-1].close, "EOD")

    return BacktestResult(realized_pnl_points=broker.realized_pnl_points, fills=broker.fills, decisions=decisions)
