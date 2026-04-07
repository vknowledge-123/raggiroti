from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from raggiroti.backtest.broker_sim import BrokerSim
from raggiroti.backtest.csv_loader import Candle
from raggiroti.backtest.prev_day_planner import PrevDayLevels, compute_prev_day_levels
from raggiroti.backtest.state_builder import StateBuilder
from raggiroti.config import get_settings
from raggiroti.llm.gemini_decider import GeminiDecider
from raggiroti.rag.rule_retriever import retrieve_rulebook_rules
from raggiroti.dhan.option_chain import derive_oi_features

from .models import DecisionOut, LiveCandle


@dataclass
class LiveSimStatus:
    running: bool
    symbol: str | None
    security_id: str | None
    started_at: str | None
    last_candle_dt: str | None
    last_action: str | None
    realized_pnl_points: float


class LiveSimEngine:
    """
    Live candle-by-candle paper trading engine:
    - Builds state (Dow/zone/proxies) using StateBuilder (upgrade later as you refine).
    - Retrieves relevant rules (RAG-lite).
    - Calls Gemini per candle (user request).
    - Simulates trades only (BrokerSim).
    """

    def __init__(self, *, symbol: str, security_id: str, gemini_api_key: str, gemini_model: str, qty: int = 65) -> None:
        self.symbol = symbol
        self.security_id = security_id
        self.qty = int(qty)
        self._settings = get_settings()
        self._broker = BrokerSim()
        self._sb = StateBuilder()
        self._prev: PrevDayLevels | None = None
        self._gemini = GeminiDecider(api_key=gemini_api_key, model=gemini_model, db_path=self._settings.db_path)

        self._started_at = datetime.now(timezone.utc).isoformat()
        self._last_candle: LiveCandle | None = None
        self._last_state: dict | None = None
        self._decisions: list[dict[str, Any]] = []
        self._candles: list[dict[str, Any]] = []
        self._oi_snapshot: dict | None = None
        self._lock = asyncio.Lock()

    def status(self) -> LiveSimStatus:
        return LiveSimStatus(
            running=True,
            symbol=self.symbol,
            security_id=self.security_id,
            started_at=self._started_at,
            last_candle_dt=None if self._last_candle is None else self._last_candle.dt.isoformat(timespec="minutes"),
            last_action=(None if not self._decisions else str(self._decisions[-1].get("action"))),
            realized_pnl_points=float(self._broker.realized_pnl_points),
        )

    def last_decisions(self, limit: int = 120) -> list[dict]:
        return list(self._decisions[-limit:])

    def last_candles(self, limit: int = 300) -> list[dict]:
        return list(self._candles[-limit:])

    def last_state(self) -> dict | None:
        return self._last_state

    def fills(self) -> list[dict]:
        return list(self._broker.fills)

    def reset(self) -> None:
        self._broker = BrokerSim()
        self._sb = StateBuilder()
        self._prev = None
        self._decisions.clear()
        self._candles.clear()
        self._last_candle = None
        self._last_state = None
        self._oi_snapshot = None

    def set_oi_snapshot(self, snapshot: dict | None) -> None:
        # Keep it raw for now; state builder/LLM can interpret.
        self._oi_snapshot = snapshot

    def _ensure_day_initialized(self, candle: LiveCandle) -> None:
        if self._sb.day_open is not None:
            return
        # In live mode, we often don't have previous-day candles at startup. If you do,
        # call set_prev_day_candles() before starting.
        self._sb.on_new_day(prev=self._prev)

    def set_prev_day_candles(self, prev_day_candles: list[Candle]) -> None:
        self._prev = compute_prev_day_levels(prev_day_candles)
        self._sb.on_new_day(prev=self._prev)

    def _force_exit_reason(self, state: dict) -> str | None:
        """
        Deterministic Dow-theory / structure based exit guard.
        This is a safety layer so positions don't depend exclusively on LLM reliability.
        """
        pos = self._broker.position
        if pos is None:
            return None
        s1 = str(state.get("dow_structure_1m") or "")
        s5 = str(state.get("dow_structure_5m") or "")
        broke_h = bool(state.get("broke_last_swing_high"))
        broke_l = bool(state.get("broke_last_swing_low"))

        if pos.side == "LONG":
            if s5 == "bear":
                return "DOW_FLIP_5M"
            if broke_l and s1 == "bear":
                return "SWING_LOW_BREAK_1M"
        if pos.side == "SHORT":
            if s5 == "bull":
                return "DOW_FLIP_5M"
            if broke_h and s1 == "bull":
                return "SWING_HIGH_BREAK_1M"
        return None

    async def on_candle_close(self, candle: LiveCandle) -> DecisionOut:
        async with self._lock:
            self._ensure_day_initialized(candle)
            self._last_candle = candle

            c = Candle(
                symbol=self.symbol,
                dt=candle.dt,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=0.0 if candle.volume is None else float(candle.volume),
            )

            state = self._sb.update(c)
            self._broker.on_candle(state["dt"], high=c.high, low=c.low)

            state["symbol"] = self.symbol
            state["security_id"] = self.security_id
            state["oi"] = self._oi_snapshot
            state.update(derive_oi_features(spot_price=float(c.close), snapshot=self._oi_snapshot))
            state["position"] = (
                None
                if self._broker.position is None
                else {
                    "side": self._broker.position.side,
                    "entry": self._broker.position.entry,
                    "sl": self._broker.position.sl,
                    "target": self._broker.position.target,
                    "qty": self._broker.position.qty,
                }
            )

            self._last_state = state

            # Deterministic structure exit (Dow flip / swing break) BEFORE LLM call.
            # This keeps latency low and ensures we exit even if Gemini fails.
            forced = self._force_exit_reason(state)
            if forced and self._broker.position is not None:
                self._broker.exit(state["dt"], c.close, forced)
                decision = DecisionOut(action="EXIT", sl=None, targets=[], raw={"action": "EXIT", "sl": None, "targets": [], "error": f"forced_exit:{forced}"})
                self._candles.append(
                    {
                        "dt": state["dt"],
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close,
                        "volume": candle.volume,
                    }
                )
                self._decisions.append(
                    {
                        "dt": state["dt"],
                        "action": decision.action,
                        "sl": decision.sl,
                        "targets": decision.targets,
                        "pos": state["position"],
                        "reason": forced,
                        "raw": {k: v for k, v in (decision.raw or {}).items() if k in {"action", "sl", "targets", "error"}},
                    }
                )
                return decision

            retrieved = retrieve_rulebook_rules(self._settings.rulebook_path, state, limit=25)
            out = self._gemini.decide(
                state=state,
                retrieved={"rulebook_version": retrieved.rulebook_version, "rules": retrieved.rules},
            )

            decision = DecisionOut(action=str(out.get("action", "WAIT")), sl=out.get("sl"), targets=out.get("targets") or [], raw=out)
            # Enforce: while in position, do NOT start a new position.
            # - If Gemini outputs the opposite side, interpret as EXIT (structure change / invalidation).
            # - If Gemini outputs same-side BUY/SELL, ignore as WAIT (hold).
            pos = self._broker.position
            override_reason: str | None = None
            if pos is not None and decision.action in {"BUY", "SELL"}:
                if (pos.side == "LONG" and decision.action == "SELL") or (pos.side == "SHORT" and decision.action == "BUY"):
                    decision = DecisionOut(action="EXIT", sl=None, targets=[], raw={**(decision.raw or {}), "error": "override:opposite_signal_exit"})
                    override_reason = "LLM_OPPOSITE_SIGNAL_EXIT"
                else:
                    decision = DecisionOut(action="WAIT", sl=None, targets=[], raw={**(decision.raw or {}), "error": "override:in_position_ignore_entry"})
                    override_reason = "IN_POSITION_IGNORE_ENTRY"

            self._candles.append(
                {
                    "dt": state["dt"],
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
            )
            self._decisions.append(
                {
                    "dt": state["dt"],
                    "action": decision.action,
                    "sl": decision.sl,
                    "targets": decision.targets,
                    "pos": state["position"],
                    "reason": override_reason,
                    "raw": {k: v for k, v in (decision.raw or {}).items() if k in {"action", "sl", "targets", "error"}},
                }
            )

            # Execute paper trades (no real orders).
            if self._broker.position is not None:
                if decision.action == "EXIT":
                    self._broker.exit(state["dt"], c.close, "LLM_EXIT")
                return decision

            if decision.action == "BUY":
                sl = float(decision.sl) if decision.sl is not None else (c.close - 15.0)
                t1 = None if not decision.targets else float(decision.targets[0])
                self._broker.enter(dt=state["dt"], side="LONG", price=c.close, sl=sl, target=t1, qty=self.qty, reason="LLM_BUY")
            elif decision.action == "SELL":
                sl = float(decision.sl) if decision.sl is not None else (c.close + 15.0)
                t1 = None if not decision.targets else float(decision.targets[0])
                self._broker.enter(dt=state["dt"], side="SHORT", price=c.close, sl=sl, target=t1, qty=self.qty, reason="LLM_SELL")

            return decision
