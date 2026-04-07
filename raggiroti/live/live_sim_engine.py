from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
import math

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
        # Live needs faster swing confirmation to avoid staying "unknown" too long.
        # We still keep it conservative with confirmed swings (uses past-only once confirmed).
        try:
            self._sb.swing_window = 1
        except Exception:
            pass
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
        out: list[dict] = []
        for f in self._broker.fills:
            try:
                out.append(asdict(f))
            except Exception:
                try:
                    out.append(dict(getattr(f, "__dict__", {}) or {}))
                except Exception:
                    out.append({"type": type(f).__name__})
        return out

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

    @staticmethod
    def _compact_oi_for_llm(snapshot: dict | None) -> dict | None:
        """
        Reduce OI snapshot size for per-candle LLM prompts.
        Keep only: walls + a small strike window (without huge volume fields).
        """
        if not isinstance(snapshot, dict) or not snapshot.get("ok"):
            return snapshot

        out: dict = {
            "ok": True,
            "mode": snapshot.get("mode"),
            "spot": snapshot.get("spot"),
            "atm_strike": snapshot.get("atm_strike"),
            "ce_walls": snapshot.get("ce_walls"),
            "pe_walls": snapshot.get("pe_walls"),
        }

        win = snapshot.get("window")
        if isinstance(win, list):
            w2 = []
            for row in win[:11]:
                if not isinstance(row, dict):
                    continue
                strike = row.get("strike")
                ce = row.get("CE") if isinstance(row.get("CE"), dict) else None
                pe = row.get("PE") if isinstance(row.get("PE"), dict) else None
                def _leg(d: dict | None) -> dict | None:
                    if not isinstance(d, dict):
                        return None
                    return {"oi": d.get("oi"), "oi_change": d.get("oi_change"), "ltp": d.get("ltp")}
                w2.append({"strike": strike, "CE": _leg(ce), "PE": _leg(pe)})
            out["window"] = w2
        return out

    @staticmethod
    def _uniq_sorted(levels: list[float]) -> list[float]:
        out: list[float] = []
        for x in sorted({float(v) for v in levels if v is not None}):
            if not out or abs(x - out[-1]) >= 0.01:
                out.append(float(x))
        return out

    @staticmethod
    def _maybe_price(v: object) -> float | None:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict) and v.get("price") is not None:
            try:
                return float(v.get("price"))
            except Exception:
                return None
        return None

    def _attach_dynamic_liquidity(self, state: dict) -> None:
        """
        Add small, high-impact derived fields so Gemini can reason like a "super brain"
        without requiring it to infer everything from raw OHLC.
        """
        try:
            price = float(state.get("price"))
        except Exception:
            return

        # Round numbers near price (50/100 steps).
        def _rns(step: float) -> list[float]:
            if step <= 0:
                return []
            base = round(price / step) * step
            return [float(base - step), float(base), float(base + step)]

        state["round_numbers_50"] = _rns(50.0)
        state["round_numbers_100"] = _rns(100.0)

        # Intraday zone based on session PDH/PDL quartiles (more dynamic than prev-day zones).
        try:
            pdh = float(state.get("pdh")) if state.get("pdh") is not None else None
            pdl = float(state.get("pdl")) if state.get("pdl") is not None else None
            if pdh is not None and pdl is not None and pdh > pdl:
                q1 = pdl + 0.25 * (pdh - pdl)
                q3 = pdl + 0.75 * (pdh - pdl)
                z = "fair"
                if price <= q1:
                    z = "discount"
                elif price >= q3:
                    z = "inflated"
                state["intraday_zone"] = z
                state["intraday_q1"] = float(q1)
                state["intraday_q3"] = float(q3)
        except Exception:
            pass

        # Liquidity levels: merge prev levels + swings + OI walls.
        buy = []
        sell = []
        buy += [self._maybe_price(state.get("prev_pdl")), self._maybe_price(state.get("prev_last_hour_low")), self._maybe_price(state.get("pdl"))]
        sell += [self._maybe_price(state.get("prev_pdh")), self._maybe_price(state.get("prev_last_hour_high")), self._maybe_price(state.get("pdh"))]
        buy += [self._maybe_price(state.get("last_swing_low_1m")), self._maybe_price(state.get("last_swing_low_5m"))]
        sell += [self._maybe_price(state.get("last_swing_high_1m")), self._maybe_price(state.get("last_swing_high_5m"))]
        buy += [self._maybe_price(state.get("oi_support"))]
        sell += [self._maybe_price(state.get("oi_resistance"))]

        # Add a couple nearest OI walls lists (already compact).
        try:
            supps = state.get("oi_supports") if isinstance(state.get("oi_supports"), list) else []
            ress = state.get("oi_resistances") if isinstance(state.get("oi_resistances"), list) else []
            buy += [float(x) for x in supps[:3] if isinstance(x, (int, float))]
            sell += [float(x) for x in ress[:3] if isinstance(x, (int, float))]
        except Exception:
            pass

        state["liquidity_buy_levels"] = self._uniq_sorted([x for x in buy if x is not None])
        state["liquidity_sell_levels"] = self._uniq_sorted([x for x in sell if x is not None])

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
            oi_compact = self._compact_oi_for_llm(self._oi_snapshot)
            state["oi"] = oi_compact
            state.update(derive_oi_features(spot_price=float(c.close), snapshot=oi_compact))
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

            # Attach derived levels + micro-context for better LLM reasoning.
            try:
                # Small candle history for pattern/context (kept compact).
                recent = list(self._candles[-11:])
                recent.append(
                    {
                        "dt": state["dt"],
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close,
                        "volume": candle.volume,
                    }
                )
                state["recent_candles"] = recent[-12:]
            except Exception:
                pass
            try:
                state["recent_decisions"] = [
                    {k: v for k, v in d.items() if k in {"dt", "action", "reason"}}
                    for d in self._decisions[-5:]
                    if isinstance(d, dict)
                ]
            except Exception:
                pass
            self._attach_dynamic_liquidity(state)

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
