from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math

from .csv_loader import Candle
from .prev_day_planner import PrevDayLevels, classify_open_scenario


@dataclass
class StateBuilder:
    """
    Minimal state builder placeholder.
    Replace with your full Nexus Ultra computations:
    - PDH/PDL tracking
    - last 30m delta
    - zone detection (discount/fair/inflated)
    - structure (Dow swings)
    - confirmation flags (reclaim/acceptance)
    """

    prev: PrevDayLevels | None = None
    pdh: float | None = None  # intraday high so far
    pdl: float | None = None  # intraday low so far
    day_open: float | None = None
    scenario: str | None = None

    # Swing-based Dow structure with confirmation lag (uses only past data once confirmed)
    swing_window: int = 2  # confirmation delay in candles
    _buf: deque[Candle] | None = None
    _swing_highs: list[float] | None = None
    _swing_lows: list[float] | None = None
    _swing_high_points: list[tuple[str, float]] | None = None  # (dt_iso, price)
    _swing_low_points: list[tuple[str, float]] | None = None  # (dt_iso, price)
    _structure: str = "unknown"  # bull|bear|range|unknown

    # 5-minute aggregation for confirmation
    _five_acc: list[Candle] | None = None
    _buf5: deque[Candle] | None = None
    _swing_highs5: list[float] | None = None
    _swing_lows5: list[float] | None = None
    _swing_high_points5: list[tuple[str, float]] | None = None
    _swing_low_points5: list[tuple[str, float]] | None = None
    _structure5: str = "unknown"

    _first_candle_color: str | None = None
    _gap_type: str | None = None
    _prev_close_seen: float | None = None

    # Rolling analytics (1m only)
    _roll: deque[Candle] | None = None
    _minute_index: int = 0
    _market_type: str = "unknown"  # trend|trap|chop|unknown
    _comfort_risk: bool = False
    _operator_exit_risk: bool = False
    _retail_participation_present: bool = True
    _overcrowding_risk: bool = False
    _gap_threshold_points: float = 30.0
    _flat_threshold_points: float = 15.0
    _prev_confirmed_long: bool = False
    _prev_confirmed_short: bool = False

    def on_new_day(
        self,
        prev: PrevDayLevels | None = None,
        gap_threshold_points: float = 30.0,
        flat_threshold_points: float = 15.0,
    ) -> None:
        self.prev = prev
        self._gap_threshold_points = float(gap_threshold_points)
        self._flat_threshold_points = float(flat_threshold_points)
        self.pdh = None
        self.pdl = None
        self.day_open = None
        self.scenario = None
        self._buf = deque(maxlen=(2 * self.swing_window + 1))
        self._swing_highs = []
        self._swing_lows = []
        self._swing_high_points = []
        self._swing_low_points = []
        self._structure = "unknown"
        self._five_acc = []
        self._buf5 = deque(maxlen=(2 * self.swing_window + 1))
        self._swing_highs5 = []
        self._swing_lows5 = []
        self._swing_high_points5 = []
        self._swing_low_points5 = []
        self._structure5 = "unknown"
        self._first_candle_color = None
        self._gap_type = None
        self._prev_close_seen = None
        self._roll = deque(maxlen=60)  # 60 minutes context
        self._minute_index = 0
        self._market_type = "unknown"
        self._comfort_risk = False
        self._operator_exit_risk = False
        self._retail_participation_present = True
        self._overcrowding_risk = False
        self._prev_confirmed_long = False
        self._prev_confirmed_short = False

    def _update_swings(self) -> None:
        if self._buf is None or self._swing_highs is None or self._swing_lows is None:
            return
        w = self.swing_window
        if len(self._buf) < (2 * w + 1):
            return
        mid = self._buf[w]
        left = list(self._buf)[:w]
        right = list(self._buf)[w + 1 :]
        is_swing_high = all(mid.high > c.high for c in left) and all(mid.high >= c.high for c in right)
        is_swing_low = all(mid.low < c.low for c in left) and all(mid.low <= c.low for c in right)
        if is_swing_high:
            self._swing_highs.append(mid.high)
            if self._swing_high_points is not None:
                self._swing_high_points.append((mid.dt.isoformat(timespec="minutes"), float(mid.high)))
        if is_swing_low:
            self._swing_lows.append(mid.low)
            if self._swing_low_points is not None:
                self._swing_low_points.append((mid.dt.isoformat(timespec="minutes"), float(mid.low)))

        if len(self._swing_highs) >= 2 and len(self._swing_lows) >= 2:
            h1, h2 = self._swing_highs[-2], self._swing_highs[-1]
            l1, l2 = self._swing_lows[-2], self._swing_lows[-1]
            if h2 > h1 and l2 > l1:
                self._structure = "bull"
            elif h2 < h1 and l2 < l1:
                self._structure = "bear"
            else:
                self._structure = "range"

    def _update_swings5(self) -> None:
        if self._buf5 is None or self._swing_highs5 is None or self._swing_lows5 is None:
            return
        w = self.swing_window
        if len(self._buf5) < (2 * w + 1):
            return
        mid = self._buf5[w]
        left = list(self._buf5)[:w]
        right = list(self._buf5)[w + 1 :]
        is_swing_high = all(mid.high > c.high for c in left) and all(mid.high >= c.high for c in right)
        is_swing_low = all(mid.low < c.low for c in left) and all(mid.low <= c.low for c in right)
        if is_swing_high:
            self._swing_highs5.append(mid.high)
            if self._swing_high_points5 is not None:
                self._swing_high_points5.append((mid.dt.isoformat(timespec="minutes"), float(mid.high)))
        if is_swing_low:
            self._swing_lows5.append(mid.low)
            if self._swing_low_points5 is not None:
                self._swing_low_points5.append((mid.dt.isoformat(timespec="minutes"), float(mid.low)))

        if len(self._swing_highs5) >= 2 and len(self._swing_lows5) >= 2:
            h1, h2 = self._swing_highs5[-2], self._swing_highs5[-1]
            l1, l2 = self._swing_lows5[-2], self._swing_lows5[-1]
            if h2 > h1 and l2 > l1:
                self._structure5 = "bull"
            elif h2 < h1 and l2 < l1:
                self._structure5 = "bear"
            else:
                self._structure5 = "range"

    def _accumulate_5m(self, candle: Candle) -> None:
        if self._five_acc is None or self._buf5 is None:
            return
        self._five_acc.append(candle)
        if len(self._five_acc) < 5:
            return
        # Use non-overlapping 5-candle blocks to build 5m candles.
        block = list(self._five_acc)
        dt0 = block[0].dt
        o = block[0].open
        h = max(c.high for c in block)
        l = min(c.low for c in block)
        c = block[-1].close
        v = sum(c.volume for c in block)
        agg = Candle(symbol=block[0].symbol, dt=dt0, open=o, high=h, low=l, close=c, volume=v)
        self._buf5.append(agg)
        self._update_swings5()
        self._five_acc.clear()

    def update(self, candle: Candle) -> dict:
        self._minute_index += 1
        if self.day_open is None:
            self.day_open = candle.open
            if self.prev is not None:
                self.scenario = classify_open_scenario(
                    self.day_open,
                    self.prev.close,
                    gap_threshold_points=self._gap_threshold_points,
                    flat_threshold_points=self._flat_threshold_points,
                )
                self._gap_type = "gap_up" if self.day_open > self.prev.close else ("gap_down" if self.day_open < self.prev.close else "flat")
                self._prev_close_seen = self.prev.close

        if self._first_candle_color is None:
            self._first_candle_color = "green" if candle.close >= candle.open else "red"

        if self._buf is not None:
            self._buf.append(candle)
            self._update_swings()
            self._accumulate_5m(candle)

        if self._roll is not None:
            self._roll.append(candle)
        self.pdh = candle.high if self.pdh is None else max(self.pdh, candle.high)
        self.pdl = candle.low if self.pdl is None else min(self.pdl, candle.low)

        # Zone heuristic:
        # prefer previous-day range as a stable anchor (better for opening decisions),
        # fallback to intraday range early in session.
        zone = "fair"
        anchor_low = self.prev.low if self.prev is not None else self.pdl
        anchor_high = self.prev.high if self.prev is not None else self.pdh
        if anchor_low is not None and anchor_high is not None and anchor_high > anchor_low:
            q1 = anchor_low + 0.25 * (anchor_high - anchor_low)
            q3 = anchor_low + 0.75 * (anchor_high - anchor_low)
            if candle.close <= q1:
                zone = "discount"
            elif candle.close >= q3:
                zone = "inflated"

        # Basic reclaim confirmations relative to previous day PDH/PDL (sweep + reclaim)
        confirmed_long = False
        confirmed_short = False
        swept_prev_pdh = False
        swept_prev_pdl = False
        reclaimed_prev_pdh = False
        reclaimed_prev_pdl = False
        if self.prev is not None:
            swept_prev_pdh = candle.high > self.prev.high
            swept_prev_pdl = candle.low < self.prev.low
            reclaimed_prev_pdh = swept_prev_pdh and candle.close < self.prev.high
            reclaimed_prev_pdl = swept_prev_pdl and candle.close > self.prev.low
            confirmed_long = reclaimed_prev_pdl
            confirmed_short = reclaimed_prev_pdh

        # Edge-triggered confirmation events (prevents calling LLM repeatedly while a condition stays true)
        event_confirmed_long = confirmed_long and not self._prev_confirmed_long
        event_confirmed_short = confirmed_short and not self._prev_confirmed_short
        self._prev_confirmed_long = confirmed_long
        self._prev_confirmed_short = confirmed_short

        # --------- High-impact sensors (1m, single-instrument) ---------
        # These are proxy-based; improve thresholds as you calibrate on BankNifty.
        self._market_type = self._classify_market_type()
        self._comfort_risk = self._detect_comfort_risk()
        self._operator_exit_risk = self._detect_operator_exit_risk()
        self._retail_participation_present = self._detect_participation()
        self._overcrowding_risk = self._detect_overcrowding()

        validity_ok = self._retail_participation_present and (self._market_type != "chop")

        last_swing_high = None
        last_swing_low = None
        if self._swing_high_points:
            last_swing_high = {"dt": self._swing_high_points[-1][0], "price": float(self._swing_high_points[-1][1])}
        if self._swing_low_points:
            last_swing_low = {"dt": self._swing_low_points[-1][0], "price": float(self._swing_low_points[-1][1])}
        last_swing_high5 = None
        last_swing_low5 = None
        if self._swing_high_points5:
            last_swing_high5 = {"dt": self._swing_high_points5[-1][0], "price": float(self._swing_high_points5[-1][1])}
        if self._swing_low_points5:
            last_swing_low5 = {"dt": self._swing_low_points5[-1][0], "price": float(self._swing_low_points5[-1][1])}

        broke_last_swing_high = False
        broke_last_swing_low = False
        reclaimed_last_swing_high = False
        reclaimed_last_swing_low = False
        if last_swing_high is not None:
            lvl = float(last_swing_high["price"])
            broke_last_swing_high = candle.close > lvl
            reclaimed_last_swing_high = candle.high > lvl and candle.close < lvl
        if last_swing_low is not None:
            lvl = float(last_swing_low["price"])
            broke_last_swing_low = candle.close < lvl
            reclaimed_last_swing_low = candle.low < lvl and candle.close > lvl

        return {
            "dt": candle.dt.isoformat(timespec="minutes"),
            "price": candle.close,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "scenario": self.scenario,
            "gap_type": self._gap_type,
            "first_candle_color": self._first_candle_color,
            "minute_index": self._minute_index,
            "prev_close": None if self.prev is None else self.prev.close,
            "prev_pdh": None if self.prev is None else self.prev.high,
            "prev_pdl": None if self.prev is None else self.prev.low,
            "prev_last_hour_high": None if self.prev is None else self.prev.last_hour_high,
            "prev_last_hour_low": None if self.prev is None else self.prev.last_hour_low,
            "swept_prev_pdh": swept_prev_pdh,
            "swept_prev_pdl": swept_prev_pdl,
            "reclaimed_prev_pdh": reclaimed_prev_pdh,
            "reclaimed_prev_pdl": reclaimed_prev_pdl,
            "pdh": self.pdh,
            "pdl": self.pdl,
            "zone": zone,
            "dow_structure_1m": self._structure,
            "dow_structure_5m": self._structure5,
            # Back-compat for older retrieval/policies
            "structure": self._structure,
            "swing_window": self.swing_window,
            "swing_highs_1m": [] if self._swing_high_points is None else [{"dt": d, "price": p} for (d, p) in self._swing_high_points[-5:]],
            "swing_lows_1m": [] if self._swing_low_points is None else [{"dt": d, "price": p} for (d, p) in self._swing_low_points[-5:]],
            "last_swing_high_1m": last_swing_high,
            "last_swing_low_1m": last_swing_low,
            "swing_highs_5m": [] if self._swing_high_points5 is None else [{"dt": d, "price": p} for (d, p) in self._swing_high_points5[-3:]],
            "swing_lows_5m": [] if self._swing_low_points5 is None else [{"dt": d, "price": p} for (d, p) in self._swing_low_points5[-3:]],
            "last_swing_high_5m": last_swing_high5,
            "last_swing_low_5m": last_swing_low5,
            "broke_last_swing_high": broke_last_swing_high,
            "broke_last_swing_low": broke_last_swing_low,
            "reclaimed_last_swing_high": reclaimed_last_swing_high,
            "reclaimed_last_swing_low": reclaimed_last_swing_low,
            "sl_available": True,
            "market_type": self._market_type,
            "comfort_risk": self._comfort_risk,
            "operator_exit_risk": self._operator_exit_risk,
            "retail_participation_present": self._retail_participation_present,
            "overcrowding_risk": self._overcrowding_risk,
            "validity_ok": validity_ok,
            "confirmed_long": confirmed_long,
            "confirmed_short": confirmed_short,
            "event_confirmed_long": event_confirmed_long,
            "event_confirmed_short": event_confirmed_short,
        }

    def _classify_market_type(self) -> str:
        if self._roll is None or len(self._roll) < 20:
            return "unknown"
        w = list(self._roll)[-20:]
        highs = [c.high for c in w]
        lows = [c.low for c in w]
        closes = [c.close for c in w]
        opens = [c.open for c in w]
        ranges = [max(0.01, c.high - c.low) for c in w]

        window_range = max(highs) - min(lows)
        avg_range = sum(ranges) / len(ranges)
        net_move = closes[-1] - closes[0]

        # Reversal count: sign changes in candle body direction
        dirs = []
        for o, c in zip(opens, closes):
            d = 1 if c > o else (-1 if c < o else 0)
            dirs.append(d)
        rev = 0
        last = 0
        for d in dirs:
            if d == 0:
                continue
            if last != 0 and d != last:
                rev += 1
            last = d

        # Trendiness ratio: net move vs total movement
        total_move = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))
        trend_ratio = abs(net_move) / max(0.01, total_move)

        # Chop: small range and many reversals
        if window_range <= max(20.0, 2.2 * avg_range) and rev >= 6:
            return "chop"
        # Trend: strong trend ratio and low reversals
        if trend_ratio >= 0.65 and rev <= 3 and window_range >= (4.0 * avg_range):
            return "trend"
        # Trap/volatile: moderate range but many reversals
        if rev >= 6 and window_range >= (3.0 * avg_range):
            return "trap"
        return "unknown"

    def _detect_comfort_risk(self) -> bool:
        # Comfort in single-instrument 1m is proxied as "obvious breakout + hold + volume support"
        if self.prev is None or self._roll is None or len(self._roll) < 25:
            return False
        w3 = list(self._roll)[-3:]
        w20 = list(self._roll)[-20:]
        avg_vol20 = sum(c.volume for c in w20) / max(1.0, len(w20))
        avg_vol3 = sum(c.volume for c in w3) / 3.0
        vol_ok = avg_vol3 >= (1.2 * avg_vol20) if avg_vol20 > 0 else True

        last_close = w3[-1].close
        all_above_pdh = all(c.close > self.prev.high for c in w3)
        all_below_pdl = all(c.close < self.prev.low for c in w3)

        # Obvious breakout beyond prev day extremes + holds for 3 mins with volume support
        if vol_ok and (all_above_pdh or all_below_pdl):
            return True

        # Also treat "round-number break + hold" as comfort (BankNifty often reacts to 100s)
        step = 100.0
        rn = round(last_close / step) * step
        above_rn = all(c.close > rn for c in w3)
        below_rn = all(c.close < rn for c in w3)
        if vol_ok and (above_rn or below_rn):
            return True
        return False

    def _detect_operator_exit_risk(self) -> bool:
        # Proxy: after a strong impulse, progress slows into choppy grind (distribution).
        if self._roll is None or len(self._roll) < 40:
            return False
        w10 = list(self._roll)[-10:]
        w30 = list(self._roll)[-30:]
        avg_range30 = sum(max(0.01, c.high - c.low) for c in w30) / len(w30)

        impulse = w10[-1].close - w10[0].close
        impulse_abs = abs(impulse)
        # Need a meaningful impulse first
        if impulse_abs < max(80.0, 8.0 * avg_range30):
            return False

        # After impulse, check last 5 mins for "stall": small net progress but lots of movement
        w5 = list(self._roll)[-5:]
        net5 = w5[-1].close - w5[0].close
        total5 = sum(abs(w5[i].close - w5[i - 1].close) for i in range(1, len(w5)))
        stall = abs(net5) <= 0.2 * impulse_abs and total5 >= 0.8 * impulse_abs

        # Extra: bodies get small (grind)
        bodies = [abs(c.close - c.open) for c in w5]
        grind = (sum(bodies) / len(bodies)) <= (0.35 * avg_range30)
        return stall and grind

    def _detect_participation(self) -> bool:
        # Proxy: volume and range not dead (avoid low-energy periods).
        if self._roll is None or len(self._roll) < 30:
            return True
        w30 = list(self._roll)[-30:]
        vols = [c.volume for c in w30]
        ranges = [max(0.01, c.high - c.low) for c in w30]
        avg_vol = sum(vols) / len(vols)
        avg_rng = sum(ranges) / len(ranges)
        # If volume is mostly zero in dataset, treat as present.
        if avg_vol == 0:
            return True
        return avg_vol > 0 and avg_rng >= 10.0

    def _detect_overcrowding(self) -> bool:
        # Single-instrument proxy: comfort + trend day + extended move implies overcrowding risk for late entries.
        if self._roll is None or len(self._roll) < 30:
            return False
        w30 = list(self._roll)[-30:]
        net = w30[-1].close - w30[0].close
        extended = abs(net) >= 200.0  # calibrate for BankNifty
        return bool(self._comfort_risk and self._market_type == "trend" and extended)
