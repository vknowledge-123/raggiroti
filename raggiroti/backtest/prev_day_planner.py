from __future__ import annotations

from dataclasses import dataclass
from datetime import time

from .csv_loader import Candle


@dataclass(frozen=True)
class PrevDayLevels:
    symbol: str
    date: str  # YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    last_hour_high: float
    last_hour_low: float


def compute_prev_day_levels(
    candles: list[Candle],
    last_hour_start: time = time(14, 30),
) -> PrevDayLevels:
    if not candles:
        raise ValueError("No candles")
    candles = sorted(candles, key=lambda c: c.dt)
    symbol = candles[0].symbol
    date_s = candles[0].dt.strftime("%Y-%m-%d")

    day_open = candles[0].open
    day_close = candles[-1].close
    day_high = max(c.high for c in candles)
    day_low = min(c.low for c in candles)

    lh = [c for c in candles if c.dt.time() >= last_hour_start]
    if not lh:
        lh = candles[-60:] if len(candles) >= 60 else candles
    last_hour_high = max(c.high for c in lh)
    last_hour_low = min(c.low for c in lh)

    return PrevDayLevels(
        symbol=symbol,
        date=date_s,
        open=day_open,
        high=day_high,
        low=day_low,
        close=day_close,
        last_hour_high=last_hour_high,
        last_hour_low=last_hour_low,
    )


def classify_open_scenario(
    day_open: float,
    prev_close: float,
    gap_up_threshold_points: float = 30.0,
    gap_down_threshold_points: float = 30.0,
    flat_threshold_points: float = 15.0,
) -> str:
    gap = day_open - prev_close
    if gap >= gap_up_threshold_points:
        return "gap_up"
    if gap <= -gap_down_threshold_points:
        return "gap_down"
    if abs(gap) <= flat_threshold_points:
        return "flat"
    return "mild_gap_up" if gap > 0 else "mild_gap_down"
