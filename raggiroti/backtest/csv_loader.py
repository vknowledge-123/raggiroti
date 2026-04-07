from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Candle:
    symbol: str
    dt: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_candles(path: str) -> list[Candle]:
    candles: list[Candle] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 7:
                continue
            symbol, date_s, time_s, o, h, l, c = row[:7]
            vol = row[7] if len(row) >= 8 else "0"
            dt = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H:%M")
            candles.append(
                Candle(
                    symbol=symbol,
                    dt=dt,
                    open=float(o),
                    high=float(h),
                    low=float(l),
                    close=float(c),
                    volume=float(vol),
                )
            )
    candles.sort(key=lambda x: x.dt)
    return candles

