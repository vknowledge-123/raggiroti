from __future__ import annotations

from collections import defaultdict

from .csv_loader import Candle


def group_by_date(candles: list[Candle]) -> dict[str, list[Candle]]:
    by: dict[str, list[Candle]] = defaultdict(list)
    for c in candles:
        by[c.dt.strftime("%Y-%m-%d")].append(c)
    return dict(by)

