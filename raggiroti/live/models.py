from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Tick:
    dt: datetime
    security_id: str
    ltp: float
    volume: float | None = None


@dataclass(frozen=True)
class LiveCandle:
    dt: datetime  # candle start (minute)
    open: float
    high: float
    low: float
    close: float
    volume: float | None


@dataclass(frozen=True)
class DecisionOut:
    action: str  # BUY|SELL|WAIT|EXIT
    sl: float | None
    targets: list[float]
    raw: dict

