from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .models import LiveCandle, Tick


def _minute_floor(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


@dataclass
class CandleBuilder1m:
    """
    Builds 1-minute candles from ticks.

    - Assumes ticks are for a single instrument stream.
    - Emits a candle when a tick arrives in the next minute.
    """

    current_minute: datetime | None = None
    o: float | None = None
    h: float | None = None
    l: float | None = None
    c: float | None = None
    vol0: float | None = None
    vol_last: float | None = None

    def update(self, tick: Tick) -> LiveCandle | None:
        m = _minute_floor(tick.dt)
        if self.current_minute is None:
            self.current_minute = m
            self.o = tick.ltp
            self.h = tick.ltp
            self.l = tick.ltp
            self.c = tick.ltp
            self.vol0 = tick.volume
            self.vol_last = tick.volume
            return None

        if m == self.current_minute:
            self.h = tick.ltp if self.h is None else max(self.h, tick.ltp)
            self.l = tick.ltp if self.l is None else min(self.l, tick.ltp)
            self.c = tick.ltp
            if tick.volume is not None:
                self.vol_last = tick.volume
            return None

        # minute changed -> emit previous candle using last known OHLC
        candle = LiveCandle(
            dt=self.current_minute,
            open=float(self.o or tick.ltp),
            high=float(self.h or tick.ltp),
            low=float(self.l or tick.ltp),
            close=float(self.c or tick.ltp),
            volume=(None if self.vol_last is None or self.vol0 is None else float(self.vol_last - self.vol0)),
        )

        # start next candle
        self.current_minute = m
        self.o = tick.ltp
        self.h = tick.ltp
        self.l = tick.ltp
        self.c = tick.ltp
        self.vol0 = tick.volume
        self.vol_last = tick.volume
        return candle

