from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Callable

from raggiroti.dhan.session import DhanUnavailable


def _now_dt() -> datetime:
    return datetime.now(tz=ZoneInfo("Asia/Kolkata"))


def _pick(d: dict, keys: list[str]) -> float | None:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                continue
    return None


@dataclass(frozen=True)
class LiveFeedInstrument:
    exchange_segment: int
    security_id: str
    subscription_type: int


class DhanLiveFeed:
    """
    DhanHQ-py MarketFeed wrapper (sync).

    Docs show:
      from dhanhq import DhanContext, MarketFeed
      data = MarketFeed(dhan_context, instruments, version="v2")
      data.run_forever()
      response = data.get_data()
    """

    def __init__(self, *, client_id: str, access_token: str, instruments: list[LiveFeedInstrument]) -> None:
        try:
            from dhanhq import DhanContext, MarketFeed  # type: ignore
        except Exception as e:  # pragma: no cover
            raise DhanUnavailable("Dhan SDK not installed. Install 'dhanhq'.") from e

        self._DhanContext = DhanContext
        self._MarketFeed = MarketFeed
        self._ctx = DhanContext(client_id, access_token)
        self._instruments = [(i.exchange_segment, i.security_id, i.subscription_type) for i in instruments]
        self._feed = MarketFeed(self._ctx, self._instruments, version="v2")

    def disconnect(self) -> None:
        try:
            self._feed.disconnect()
        except Exception:
            pass

    def run_forever(self) -> None:
        self._feed.run_forever()

    def get_data(self) -> dict:
        return self._feed.get_data()

    def iter_ticks(self, on_tick: Callable[[dict], None], sleep_s: float = 0.0) -> None:
        """
        Blocking loop that calls on_tick(raw_message).
        """
        while True:
            self.run_forever()
            msg = self.get_data()
            if msg:
                on_tick(msg)
            if sleep_s > 0:
                time.sleep(sleep_s)


def parse_marketfeed_tick(msg: dict) -> tuple[str, float, float | None, datetime]:
    """
    Best-effort parse of MarketFeed message into (security_id, ltp, volume, dt).

    Dhan payload keys can vary by subscription type.
    We attempt common keys; if dt not present, use now.
    """
    data = msg.get("Data") if isinstance(msg.get("Data"), dict) else msg
    if not isinstance(data, dict):
        data = {}
    sec = str(data.get("security_id") or data.get("SecurityId") or data.get("securityId") or "")

    ltp = _pick(data, ["LTP", "ltp", "last_traded_price", "LastTradedPrice", "lastTradedPrice"])
    if ltp is None:
        raise ValueError(f"missing ltp in tick: keys={list(data.keys())[:12]}")

    vol = _pick(data, ["volume", "Volume", "total_volume", "TotalVolume", "totalVolume"])

    # Some feeds include exchange time; if not, use now.
    dt = _now_dt()
    return sec, float(ltp), (None if vol is None else float(vol)), dt
