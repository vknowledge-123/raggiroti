from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Callable

from raggiroti.dhan.session import DhanUnavailable


try:
    _IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    # Some minimal Docker images don't ship tzdata/zoneinfo files.
    _IST = timezone(timedelta(hours=5, minutes=30))


def _now_dt() -> datetime:
    return datetime.now(tz=_IST)


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
        """
        DhanHQ PyPI latest stable is 2.0.2 (as of 2026-04), which exposes live feed via:
          from dhanhq import marketfeed
          marketfeed.DhanFeed(client_id, access_token, instruments, version="v2")

        DhanHQ repo mentions newer import paths (DhanContext/MarketFeed) but those releases
        were yanked/pre-release; so we support the stable path primarily.
        """
        self._instruments = [(i.exchange_segment, i.security_id, i.subscription_type) for i in instruments]

        # Prefer stable marketfeed module.
        try:
            # Ensure an event loop exists for libraries that call asyncio.get_event_loop().
            try:
                import asyncio as _asyncio

                try:
                    _asyncio.get_event_loop()
                except RuntimeError:
                    _asyncio.set_event_loop(_asyncio.new_event_loop())
            except Exception:
                pass

            from dhanhq import marketfeed  # type: ignore
            self._feed = marketfeed.DhanFeed(client_id, access_token, self._instruments, version="v2")
            self._mode = "marketfeed"
            return
        except Exception as e:
            last_err = e

        # Fallback: try newer interface if present.
        try:
            from dhanhq import DhanContext, MarketFeed  # type: ignore
            ctx = DhanContext(client_id, access_token)
            self._feed = MarketFeed(ctx, self._instruments, version="v2")
            self._mode = "class"
            return
        except Exception as e:  # pragma: no cover
            raise DhanUnavailable(f"Dhan SDK import/init failed. Install/upgrade 'dhanhq'. Root error: {last_err} / {e}") from e

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
        backoff = 0.5
        # Connect once; reconnect on exceptions.
        self.run_forever()
        while True:
            try:
                msg = self.get_data()
                if msg:
                    on_tick(msg)
                backoff = 0.5
            except Exception:
                # Reconnect loop: disconnect, wait a bit, and connect again.
                try:
                    self.disconnect()
                except Exception:
                    pass
                time.sleep(backoff)
                backoff = min(backoff * 1.6, 8.0)
                try:
                    self.run_forever()
                except Exception:
                    continue
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
        # MarketFeed also emits non-trade packets (prev_close / status / OI etc).
        # These do not contain LTP and should be ignored by candle builders.
        if any(k in data for k in ("prev_close", "prev_OI", "prev_oi", "OI", "oi", "status", "Status")) or str(data.get("type") or "").lower().find("prev") >= 0:
            raise ValueError(f"non_ltp_packet:{data.get('type') or ''}")
        raise ValueError(f"missing ltp in tick: keys={list(data.keys())[:12]}")

    vol = _pick(data, ["volume", "Volume", "total_volume", "TotalVolume", "totalVolume"])

    # Some feeds include exchange time; if not, use now.
    dt = _now_dt()
    return sec, float(ltp), (None if vol is None else float(vol)), dt
