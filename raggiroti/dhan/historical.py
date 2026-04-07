from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from raggiroti.backtest.csv_loader import Candle


DHAN_BASE = "https://api.dhan.co/v2"


@dataclass(frozen=True)
class DhanIntradayRequest:
    security_id: str
    exchange_segment: str
    instrument: str
    interval: str = "1"
    oi: bool = False
    from_dt: datetime | None = None
    to_dt: datetime | None = None


def _fmt_dt(dt: datetime) -> str:
    # Docs: "YYYY-MM-DD HH:MM:SS"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _post_json(url: str, access_token: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": access_token,
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def fetch_intraday_candles(req: DhanIntradayRequest, access_token: str) -> list[Candle]:
    """
    Fetch intraday candles from Dhan /charts/intraday and return as Candle objects.

    Dhan docs: intraday supports 1/5/15/25/60 minute; max 90 days per call.
    """
    if req.from_dt is None or req.to_dt is None:
        raise ValueError("from_dt and to_dt are required")
    if req.from_dt >= req.to_dt:
        raise ValueError("from_dt must be < to_dt")

    url = f"{DHAN_BASE}/charts/intraday"

    # Chunk into <= 90 days windows (docs).
    max_days = 90
    candles: list[Candle] = []

    cur_from = req.from_dt
    while cur_from < req.to_dt:
        cur_to = min(req.to_dt, cur_from + timedelta(days=max_days))
        payload = {
            "securityId": req.security_id,
            "exchangeSegment": req.exchange_segment,
            "instrument": req.instrument,
            "interval": req.interval,
            "oi": req.oi,
            "fromDate": _fmt_dt(cur_from),
            "toDate": _fmt_dt(cur_to),
        }
        data = _post_json(url, access_token, payload)

        opens = data.get("open") or []
        highs = data.get("high") or []
        lows = data.get("low") or []
        closes = data.get("close") or []
        vols = data.get("volume") or []
        ts = data.get("timestamp") or []

        n = min(len(opens), len(highs), len(lows), len(closes), len(vols), len(ts))
        symbol = req.security_id  # we keep security_id in symbol field for now
        for i in range(n):
            # Dhan returns epoch timestamps; normalize to IST for consistent candle boundaries on servers (GCP is often UTC).
            dt = datetime.fromtimestamp(int(ts[i]), tz=timezone.utc).astimezone(ZoneInfo("Asia/Kolkata"))
            candles.append(
                Candle(
                    symbol=symbol,
                    dt=dt,
                    open=float(opens[i]),
                    high=float(highs[i]),
                    low=float(lows[i]),
                    close=float(closes[i]),
                    volume=float(vols[i]),
                )
            )

        cur_from = cur_to

    candles.sort(key=lambda c: c.dt)
    return candles
