from __future__ import annotations

from dataclasses import dataclass

from raggiroti.dhan.session import DhanUnavailable


@dataclass(frozen=True)
class OIWall:
    strike: float
    side: str  # CE|PE
    oi: float
    oi_change: float | None
    ltp: float | None


def _to_float(x) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


class DhanOptionChainClient:
    def __init__(self, *, client_id: str, access_token: str) -> None:
        """
        Supports stable DhanHQ-py (PyPI 2.0.2) style:
          from dhanhq import dhanhq
          dhan = dhanhq(client_id, access_token)
          dhan.expiry_list(...)
          dhan.option_chain(...)

        Also supports newer modular SDK style if present.
        """
        last_err: Exception | None = None
        try:
            from dhanhq import dhanhq  # type: ignore
            self._client = dhanhq(client_id, access_token)
            self._mode = "client"
            return
        except Exception as e:
            last_err = e

        try:
            from dhanhq import DhanContext, OptionChain  # type: ignore
            ctx = DhanContext(client_id, access_token)
            self._client = OptionChain(ctx)
            self._mode = "class"
            return
        except Exception as e:  # pragma: no cover
            raise DhanUnavailable(f"Dhan SDK import/init failed. Root error: {last_err} / {e}") from e

    def expiry_list(self, *, under_security_id: int, under_exchange_segment: str) -> dict:
        if getattr(self, "_mode", "") == "client":
            return self._client.expiry_list(under_security_id=under_security_id, under_exchange_segment=under_exchange_segment)
        return self._client.expiry_list(under_security_id, under_exchange_segment)

    def option_chain(self, *, under_security_id: int, under_exchange_segment: str, expiry: str) -> dict:
        if getattr(self, "_mode", "") == "client":
            return self._client.option_chain(under_security_id=under_security_id, under_exchange_segment=under_exchange_segment, expiry=expiry)
        return self._client.option_chain(under_security_id, under_exchange_segment, expiry)


def summarize_oi_walls(option_chain_resp: dict, top_n: int = 5) -> dict:
    """
    Converts the full option chain payload into a compact OI "wall" summary.
    Assumes response contains strike rows with CE/PE legs (common in Dhan).
    """
    rows = option_chain_resp.get("data") or option_chain_resp.get("Data") or option_chain_resp.get("option_data") or option_chain_resp.get("optionData")
    if not isinstance(rows, list):
        return {"ok": False, "error": "unexpected option_chain format", "raw_keys": list(option_chain_resp.keys())[:20]}

    ce: list[OIWall] = []
    pe: list[OIWall] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        strike = _to_float(r.get("strikePrice") or r.get("strike_price") or r.get("strike"))
        if strike is None:
            continue
        ce_leg = r.get("CE") or r.get("ce") or r.get("call") or {}
        pe_leg = r.get("PE") or r.get("pe") or r.get("put") or {}
        if isinstance(ce_leg, dict):
            ce.append(
                OIWall(
                    strike=strike,
                    side="CE",
                    oi=_to_float(ce_leg.get("oi") or ce_leg.get("openInterest") or ce_leg.get("open_interest") or 0.0) or 0.0,
                    oi_change=_to_float(ce_leg.get("oiChange") or ce_leg.get("changeinOpenInterest") or ce_leg.get("change_in_oi")),
                    ltp=_to_float(ce_leg.get("ltp") or ce_leg.get("lastPrice")),
                )
            )
        if isinstance(pe_leg, dict):
            pe.append(
                OIWall(
                    strike=strike,
                    side="PE",
                    oi=_to_float(pe_leg.get("oi") or pe_leg.get("openInterest") or pe_leg.get("open_interest") or 0.0) or 0.0,
                    oi_change=_to_float(pe_leg.get("oiChange") or pe_leg.get("changeinOpenInterest") or pe_leg.get("change_in_oi")),
                    ltp=_to_float(pe_leg.get("ltp") or pe_leg.get("lastPrice")),
                )
            )

    ce_sorted = sorted(ce, key=lambda x: x.oi, reverse=True)[:top_n]
    pe_sorted = sorted(pe, key=lambda x: x.oi, reverse=True)[:top_n]
    return {
        "ok": True,
        "ce_walls": [w.__dict__ for w in ce_sorted],
        "pe_walls": [w.__dict__ for w in pe_sorted],
    }


def derive_oi_features(*, spot_price: float | None, snapshot: dict | None) -> dict:
    """
    Best-effort derived features from a compact OI snapshot.

    We intentionally keep this light-weight and robust:
    - Works even if OI change / LTP is missing
    - Produces small fields suitable for per-candle prompts
    """
    if not isinstance(snapshot, dict) or not snapshot.get("ok"):
        return {"oi_bias": "unknown", "oi_support": None, "oi_resistance": None, "oi_supports": [], "oi_resistances": []}

    ce = snapshot.get("ce_walls") or []
    pe = snapshot.get("pe_walls") or []
    if not isinstance(ce, list) or not isinstance(pe, list):
        return {"oi_bias": "unknown", "oi_support": None, "oi_resistance": None, "oi_supports": [], "oi_resistances": []}

    def _sum(xs: list[dict], k: str) -> float:
        s = 0.0
        for x in xs:
            try:
                v = x.get(k)
                if v is None:
                    continue
                s += float(v)
            except Exception:
                continue
        return s

    ce_oi = _sum(ce, "oi")
    pe_oi = _sum(pe, "oi")
    ce_chg = _sum(ce, "oi_change")
    pe_chg = _sum(pe, "oi_change")

    # Bias by OI dominance + change-in-OI agreement (secondary only).
    bias = "range"
    if ce_oi > 1.2 * max(1.0, pe_oi) and ce_chg >= pe_chg:
        bias = "bearish"
    elif pe_oi > 1.2 * max(1.0, ce_oi) and pe_chg >= ce_chg:
        bias = "bullish"

    supports = []
    resistances = []
    try:
        supports = sorted({float(x.get("strike")) for x in pe if x.get("strike") is not None})
        resistances = sorted({float(x.get("strike")) for x in ce if x.get("strike") is not None})
    except Exception:
        supports = []
        resistances = []

    support = supports[-1] if supports else None
    resistance = resistances[0] if resistances else None
    # If spot_price is known, pick nearest relevant.
    if spot_price is not None:
        try:
            sp = float(spot_price)
            below = [s for s in supports if s <= sp]
            above = [r for r in resistances if r >= sp]
            support = (below[-1] if below else support)
            resistance = (above[0] if above else resistance)
        except Exception:
            pass

    return {
        "oi_bias": bias,
        "oi_support": support,
        "oi_resistance": resistance,
        "oi_supports": supports[:5],
        "oi_resistances": resistances[:5],
        "oi_ce_oi_sum": ce_oi,
        "oi_pe_oi_sum": pe_oi,
        "oi_ce_chg_sum": ce_chg,
        "oi_pe_chg_sum": pe_chg,
    }
