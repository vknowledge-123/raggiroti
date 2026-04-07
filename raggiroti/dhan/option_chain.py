from __future__ import annotations

from dataclasses import dataclass
import re

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


def _to_float_india(x) -> float | None:
    """
    Parses NSE-style numeric strings:
    - commas as thousand separators: "2,70,790" -> 270790
    - missing: "-" -> None
    """
    if x is None:
        return None
    s = str(x).strip()
    if not s or s == "-" or s.lower() == "na":
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def _unwrap_dhanhq(payload: object) -> object:
    """
    dhanhq client methods often return a wrapper:
      {"status":"success|failure","remarks":...,"data":<payload>}
    """
    if isinstance(payload, dict) and "data" in payload:
        try:
            st = str(payload.get("status") or "").strip().lower()
        except Exception:
            st = ""
        if st in {"success", "failure"}:
            return payload.get("data")
    return payload


def _looks_like_leg_dict(v: object) -> bool:
    if not isinstance(v, dict):
        return False
    for k in ("oi", "openInterest", "open_interest", "OI"):
        if k in v and v.get(k) is not None:
            return True
    for k in ("oiChange", "changeinOpenInterest", "change_in_oi", "oi_change"):
        if k in v and v.get(k) is not None:
            return True
    return False


def _strike_from_any(d: dict) -> float | None:
    return _to_float(d.get("strikePrice") or d.get("strike_price") or d.get("strike") or d.get("StrikePrice"))


def _side_from_any(d: dict) -> str | None:
    """
    Returns "CE" or "PE" if the object looks like a single option leg row.
    """
    # Common explicit fields
    for k in ("option_type", "optionType", "right", "side", "instrument_type", "cpType", "call_put", "callPut"):
        if k in d and d.get(k) is not None:
            s = str(d.get(k) or "").strip().upper()
            if s in {"CE", "CALL", "C"}:
                return "CE"
            if s in {"PE", "PUT", "P"}:
                return "PE"
    # Fallback: infer from trading symbol if present.
    for k in ("trading_symbol", "tradingsymbol", "symbol", "tsym", "instrument", "name"):
        if k in d and d.get(k) is not None:
            s = str(d.get(k) or "").upper()
            # Keep this intentionally simple (avoid false positives from words like "PRICE").
            if " CE" in s or s.endswith("CE") or re.search(r"\bCE\b", s):
                return "CE"
            if " PE" in s or s.endswith("PE") or re.search(r"\bPE\b", s):
                return "PE"
    return None


def _leg_from_any(d: dict) -> dict:
    """
    Normalizes a single option-leg dict to our internal keys.
    """
    return {
        "oi": _to_float(d.get("oi") or d.get("openInterest") or d.get("open_interest") or d.get("OI")) or 0.0,
        "oiChange": _to_float(d.get("oiChange") or d.get("changeinOpenInterest") or d.get("change_in_oi") or d.get("oi_change")),
        "ltp": _to_float(d.get("ltp") or d.get("lastPrice") or d.get("LTP") or d.get("lp")),
        "volume": _to_float(d.get("volume") or d.get("Volume")),
    }


def _merge_leg_rows(legs: list[dict]) -> list[dict] | None:
    """
    Some APIs return a flat list where each row is a single option leg
    (strike + CE/PE + oi fields). Merge those into strike rows:
      {"strikePrice":..., "CE":{...}, "PE":{...}}
    """
    if not isinstance(legs, list) or not legs:
        return None
    out: dict[float, dict] = {}
    any_leg = False
    for x in legs:
        if not isinstance(x, dict):
            continue
        strike = _strike_from_any(x)
        if strike is None:
            continue
        side = _side_from_any(x)
        if side not in {"CE", "PE"}:
            continue
        any_leg = True
        st = float(strike)
        row = out.get(st) or {"strikePrice": st}
        row[side] = _leg_from_any(x)
        out[st] = row
    if not any_leg:
        return None
    return [out[k] for k in sorted(out.keys())]


def _extract_option_rows_any(payload: object, *, max_nodes: int = 3000) -> list[dict] | None:
    """
    Best-effort extraction of option-chain strike rows as a list[dict] from different vendor shapes.
    Keeps traversal bounded.
    """
    payload = _unwrap_dhanhq(payload)

    # Common direct keys.
    if isinstance(payload, dict):
        # Some vendors keep CE/PE as two lists under separate keys.
        for ck, pk in (
            ("CE", "PE"),
            ("ce", "pe"),
            ("calls", "puts"),
            ("call", "put"),
            ("CALL", "PUT"),
        ):
            c = payload.get(ck)
            p = payload.get(pk)
            if isinstance(c, list) and isinstance(p, list):
                merged = _merge_leg_rows([*c, *p])
                if merged:
                    return merged
        for k in ("data", "Data", "option_data", "optionData"):
            v = payload.get(k)
            if isinstance(v, list):
                # First, see if these are already strike rows.
                rows = [x for x in v if isinstance(x, dict)]
                if rows:
                    # If list contains flat leg rows, merge.
                    merged = _merge_leg_rows(rows)
                    if merged:
                        return merged
                    return rows
            if isinstance(v, dict):
                # Sometimes nested dict holds CE/PE lists.
                merged = _extract_option_rows_any(v, max_nodes=max_nodes)
                if merged:
                    return merged
        # NSE records.data
        rec = payload.get("records")
        if isinstance(rec, dict) and isinstance(rec.get("data"), list):
            rows = [x for x in rec.get("data") if isinstance(x, dict)]
            if rows:
                merged = _merge_leg_rows(rows)
                if merged:
                    return merged
                return rows

    def _looks_like_row(x: object) -> bool:
        if not isinstance(x, dict):
            return False
        strike = x.get("strikePrice") or x.get("strike_price") or x.get("strike") or x.get("StrikePrice")
        if strike is None:
            return False
        # Any leg present
        if _looks_like_leg_dict(x.get("CE")) or _looks_like_leg_dict(x.get("PE")):
            return True
        if _looks_like_leg_dict(x.get("ce")) or _looks_like_leg_dict(x.get("pe")):
            return True
        if _looks_like_leg_dict(x.get("call")) or _looks_like_leg_dict(x.get("put")):
            return True
        # Generic nested dict looks like leg.
        for v in x.values():
            if _looks_like_leg_dict(v):
                return True
        # Flat leg row (strike + oi + side)
        if _side_from_any(x) in {"CE", "PE"} and _looks_like_leg_dict(x):
            return True
        return False

    # Bounded BFS/DFS for nested payloads.
    seen = 0
    stack = [payload]
    while stack and seen < int(max_nodes):
        cur = stack.pop()
        seen += 1
        cur = _unwrap_dhanhq(cur)
        if isinstance(cur, list):
            sample = cur[:25]
            if sample:
                # If list already looks like strike rows (or flat leg rows), return it/merge it.
                if any(_looks_like_row(x) for x in sample):
                    rows = [x for x in cur if isinstance(x, dict)]
                    if rows:
                        merged = _merge_leg_rows(rows)
                        if merged:
                            return merged
                        return rows
                # Otherwise, still try merging (some feeds don't have clear CE/PE nesting).
                rows = [x for x in cur if isinstance(x, dict)]
                merged = _merge_leg_rows(rows)
                if merged:
                    return merged
            for x in sample:
                stack.append(x)
            continue
        if isinstance(cur, dict):
            for v in list(cur.values())[:80]:
                stack.append(v)
            continue
    return None


def summarize_oi_snapshot_any(payload: dict, *, spot_price: float, strikes_each_side: int = 5, top_n_walls: int = 5) -> dict:
    """
    Live-simulation friendly OI snapshot:
    - Always tries to compact to (2N+1) strikes around spot.
    - Also includes ce_walls/pe_walls computed from the window so downstream
      code (derive_oi_features) keeps working without special-casing.
    """
    win = summarize_oi_window_any(payload, spot_price=float(spot_price), strikes_each_side=int(strikes_each_side))
    if not isinstance(win, dict) or not win.get("ok"):
        # Fallback to global walls if we couldn't build a window.
        return summarize_oi_walls_any(payload, top_n=int(top_n_walls))

    ce: list[OIWall] = []
    pe: list[OIWall] = []
    for row in win.get("window") or []:
        if not isinstance(row, dict):
            continue
        strike = _to_float(row.get("strike"))
        if strike is None:
            continue
        ce_leg = row.get("CE") if isinstance(row.get("CE"), dict) else None
        pe_leg = row.get("PE") if isinstance(row.get("PE"), dict) else None
        if ce_leg is not None:
            ce.append(
                OIWall(
                    strike=float(strike),
                    side="CE",
                    oi=float(_to_float(ce_leg.get("oi")) or 0.0),
                    oi_change=_to_float(ce_leg.get("oi_change")),
                    ltp=_to_float(ce_leg.get("ltp")),
                )
            )
        if pe_leg is not None:
            pe.append(
                OIWall(
                    strike=float(strike),
                    side="PE",
                    oi=float(_to_float(pe_leg.get("oi")) or 0.0),
                    oi_change=_to_float(pe_leg.get("oi_change")),
                    ltp=_to_float(pe_leg.get("ltp")),
                )
            )

    ce_sorted = sorted(ce, key=lambda x: x.oi, reverse=True)[: int(top_n_walls)]
    pe_sorted = sorted(pe, key=lambda x: x.oi, reverse=True)[: int(top_n_walls)]

    return {
        "ok": True,
        "mode": "window",
        "spot": win.get("spot"),
        "atm_strike": win.get("atm_strike"),
        "window": win.get("window"),
        "ce_walls": [w.__dict__ for w in ce_sorted],
        "pe_walls": [w.__dict__ for w in pe_sorted],
    }


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
    rows = _extract_option_rows_any(option_chain_resp)
    if not isinstance(rows, list):
        return {"ok": False, "error": "unexpected option_chain format", "raw_keys": list(option_chain_resp.keys())[:20] if isinstance(option_chain_resp, dict) else []}

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


def summarize_oi_walls_any(payload: dict, top_n: int = 5) -> dict:
    """
    Accepts:
    - Our compact wall summary format: {"ok":true,"ce_walls":[...],"pe_walls":[...]}
    - dhanhq wrapper responses: {"status":"success|failure","remarks":...,"data":<payload>}
    - Dhan option chain format (handled by summarize_oi_walls)
    - NSE option chain JSON (common keys: records.data[*].CE/PE)
    Returns our compact summary format.
    """
    if isinstance(payload, dict) and payload.get("ok") is True and isinstance(payload.get("ce_walls"), list) and isinstance(payload.get("pe_walls"), list):
        return payload

    # dhanhq wrapper: unwrap recursively
    try:
        inner = _unwrap_dhanhq(payload)
        if inner is not payload and isinstance(inner, dict):
            return summarize_oi_walls_any(inner, top_n=top_n)
        if isinstance(inner, list):
            return summarize_oi_walls({"data": inner}, top_n=top_n)
    except Exception:
        pass

    # NSE option chain API style: {"records":{"data":[{"strikePrice":...,"CE":{...},"PE":{...}}, ...]}}
    try:
        rec = payload.get("records") if isinstance(payload, dict) else None
        data = None
        if isinstance(rec, dict):
            data = rec.get("data")
        if isinstance(data, list):
            return summarize_oi_walls({"data": data}, top_n=top_n)
    except Exception:
        pass

    # Dhan / other nested shapes: locate the strike rows list anywhere in the payload.
    try:
        rows = _extract_option_rows_any(payload)
        if isinstance(rows, list) and rows:
            return summarize_oi_walls({"data": rows}, top_n=top_n)
    except Exception:
        pass

    return summarize_oi_walls(payload, top_n=top_n)


def summarize_oi_window_any(payload: dict, *, spot_price: float, strikes_each_side: int = 5) -> dict:
    """
    Compact snapshot around spot:
      - ATM strike
      - N strikes below + N strikes above (plus ATM) => total 2N+1
    Output shape is intentionally small for LLM prompts.
    """
    rows = _extract_option_rows_any(payload)
    if not isinstance(rows, list) or not rows:
        return {"ok": False, "error": "unexpected option_chain format"}

    # Map strike -> row
    strike_to_row: dict[float, dict] = {}
    strikes: list[float] = []
    for r in rows:
        strike = _to_float(r.get("strikePrice") or r.get("strike_price") or r.get("strike") or r.get("StrikePrice"))
        if strike is None:
            continue
        strike = float(strike)
        strike_to_row[strike] = r
        strikes.append(strike)
    strikes = sorted(set(strikes))
    if not strikes:
        return {"ok": False, "error": "no_strikes"}

    sp = float(spot_price)
    atm = min(strikes, key=lambda x: abs(x - sp))
    i = strikes.index(atm)
    lo = max(0, i - int(strikes_each_side))
    hi = min(len(strikes), i + int(strikes_each_side) + 1)
    sel = strikes[lo:hi]

    def _leg(d: object) -> dict | None:
        if not isinstance(d, dict):
            return None
        return {
            "oi": _to_float(d.get("oi") or d.get("openInterest") or d.get("open_interest") or d.get("OI")),
            "oi_change": _to_float(d.get("oiChange") or d.get("changeinOpenInterest") or d.get("change_in_oi") or d.get("oi_change")),
            "ltp": _to_float(d.get("ltp") or d.get("lastPrice") or d.get("LTP")),
            "volume": _to_float(d.get("volume") or d.get("Volume")),
        }

    window = []
    for s in sel:
        r = strike_to_row.get(s) or {}
        ce_leg = r.get("CE") or r.get("ce") or r.get("call") or r.get("CALL") or r.get("Call") or {}
        pe_leg = r.get("PE") or r.get("pe") or r.get("put") or r.get("PUT") or r.get("Put") or {}
        window.append({"strike": float(s), "CE": _leg(ce_leg), "PE": _leg(pe_leg)})

    return {"ok": True, "spot": sp, "atm_strike": float(atm), "window": window}


def summarize_oi_walls_plaintext(text: str, top_n: int = 5) -> dict:
    """
    Parses NSE option-chain table copied as plain text (tab-separated) and converts it into our compact OI walls summary.

    Expected columns (typical NSE):
    CALL: OI, Chng in OI, Volume, IV, LTP, Chng, BID QTY, BID, ASK, ASK QTY,
    MID: Strike,
    PUT: BID QTY, BID, ASK, ASK QTY, Chng, LTP, IV, Volume, Chng in OI, OI

    We only need: strike, CE OI (+ change), PE OI (+ change), and optional LTP.
    """
    raw = (text or "").strip()
    if not raw:
        return {"ok": False, "error": "empty plaintext"}

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {"ok": False, "error": "not enough lines"}

    header = lines[0]
    # Detect delimiter (NSE copy is usually tabs).
    delim = "\t" if "\t" in header else None

    def split_line(ln: str) -> list[str]:
        if delim:
            return [x.strip() for x in ln.split("\t")]
        # fallback: split on 2+ spaces
        return [x.strip() for x in re.split(r"\s{2,}", ln.strip()) if x.strip()]

    head_cols = split_line(header)
    strike_idx = None
    for i, c in enumerate(head_cols):
        if c.lower() == "strike":
            strike_idx = i
            break
    # Fallback: NSE sometimes has "Strike" in the middle; assume 10 if unknown.
    if strike_idx is None:
        strike_idx = 10

    ce: list[OIWall] = []
    pe: list[OIWall] = []

    for ln in lines[1:]:
        cols = split_line(ln)
        # Skip clearly malformed lines
        if len(cols) < (strike_idx + 1):
            continue
        strike = _to_float_india(cols[strike_idx])
        if strike is None:
            continue

        # CALL side indices (by convention)
        ce_oi = _to_float_india(cols[0]) if len(cols) > 0 else None
        ce_oi_chg = _to_float_india(cols[1]) if len(cols) > 1 else None
        ce_ltp = _to_float_india(cols[4]) if len(cols) > 4 else None

        # PUT side indices relative to strike column
        # After strike: BID QTY, BID, ASK, ASK QTY, Chng, LTP, IV, Volume, Chng in OI, OI
        pe_ltp = _to_float_india(cols[strike_idx + 6]) if len(cols) > (strike_idx + 6) else None
        pe_oi_chg = _to_float_india(cols[strike_idx + 9]) if len(cols) > (strike_idx + 9) else None
        pe_oi = _to_float_india(cols[strike_idx + 10]) if len(cols) > (strike_idx + 10) else None
        # Fallback from the end (if there are extra spaces/columns, last 2 should still be chg_oi, oi)
        if pe_oi is None and len(cols) >= 2:
            pe_oi = _to_float_india(cols[-1])
        if pe_oi_chg is None and len(cols) >= 2:
            pe_oi_chg = _to_float_india(cols[-2])

        if ce_oi is not None:
            ce.append(OIWall(strike=float(strike), side="CE", oi=float(ce_oi), oi_change=ce_oi_chg, ltp=ce_ltp))
        if pe_oi is not None:
            pe.append(OIWall(strike=float(strike), side="PE", oi=float(pe_oi), oi_change=pe_oi_chg, ltp=pe_ltp))

    if not ce and not pe:
        return {"ok": False, "error": "no strike rows parsed from plaintext"}

    ce_sorted = sorted(ce, key=lambda x: x.oi, reverse=True)[: int(top_n)]
    pe_sorted = sorted(pe, key=lambda x: x.oi, reverse=True)[: int(top_n)]
    return {"ok": True, "ce_walls": [w.__dict__ for w in ce_sorted], "pe_walls": [w.__dict__ for w in pe_sorted]}


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
