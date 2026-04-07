from __future__ import annotations

import argparse

from raggiroti.backtest.csv_loader import load_candles
from raggiroti.backtest.engine import run_backtest
from raggiroti.backtest.rulebook_scoring_policy import RulebookScoringPolicy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to 1m CSV")
    ap.add_argument("--qty", type=int, default=65)
    args = ap.parse_args()

    candles = load_candles(args.csv)
    policy = RulebookScoringPolicy()
    result = run_backtest(candles, policy=policy, qty=args.qty)

    print(f"Realized PnL (points*qty): {result.realized_pnl_points:.2f}")
    for f in result.fills[:20]:
        print(f"{f.dt} {f.side} {f.price} x{f.qty} ({f.reason})")
    if len(result.fills) > 20:
        print(f"... {len(result.fills)-20} more fills")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
