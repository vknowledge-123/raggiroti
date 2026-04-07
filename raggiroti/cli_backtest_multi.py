from __future__ import annotations

import argparse

from raggiroti.backtest.csv_loader import load_candles
from raggiroti.backtest.day_split import group_by_date
from raggiroti.backtest.engine import run_backtest
from raggiroti.backtest.prev_day_planner import classify_open_scenario, compute_prev_day_levels
from raggiroti.backtest.scenario_policy import ScenarioPolicy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Multi-day 1m CSV")
    ap.add_argument("--qty", type=int, default=65)
    ap.add_argument("--gap", type=float, default=30.0, help="Gap threshold points")
    ap.add_argument("--flat", type=float, default=15.0, help="Flat threshold points")
    ap.add_argument("--start", default=None, help="Warmup start date (YYYY-MM-DD). Trades start the next day.")
    ap.add_argument("--end", default=None, help="End date inclusive (YYYY-MM-DD)")
    args = ap.parse_args()

    candles = load_candles(args.csv)
    by_date = group_by_date(candles)
    dates = sorted(by_date.keys())
    if len(dates) < 2:
        raise SystemExit("Need at least 2 days in CSV (prev day used for levels).")

    if args.start is not None and args.start not in dates:
        raise SystemExit(f"--start not found in CSV: {args.start}")
    if args.end is not None and args.end not in dates:
        raise SystemExit(f"--end not found in CSV: {args.end}")

    # Warmup day: if start provided, first traded day is the day after start.
    start_idx = 1
    if args.start is not None:
        start_idx = dates.index(args.start) + 1
    end_idx = len(dates) - 1
    if args.end is not None:
        end_idx = dates.index(args.end)

    if start_idx > end_idx:
        raise SystemExit("Invalid range: trades start after end date.")

    total = 0.0
    for i in range(start_idx, end_idx + 1):
        prev_date = dates[i - 1]
        date = dates[i]
        prev_levels = compute_prev_day_levels(by_date[prev_date])
        day = sorted(by_date[date], key=lambda c: c.dt)
        scenario = classify_open_scenario(
            day[0].open,
            prev_levels.close,
            gap_up_threshold_points=args.gap,
            gap_down_threshold_points=args.gap,
            flat_threshold_points=args.flat,
        )
        policy = ScenarioPolicy(prev=prev_levels, scenario=scenario)
        res = run_backtest(day, policy=policy, qty=args.qty, prev=prev_levels)
        print(f"{date} scenario={scenario} pnl(points*qty)={res.realized_pnl_points:.2f}")
        total += res.realized_pnl_points

    print(f"TOTAL pnl(points*qty)={total:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
