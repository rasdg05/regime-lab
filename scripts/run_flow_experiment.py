#!/usr/bin/env python3
"""Bonus experiment: does order flow anticipate regime transitions?

    python scripts/run_flow_experiment.py --symbol BTCUSDT --days 30

Downloads klines with taker order-flow columns, defines a regime transition as a
flip of the irreversibility label, and tests whether flow features predict a
transition in the next `horizon` bars — vs a price-only control. Purged
walk-forward, pooled OOS AUC. No API key.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regimelab.data import load_binance_vision
from regimelab.flow_experiment import run_flow_experiment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--n-splits", type=int, default=6)
    ap.add_argument("--embargo", type=int, default=48)
    args = ap.parse_args()

    end = date.today() - timedelta(days=2)
    start = end - timedelta(days=args.days)
    s = load_binance_vision(args.symbol, start, end, "5m", with_flow=True)
    print(f"{args.symbol} 5m: {len(s)} bars {start}..{end}")

    res = run_flow_experiment(s, window=args.window, horizon=args.horizon,
                              n_splits=args.n_splits, embargo=args.embargo,
                              min_train=max(2000, len(s) // (args.n_splits + 2)))
    print(f"\nregime-transition rate = {res['transition_rate']:.3f}  "
          f"(a transition within {args.horizon} bars)  |  folds = {res['n_folds']}\n")
    print(f"OOS AUC  price only        = {res['auc_price_only']:.4f}")
    print(f"OOS AUC  price + flow      = {res['auc_price_plus_flow']:.4f}  "
          f"(uplift {res['auc_uplift']:+.4f})")
    print(f"OOS AUC  flow only         = {res['auc_flow_only']:.4f}")


if __name__ == "__main__":
    main()
