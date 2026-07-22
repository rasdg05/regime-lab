#!/usr/bin/env python3
"""Reproduce the benchmark table on public Binance Vision data.

    python scripts/run_benchmark.py --symbol BTCUSDT --days 45 --rule reversion

Downloads 5-minute klines (no API key), runs every detector's regime-gated
strategy through purged walk-forward, and prints Sharpe / Deflated Sharpe /
max-drawdown per strategy plus the Probability of Backtest Overfitting across
the candidate set. Everything is causal and leak-checked.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

# make the repo importable when run as `python scripts/run_benchmark.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from regimelab import (
    IrreversibilityDetector, PermutationEntropyDetector,
    RealizedVolDetector, TrendSlopeDetector, run_benchmark,
)
from regimelab.data import load_binance_vision, synthetic_regimes
from regimelab.evaluation.deflated_sharpe import probability_backtest_overfitting


def build_detectors(window: int) -> dict:
    dets = {
        "irreversibility": IrreversibilityDetector(window),
        "perm_entropy": PermutationEntropyDetector(window),
        "realized_vol": RealizedVolDetector(window),
        "trend_slope": TrendSlopeDetector(window),
    }
    try:
        from regimelab import HMMDetector
        dets["hmm_2state"] = HMMDetector()
    except Exception:
        pass                    # hmmlearn optional
    return dets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--rule", choices=["reversion", "momentum"], default="reversion")
    ap.add_argument("--n-splits", type=int, default=6)
    ap.add_argument("--embargo", type=int, default=48)
    ap.add_argument("--cost", type=float, default=0.0002)
    ap.add_argument("--synthetic", action="store_true",
                    help="use the synthetic regime generator instead of real data")
    args = ap.parse_args()

    if args.synthetic:
        s = synthetic_regimes(n_blocks=40, block_len=300, seed=1)
        print(f"synthetic regimes: {len(s)} bars")
    else:
        end = date.today() - timedelta(days=2)
        start = end - timedelta(days=args.days)
        s = load_binance_vision(args.symbol, start, end, "5m")
        print(f"{args.symbol} 5m: {len(s)} bars (~{len(s)/288:.0f} days) "
              f"{start}..{end}")

    dets = build_detectors(args.window)
    min_train = max(2000, args.window * 30)
    res = run_benchmark(s.close, dets, rule=args.rule, n_splits=args.n_splits,
                        embargo=args.embargo, min_train=min_train, cost=args.cost)

    print(f"\nbase rule = {args.rule} | cost = {args.cost} | "
          f"walk-forward folds = {args.n_splits}, embargo = {args.embargo}\n")
    print(f"{'strategy':18} {'sharpe':>9} {'DSR':>7} {'maxDD':>9} {'turnover':>9}")
    print("-" * 56)
    for k, r in sorted(res.items(), key=lambda kv: -kv[1].sharpe):
        print(f"{k:18} {r.sharpe:9.4f} {r.dsr:7.3f} {r.max_drawdown:9.4f} "
              f"{r.turnover:9.3f}")

    # PBO across the *competing* strategies. Exclude the degenerate do-nothing
    # floor (a zero-return column would trivially always "win" and force PBO->0).
    names = [k for k in res if len(res[k].oos_returns) > 0 and k != "always_flat"]
    if len(names) >= 2:
        m = min(len(res[k].oos_returns) for k in names)
        M = np.column_stack([res[k].oos_returns[-m:] for k in names])
        pbo = probability_backtest_overfitting(M, n_splits=8)
        print(f"\nPBO (prob. the in-sample winner is overfit) = {pbo:.3f}")


if __name__ == "__main__":
    main()
