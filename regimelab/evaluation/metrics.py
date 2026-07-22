"""Small, dependency-free performance metrics used by the benchmark."""
from __future__ import annotations

import numpy as np

__all__ = ["sharpe", "max_drawdown", "hit_rate", "turnover", "annualize_factor"]


def sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    return float(r.mean() / sd) if sd > 0 else 0.0


def max_drawdown(returns: np.ndarray) -> float:
    """Worst peak-to-trough of the cumulative (sum) equity curve, as a fraction."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq - peak))


def hit_rate(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    nz = r[r != 0]
    return float((nz > 0).mean()) if len(nz) else 0.0


def turnover(positions: np.ndarray) -> float:
    """Average |position change| per bar — proxy for trading cost exposure."""
    p = np.asarray(positions, dtype=float)
    if len(p) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(p))))


def annualize_factor(bar_seconds: int = 300) -> float:
    """sqrt(bars per year) to scale a per-bar Sharpe to annual, if desired."""
    bars_per_year = (365 * 24 * 3600) / bar_seconds
    return float(np.sqrt(bars_per_year))
