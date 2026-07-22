"""Deflated Sharpe Ratio and Probability of Backtest Overfitting.

Both come from Bailey & López de Prado. The point they make — and the reason
this repo puts them front and centre — is that if you try N strategy
configurations and keep the best Sharpe, that best Sharpe is inflated by
multiple testing, even if none of the configs has any real edge. The DSR
discounts a Sharpe for (a) how many trials you ran, (b) the non-normality
(skew/kurtosis) of the returns, and (c) the sample length. A DSR >= 0.95 is the
usual "this survived an honest multiple-testing correction" bar.

References
----------
* Bailey, López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality", J. Portfolio
  Management 40(5):94-107.
* Bailey, Borwein, López de Prado, Zhu (2017), "The Probability of Backtest
  Overfitting", J. Computational Finance 20(4):39-69.
"""
from __future__ import annotations

from math import erf, sqrt

import numpy as np

__all__ = [
    "sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "probability_backtest_overfitting",
]

_EULER = 0.5772156649015329


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _phi_inv(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = sqrt(-2 * np.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def sharpe_ratio(returns: np.ndarray) -> float:
    """Per-period Sharpe (mean/std, ddof=1). Not annualized — keep it in the
    same units the DSR math expects."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    return float(r.mean() / sd) if sd > 0 else 0.0


def expected_max_sharpe(n_trials: int, trials_sr_var: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent strategies whose
    Sharpes have variance ``trials_sr_var`` (the multiple-testing benchmark)."""
    n = max(int(n_trials), 1)
    if n == 1 or trials_sr_var <= 0:
        return 0.0
    v = sqrt(trials_sr_var)
    return v * ((1 - _EULER) * _phi_inv(1 - 1.0 / n) +
                _EULER * _phi_inv(1 - 1.0 / (n * np.e)))


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int,
                          trials_sr_var: float | None = None) -> dict:
    """Deflated Sharpe Ratio in [0, 1].

    ``n_trials`` is how many configurations were tried; ``trials_sr_var`` is the
    variance of those trials' Sharpes (if unknown we fall back to a unit-variance
    proxy, which is conservative). Returns a dict with the raw Sharpe, the
    selection benchmark ``sr0``, and the deflated probability ``dsr``.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 4:
        return {"sr": 0.0, "T": T, "skew": 0.0, "kurt": 3.0, "sr0": 0.0, "dsr": 0.0}
    sr = sharpe_ratio(r)
    sd = float(r.std(ddof=1))
    z = (r - r.mean()) / (sd if sd > 0 else 1.0)
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))               # non-excess kurtosis
    v = trials_sr_var if trials_sr_var is not None else 1.0
    sr0 = expected_max_sharpe(n_trials, v)
    denom = sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    z_stat = (sr - sr0) * sqrt(T - 1) / denom
    dsr = float(min(max(_phi(z_stat), 0.0), 1.0))
    return {"sr": sr, "T": T, "skew": skew, "kurt": kurt, "sr0": float(sr0), "dsr": dsr}


def probability_backtest_overfitting(config_returns: np.ndarray,
                                     n_splits: int = 8) -> float:
    """PBO via combinatorially-symmetric cross-validation (CSCV).

    ``config_returns`` is a (T, N) matrix: T periods, N candidate configs. PBO is
    the fraction of CSCV splits where the config that was best in-sample ranks
    below median out-of-sample — i.e. the probability your "winner" is overfit.
    """
    from itertools import combinations

    M = np.asarray(config_returns, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return float("nan")
    T, N = M.shape
    s = n_splits - (n_splits % 2)               # must be even
    if s < 2:
        s = 2
    rows = np.array_split(np.arange(T), s)
    blocks = [M[idx] for idx in rows]
    half = s // 2
    logits = []
    for combo in combinations(range(s), half):
        is_idx = list(combo)
        oos_idx = [i for i in range(s) if i not in combo]
        is_ret = np.vstack([blocks[i] for i in is_idx])
        oos_ret = np.vstack([blocks[i] for i in oos_idx])
        is_sr = np.array([sharpe_ratio(is_ret[:, c]) for c in range(N)])
        oos_sr = np.array([sharpe_ratio(oos_ret[:, c]) for c in range(N)])
        best = int(np.argmax(is_sr))
        # rank (0..1) of the in-sample winner out-of-sample
        rank = float((oos_sr < oos_sr[best]).sum()) / (N - 1)
        w = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
    logits = np.asarray(logits)
    return float(np.mean(logits < 0))           # P(winner ranks below median OOS)
