"""Time-irreversibility regime detector via the Horizontal Visibility Graph (HVG).

The idea, in one paragraph
--------------------------
Some markets look roughly the same played forwards or backwards in time — they
are *time-reversible* (a driftless Gaussian random walk is the textbook case).
Others do not: the way price rises differs from the way it falls — the classic
"stairs up, elevator down" asymmetry — so the series carries a temporal arrow.
We quantify that arrow without any tunable parameters by mapping the price
window to a **directed Horizontal Visibility Graph** and measuring how different
the in-degree and out-degree distributions are (Kullback–Leibler divergence).
Low divergence -> reversible / efficient regime; high divergence -> irreversible
/ asymmetric-structure regime.

Important nuance (measured, not assumed): irreversibility is NOT a trend
detector. A clean linear ramp is perfectly *reversible* (irreversibility ~ 0);
what lights the statistic up is temporal *asymmetry* between up- and down-moves.
That is precisely why it is worth benchmarking separately from realized-vol or
slope baselines — it sees a different thing.

Why this is a good detector to open-source
------------------------------------------
* Parameter-free: no lookback-specific thresholds baked into the statistic
  itself (you still choose the *window length*, but the KL value is intrinsic).
* Grounded in published work rather than folklore:
    - Lacasa, Luque, Ballesteros, Luque, Nuño (2008), "From time series to
      complex networks: the visibility graph", PNAS 105(13):4972.
    - Lacasa, Nuñez, Roldán, Parrondo, Luque (2012), "Time series
      irreversibility: a visibility graph approach", Eur. Phys. J. B 85:217.
* Cheap: O(n * avg_visible) per window; a 64-bar window is sub-millisecond.

This module is intentionally free of any trading-strategy secret sauce. It only
turns a price window into a scalar (irreversibility) and a coarse regime label.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["hvg_degrees", "irreversibility", "IrreversibilityDetector"]


def hvg_degrees(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In/out degree of the directed Horizontal Visibility Graph of ``y``.

    Two samples i < j are connected iff every sample strictly between them is
    lower than ``min(y_i, y_j)`` (the horizontal-visibility criterion). The edge
    is *directed* i -> j (time's arrow), so i contributes to ``outdeg`` and j to
    ``indeg``. Returns ``(indeg, outdeg)``, each length ``len(y)``.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    indeg = np.zeros(n, dtype=float)
    outdeg = np.zeros(n, dtype=float)
    for i in range(n):
        mid_max = -np.inf
        yi = y[i]
        for j in range(i + 1, n):
            yj = y[j]
            # visible iff no intermediate bar reaches min(yi, yj)
            if mid_max < (yi if yi < yj else yj):
                outdeg[i] += 1.0
                indeg[j] += 1.0
            if yj > mid_max:
                mid_max = yj
            if yj >= yi:            # yi can no longer see anything further right
                break
    return indeg, outdeg


def irreversibility(y: np.ndarray) -> float:
    """KL(P_in || P_out) of the HVG degree distributions.

    0.0 means the in- and out-degree distributions coincide (time-symmetric /
    reversible). Larger values mean a stronger temporal arrow (trending). The
    statistic has no free parameters; the only choice is the length of ``y``.
    Returns 0.0 for degenerate windows (constant series, < 8 points).
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 8:
        return 0.0
    indeg, outdeg = hvg_degrees(y)
    kmax = int(max(indeg.max(), outdeg.max()))
    if kmax < 1:
        return 0.0
    bins = np.arange(0, kmax + 2)
    pin, _ = np.histogram(indeg, bins=bins)
    pout, _ = np.histogram(outdeg, bins=bins)
    pin = pin.astype(float) + 1e-9          # Laplace smoothing keeps KL finite
    pout = pout.astype(float) + 1e-9
    pin /= pin.sum()
    pout /= pout.sum()
    return float(np.sum(pin * np.log(pin / pout)))


@dataclass
class IrreversibilityDetector:
    """Rolling time-irreversibility regime labeller.

    Parameters
    ----------
    window : int
        Number of closes in each causal window (default 64).
    threshold : float | None
        Split point between "reversible" (<= threshold) and "irreversible"
        (> threshold). If ``None`` it is fit as the median of the training
        irreversibility series (``fit``), which keeps the class label-balanced
        and avoids hand-tuning. The *statistic* is parameter-free; only this cut
        is learned, on training data only.
    """

    window: int = 64
    threshold: float | None = None

    name: str = "irreversibility"

    def score(self, closes: np.ndarray) -> np.ndarray:
        """Causal rolling irreversibility. ``score[t]`` uses closes (t-window, t]
        only, so there is no look-ahead. Positions before a full window are NaN.
        """
        closes = np.asarray(closes, dtype=float)
        n = len(closes)
        out = np.full(n, np.nan, dtype=float)
        for t in range(self.window, n + 1):
            out[t - 1] = irreversibility(closes[t - self.window:t])
        return out

    def fit(self, closes: np.ndarray) -> "IrreversibilityDetector":
        """Learn ``threshold`` as the median of the (finite) training scores."""
        s = self.score(closes)
        s = s[np.isfinite(s)]
        if len(s):
            self.threshold = float(np.median(s))
        return self

    def label(self, closes: np.ndarray) -> np.ndarray:
        """Regime label per bar: 0 = reversible/efficient, 1 = irreversible/
        asymmetric, -1 = warming up (window not yet full). Uses ``threshold``
        (fit first, or pass one in)."""
        thr = self.threshold if self.threshold is not None else 0.34
        s = self.score(closes)
        lab = np.where(np.isfinite(s), (s > thr).astype(int), -1)
        return lab
