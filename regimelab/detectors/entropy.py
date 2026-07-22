"""Permutation-entropy regime detector.

Permutation entropy (Bandt & Pompe 2002) measures the *complexity* of the
ordinal patterns in a series. A clean trend produces very few distinct ordinal
patterns (mostly monotone), so entropy is low; choppy, range-bound price action
visits many patterns, so entropy is high. That makes normalized permutation
entropy a natural, cheap, parameter-light complement to the visibility-graph
irreversibility statistic — and a useful *second* detector to benchmark against.

Reference
---------
Bandt, Pompe (2002), "Permutation entropy: a natural complexity measure for
time series", Phys. Rev. Lett. 88(17):174102.

Note the label convention matches the rest of the package: 1 = trend
(low entropy / high order), 0 = range (high entropy). We invert the raw
entropy so that, like irreversibility, a *higher* detector score means "more
trend".
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import log

import numpy as np

__all__ = ["permutation_entropy", "PermutationEntropyDetector"]


def permutation_entropy(y: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Normalized permutation entropy of ``y`` in [0, 1].

    ``order`` is the embedding dimension (pattern length); ``delay`` the time
    gap between points in a pattern. 0 = perfectly ordered (single pattern),
    1 = all ``order!`` patterns equally likely (maximally complex).
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    m = n - (order - 1) * delay
    if m < 2:
        return 0.0
    # count ordinal patterns
    perm_index = {p: k for k, p in enumerate(permutations(range(order)))}
    counts = np.zeros(len(perm_index), dtype=float)
    for i in range(m):
        window = y[i:i + order * delay:delay]
        pattern = tuple(np.argsort(window, kind="mergesort"))
        counts[perm_index[pattern]] += 1.0
    p = counts[counts > 0]
    p /= p.sum()
    h = -np.sum(p * np.log(p))
    hmax = log(len(perm_index))
    return float(h / hmax) if hmax > 0 else 0.0


@dataclass
class PermutationEntropyDetector:
    """Rolling regime labeller: low permutation entropy -> trend (label 1)."""

    window: int = 64
    order: int = 3
    delay: int = 1
    threshold: float | None = None      # on the *trend score* = 1 - entropy
    name: str = "permutation_entropy"

    def score(self, closes: np.ndarray) -> np.ndarray:
        """Causal rolling trend score = 1 - normalized permutation entropy."""
        closes = np.asarray(closes, dtype=float)
        n = len(closes)
        out = np.full(n, np.nan, dtype=float)
        for t in range(self.window, n + 1):
            h = permutation_entropy(closes[t - self.window:t], self.order, self.delay)
            out[t - 1] = 1.0 - h
        return out

    def fit(self, closes: np.ndarray) -> "PermutationEntropyDetector":
        s = self.score(closes)
        s = s[np.isfinite(s)]
        if len(s):
            self.threshold = float(np.median(s))
        return self

    def label(self, closes: np.ndarray) -> np.ndarray:
        thr = self.threshold if self.threshold is not None else 0.5
        s = self.score(closes)
        return np.where(np.isfinite(s), (s > thr).astype(int), -1)
