"""Common interface every regime detector implements.

A detector maps a 1-D array of closes to a per-bar regime label in {-1, 0, 1}:
    -1  warming up (not enough history yet)
     0  "range" / mean-reverting / low-directionality regime
     1  "trend" / directional regime

Keeping the contract this small is deliberate: the benchmark harness
(``regimelab.evaluation.benchmark``) treats every detector — the visibility-graph
one and the baselines — through the exact same interface, so the comparison is
apples-to-apples. Detectors may optionally expose ``fit`` (to learn a split
point on training data only) and ``score`` (a continuous statistic), but only
``label`` and ``name`` are required.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RegimeDetector(Protocol):
    name: str

    def label(self, closes: np.ndarray) -> np.ndarray:
        """Return an int array of regime labels, same length as ``closes``."""
        ...


def fit_if_supported(detector, closes: np.ndarray):
    """Call ``detector.fit(closes)`` when present; return the detector.

    Used by the walk-forward harness so any threshold a detector needs is
    learned on the *training* fold only — never on the test fold.
    """
    fit = getattr(detector, "fit", None)
    if callable(fit):
        fit(closes)
    return detector
