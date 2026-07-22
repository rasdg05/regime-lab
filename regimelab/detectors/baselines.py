"""Baseline regime detectors — the honest comparison points.

A regime detector is only interesting if it beats the obvious alternatives. We
include three:

* ``RealizedVolDetector`` — the "vol regime" folklore: high realized volatility
  labelled one way, low the other. Cheap, widely used, hard to beat on some
  assets.
* ``TrendSlopeDetector`` — a moving-average slope rule. If the MA is rising/
  falling steeply it's a "trend"; flat is "range". This is what a discretionary
  trader eyeballs.
* ``HMMDetector`` — a 2-state Gaussian Hidden Markov Model on returns, the
  textbook statistical regime model. Optional dependency (``hmmlearn``); if it
  is not installed the detector raises a clear error only when instantiated, so
  the rest of the package works without it.

All three implement the same ``label`` (and ``fit``) contract as the
visibility-graph detector, so the benchmark compares them head-to-head.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RealizedVolDetector", "TrendSlopeDetector", "HMMDetector"]


def _log_returns(closes: np.ndarray) -> np.ndarray:
    closes = np.asarray(closes, dtype=float)
    r = np.zeros_like(closes)
    r[1:] = np.diff(np.log(np.clip(closes, 1e-12, None)))
    return r


@dataclass
class RealizedVolDetector:
    """High realized volatility -> label 1 ("trend"/active), low -> 0.

    Note: whether "high vol == trend" holds is an empirical question — that's
    exactly why it's a baseline. On many crypto series high vol coincides with
    trends; on others it coincides with chop. Letting the data decide via the
    fitted median threshold keeps it a fair opponent.
    """

    window: int = 64
    threshold: float | None = None
    name: str = "realized_vol"

    def score(self, closes: np.ndarray) -> np.ndarray:
        r = _log_returns(closes)
        n = len(r)
        out = np.full(n, np.nan, dtype=float)
        for t in range(self.window, n + 1):
            out[t - 1] = float(np.std(r[t - self.window:t], ddof=1))
        return out

    def fit(self, closes: np.ndarray) -> "RealizedVolDetector":
        s = self.score(closes)
        s = s[np.isfinite(s)]
        if len(s):
            self.threshold = float(np.median(s))
        return self

    def label(self, closes: np.ndarray) -> np.ndarray:
        thr = self.threshold if self.threshold is not None else 0.0
        s = self.score(closes)
        return np.where(np.isfinite(s), (s > thr).astype(int), -1)


@dataclass
class TrendSlopeDetector:
    """|slope of a moving average|, normalized by realized vol -> trend score.

    Steep MA (relative to noise) = trend (label 1); flat = range (0). The
    normalization makes the score scale-free across assets.
    """

    window: int = 64
    threshold: float | None = None
    name: str = "trend_slope"

    def score(self, closes: np.ndarray) -> np.ndarray:
        closes = np.asarray(closes, dtype=float)
        n = len(closes)
        out = np.full(n, np.nan, dtype=float)
        x = np.arange(self.window, dtype=float)
        xc = x - x.mean()
        denom = float(np.sum(xc * xc))
        for t in range(self.window, n + 1):
            w = closes[t - self.window:t]
            slope = float(np.sum(xc * (w - w.mean())) / denom)   # OLS slope
            vol = float(np.std(np.diff(w), ddof=1)) or 1e-12
            out[t - 1] = abs(slope) / vol
        return out

    def fit(self, closes: np.ndarray) -> "TrendSlopeDetector":
        s = self.score(closes)
        s = s[np.isfinite(s)]
        if len(s):
            self.threshold = float(np.median(s))
        return self

    def label(self, closes: np.ndarray) -> np.ndarray:
        thr = self.threshold if self.threshold is not None else 0.0
        s = self.score(closes)
        return np.where(np.isfinite(s), (s > thr).astype(int), -1)


@dataclass
class HMMDetector:
    """2-state Gaussian HMM on log-returns (textbook statistical regime model).

    The state with the larger absolute mean return is mapped to "trend" (label
    1), the other to "range" (0), so the label semantics match the other
    detectors regardless of which hidden state the fit happens to number first.
    Requires ``hmmlearn``; instantiating without it raises ImportError.
    """

    n_iter: int = 100
    random_state: int = 0
    name: str = "hmm_2state"
    _model: object = None
    _trend_state: int = 0

    def __post_init__(self):
        try:
            import hmmlearn  # noqa: F401
        except Exception as e:   # pragma: no cover - only when dep missing
            raise ImportError(
                "HMMDetector needs `hmmlearn` (pip install hmmlearn)"
            ) from e

    def fit(self, closes: np.ndarray) -> "HMMDetector":
        from hmmlearn.hmm import GaussianHMM
        r = _log_returns(closes)[1:].reshape(-1, 1)
        model = GaussianHMM(n_components=2, covariance_type="full",
                            n_iter=self.n_iter, random_state=self.random_state)
        model.fit(r)
        self._model = model
        self._trend_state = int(np.argmax(np.abs(model.means_.ravel())))
        return self

    def label(self, closes: np.ndarray) -> np.ndarray:
        n = len(closes)
        if self._model is None:
            self.fit(closes)
        r = _log_returns(closes)[1:].reshape(-1, 1)
        states = self._model.predict(r)
        lab = (states == self._trend_state).astype(int)
        out = np.full(n, -1, dtype=int)
        out[1:] = lab                       # returns start at index 1
        return out
