"""The benchmark: is a regime signal worth anything, out of sample?

Design of the honest test
-------------------------
We do NOT ask "does the detector label bars correctly" (there is no ground-truth
regime in real markets). We ask the only question that matters economically:

    A fixed, trivial trading rule works in some conditions and bleeds in others.
    Does *gating that rule on the detector's regime call* earn more out-of-sample
    than running the rule all the time (or than gating on a baseline detector)?

The base rule is deliberately trivial, so any edge comes from the *gating*, not
from a clever strategy. Default is mean-reversion:

    reversion:  position = -sign(logreturn_t)      (fade the last bar)

The regime-gated strategy trades the rule only when the detector flags its
"active" regime (label 1 — for the irreversibility detector that's the
asymmetric / "elevator down snaps back" regime where fading pays) and stays flat
otherwise. Realized return at bar t is ``position_t * logreturn_{t+1}`` minus a
per-turn cost. Everything is causal; any detector threshold is fit on the
training fold only; evaluation is on purged walk-forward test folds. We report
the Deflated Sharpe Ratio so the comparison is corrected for how many strategies
we tried.

The controls tell you whether the regime call helped at all:
* ``always_on``   — the base rule every bar (no gating).
* ``always_flat`` — never trade (the do-nothing floor; Sharpe 0 by construction).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .evaluation.deflated_sharpe import deflated_sharpe_ratio
from .evaluation.metrics import hit_rate, max_drawdown, sharpe, turnover
from .evaluation.walkforward import assert_no_leakage, purged_walk_forward_splits

__all__ = ["StrategyResult", "run_benchmark", "base_rule_signal"]


def _log_returns(closes: np.ndarray) -> np.ndarray:
    c = np.asarray(closes, dtype=float)
    r = np.zeros_like(c)
    r[1:] = np.diff(np.log(np.clip(c, 1e-12, None)))
    return r


def base_rule_signal(closes: np.ndarray, rule: str = "reversion",
                     momentum_k: int = 12) -> np.ndarray:
    """The fixed base rule's position per bar, causal.

    ``reversion`` fades the last bar; ``momentum`` rides the last ``k`` bars.
    """
    c = np.asarray(closes, dtype=float)
    if rule == "reversion":
        return -np.sign(_log_returns(c))
    if rule == "momentum":
        sig = np.zeros(len(c))
        sig[momentum_k:] = np.sign(c[momentum_k:] - c[:-momentum_k])
        return sig
    raise ValueError(f"unknown rule {rule!r}")


@dataclass
class StrategyResult:
    name: str
    oos_returns: np.ndarray
    n_trials: int
    cost: float
    sharpe: float = 0.0
    dsr: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    turnover: float = 0.0
    extra: dict = field(default_factory=dict)

    def finalize(self) -> "StrategyResult":
        self.sharpe = sharpe(self.oos_returns)
        self.dsr = deflated_sharpe_ratio(self.oos_returns, self.n_trials)["dsr"]
        self.max_drawdown = max_drawdown(self.oos_returns)
        self.hit_rate = hit_rate(self.oos_returns)
        return self


def _oos_returns(closes, positions, folds, cost):
    """Concatenate per-fold OOS bar returns for a fixed position series."""
    r = _log_returns(closes)
    fwd = np.zeros_like(r)
    fwd[:-1] = r[1:]                    # return realized AFTER acting at bar t
    n = len(closes)
    rets, poss = [], []
    for f in folds:
        idx = f.test[f.test < n - 1]                # need t+1 to exist
        if len(idx) == 0:
            continue
        p = positions[idx]
        prev = np.concatenate([[0.0], p[:-1]])
        ret = p * fwd[idx] - cost * np.abs(p - prev)
        rets.append(ret)
        poss.append(p)
    if not rets:
        return np.array([]), np.array([])
    return np.concatenate(rets), np.concatenate(poss)


def run_benchmark(closes, detectors, *, rule="reversion", n_splits=5, embargo=32,
                  min_train=256, momentum_k=12, cost=0.0002):
    """Compare regime-gated strategies (one per detector) against the no-gating
    controls, out-of-sample.

    Parameters
    ----------
    closes : array of prices.
    detectors : dict[str, detector] — each implements ``label`` (and optionally
        ``fit``; if so it is fit on each training fold only).
    rule : "reversion" (default) or "momentum" — the fixed base rule being gated.
    Returns dict[name -> StrategyResult], including controls ``always_on`` and
    ``always_flat``.
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    folds = purged_walk_forward_splits(n, n_splits=n_splits, embargo=embargo,
                                       min_train=min_train)
    assert_no_leakage(folds, embargo=embargo)
    if not folds:
        raise ValueError("not enough data for the requested walk-forward config")

    base = base_rule_signal(closes, rule=rule, momentum_k=momentum_k)
    n_trials = len(detectors) + 2                   # +2 controls, for the DSR bar
    results: dict[str, StrategyResult] = {}

    # controls
    ret, pos = _oos_returns(closes, base, folds, cost)
    results["always_on"] = StrategyResult("always_on", ret, n_trials, cost)
    results["always_on"].turnover = turnover(pos)
    results["always_on"].finalize()

    ret0, _ = _oos_returns(closes, np.zeros(n), folds, cost)
    results["always_flat"] = StrategyResult("always_flat", ret0, n_trials, cost).finalize()

    # one regime-gated strategy per detector, fit per fold (no leakage)
    for name, det in detectors.items():
        fold_ret, fold_pos = [], []
        r = _log_returns(closes)
        fwd = np.zeros_like(r); fwd[:-1] = r[1:]
        for f in folds:
            fitter = getattr(det, "fit", None)
            if callable(fitter):
                det.fit(closes[f.train])            # threshold learned on train only
            labels = det.label(closes)              # causal labels
            gated = np.where(labels == 1, base, 0.0)
            idx = f.test[f.test < n - 1]
            if len(idx) == 0:
                continue
            p = gated[idx]
            prev = np.concatenate([[0.0], p[:-1]])
            fold_ret.append(p * fwd[idx] - cost * np.abs(p - prev))
            fold_pos.append(p)
        ret = np.concatenate(fold_ret) if fold_ret else np.array([])
        pos = np.concatenate(fold_pos) if fold_pos else np.array([])
        res = StrategyResult(name, ret, n_trials, cost)
        res.turnover = turnover(pos)
        results[name] = res.finalize()

    return results
